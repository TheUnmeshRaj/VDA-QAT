"""
QAT Fine-Tuning Loop — VDA log-disparity output, VideoDepthLoss, no clamp on pred.
"""

import os
import sys
import time
import math
import torch
import torch.nn as nn

# Add Video-Depth-Anything to sys.path to enable loading loss package
vda_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Video-Depth-Anything")
if vda_path not in sys.path:
    sys.path.insert(0, vda_path)

from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from loss.loss import VideoDepthLoss

VDA_CUDA_DEVICE = os.environ.get("VDA_CUDA_DEVICE", "1")
DEVICE          = torch.device(f"cuda:{VDA_CUDA_DEVICE}")
CKPT_DIR        = "/media/rvcse22/CSERV/vdaproj/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

EPOCHS       = 30
BATCH_SIZE   = 6
SEQ_LEN      = 4
LR_MAX       = 1e-7
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0
AMP_ENABLED  = True
SAVE_EVERY_N = 1
WARMUP_PCT   = 0.20
GRAD_ACCUM   = 3
MIN_DEPTH    = 0.1
MAX_DEPTH    = 80.0


# ── Loss ──────────────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vda_loss = VideoDepthLoss()

    def forward(self, pred_seq: torch.Tensor,
                gt_seq:   torch.Tensor) -> tuple[torch.Tensor, dict]:
        # pred_seq: [B, T, 1, H, W] raw log-disparity, NO clamp applied
        # gt_seq:   [B, T, 1, H, W] metric metres
        prediction = pred_seq.squeeze(2)          # [B, T, H, W]
        target     = gt_seq  .squeeze(2)          # [B, T, H, W]
        mask       = (target > MIN_DEPTH) & (target < MAX_DEPTH)   # bool

        if mask.sum() < 10:
            loss = pred_seq.sum() * 0.0
            return loss, {"spatial": 0.0, "temporal": 0.0}

        loss_dict  = self.vda_loss(prediction, target, mask)
        loss       = loss_dict["total_loss"]

        # Guard: if loss somehow lost grad (e.g. all-invalid mask batch)
        if not loss.requires_grad:
            loss = pred_seq.sum() * 0.0

        parts = {
            "spatial":  loss_dict.get("spatial_loss",  torch.tensor(0.0)).item(),
            "temporal": loss_dict.get("stable_loss",   torch.tensor(0.0)).item(),
        }
        return loss, parts


# ── Metrics (median scale-aligned, no grad needed) ────────────────────────────

@torch.no_grad()
def compute_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    # pred: [N, 1, H, W] log-disparity    gt: [N, 1, H, W] metric metres
    mask = (gt > MIN_DEPTH) & (gt < MAX_DEPTH)
    if mask.sum() < 10:
        return {}
    p_raw = pred[mask]
    g     = gt[mask]
    # median scale alignment — single scalar, always has grad=False here (no_grad context)
    scale = g.median() / p_raw.median().clamp(min=1e-8)
    p     = p_raw * scale
    thresh    = torch.max(p / g.clamp(1e-8), g / p.clamp(1e-8))
    d1        = (thresh < 1.25).float().mean().item()
    abs_rel   = ((p - g).abs() / g.clamp(1e-8)).mean().item()
    rmse      = (p - g).pow(2).mean().sqrt().item()
    rmse_log  = (
        torch.log(p.clamp(1e-3)) - torch.log(g.clamp(1e-3))
    ).pow(2).mean().sqrt().item()
    return {"abs_rel": abs_rel, "d1": d1, "rmse": rmse, "rmse_log": rmse_log}


# ── Pred normalisation ────────────────────────────────────────────────────────

def normalise_pred(pred: torch.Tensor) -> torch.Tensor:
    """[B,T,H,W] → [B,T,1,H,W]. No clamp — loss handles alignment."""
    if isinstance(pred, dict):
        pred = pred["depth"]
    if pred.dim() == 4:
        pred = pred.unsqueeze(2)
    return pred


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(quant_sim, optimizer, scheduler, epoch, metrics, tag=None):
    name = f"vda_qat_epoch{epoch:03d}.pt" if tag is None else f"vda_qat_{tag}.pt"
    path = os.path.join(CKPT_DIR, name)
    torch.save({
        "epoch"      : epoch,
        "model_state": quant_sim.model.state_dict(),
        "optimizer"  : optimizer.state_dict(),
        "scheduler"  : scheduler.state_dict(),
        "metrics"    : metrics,
    }, path)
    print(f"  [Ckpt] → {path}")


def load_checkpoint(quant_sim, optimizer, scheduler, path: str) -> int:
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    quant_sim.model.load_state_dict(ck["model_state"])
    if "optimizer" in ck:
        optimizer.load_state_dict(ck["optimizer"])
    else:
        print("[Resume] WARNING: no optimizer state in checkpoint")
    if "scheduler" in ck:
        scheduler.load_state_dict(ck["scheduler"])
    else:
        print("[Resume] WARNING: no scheduler state in checkpoint")
    print(f"[Resume] epoch={ck['epoch']}  metrics={ck.get('metrics', {})}")
    return ck["epoch"] + 1


# ── Collapse detector ─────────────────────────────────────────────────────────

def check_pred_health(pred: torch.Tensor, step: int, epoch: int):
    """Warn if prediction std collapses — early sign of flat-depth degenerate minimum."""
    with torch.no_grad():
        std = pred.std().item()
    if std < 0.05:
        print(f"  [COLLAPSE WARNING] Ep{epoch:03d} step{step:04d} "
              f"pred.std()={std:.4f} — model may be converging to flat depth")


# ── Training loop ─────────────────────────────────────────────────────────────

def train_qat(quant_sim, train_loader, val_loader,
              resume_ckpt: str | None = None):

    criterion = CombinedLoss().to(DEVICE)
    scaler    = GradScaler("cuda", enabled=AMP_ENABLED)

    optimizer = AdamW(
        [p for p in quant_sim.model.parameters() if p.requires_grad],
        lr=LR_MAX, weight_decay=WEIGHT_DECAY,
    )

    total_steps = EPOCHS * math.ceil(len(train_loader) / GRAD_ACCUM)
    warmup    = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=200)
    cosine    = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - 200), eta_min=1e-9)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[200])

    start_epoch = 0
    if resume_ckpt:
        start_epoch = load_checkpoint(quant_sim, optimizer, scheduler, resume_ckpt)

    best_abs_rel = float("inf")

    for epoch in range(start_epoch, EPOCHS):
        quant_sim.model.train()
        epoch_loss  = 0.0
        skipped     = 0
        t0          = time.time()
        optimizer.zero_grad()

        for step, (rgb, depth) in enumerate(train_loader):
            # Skip batches with no valid GT depth before touching GPU
            if ((depth > MIN_DEPTH) & (depth < MAX_DEPTH)).sum() < 10:
                skipped += 1
                continue

            rgb   = rgb  .to(DEVICE, non_blocking=True)
            depth = depth.to(DEVICE, non_blocking=True)

            with autocast("cuda", enabled=AMP_ENABLED):
                pred = normalise_pred(quant_sim.model(rgb))   # [B,T,1,H,W], no clamp
                loss, parts = criterion(pred, depth)
                loss = loss / GRAD_ACCUM

            # Hard guard — should never trigger with VideoDepthLoss but be safe
            if not loss.requires_grad:
                print(f"  [ERROR] Ep{epoch:03d} step{step:04d} loss.requires_grad=False — skipping")
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(quant_sim.model.parameters(), GRAD_CLIP)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # Only step scheduler when optimizer actually updated (no inf/nan)
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * GRAD_ACCUM

            if step % 50 == 0:
                check_pred_health(pred, step, epoch)
                print(f"  Ep{epoch:03d} step{step:04d}  "
                      f"loss={loss.item()*GRAD_ACCUM:.4f}  "
                      f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                      f"spatial={parts['spatial']:.4f}  "
                      f"temporal={parts['temporal']:.4f}  "
                      f"pred_std={pred.std().item():.3f}")

        if skipped:
            print(f"  [Info] {skipped} batches skipped (no valid GT depth)")

        # ── Validation ───────────────────────────────────────────────────────
        quant_sim.model.eval()
        val_metrics = {"abs_rel": 0.0, "d1": 0.0, "rmse": 0.0, "rmse_log": 0.0}
        n_val = 0

        with torch.no_grad():
            for rgb, depth in val_loader:
                rgb   = rgb  .to(DEVICE, non_blocking=True)
                depth = depth.to(DEVICE, non_blocking=True)
                with autocast("cuda", enabled=AMP_ENABLED):
                    pred = normalise_pred(quant_sim.model(rgb))
                m = compute_metrics(pred.flatten(0, 1), depth.flatten(0, 1))
                for k in val_metrics:
                    val_metrics[k] += m.get(k, 0.0)
                n_val += 1

        for k in val_metrics:
            val_metrics[k] /= max(n_val, 1)

        elapsed = time.time() - t0
        print(f"[Epoch {epoch:03d}] loss={epoch_loss/max(len(train_loader)-skipped,1):.4f}  "
              f"abs_rel={val_metrics['abs_rel']:.4f}  "
              f"d1={val_metrics['d1']:.4f}  "
              f"rmse={val_metrics['rmse']:.4f}  "
              f"rmse_log={val_metrics['rmse_log']:.4f}  "
              f"time={elapsed:.0f}s")

        if (epoch + 1) % SAVE_EVERY_N == 0:
            save_checkpoint(quant_sim, optimizer, scheduler, epoch, val_metrics)

        if val_metrics["abs_rel"] < best_abs_rel:
            best_abs_rel = val_metrics["abs_rel"]
            save_checkpoint(quant_sim, optimizer, scheduler, epoch, val_metrics, tag="best")
            print(f"  [Best] abs_rel={best_abs_rel:.4f}")

    print(f"[QAT] Done. Best abs_rel={best_abs_rel:.4f}")
    return quant_sim


if __name__ == "__main__":
    from model_patch      import build_patched_vda
    from aimet_qat_init   import build_quant_sim, calibrate_encodings
    from dataset_pipeline import build_loaders

    model = build_patched_vda()
    dummy = torch.randn(1, 2, 3, 392, 518, device=DEVICE)
    qsim  = build_quant_sim(model, dummy)
    train_loader, val_loader = build_loaders(batch_size=BATCH_SIZE, seq_len=SEQ_LEN, stride=12, num_workers=12)
    calibrate_encodings(qsim, train_loader, n_batches=64)
    train_qat(qsim, train_loader, val_loader, resume_ckpt=None)