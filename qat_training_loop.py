"""
QAT Fine-Tuning Loop — optimised for 40 GB A100 VRAM.
Scale-invariant gradient + SILog + edge-aware loss.
Mixed precision via torch.cuda.amp (FP16 AMP on top of AIMET QAT).
"""

import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.cuda.amp.grad_scaler import OptState
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

DEVICE      = torch.device("cuda:1")
CKPT_DIR    = "/media/rvcse22/CSERV/vdaproj/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Training hyper-parameters ─────────────────────────────────────────────────
EPOCHS          = 30
BATCH_SIZE      = 6          # fits comfortably in 40 GB with seq_len=4, 392×518
SEQ_LEN         = 4
LR_MAX          = 1e-6       # QAT fine-tune: conservative but workable; 3e-7 was too small
WEIGHT_DECAY    = 1e-4
GRAD_CLIP       = 1.0
AMP_ENABLED     = True
SAVE_EVERY_N    = 1          # checkpoint every N epochs
WARMUP_PCT      = 0.05
GRAD_ACCUM      = 2          # effective batch = BATCH_SIZE * GRAD_ACCUM = 12
MIN_DEPTH       = 1e-3
MAX_DEPTH       = 80.0


# ── Loss functions ─────────────────────────────────────────────────────────────

class SILogLoss(nn.Module):
    """Scale-Invariant Log loss — standard for monocular depth."""
    def __init__(self, lam: float = 0.5):
        super().__init__()
        self.lam = lam

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        mask = (gt > MIN_DEPTH) & (gt < MAX_DEPTH)
        if mask.sum() < 10:
            return pred[mask].sum() * 0.0
        d    = torch.log(pred[mask].clamp(MIN_DEPTH)) - torch.log(gt[mask].clamp(MIN_DEPTH))
        loss = d.pow(2).mean() - self.lam * d.mean().pow(2)
        return loss


class EdgeAwareLoss(nn.Module):
    """Gradient-based smoothness preserving temporal edges."""
    def forward(self, pred: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        # pred/img: [B*T, 1, H, W] and [B*T, 3, H, W]
        pred_dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        pred_dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        img_dx  = torch.abs(img[:, :, :, :-1]  - img[:, :, :, 1:] ).mean(1, keepdim=True)
        img_dy  = torch.abs(img[:, :, :-1, :]  - img[:, :, 1:, :] ).mean(1, keepdim=True)
        loss    = (pred_dx * torch.exp(-img_dx)).mean() + \
                  (pred_dy * torch.exp(-img_dy)).mean()
        return loss


class TemporalConsistencyLoss(nn.Module):
    """Penalise frame-to-frame depth jumps (TAE proxy)."""
    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        # pred: [B, T, 1, H, W]
        if pred.size(1) < 2:
            return pred.sum() * 0.0
        diff = pred[:, 1:] - pred[:, :-1]
        return diff.abs().mean()


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.silog   = SILogLoss(lam=0.5)
        self.edge    = EdgeAwareLoss()
        self.temporal = TemporalConsistencyLoss()

    @staticmethod
    def _align_to_gt(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Least-squares scale+shift alignment of relative `pred` to metric `gt`.
        VDA outputs affine-invariant depth — it has no absolute scale.
        Without this step, SILog is forced to fight both quantization noise
        AND an impossible absolute-scale constraint, collapsing the network.

        Solves:  pred_aligned = scale * pred + shift
        where (scale, shift) = argmin ||scale*pred + shift - gt||^2  over valid pixels.

        IMPORTANT: scale and shift are computed with torch.no_grad() so they are
        treated as *constants* during backprop. Keeping them in the graph creates
        a self-referential gradient (d(loss)/d(pred) involves d(scale)/d(pred))
        which is ill-conditioned and causes the loss to increase.
        """
        with torch.no_grad():
            mask = (gt > MIN_DEPTH) & (gt < MAX_DEPTH) & (pred > 0)
            if mask.sum() < 10:
                return pred  # not enough pixels — return as-is

            p = pred[mask].float()  # [N]
            g = gt[mask].float()    # [N]

            # closed-form least-squares: [scale, shift] = (A^T A)^{-1} A^T g
            # A = [p, 1]  →  A^T A = [[sum(p^2), sum(p)], [sum(p), N]]
            n    = torch.tensor(p.numel(), dtype=p.dtype, device=p.device)
            sp   = p.sum()
            sp2  = (p * p).sum()
            sg   = g.sum()
            spg  = (p * g).sum()
            det  = sp2 * n - sp * sp
            if det.abs() < 1e-6:
                # Degenerate: only shift
                shift = g.mean() - p.mean()
                return (pred + shift).clamp(MIN_DEPTH, MAX_DEPTH)
            scale = (spg * n - sp * sg) / det
            shift = (sg - scale * sp) / n

        # Apply alignment — gradient only flows through `pred`, scale/shift are detached
        return (scale * pred + shift).clamp(MIN_DEPTH, MAX_DEPTH)

    def forward(self, pred_seq, gt_seq, rgb_seq):
        # shapes: [B, T, 1, H, W], [B, T, 1, H, W], [B, T, 3, H, W]
        B, T = pred_seq.shape[:2]
        pred_flat = pred_seq.reshape(B * T, 1, *pred_seq.shape[3:])
        gt_flat   = gt_seq  .reshape(B * T, 1, *gt_seq  .shape[3:])
        rgb_flat  = rgb_seq .reshape(B * T, 3, *rgb_seq .shape[3:])

        # Align relative predictions to metric GT before loss computation.
        # This is the KEY step — without it, SILog tries to enforce absolute
        # metric scale on an affine-invariant model and destroys the network.
        pred_aligned = self._align_to_gt(pred_flat, gt_flat)
        # Also align the sequence-shaped tensor for temporal loss
        pred_seq_aligned = pred_aligned.reshape(B, T, 1, *pred_seq.shape[3:])

        l_silog    = self.silog(pred_aligned, gt_flat)
        l_edge     = self.edge(pred_aligned, rgb_flat)
        l_temporal = self.temporal(pred_seq_aligned)

        loss = l_silog + 0.1 * l_edge + 0.05 * l_temporal
        return loss, {"silog": l_silog.item(), "edge": l_edge.item(),
                      "temporal": l_temporal.item()}


# ── Metric helpers ─────────────────────────────────────────────────────────────

def compute_metrics(pred: torch.Tensor, gt: torch.Tensor):
    mask = (gt > MIN_DEPTH) & (gt < MAX_DEPTH)
    if mask.sum() < 10:
        return {}
    p, g = pred[mask], gt[mask]
    thresh   = torch.max(p / g, g / p)
    d1       = (thresh < 1.25   ).float().mean().item()
    abs_rel  = ((p - g).abs() / g).mean().item()
    rmse     = torch.sqrt(((p - g) ** 2).mean()).item()
    rmse_log = torch.sqrt(
        ((torch.log(p.clamp(MIN_DEPTH)) - torch.log(g.clamp(MIN_DEPTH))) ** 2).mean()
    ).item()
    return {"abs_rel": abs_rel, "d1": d1, "rmse": rmse, "rmse_log": rmse_log}


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_checkpoint(quant_sim, optimizer, scheduler, epoch, metrics):
    path = os.path.join(CKPT_DIR, f"vda_qat_epoch{epoch:03d}.pt")
    torch.save({
        "epoch"         : epoch,
        "model_state"   : quant_sim.model.state_dict(),
        "optimizer"     : optimizer.state_dict(),
        "scheduler"     : scheduler.state_dict(),
        "metrics"       : metrics,
    }, path)
    print(f"  [Ckpt] saved → {path}")


def load_checkpoint(quant_sim, optimizer, scheduler, path: str) -> int:
    ck = torch.load(path, map_location=DEVICE)
    quant_sim.model.load_state_dict(ck["model_state"])
    optimizer.load_state_dict(ck["optimizer"])
    scheduler.load_state_dict(ck["scheduler"])
    print(f"[Resume] epoch {ck['epoch']}  metrics={ck.get('metrics', {})}")
    return ck["epoch"] + 1


# ── Training loop ──────────────────────────────────────────────────────────────

def train_qat(quant_sim, train_loader, val_loader,
              resume_ckpt: str | None = None):

    criterion = CombinedLoss().to(DEVICE)
    scaler    = GradScaler(enabled=AMP_ENABLED)

    # Freeze backbone BN / non-quantized params; only fine-tune quantized weights
    optimizer = AdamW(
        [p for p in quant_sim.model.parameters() if p.requires_grad],
        lr=LR_MAX, weight_decay=WEIGHT_DECAY,
    )

    total_steps = EPOCHS * math.ceil(len(train_loader) / GRAD_ACCUM)
    scheduler   = OneCycleLR(
        optimizer, max_lr=LR_MAX, total_steps=total_steps,
        pct_start=WARMUP_PCT, anneal_strategy="cos", div_factor=25,
        final_div_factor=1e4,
    )

    start_epoch = 0
    if resume_ckpt:
        start_epoch = load_checkpoint(quant_sim, optimizer, scheduler, resume_ckpt)

    best_abs_rel = float("inf")

    for epoch in range(start_epoch, EPOCHS):
        quant_sim.model.train()
        epoch_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad()

        for step, (rgb, depth) in enumerate(train_loader):
            rgb   = rgb  .to(DEVICE, non_blocking=True)   # [B, T, 3, H, W]
            depth = depth.to(DEVICE, non_blocking=True)   # [B, T, 1, H, W]

            with autocast(device_type="cuda", enabled=AMP_ENABLED):
                pred = quant_sim.model(rgb)               # [B, T, H, W] or [B*T, 1, H, W]

                # VDA may return dict or raw tensor — normalise
                if isinstance(pred, dict):
                    pred = pred["depth"]
                if pred.dim() == 4 and pred.shape[0] == rgb.shape[0]:
                    # Shape is [B, T, H, W] — add channel dim
                    pred = pred.unsqueeze(2)              # [B, T, 1, H, W]
                elif pred.dim() == 4:
                    # Shape is [B*T, 1, H, W]
                    B, T = rgb.shape[:2]
                    pred = pred.reshape(B, T, 1, *pred.shape[2:])

                # VDA outputs affine-invariant (relative) depth — keep raw for loss;
                # the CombinedLoss will align scale+shift internally.
                pred = pred.clamp(MIN_DEPTH, MAX_DEPTH)

                loss, loss_parts = criterion(pred, depth, rgb)
                loss = loss / GRAD_ACCUM

            if loss.requires_grad:
                scaler.scale(loss).backward()
            else:
                print(f"  [Warning] Ep{epoch:03d} step{step:04d} loss does not require grad (no valid GT pixels). Skipping backward.")
                continue

            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(quant_sim.model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                if scaler._per_optimizer_states[id(optimizer)]["stage"] == OptState.STEPPED:
                    optimizer._opt_called = True
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * GRAD_ACCUM
            if step % 50 == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"  Ep{epoch:03d} step{step:04d}  "
                      f"loss={loss.item()*GRAD_ACCUM:.4f}  lr={lr_now:.2e}  "
                      f"silog={loss_parts['silog']:.4f}  "
                      f"temp={loss_parts['temporal']:.4f}")

        # ── Validation ──────────────────────────────────────────────────────
        quant_sim.model.eval()
        val_metrics = {"abs_rel": 0, "d1": 0, "rmse": 0, "rmse_log": 0}
        n_val = 0
        with torch.no_grad():
            for rgb, depth in val_loader:
                rgb   = rgb  .to(DEVICE, non_blocking=True)
                depth = depth.to(DEVICE, non_blocking=True)
                with autocast(device_type="cuda", enabled=AMP_ENABLED):
                    pred = quant_sim.model(rgb)
                    if isinstance(pred, dict):
                        pred = pred["depth"]
                    if pred.dim() == 4 and pred.shape[0] == rgb.shape[0]:
                        pred = pred.unsqueeze(2)          # [B, T, 1, H, W]
                    elif pred.dim() == 4:
                        B, T = rgb.shape[:2]
                        pred = pred.reshape(B, T, 1, *pred.shape[2:])
                    pred = pred.clamp(MIN_DEPTH, MAX_DEPTH)
                    # Align relative depth to metric GT for evaluation metrics
                    pred_flat = pred.flatten(0, 1)        # [B*T, 1, H, W]
                    depth_flat = depth.flatten(0, 1)      # [B*T, 1, H, W]
                    pred_flat = criterion._align_to_gt(pred_flat, depth_flat)
                m = compute_metrics(pred_flat, depth_flat)
                for k in val_metrics:
                    val_metrics[k] += m.get(k, 0)
                n_val += 1

        for k in val_metrics:
            val_metrics[k] /= max(n_val, 1)

        elapsed = time.time() - t0
        print(f"[Epoch {epoch:03d}] loss={epoch_loss/len(train_loader):.4f}  "
              f"abs_rel={val_metrics['abs_rel']:.4f}  "
              f"d1={val_metrics['d1']:.4f}  "
              f"rmse={val_metrics['rmse']:.4f}  "
              f"rmse_log={val_metrics['rmse_log']:.4f}  "
              f"time={elapsed:.0f}s")

        if (epoch + 1) % SAVE_EVERY_N == 0:
            save_checkpoint(quant_sim, optimizer, scheduler, epoch, val_metrics)

        if val_metrics["abs_rel"] < best_abs_rel:
            best_abs_rel = val_metrics["abs_rel"]
            best_path    = os.path.join(CKPT_DIR, "vda_qat_best.pt")
            torch.save({"epoch": epoch, "model_state": quant_sim.model.state_dict(),
                        "metrics": val_metrics}, best_path)
            print(f"  [Best] abs_rel={best_abs_rel:.4f}  saved → {best_path}")

    print(f"[QAT] Training complete. Best abs_rel = {best_abs_rel:.4f}")
    return quant_sim


if __name__ == "__main__":
    from model_patch    import build_patched_vda
    from old.aimet_qat_init import build_quant_sim, calibrate_encodings
    from dataset_pipeline import build_loaders

    model        = build_patched_vda()
    dummy        = torch.randn(1, 2, 3, 392, 518, device=DEVICE)
    qsim         = build_quant_sim(model, dummy)
    train_loader, val_loader = build_loaders(batch_size=BATCH_SIZE, seq_len=SEQ_LEN)

    calibrate_encodings(qsim, train_loader, n_batches=64)

    # To resume: pass resume_ckpt="path/to/checkpoint.pt"
    train_qat(qsim, train_loader, val_loader, resume_ckpt=None)