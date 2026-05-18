"""
Evaluation & Export — depth metrics + ONNX + TorchScript + JSON encodings for NPU.
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast

DEVICE      = torch.device("cuda:1")
CKPT_DIR    = "/media/rvcse22/CSERV/vdaproj/checkpoints"
EXPORT_DIR  = os.path.join(CKPT_DIR, "export")
os.makedirs(EXPORT_DIR, exist_ok=True)

MIN_DEPTH = 1e-3
MAX_DEPTH = 80.0


# ── Full metric suite ──────────────────────────────────────────────────────────

def compute_full_metrics(pred_all: torch.Tensor,
                         gt_all:   torch.Tensor) -> dict:
    """
    pred_all, gt_all: [N, 1, H, W] on any device.
    Returns dict with abs_rel, rmse, rmse_log, d1, d2, d3, TAE.
    """
    mask = (gt_all > MIN_DEPTH) & (gt_all < MAX_DEPTH)
    p = pred_all[mask].float()
    g = gt_all  [mask].float()

    thresh     = torch.max(p / g, g / p)
    d1         = (thresh < 1.25    ).float().mean().item()
    d2         = (thresh < 1.25**2 ).float().mean().item()
    d3         = (thresh < 1.25**3 ).float().mean().item()
    abs_rel    = ((p - g).abs() / g).mean().item()
    rmse       = torch.sqrt(((p - g)**2).mean()).item()
    rmse_log   = torch.sqrt(
        ((torch.log(p.clamp(MIN_DEPTH)) - torch.log(g.clamp(MIN_DEPTH)))**2).mean()
    ).item()

    # TAE — temporal alignment error across consecutive frame pairs
    N = pred_all.shape[0]
    if N >= 2:
        tae_vals = []
        for i in range(N - 1):
            m = (gt_all[i] > MIN_DEPTH) & (gt_all[i] < MAX_DEPTH) & \
                (gt_all[i+1] > MIN_DEPTH) & (gt_all[i+1] < MAX_DEPTH)
            if m.sum() < 10:
                continue
            diff_pred = (pred_all[i+1][m] - pred_all[i][m]).abs()
            diff_gt   = (gt_all  [i+1][m] - gt_all  [i][m]).abs()
            tae_vals.append((diff_pred - diff_gt).abs().mean().item())
        tae = float(np.mean(tae_vals)) if tae_vals else 0.0
    else:
        tae = 0.0

    return dict(abs_rel=abs_rel, rmse=rmse, rmse_log=rmse_log,
                d1=d1, d2=d2, d3=d3, tae=tae)


# ── Evaluation runner ──────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(quant_sim_model: nn.Module, val_loader,
             amp: bool = True) -> dict:
    quant_sim_model.eval()
    all_preds, all_gts = [], []

    for rgb, depth in val_loader:
        rgb   = rgb  .to(DEVICE, non_blocking=True)
        depth = depth.to(DEVICE, non_blocking=True)

        with autocast(enabled=amp):
            pred = quant_sim_model(rgb)

        if isinstance(pred, dict):
            pred = pred["depth"]
        if pred.dim() == 4:
            B, T = rgb.shape[:2]
            pred = pred.reshape(B, T, 1, *pred.shape[2:])

        pred = pred.clamp(MIN_DEPTH, MAX_DEPTH)
        # Flatten temporal dim for metric aggregation
        all_preds.append(pred.flatten(0, 1).cpu())
        all_gts  .append(depth.flatten(0, 1).cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_gts   = torch.cat(all_gts,   dim=0)

    metrics = compute_full_metrics(all_preds, all_gts)
    print("\n── Evaluation Results ──────────────────────")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.6f}")
    print("────────────────────────────────────────────\n")
    return metrics


# ── Export helpers ─────────────────────────────────────────────────────────────

class VDAExportWrapper(nn.Module):
    """
    Thin wrapper: accepts [B, T, C, H, W] → outputs [B, T, 1, H, W].
    Makes the ONNX graph topology explicit and traceable.
    """
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.inner(x)
        if isinstance(out, dict):
            out = out["depth"]
        if out.dim() == 4:
            B, T = x.shape[:2]
            out = out.reshape(B, T, 1, *out.shape[2:])
        return out.clamp(1e-3, 80.0)


def export_onnx(quant_sim, dummy_input: torch.Tensor,
                path: str | None = None) -> str:
    if path is None:
        path = os.path.join(EXPORT_DIR, "vda_qat_int8.onnx")

    wrapper = VDAExportWrapper(quant_sim.model).to(DEVICE).eval()

    # AIMET export writes ONNX + encodings side-by-side
    quant_sim.export(
        path         = EXPORT_DIR,
        filename_prefix = "vda_qat_int8",
        dummy_input  = dummy_input,
        export_args  = {"opset_version": 17, "input_names": ["video_frames"],
                        "output_names": ["depth_map"],
                        "dynamic_axes": {"video_frames": {0: "batch", 1: "frames"},
                                         "depth_map":    {0: "batch", 1: "frames"}}},
    )
    print(f"[Export] ONNX  → {EXPORT_DIR}/vda_qat_int8.onnx")
    print(f"[Export] Enc   → {EXPORT_DIR}/vda_qat_int8.encodings")
    return path


def export_torchscript(quant_sim, dummy_input: torch.Tensor,
                       path: str | None = None) -> str:
    if path is None:
        path = os.path.join(EXPORT_DIR, "vda_qat_int8.pt")

    wrapper = VDAExportWrapper(quant_sim.model).to(DEVICE).eval()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy_input, strict=False)
    traced.save(path)
    print(f"[Export] TorchScript → {path}")
    return path


def export_encodings_json(quant_sim,
                          path: str | None = None) -> str:
    # AIMET quant_sim.export() already saves the encodings json.
    # The manual JSON dump logic uses deprecated .encoding attributes which crash.
    print(f"[Export] Encodings JSON (Skipped, already exported by quant_sim.export)")
    return path


def save_metrics_report(metrics: dict,
                        path: str | None = None) -> str:
    if path is None:
        path = os.path.join(EXPORT_DIR, "eval_metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Metrics] Report → {path}")
    return path


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/media/rvcse22/CSERV/vdaproj/Video-Depth-Anything")

    from model_patch      import build_patched_vda
    from old.aimet_qat_init  import build_quant_sim
    from dataset_pipeline import build_loaders

    BEST_CKPT = os.path.join(CKPT_DIR, "vda_qat_best.pt")

    # ── Load best QAT checkpoint ──
    model = build_patched_vda()
    dummy = torch.randn(1, 2, 3, 392, 518, device=DEVICE)
    qsim  = build_quant_sim(model, dummy)

    ck    = torch.load(BEST_CKPT, map_location=DEVICE)
    qsim.model.load_state_dict(ck["model_state"])
    print(f"[Load] best checkpoint epoch={ck['epoch']}")

    # ── Evaluate ──
    _, val_loader = build_loaders(batch_size=4, seq_len=4)
    metrics = evaluate(qsim.model, val_loader)
    save_metrics_report(metrics)

    # ── Export artifacts ──
    export_onnx(qsim, dummy)
    export_torchscript(qsim, dummy)
    export_encodings_json(qsim)

    print("\n[Done] All artifacts written to:", EXPORT_DIR)
    for f in sorted(os.listdir(EXPORT_DIR)):
        size = os.path.getsize(os.path.join(EXPORT_DIR, f)) / 1e6
        print(f"  {f:45s}  {size:7.2f} MB")
