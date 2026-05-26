"""
infer.py — Quick inference test for the QAT-trained VDA model.
Loads vda_qat_best.pt, runs depth estimation on a sample clip,
and saves colorized depth map images to checkpoints/infer_output/.

Usage:
    python infer.py
    python infer.py --ckpt checkpoints/vda_qat_best.pt --n_clips 3
"""

import argparse
import os
import sys
import glob
import re

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

DEVICE   = torch.device("cuda:0")
VDA_REPO = "/media/rvcse22/CSERV/vdaproj/Video-Depth-Anything"
RGB_ROOT = "/media/rvcse22/CSERV/vdaproj/dataset/vkitti_2.0.3_rgb-001"
OUT_DIR  = "/media/rvcse22/CSERV/vdaproj/checkpoints/infer_output"
IMG_H, IMG_W = 392, 518
SEQ_LEN = 4

sys.path.insert(0, VDA_REPO)
sys.path.insert(0, os.path.dirname(__file__))

# ── helpers ───────────────────────────────────────────────────────────────────

def _sorted_frames(directory):
    paths = glob.glob(os.path.join(directory, "*.jpg")) + \
            glob.glob(os.path.join(directory, "*.png"))
    paths.sort(key=lambda p: [int(t) if t.isdigit() else t
                               for t in re.split(r"(\d+)", p)])
    return paths


def colorize_depth(depth_np: np.ndarray) -> np.ndarray:
    """Map a 2-D depth array to a plasma colormap RGB image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    d = depth_np.copy()
    d = np.clip(d, 0, 80)
    valid = d > 0
    if valid.any():
        d_min, d_max = d[valid].min(), d[valid].max()
        d = (d - d_min) / (d_max - d_min + 1e-8)
    colored = cm.plasma(d)[:, :, :3]          # H×W×3, float64 [0,1]
    return (colored * 255).astype(np.uint8)


rgb_transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, no_quant: bool = False):
    import model_patch
    model_patch.DEVICE = DEVICE          
    # override hardcoded cuda:1
    from model_patch import build_patched_vda

    print(f"[Infer] Building patched VDA model on {DEVICE} ...")
    model = build_patched_vda(verify=False)

    if no_quant:
        print("[Infer] Running in FLOAT mode (Original Pretrained Baseline)")
        model.eval()
        return model

    from aimet_qat_init import build_quant_sim
    print(f"[Infer] Initializing QuantSim wrapper for loading QAT checkpoint ...")
    dummy = torch.randn(1, 2, 3, 392, 518, device=DEVICE)
    qsim = build_quant_sim(model, dummy)

    if ckpt_path and os.path.isfile(ckpt_path):
        print(f"[Infer] Loading QAT weights from {ckpt_path}")
        ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        qsim.model.load_state_dict(ck["model_state"])
        epoch = ck.get("epoch", "?")
        metrics = ck.get("metrics", {})
        print(f"[Infer] Checkpoint loaded: epoch={epoch}  abs_rel={metrics.get('abs_rel', 'N/A'):.4f}" if metrics else f"[Infer] Checkpoint loaded: epoch={epoch}")
    else:
        print("[Infer] No checkpoint found — calibrating activation encodings dynamically ...")
        from dataset_pipeline import build_loaders
        from aimet_qat_init import calibrate_encodings
        train_loader, _ = build_loaders(batch_size=2)
        calibrate_encodings(qsim, train_loader, n_batches=32)

    qsim.model.eval()
    return qsim.model


def run_inference(model, n_clips: int = 3, out_dir: str = OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    # Collect some clip directories
    all_rgb_dirs = sorted(glob.glob(
        os.path.join(RGB_ROOT, "*", "*", "frames", "rgb", "Camera_*")
    ))
    if not all_rgb_dirs:
        print(f"[Infer] ERROR: No RGB directories found under {RGB_ROOT}")
        return

    clip_count = 0
    for rgb_dir in all_rgb_dirs:
        scene_tag = os.path.relpath(rgb_dir, RGB_ROOT).replace(os.sep, "_")
        if scene_tag != "Scene01_15-deg-left_frames_rgb_Camera_0":
            continue

        frames = _sorted_frames(rgb_dir)
        if len(frames) < SEQ_LEN:
            continue

        # Take the first SEQ_LEN frames of this directory
        clip_frames = frames[:SEQ_LEN]

        print(f"\n[Infer] Target Clip: {scene_tag}")

        # Load and preprocess frames
        rgb_tensors = []
        original_frames = []
        for fp in clip_frames:
            img = Image.open(fp).convert("RGB")
            original_frames.append(img)
            rgb_tensors.append(rgb_transform(img))

        rgb_batch = torch.stack(rgb_tensors).unsqueeze(0).to(DEVICE)  # [1,T,3,H,W]

        with torch.no_grad():
            out = model(rgb_batch)

        # Handle dict or tensor output
        if isinstance(out, dict):
            depth = out.get("depth", out.get("pred", next(iter(out.values()))))
        else:
            depth = out

        # VDA forward() returns [B, T, H, W] (no channel dim).
        # After QAT wrapping it may also be [B, T, 1, H, W] or [B*T, 1, H, W].
        depth = depth.float()
        if depth.dim() == 5:                         # [B, T, 1, H, W]
            depth = depth.squeeze(0).squeeze(1)      # → [T, H, W]
        elif depth.dim() == 4 and depth.shape[0] == 1:
            depth = depth.squeeze(0)                 # [1, T, H, W] → [T, H, W]
        elif depth.dim() == 4:                       # [B*T, 1, H, W]
            depth = depth.squeeze(1).reshape(-1, rgb_batch.shape[3], rgb_batch.shape[4])
        # depth is now [T, H, W]

        depth_np = depth.cpu().numpy()               # [T, H, W]
        print(f"  [Stats] depth min={depth_np.min():.4f}  max={depth_np.max():.4f}  "
              f"mean={depth_np.mean():.4f}  shape={depth_np.shape}")

        # Save side-by-side RGB | Depth for frame 00 only
        for t, (frame_img, d_map) in enumerate(zip(original_frames, depth_np)):
            if t != 0:
                continue
            frame_rgb = frame_img.resize((IMG_W, IMG_H))
            depth_color = Image.fromarray(colorize_depth(d_map))

            # Side-by-side composite
            composite = Image.new("RGB", (IMG_W * 2, IMG_H))
            composite.paste(frame_rgb,   (0, 0))
            composite.paste(depth_color, (IMG_W, 0))

            out_name = f"clip{clip_count:02d}_frame{t:02d}_{scene_tag[:40]}.png"
            out_path = os.path.join(out_dir, out_name)
            composite.save(out_path)
            print(f"  [Saved] {out_path}")

        clip_count += 1
        break # Only processing our single target clip

    print(f"\n[Infer] Done! Single target frame saved to {out_dir}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",    default="checkpoints/vda_qat_best.pt",
                        help="Path to checkpoint (.pt)")
    parser.add_argument("--n_clips", type=int, default=3,
                        help="Number of clips to run inference on")
    parser.add_argument("--no-quant", action="store_true",
                        help="Run original float model without quantization (pretrained baseline)")
    args = parser.parse_args()

    model = load_model(args.ckpt, no_quant=args.no_quant)
    out_dir = "checkpoints/infer_output_float" if args.no_quant else "checkpoints/infer_output_qat/new"
    run_inference(model, n_clips=args.n_clips, out_dir=out_dir)
