"""
video_infer.py — High-performance video inference pipeline for QAT-trained VDA.
Accepts any custom video (.mp4, .avi, .mov), extracts frames, estimates depth,
colorizes them, and compiles them back into a premium H.264 depth video.

Usage:
    python video_infer.py --video my_input.mp4 --ckpt checkpoints/vda_qat_best.pt
"""

import argparse
import os
import sys
import tempfile
import subprocess
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

DEVICE   = torch.device("cuda:0")
VDA_REPO = "/media/rvcse22/CSERV/vdaproj/Video-Depth-Anything"
IMG_H, IMG_W = 392, 518
SEQ_LEN = 4

sys.path.insert(0, VDA_REPO)
sys.path.insert(0, os.path.dirname(__file__))

rgb_transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

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

def load_model(ckpt_path: str, no_quant: bool = False):
    import model_patch
    model_patch.DEVICE = DEVICE          
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
        print(f"[Infer] Checkpoint loaded: epoch={epoch}")
    else:
        print("[Infer] No checkpoint found — calibrating activation encodings dynamically ...")
        from dataset_pipeline import build_loaders
        from aimet_qat_init import calibrate_encodings
        train_loader, _ = build_loaders(batch_size=2)
        calibrate_encodings(qsim, train_loader, n_batches=32)

    qsim.model.eval()
    return qsim.model

def process_video(model, video_path: str, output_path: str, side_by_side: bool = True):
    if not os.path.isfile(video_path):
        print(f"[Infer] ERROR: Input video file '{video_path}' does not exist.")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Infer] Opened video: {video_path} | FPS: {fps} | Total Frames: {total_frames}")

    temp_dir = tempfile.mkdtemp(prefix="vda_video_infer_")
    print(f"[Infer] Extracting and processing frames ...")

    frame_idx = 0
    frame_buffer = []
    orig_buffer = []
    depth_frames_saved = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # cv2 reads as BGR, convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        orig_buffer.append(pil_img)
        frame_buffer.append(rgb_transform(pil_img))
        
        # When we hit SEQ_LEN, run batch inference
        if len(frame_buffer) == SEQ_LEN:
            rgb_batch = torch.stack(frame_buffer).unsqueeze(0).to(DEVICE)  # [1, T, 3, H, W]

            with torch.no_grad():
                out = model(rgb_batch)

            if isinstance(out, dict):
                depth = out.get("depth", out.get("pred", next(iter(out.values()))))
            else:
                depth = out

            depth = depth.float()
            if depth.dim() == 5:
                depth = depth.squeeze(0).squeeze(1)
            elif depth.dim() == 4 and depth.shape[0] == 1:
                depth = depth.squeeze(0)
            elif depth.dim() == 4:
                depth = depth.squeeze(1).reshape(-1, rgb_batch.shape[3], rgb_batch.shape[4])

            depth_np = depth.cpu().numpy()  # [T, H, W]

            # Save frames
            for t, (orig_pil, d_map) in enumerate(zip(orig_buffer, depth_np)):
                depth_color_np = colorize_depth(d_map)
                
                if side_by_side:
                    # Resize original to match depth dimensions
                    orig_resized = orig_pil.resize((IMG_W, IMG_H))
                    composite = Image.new("RGB", (IMG_W * 2, IMG_H))
                    composite.paste(orig_resized, (0, 0))
                    composite.paste(Image.fromarray(depth_color_np), (IMG_W, 0))
                    out_frame_path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.png")
                    composite.save(out_frame_path)
                else:
                    out_frame_path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.png")
                    Image.fromarray(depth_color_np).save(out_frame_path)

                depth_frames_saved.append(out_frame_path)
                frame_idx += 1

            frame_buffer = []
            orig_buffer = []
            print(f"  Processed {frame_idx}/{total_frames} frames ...", end="\r")

    # Process remaining frames in buffer if any
    if len(frame_buffer) > 0:
        # Pad buffer to SEQ_LEN using duplicates of the last frame
        while len(frame_buffer) < SEQ_LEN:
            frame_buffer.append(frame_buffer[-1])
            orig_buffer.append(orig_buffer[-1])

        rgb_batch = torch.stack(frame_buffer).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(rgb_batch)
        if isinstance(out, dict):
            depth = out.get("depth", out.get("pred", next(iter(out.values()))))
        else:
            depth = out
        depth = depth.float()
        if depth.dim() == 5:
            depth = depth.squeeze(0).squeeze(1)
        elif depth.dim() == 4 and depth.shape[0] == 1:
            depth = depth.squeeze(0)
        elif depth.dim() == 4:
            depth = depth.squeeze(1).reshape(-1, rgb_batch.shape[3], rgb_batch.shape[4])

        depth_np = depth.cpu().numpy()

        for t, (orig_pil, d_map) in enumerate(zip(orig_buffer, depth_np)):
            if frame_idx >= total_frames:
                break  # don't save padded frames past total count
            depth_color_np = colorize_depth(d_map)
            
            if side_by_side:
                orig_resized = orig_pil.resize((IMG_W, IMG_H))
                composite = Image.new("RGB", (IMG_W * 2, IMG_H))
                composite.paste(orig_resized, (0, 0))
                composite.paste(Image.fromarray(depth_color_np), (IMG_W, 0))
                out_frame_path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.png")
                composite.save(out_frame_path)
            else:
                out_frame_path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.png")
                Image.fromarray(depth_color_np).save(out_frame_path)

            depth_frames_saved.append(out_frame_path)
            frame_idx += 1

    cap.release()
    print(f"\n[Infer] All {frame_idx} frames processed successfully.")

    # Compile frames back to video using ffmpeg
    print(f"[Infer] Compiling frames to final H.264 video '{output_path}' using FFmpeg ...")
    
    input_pattern = os.path.join(temp_dir, "frame_%06d.png")
    
    # ffmpeg command to compile H.264 mp4
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", input_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "18",
        output_path
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"[Infer] SUCCESS! Saved video depth results to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"[Infer] ERROR: FFmpeg compilation failed. Error log:\n{e.stderr.decode()}")
    finally:
        # Clean up temporary frames
        for fp in depth_frames_saved:
            if os.path.exists(fp):
                os.remove(fp)
        os.rmdir(temp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",   required=True, help="Path to input video file (.mp4, .avi, etc.)")
    parser.add_argument("--ckpt",    default="checkpoints/vda_qat_best.pt", help="Path to QAT checkpoint (.pt)")
    parser.add_argument("--output",  default=None, help="Path to save output video (defaults to custom suffix based on mode)")
    parser.add_argument("--no-quant", action="store_true", help="Run original float baseline")
    parser.add_argument("--single",   action="store_true", help="Only output the depth map, without side-by-side RGB")
    args = parser.parse_args()

    if args.output is None:
        if args.no_quant:
            args.output = "checkpoints/depth_video_output_float.mp4"
        elif args.ckpt == "":
            args.output = "checkpoints/depth_video_output_calib.mp4"
        else:
            args.output = "checkpoints/depth_video_output_qat.mp4"

    model = load_model(args.ckpt, no_quant=args.no_quant)
    process_video(model, args.video, args.output, side_by_side=not args.single)
