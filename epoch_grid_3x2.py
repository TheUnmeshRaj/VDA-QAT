#!/usr/bin/env python3
"""
epoch_grid_3x2.py — Multi-Epoch Visual Comparison Grid Compiler.

Compiles a premium 3x2 stacked layout comparing 6 different model states over a test video:
  +--------------------------------+--------------------------------+--------------------------------+
  |  1. Pretrained Baseline (FP32) |  2. QAT Epoch {start}          |  3. QAT Epoch {start + 1}      |
  +--------------------------------+--------------------------------+--------------------------------+
  |  4. QAT Epoch {start + 2}      |  5. QAT Epoch {end}            |  6. Fine-tuned QAT Best        |
  +--------------------------------+--------------------------------+--------------------------------+

Dynamic Skip-If-Exists:
  Individual panels already created in the output directory will be automatically cached
  and skipped to ensure ultra-fast execution times. Only the changing QAT best panel
  is generated fresh by default!
"""

import os
import argparse
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Compile VDA QAT 3x2 Multi-Epoch Comparison Grid Video")
    parser.add_argument(
        "--video",
        default="video.mp4",
        help="Path to original input RGB video file (.mp4, .avi, etc.)"
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="The first epoch of the middle 4 panels (e.g. 0)"
    )
    parser.add_argument(
        "--end-epoch",
        type=int,
        default=3,
        help="The last epoch of the middle 4 panels (e.g. 3)"
    )
    parser.add_argument(
        "--output",
        default="checkpoints/epoch_grid_3x2",
        help="Path to save the final compiled grid and panel videos (or folder path)"
    )
    parser.add_argument(
        "--gpu",
        default="0",
        help="CUDA device index to target (default: 0)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of all panel videos (bypass cache)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Determine paths and output directories dynamically
    output_path = os.path.abspath(args.output)
    
    if not output_path.lower().endswith(".mp4"):
        dir_path = output_path
        grid_output = os.path.join(dir_path, "vda_qat_epoch_comparison_3x2.mp4")
    else:
        dir_path = os.path.dirname(output_path)
        grid_output = output_path

    os.makedirs(dir_path, exist_ok=True)

    if not os.path.isfile(args.video):
        print(f"ERROR: Input RGB video file '{args.video}' does not exist.")
        sys.exit(1)

    # Calculate intermediate epoch numbers
    # We need exactly 4 epochs between start and end. If range is small, we interpolate.
    ep0 = args.start_epoch
    ep3 = args.end_epoch
    
    # Simple linear interpolation or stepping to get 4 intermediate epochs
    if ep3 - ep0 >= 3:
        ep1 = ep0 + 1
        ep2 = ep0 + 2
    else:
        # Fallback if range is too small
        ep1 = ep0
        ep2 = ep3
        print(f"Warning: Range between start ({ep0}) and end ({ep3}) is too small. Duplicating slots.")

    epochs_list = [ep0, ep1, ep2, ep3]

    # Map paths to all 6 panel videos
    fp32_output = os.path.join(dir_path, "single_float.mp4")
    qat_best_output = os.path.join(dir_path, "single_qat_best.mp4")
    
    intermediate_outputs = []
    for ep in epochs_list:
        intermediate_outputs.append(os.path.join(dir_path, f"single_qat_epoch{ep:03d}.mp4"))

    # Set CUDA environment
    os.environ["VDA_CUDA_DEVICE"] = args.gpu
    print("=" * 75)
    print("        VDA 3X2 MULTI-EPOCH VISUAL COMPARISON PIPELINE        ")
    print("=" * 75)
    print(f"Targeting CUDA Device    : cuda:{args.gpu}")
    print(f"Input RGB Video          : {args.video}")
    print(f"Intermediate Epochs      : {epochs_list}")
    print(f"Output Directory         : {dir_path}")
    print(f"Final 3x2 Grid Output    : {grid_output}\n")

    # 1. Generate FP32 Float (Panel 1)
    print("--- [Panel 1/6] Generating FP32 Float Baseline Depth Map ---")
    if os.path.isfile(fp32_output) and not args.force:
        print(f"  [Skip] {os.path.basename(fp32_output)} already exists — skipping generation.")
    else:
        cmd_float = [
            "./vda_env/bin/python", "video_infer.py",
            "--video", args.video,
            "--no-quant",
            "--single",
            "--output", fp32_output
        ]
        subprocess.run(cmd_float, check=True)

    # 2. Generate 4 Intermediate Epochs (Panels 2, 3, 4, 5)
    for idx, ep in enumerate(epochs_list):
        panel_num = idx + 2
        p_out = intermediate_outputs[idx]
        ckpt_path = f"checkpoints/vda_qat_epoch{ep:03d}.pt"
        
        print(f"\n--- [Panel {panel_num}/6] Generating INT8 QAT Epoch {ep:03d} Depth Map ---")
        
        if os.path.isfile(p_out) and not args.force:
            print(f"  [Skip] {os.path.basename(p_out)} already exists — skipping generation.")
            continue

        if not os.path.isfile(ckpt_path):
            print(f"  ERROR: Checkpoint file '{ckpt_path}' does not exist. Cannot generate panel {panel_num}.")
            sys.exit(1)

        cmd_ep = [
            "./vda_env/bin/python", "video_infer.py",
            "--video", args.video,
            "--ckpt", ckpt_path,
            "--single",
            "--output", p_out
        ]
        subprocess.run(cmd_ep, check=True)

    # 3. Generate QAT Best (Panel 6)
    print("\n--- [Panel 6/6] Generating Fine-tuned QAT Best Depth Map ---")
    # Always regenerate the best QAT model to capture the latest training updates
    cmd_best = [
        "./vda_env/bin/python", "video_infer.py",
        "--video", args.video,
        "--ckpt", "checkpoints/vda_qat_best.pt",
        "--single",
        "--output", qat_best_output
    ]
    subprocess.run(cmd_best, check=True)

    # 4. Assemble the 3x2 Grid via FFmpeg
    print("\n--- [Grid Assembly] Stacking and Overlaying 3x2 Layout ---")
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", fp32_output,
        "-i", intermediate_outputs[0],
        "-i", intermediate_outputs[1],
        "-i", intermediate_outputs[2],
        "-i", intermediate_outputs[3],
        "-i", qat_best_output,
        "-filter_complex", (
            "[0:v]scale=518:392,drawtext=text='1. Pretrained Float (FP32)':fontcolor=white:fontsize=18:box=1:boxcolor=black@0.6:boxborderw=5:x=15:y=15[t0];"
            f"[1:v]scale=518:392,drawtext=text='2. QAT Epoch {epochs_list[0]}':fontcolor=white:fontsize=18:box=1:boxcolor=black@0.6:boxborderw=5:x=15:y=15[t1];"
            f"[2:v]scale=518:392,drawtext=text='3. QAT Epoch {epochs_list[1]}':fontcolor=white:fontsize=18:box=1:boxcolor=black@0.6:boxborderw=5:x=15:y=15[t2];"
            f"[3:v]scale=518:392,drawtext=text='4. QAT Epoch {epochs_list[2]}':fontcolor=white:fontsize=18:box=1:boxcolor=black@0.6:boxborderw=5:x=15:y=15[t3];"
            f"[4:v]scale=518:392,drawtext=text='5. QAT Epoch {epochs_list[3]}':fontcolor=white:fontsize=18:box=1:boxcolor=black@0.6:boxborderw=5:x=15:y=15[t4];"
            "[5:v]scale=518:392,drawtext=text='6. QAT Best Model':fontcolor=white:fontsize=18:box=1:boxcolor=black@0.6:boxborderw=5:x=15:y=15[t5];"
            "[t0][t1][t2][t3][t4][t5]xstack=inputs=6:layout=0_0|518_0|1036_0|0_392|518_392|1036_392[v]"
        ),
        "-map", "[v]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "medium",
        grid_output
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print("=" * 75)
        print(f" SUCCESS! Saved all 3x2 multi-epoch outputs in: {dir_path}")
        print(f"   -> Final 3x2 Grid Video: {grid_output}")
        print("=" * 75)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: FFmpeg grid compilation failed.")

if __name__ == "__main__":
    main()
