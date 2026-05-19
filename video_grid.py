#!/usr/bin/env python3
"""
compile_grid_2x2.py — Automated Visual Comparison Grid Compiler.

Generates 3 separate single-panel depth videos:
  1. FP32 Float (Original pretrained model)
  2. INT8 PTQ (Calibration only, without QAT)
  3. INT8 QAT (Fully fine-tuned checkpoint)
Then, stacks them with the original RGB video into a highly stylized 2x2 grid comparison:
  +--------------------------------+--------------------------------+
  |  1. Original RGB Video         |  2. Pretrained Float (FP32)    |
  +--------------------------------+--------------------------------+
  |  3. PTQ Calibration (INT8)     |  4. Fine-tuned QAT (INT8)      |
  +--------------------------------+--------------------------------+

Both the final 2x2 grid video and the individual single-panel videos will be saved
dynamically inside the folder determined by your '--output' path.
"""

import os
import argparse
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Compile VDA QAT 2x2 Visual Comparison Grid Video")
    parser.add_argument(
        "--video",
        default="video.mp4",
        help="Path to original input RGB video file (.mp4, .avi, etc.)"
    )
    parser.add_argument(
        "--ckpt",
        default="checkpoints/vda_qat_best.pt",
        help="Path to final fine-tuned QAT model checkpoint (.pt)"
    )
    parser.add_argument(
        "--output",
        default="checkpoints/vda_qat_comparison_2x2.mp4",
        help="Path to save the final grid video (or output folder path)"
    )
    parser.add_argument(
        "--gpu",
        default="0",
        help="CUDA device index to target (default: 0)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Determine paths and output directories dynamically
    output_path = os.path.abspath(args.output)
    
    # If the output path is a directory (doesn't end with .mp4)
    if not output_path.lower().endswith(".mp4"):
        dir_path = output_path
        grid_output = os.path.join(dir_path, "vda_qat_comparison_2x2.mp4")
    else:
        dir_path = os.path.dirname(output_path)
        grid_output = output_path

    os.makedirs(dir_path, exist_ok=True)

    fp32_output = os.path.join(dir_path, "single_float.mp4")
    ptq_output  = os.path.join(dir_path, "single_calib.mp4")
    qat_output  = os.path.join(dir_path, "single_qat.mp4")

    # Set CUDA environment
    os.environ["VDA_CUDA_DEVICE"] = args.gpu
    print("=" * 75)
    print("        VDA 2X2 VISUAL COMPARISON GRID COMPILATION PIPELINE        ")
    print("=" * 75)
    print(f"Targeting CUDA Device   : cuda:{args.gpu}")
    print(f"Input RGB Video         : {args.video}")
    print(f"QAT Checkpoint          : {args.ckpt}")
    print(f"Output Directory        : {dir_path}")
    print(f"Individual FP32 Output  : {fp32_output}")
    print(f"Individual PTQ Output   : {ptq_output}")
    print(f"Individual QAT Output   : {qat_output}")
    print(f"Final 2x2 Grid Output   : {grid_output}\n")

    if not os.path.isfile(args.video):
        print(f"ERROR: Input RGB video file '{args.video}' does not exist.")
        sys.exit(1)

    # 1. FP32 Float Single Panel
    print("--- [Panel 1/3] Generating Single FP32 Float Depth Map ---")
    if os.path.isfile(fp32_output):
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

    # 2. PTQ Calibration-Only Single Panel
    print("\n--- [Panel 2/3] Generating Single PTQ Calibration INT8 Depth Map ---")
    if os.path.isfile(ptq_output):
        print(f"  [Skip] {os.path.basename(ptq_output)} already exists — skipping generation.")
    else:
        cmd_calib = [
            "./vda_env/bin/python", "video_infer.py",
            "--video", args.video,
            "--ckpt", "",
            "--single",
            "--output", ptq_output
        ]
        subprocess.run(cmd_calib, check=True)

    # 3. QAT Fine-Tuned Single Panel
    print("\n--- [Panel 3/3] Generating Single QAT Fine-Tuned INT8 Depth Map ---")
    cmd_qat = [
        "./vda_env/bin/python", "video_infer.py",
        "--video", args.video,
        "--ckpt", args.ckpt,
        "--single",
        "--output", qat_output
    ]
    subprocess.run(cmd_qat, check=True)

    # 4. FFmpeg Stacking and Overlay
    print("\n--- [Grid Assembly] Stacking and Overlaying 2x2 Layout ---")
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", args.video,
        "-i", fp32_output,
        "-i", ptq_output,
        "-i", qat_output,
        "-filter_complex", (
            "[0:v]scale=518:392,drawtext=text='1. Original RGB':fontcolor=white:fontsize=18:box=1:boxcolor=black@0.6:boxborderw=5:x=15:y=15[t0];"
            "[1:v]scale=518:392,drawtext=text='2. Pretrained Float (FP32)':fontcolor=white:fontsize=18:box=1:boxcolor=black@0.6:boxborderw=5:x=15:y=15[t1];"
            "[2:v]scale=518:392,drawtext=text='3. PTQ Calibration (INT8)':fontcolor=white:fontsize=18:box=1:boxcolor=black@0.6:boxborderw=5:x=15:y=15[t2];"
            "[3:v]scale=518:392,drawtext=text='4. Fine-tuned QAT (INT8)':fontcolor=white:fontsize=18:box=1:boxcolor=black@0.6:boxborderw=5:x=15:y=15[t3];"
            "[t0][t1][t2][t3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]"
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
        print(f" SUCCESS! Saved all visual validation outputs in: {dir_path}")
        print(f"   -> Final 2x2 Comparison Grid : {grid_output}")
        print("=" * 75)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: FFmpeg grid compilation failed.")

if __name__ == "__main__":
    main()
