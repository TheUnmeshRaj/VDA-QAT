#!/usr/bin/env python3
"""
epoch_grid_3x3.py — Multi-Epoch Visual Comparison Grid Compiler.

Compiles a 3x3 stacked layout comparing 9 different model states over a test video:
  +--------------------------------+--------------------------------+--------------------------------+
  |  1. Pretrained Baseline (FP32) |  2. QAT Epoch {e0}           |  3. QAT Epoch {e1}               |
  +--------------------------------+--------------------------------+--------------------------------+
  |  4. QAT Epoch {e2}             |  5. QAT Epoch {e3}           |  6. QAT Epoch {e4}               |
  +--------------------------------+--------------------------------+--------------------------------+
  |  7. QAT Epoch {e5}             |  8. QAT Epoch {e6}           |  9. Fine-tuned QAT Best          |
  +--------------------------------+--------------------------------+--------------------------------+

Dynamic caching is used for existing panel videos. Only missing panels are regenerated,
so repeat runs are fast once panels already exist.
"""

import os
import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Compile VDA QAT 3x3 Multi-Epoch Comparison Grid Video")
    parser.add_argument(
        "--video",
        default="video.mp4",
        help="Path to original input RGB video file (.mp4, .avi, etc.)"
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="First epoch for the QAT comparison range"
    )
    parser.add_argument(
        "--end-epoch",
        type=int,
        default=6,
        help="Last epoch for the QAT comparison range"
    )
    parser.add_argument(
        "--output",
        default="checkpoints/epoch_grid_3x3",
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


def compute_epoch_sequence(start_epoch, end_epoch, count=7):
    if start_epoch > end_epoch:
        print(f"ERROR: start-epoch ({start_epoch}) must be less than or equal to end-epoch ({end_epoch}).")
        sys.exit(1)

    if start_epoch == end_epoch:
        return [start_epoch] * count

    span = end_epoch - start_epoch
    if span >= count - 1:
        return [start_epoch + i for i in range(count)]

    epochs = []
    for idx in range(count):
        value = start_epoch + round(idx * span / (count - 1))
        epochs.append(value)

    if len(set(epochs)) != len(epochs):
        print(
            f"Warning: Epoch range [{start_epoch}, {end_epoch}] is too small to fill 7 unique slots. "
            "Some panels may reuse the same epoch file."
        )
    return epochs


def main():
    args = parse_args()

    output_path = os.path.abspath(args.output)
    if not output_path.lower().endswith(".mp4"):
        dir_path = output_path
        grid_output = os.path.join(dir_path, "vda_qat_epoch_comparison_3x3.mp4")
    else:
        dir_path = os.path.dirname(output_path)
        grid_output = output_path

    os.makedirs(dir_path, exist_ok=True)

    if not os.path.isfile(args.video):
        print(f"ERROR: Input RGB video file '{args.video}' does not exist.")
        sys.exit(1)

    epoch_sequence = compute_epoch_sequence(args.start_epoch, args.end_epoch, count=7)

    fp32_output = os.path.join(dir_path, "single_float.mp4")
    qat_best_output = os.path.join(dir_path, "single_qat_best.mp4")

    intermediate_outputs = []
    for ep in epoch_sequence:
        intermediate_outputs.append(os.path.join(dir_path, f"single_qat_epoch{ep:03d}.mp4"))

    os.environ["VDA_CUDA_DEVICE"] = args.gpu
    print("=" * 75)
    print("        VDA 3X3 MULTI-EPOCH VISUAL COMPARISON PIPELINE        ")
    print("=" * 75)
    print(f"Targeting CUDA Device    : cuda:{args.gpu}")
    print(f"Input RGB Video          : {args.video}")
    print(f"Epoch Range              : {args.start_epoch} -> {args.end_epoch}")
    print(f"Intermediate Epochs      : {epoch_sequence}")
    print(f"Output Directory         : {dir_path}")
    print(f"Final 3x3 Grid Output    : {grid_output}\n")

    print("--- [Panel 1/9] Generating FP32 Float Baseline Depth Map ---")
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

    for idx, ep in enumerate(epoch_sequence):
        panel_num = idx + 2
        output_file = intermediate_outputs[idx]
        ckpt_path = f"checkpoints/vda_qat_epoch{ep:03d}.pt"

        print(f"\n--- [Panel {panel_num}/9] Generating INT8 QAT Epoch {ep:03d} Depth Map ---")
        if os.path.isfile(output_file) and not args.force:
            print(f"  [Skip] {os.path.basename(output_file)} already exists — skipping generation.")
            continue

        if not os.path.isfile(ckpt_path):
            print(f"ERROR: Checkpoint file '{ckpt_path}' does not exist. Cannot generate panel {panel_num}.")
            sys.exit(1)

        cmd_ep = [
            "./vda_env/bin/python", "video_infer.py",
            "--video", args.video,
            "--ckpt", ckpt_path,
            "--single",
            "--output", output_file
        ]
        subprocess.run(cmd_ep, check=True)

    print("\n--- [Panel 9/9] Generating Fine-tuned QAT Best Depth Map ---")
    cmd_best = [
        "./vda_env/bin/python", "video_infer.py",
        "--video", args.video,
        "--ckpt", "checkpoints/vda_qat_best.pt",
        "--single",
        "--output", qat_best_output
    ]
    subprocess.run(cmd_best, check=True)

    print("\n--- [Grid Assembly] Stacking and Overlaying 3x3 Layout ---")
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", fp32_output,
        "-i", intermediate_outputs[0],
        "-i", intermediate_outputs[1],
        "-i", intermediate_outputs[2],
        "-i", intermediate_outputs[3],
        "-i", intermediate_outputs[4],
        "-i", intermediate_outputs[5],
        "-i", intermediate_outputs[6],
        "-i", qat_best_output,
        "-filter_complex", (
            "[0:v]scale=518:392,drawtext=text='1. Pretrained Float (FP32)':fontcolor=white:fontsize=18:box=1:boxcolor=black@0.6:boxborderw=5:x=15:y=15[t0];"
            f"[1:v]scale=518:392,drawtext=text='2. QAT Epoch {epoch_sequence[0]}':fontcolor=white:fontsize=18:box=1:boxcolor=black@0.6:boxborderw=5:x=15:y=15[t1];"
            f"[2:v]scale=518:392,drawtext=text='3. QAT Epoch {epoch_sequence[1]}':fontcolor=white:fontsize=18:box=1:boxcolor=black@0.6:boxborderw=5:x=15:y=15[t2];"
            f"[3:v]scale=518:392,drawtext=text='4. QAT Epoch {epoch_sequence[2]}':fontcolor=white:fontsize=18:box=1:boxcolor=black@0.6:boxborderw=5:x=15:y=15[t3];"
            f"[4:v]scale=518:392,drawtext=text='5. QAT Epoch {epoch_sequence[3]}':fontcolor=white:fontsize=18:box=1:boxcolor=black@0.6:boxborderw=5:x=15:y=15[t4];"
            f"[5:v]scale=518:392,drawtext=text='6. QAT Epoch {epoch_sequence[4]}':fontcolor=white:fontsize=18:box=1:boxcolor=black@0.6:boxborderw=5:x=15:y=15[t5];"
            f"[6:v]scale=518:392,drawtext=text='7. QAT Epoch {epoch_sequence[5]}':fontcolor=white:fontsize=18:box=1:boxcolor=black@0.6:boxborderw=5:x=15:y=15[t6];"
            f"[7:v]scale=518:392,drawtext=text='8. QAT Epoch {epoch_sequence[6]}':fontcolor=white:fontsize=18:box=1:boxcolor=black@0.6:boxborderw=5:x=15:y=15[t7];"
            "[8:v]scale=518:392,drawtext=text='9. QAT Best Model':fontcolor=white:fontsize=18:box=1:boxcolor=black@0.6:boxborderw=5:x=15:y=15[t8];"
            "[t0][t1][t2][t3][t4][t5][t6][t7][t8]xstack=inputs=9:layout=0_0|518_0|1036_0|0_392|518_392|1036_392|0_784|518_784|1036_784[v]"
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
        print(f" SUCCESS! Saved all 3x3 multi-epoch outputs in: {dir_path}")
        print(f"   -> Final 3x3 Grid Video: {grid_output}")
        print("=" * 75)
    except subprocess.CalledProcessError:
        print(f"\nERROR: FFmpeg grid compilation failed.")


if __name__ == "__main__":
    main()
