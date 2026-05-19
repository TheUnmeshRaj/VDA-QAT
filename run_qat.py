"""
run_qat_pipeline.py — Master orchestrator. Run this to execute all 5 stages.
Usage:
    python run_qat_pipeline.py                    # full pipeline
    python run_qat_pipeline.py --resume path.pt   # resume from checkpoint
    python run_qat_pipeline.py --eval-only        # eval + export only
"""

import argparse
import sys
import os
import torch

sys.path.insert(0, "/media/rvcse22/CSERV/vdaproj/Video-Depth-Anything")
sys.path.insert(0, os.path.dirname(__file__))

from dataset_pipeline import build_loaders
from model_patch       import build_patched_vda
from aimet_qat_init    import build_quant_sim, calibrate_encodings, save_encodings
from qat_training_loop import train_qat, DEVICE, BATCH_SIZE, SEQ_LEN
from eval_and_export   import evaluate, export_onnx, export_torchscript, \
                              export_encodings_json, save_metrics_report, CKPT_DIR

DUMMY_INPUT = lambda: torch.randn(1, 2, 3, 392, 518, device=DEVICE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",    type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--calib-batches", type=int, default=64)
    args = parser.parse_args()

    print("=" * 60)
    print("VDA QAT Pipeline  —  AIMET INT8 / FP16 Mixed Precision")
    print("=" * 60)

    # ── Stage 1: Dataset ─────────────────────────────────────────────────────
    print("\n[Stage 1] Building data loaders …")
    train_loader, val_loader = build_loaders(batch_size=BATCH_SIZE, seq_len=SEQ_LEN, stride=12, num_workers=12)

    # ── Stage 2: Model patch ─────────────────────────────────────────────────
    print("\n[Stage 2] Patching VDA attention blocks …")
    model = build_patched_vda(verify=(not args.eval_only))

    # ── Stage 3: AIMET QAT init ──────────────────────────────────────────────
    print("\n[Stage 3] Initialising QuantSim (Policy B) …")
    dummy = DUMMY_INPUT()
    qsim  = build_quant_sim(model, dummy)

    if not args.eval_only:
        calibrate_encodings(qsim, train_loader, n_batches=args.calib_batches)
        save_encodings(qsim)

    # ── Stage 4: QAT training ────────────────────────────────────────────────
    if not args.eval_only:
        print("\n[Stage 4] Starting QAT fine-tuning …")
        train_qat(qsim, train_loader, val_loader, resume_ckpt=args.resume)

    # ── Stage 5: Eval + Export ───────────────────────────────────────────────
    print("\n[Stage 5] Evaluation & Export …")

    best_ckpt = os.path.join(CKPT_DIR, "vda_qat_best.pt")
    if os.path.isfile(best_ckpt):
        ck = torch.load(best_ckpt, map_location=DEVICE)
        qsim.model.load_state_dict(ck["model_state"])
        print(f"  Loaded best ckpt  epoch={ck['epoch']}  "
              f"abs_rel={ck.get('metrics', {}).get('abs_rel', '?'):.4f}")

    metrics = evaluate(qsim.model, val_loader)
    save_metrics_report(metrics)

    dummy = DUMMY_INPUT()
    export_onnx(qsim, dummy)
    export_torchscript(qsim, dummy)
    export_encodings_json(qsim)

    print("\n[Pipeline] Complete ✓")


if __name__ == "__main__":
    main()
