"""
AIMET QAT Initialization — Policy B mixed-precision (top-10 sensitive in FP16, rest INT8).
LayerNorm, Softmax, resize/decoder critical ops also kept FP16 per quantisation strategy table.
"""

import os
import torch
import torch.nn as nn
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.common.defs import QuantScheme
from aimet_torch.mixed_precision import MixedPrecisionConfigurator
from aimet_torch.onnx_utils import OnnxExportApiArgs
from aimet_torch.nn import QuantizationMixin
from video_depth_anything.dinov2_layers.layer_scale import LayerScale
import json

VDA_CUDA_DEVICE = os.environ.get("VDA_CUDA_DEVICE", "1")
DEVICE          = torch.device(f"cuda:{VDA_CUDA_DEVICE}")

# ── Policy B: top-10 sensitive layers kept FP16 ──────────────────────────────
POLICY_B_FP16_LAYERS = [
    "pretrained.blocks.10.mlp.fc2",
    "pretrained.blocks.8.mlp.fc2",
    "pretrained.blocks.3.attn.qkv",
    "pretrained.blocks.7.mlp.fc2",
    "pretrained.blocks.4.mlp.fc1",
    "pretrained.blocks.8.attn.qkv",
    "pretrained.blocks.3.mlp.fc2",
    "pretrained.blocks.2.mlp.fc1",
    "pretrained.blocks.0.mlp.fc2",
    "pretrained.blocks.6.attn.qkv",
]

# Additional FP16 from architecture strategy table
ARCH_FP16_MODULE_TYPES = (nn.LayerNorm, nn.Softmax)

# Decoder / resize critical op name substrings
DECODER_FP16_SUBSTRINGS = [
    "depth_head", "decode_head", "scratch", "refinenet",
    "head.resize", "head.conv_depth",
]


def _collect_fp16_names(model: nn.Module) -> set[str]:
    fp16_names: set[str] = set(POLICY_B_FP16_LAYERS)

    for full_name, module in model.named_modules():
        # LayerNorm / Softmax → FP16
        if isinstance(module, ARCH_FP16_MODULE_TYPES):
            fp16_names.add(full_name)
        # Decoder critical ops → FP16
        for substr in DECODER_FP16_SUBSTRINGS:
            if substr in full_name:
                fp16_names.add(full_name)
                break

    return fp16_names


QuantizationMixin.ignore(LayerScale)

def build_quant_sim(model: nn.Module, dummy_input: torch.Tensor) -> QuantizationSimModel:
    """
    Creates AIMET QuantizationSimModel in QAT mode.
    Global config: INT8 weights + activations.
    Policy B FP16 overrides applied after construction.
    """
    quant_sim = QuantizationSimModel(
        model          = model,
        quant_scheme   = QuantScheme.training_range_learning_with_tf_init,
        default_param_bw      = 8,
        default_output_bw     = 8,
        dummy_input    = dummy_input,
        in_place       = False,
    )

    fp16_names = _collect_fp16_names(model)
    n_overridden = 0

    for name, wrapper in quant_sim.model.named_modules():
        # AIMET wraps quantizable layers; access via module attribute
        bare_name = name.replace("._module_to_wrap", "")
        if bare_name in fp16_names:
            if hasattr(wrapper, "param_quantizers"):
                for pq in wrapper.param_quantizers.values():
                    if pq is not None:
                        pq.bitwidth = 16
            if hasattr(wrapper, "input_quantizers"):
                for iq in wrapper.input_quantizers:
                    if iq is not None:
                        iq.bitwidth = 16
            if hasattr(wrapper, "output_quantizers"):
                for oq in wrapper.output_quantizers:
                    if oq is not None:
                        oq.bitwidth = 16
            n_overridden += 1

    print(f"[QuantSim] Base model patched  |  {n_overridden} modules set to 16-bit (weights + inputs + outputs)")
    return quant_sim


def calibrate_encodings(quant_sim: QuantizationSimModel,
                        calib_loader,
                        n_batches: int = 64):
    """
    Run forward passes for activation range collection before QAT.
    This seeds the learned ranges with good initial values.
    """
    quant_sim.model.eval()
    device = next(quant_sim.model.parameters()).device

    def forward_fn(model, _):
        with torch.no_grad():
            for i, (rgb, _) in enumerate(calib_loader):
                if i >= n_batches:
                    break
                rgb = rgb.to(device)          # [B, T, C, H, W]
                model(rgb)

    quant_sim.compute_encodings(forward_fn, forward_pass_callback_args=None)
    print(f"[Calibrate] encodings computed over {n_batches} batches")


def save_encodings(quant_sim: QuantizationSimModel,
                   path: str = "/media/rvcse22/CSERV/vdaproj/checkpoints/vda_qat_init.encodings"):
    """Dump initial encoding JSON (useful for debugging before full QAT)."""
    # AIMET 2.0+ handles encoding exports via quant_sim.export()
    # The manual JSON dump logic uses deprecated .encoding attributes which crash.
    print(f"[Encodings] Skipping manual save, will be exported at the end of pipeline.")


if __name__ == "__main__":
    from model_patch import build_patched_vda
    from dataset_pipeline import build_loaders

    model = build_patched_vda()
    dummy = torch.randn(1, 2, 3, 392, 518, device=DEVICE)
    qsim  = build_quant_sim(model, dummy)

    train_loader, _ = build_loaders(batch_size=2)
    calibrate_encodings(qsim, train_loader, n_batches=64)
    save_encodings(qsim)
    print("[Init] QuantSim ready for QAT")