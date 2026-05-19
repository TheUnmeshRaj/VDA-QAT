"""
Model Graph Patching: VDA → replace xFormers attention → load weights → cuda:1
Uses cuda:1 (second A100, 0-indexed) which corresponds to the "GPU 2" mentioned.
"""

import math
import copy
import torch
import torch.nn as nn
from torch import Tensor

import os
VDA_CUDA_DEVICE = os.environ.get("VDA_CUDA_DEVICE", "1")
DEVICE          = torch.device(f"cuda:{VDA_CUDA_DEVICE}")
CKPT_PATH       = "/media/rvcse22/CSERV/vdaproj/checkpoints/vda_small_pretrained.pth"
VDA_REPO        = "/media/rvcse22/CSERV/vdaproj/Video-Depth-Anything"

import sys
sys.path.insert(0, VDA_REPO)

from video_depth_anything.motion_module.motion_module import TemporalAttention


# ─────────────────────────────────────────────────────────────────────────────
# Explicit attention modules (AIMET-transparent, no fused kernels)
# ─────────────────────────────────────────────────────────────────────────────

class QuantizableAttention(nn.Module):
    """Drops-in for MemoryEfficientAttention / xFormers self-attention."""

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads  = num_heads
        self.head_dim   = dim // num_heads
        self.scale      = math.sqrt(self.head_dim)

        self.qkv  = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax   = nn.Softmax(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        H       = self.num_heads
        D       = self.head_dim

        qkv = self.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                              # each [B,H,N,D]

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)                          # [B,H,N,D]
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    @classmethod
    def from_mem_eff_attention(cls, src: nn.Module) -> "QuantizableAttention":
        """Copy weights from an xFormers / timm attention block."""
        dim       = src.qkv.in_features
        num_heads = src.num_heads
        has_bias  = src.qkv.bias is not None
        attn_drop = src.attn_drop.p if hasattr(src, "attn_drop") else 0.0
        proj_drop = src.proj_drop.p if hasattr(src, "proj_drop") else 0.0

        new = cls(dim=dim, num_heads=num_heads, qkv_bias=has_bias,
                  attn_drop=attn_drop, proj_drop=proj_drop)
        new.qkv.weight.data.copy_(src.qkv.weight.data)
        if has_bias:
            new.qkv.bias.data.copy_(src.qkv.bias.data)
        new.proj.weight.data.copy_(src.proj.weight.data)
        if src.proj.bias is not None:
            new.proj.bias.data.copy_(src.proj.bias.data)
        return new


class QuantizableCrossAttention(nn.Module):
    """Explicit cross-attention for VDA temporal / motion modules."""

    def __init__(self, query_dim: int, context_dim: int, num_heads: int,
                 head_dim: int, dropout: float = 0.0):
        super().__init__()
        inner_dim    = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim  = head_dim
        self.scale     = math.sqrt(head_dim)

        self.to_q   = nn.Linear(query_dim,   inner_dim, bias=False)
        self.to_k   = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v   = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim),
                                    nn.Dropout(dropout))
        self.softmax = nn.Softmax(dim=-1)

            
    def forward(self, hidden_states: Tensor, encoder_hidden_states: Tensor | None = None, attention_mask=None, video_length=None, **kwargs) -> Tensor:
        
        x = hidden_states

        if encoder_hidden_states is None:
            context = x
        else:
            context = encoder_hidden_states
        
        B, N, _ = hidden_states.shape
        H, D    = self.num_heads, self.head_dim

        q = self.to_q(x      ).reshape(B, N,              H, D).transpose(1, 2)
        k = self.to_k(context).reshape(B, context.size(1), H, D).transpose(1, 2)
        v = self.to_v(context).reshape(B, context.size(1), H, D).transpose(1, 2)

        attn = self.softmax(torch.matmul(q, k.transpose(-2, -1)) / self.scale)
        out  = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, H * D)
        return self.to_out(out)

    @classmethod
    def from_cross_attention(cls, src: nn.Module) -> "QuantizableCrossAttention":
        query_dim   = src.to_q.in_features
        context_dim = src.to_k.in_features
        inner_dim   = src.to_q.out_features

        # Resolve to_out linear and dropout
        if isinstance(src.to_out, (nn.Sequential, nn.ModuleList)):
            to_out_lin = src.to_out[0]
            dropout    = src.to_out[1].p if len(src.to_out) > 1 and hasattr(src.to_out[1], "p") else 0.0
        else:
            to_out_lin = src.to_out
            dropout    = 0.0

        # Infer num_heads reliably from weight shape rather than a fragile attribute.
        # to_out maps inner_dim -> query_dim; inner_dim == num_heads * head_dim.
        # We also cross-check with a .heads attribute if present.
        if hasattr(src, "heads") and src.heads > 0:
            num_heads = src.heads
        else:
            # Derive from to_out: out_features == query_dim, in_features == inner_dim
            # Try common divisors that produce integer head_dim values.
            candidate_heads = [h for h in [1, 2, 4, 8, 12, 16, 32]
                               if inner_dim % h == 0]
            # Pick the largest that gives head_dim >= 32 (typical minimum)
            num_heads = next(
                (h for h in reversed(candidate_heads) if inner_dim // h >= 32),
                candidate_heads[-1]
            )
        head_dim = inner_dim // num_heads

        new = cls(query_dim, context_dim, num_heads, head_dim, dropout)
        new.to_q.weight.data.copy_(src.to_q.weight.data)
        new.to_k.weight.data.copy_(src.to_k.weight.data)
        new.to_v.weight.data.copy_(src.to_v.weight.data)
        to_out_lin_new = new.to_out[0]
        to_out_lin_new.weight.data.copy_(to_out_lin.weight.data)
        if to_out_lin.bias is not None and to_out_lin_new.bias is not None:
            to_out_lin_new.bias.data.copy_(to_out_lin.bias.data)
        return new


# ─────────────────────────────────────────────────────────────────────────────
# Graph patching helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_xformers_self_attn(module: nn.Module) -> bool:
    """True iff this is an xFormers/timm attention block that has NOT yet been patched."""
    return (
        not isinstance(module, (QuantizableAttention, QuantizableCrossAttention))
        and hasattr(module, "qkv")
        and hasattr(module, "proj")
        and hasattr(module, "num_heads")
    )

def _is_cross_attn(module: nn.Module) -> bool:
    """True iff this is a cross-attention block that has NOT yet been patched."""
    return (
        not isinstance(module, (QuantizableAttention, QuantizableCrossAttention))
        and hasattr(module, "to_q")
        and hasattr(module, "to_k")
        and hasattr(module, "to_v")
        and hasattr(module, "to_out")
    )


def _is_temporal_attn(module: nn.Module) -> bool:
    return isinstance(module, TemporalAttention)
    
def patch_attention_blocks(model: nn.Module) -> tuple[int, int]:
    """
    Recursively walks the model and replaces xFormers / cross-attention
    blocks with their Quantizable equivalents. Returns (n_self, n_cross).
    """
    n_self = n_cross = 0
    for name, child in list(model.named_children()):
        if _is_xformers_self_attn(child):
            setattr(model, name, QuantizableAttention.from_mem_eff_attention(child))
            n_self += 1
        elif _is_temporal_attn(child):
            # Recurse into TemporalAttention to find nested cross-attn/self-attn modules
            s, c = patch_attention_blocks(child)
            n_self += s; n_cross += c
        elif _is_cross_attn(child):
            setattr(model, name, QuantizableCrossAttention.from_cross_attention(child))
            n_cross += 1
        else:
            s, c = patch_attention_blocks(child)
            n_self += s; n_cross += c
    return n_self, n_cross


def verify_patch_parity(original: nn.Module, patched: nn.Module,
                         dummy_input: Tensor, tol: float = 1e-3) -> bool:
    """Forward pass numeric equivalence check on CPU."""
    original.eval(); patched.eval()
    with torch.no_grad():
        out_orig   = original(dummy_input)
        out_patched = patched(dummy_input)
    if isinstance(out_orig, (tuple, list)):
        out_orig    = out_orig[0]
        out_patched = out_patched[0]
    max_err = (out_orig - out_patched).abs().max().item()
    print(f"[Verify] max abs error = {max_err:.6f}  →  {'PASS' if max_err < tol else 'FAIL'}")
    return max_err < tol


# ─────────────────────────────────────────────────────────────────────────────
# Main build function
# ─────────────────────────────────────────────────────────────────────────────

def build_patched_vda(verify: bool = False) -> nn.Module:
    from video_depth_anything.video_depth import VideoDepthAnything

    cfg = dict(
        encoder     = "vits",
        features    = 64,
        out_channels= [48, 96, 192, 384],
    )
    model_orig = VideoDepthAnything(**cfg)

    # Load pre-trained weights
    state = torch.load(CKPT_PATH, map_location="cpu")
    if "model" in state:
        state = state["model"]
    missing, unexpected = model_orig.load_state_dict(state, strict=False)
    if missing:
        print(f"[Load] missing keys  : {len(missing)}")
    if unexpected:
        print(f"[Load] unexpected    : {len(unexpected)}")

    # Keep a CPU copy for parity check before patching moves weights
    if verify:
        model_ref = copy.deepcopy(model_orig).eval()

    # Patch attention
    n_self, n_cross = patch_attention_blocks(model_orig)
    print(f"[Patch] replaced {n_self} self-attn  +  {n_cross} cross-attn blocks")

    if verify:
        B, T, C, H, W = 1, 2, 3, 392, 518
        dummy = torch.randn(B, T, C, H, W, device=DEVICE)
        model_ref = model_ref.to(DEVICE)
        model_orig = model_orig.to(DEVICE)
        verify_patch_parity(model_ref, model_orig, dummy)

    model_orig = model_orig.to(DEVICE)
    print(f"[Model] on {DEVICE}  |  params: {sum(p.numel() for p in model_orig.parameters()):,}")
    return model_orig


if __name__ == "__main__":
    model = build_patched_vda(verify=True)
    print(model)