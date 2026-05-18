import torch, sys
sys.path.insert(0, "/media/rvcse22/CSERV/vdaproj/Video-Depth-Anything")
from model_patch2 import build_patched_vda

model = build_patched_vda().eval()
dummy = torch.randn(1, 2, 3, 392, 518).to("cuda:1")

with torch.no_grad():
    out = model(dummy)
print(type(out), out.shape if hasattr(out, 'shape') else {k: v.shape for k,v in out.items()})
print("min:", out.min().item(), "max:", out.max().item(), "mean:", out.mean().item())