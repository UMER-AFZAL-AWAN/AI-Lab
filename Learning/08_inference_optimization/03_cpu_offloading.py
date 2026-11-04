# =============================================================================
# 03 — CPU OFFLOADING: USING SYSTEM RAM AS OVERFLOW
# =============================================================================
#
# LIKE YOU ARE 5:
# Your GPU VRAM is like your desk (small, but fast to reach).
# Your system RAM is like a bookshelf next to the desk (larger, slightly slower).
# Your hard drive is like a warehouse (huge, but slow to fetch).
#
# CPU offloading: when you don't need a part of the model RIGHT NOW,
# move it from the desk (GPU VRAM) to the bookshelf (CPU RAM).
# When you need it again, move it back.
#
# This lets you run a model that is TOO BIG for your VRAM,
# as long as your system RAM is large enough.
# Cost: extra data transfer (CPU RAM -> GPU VRAM). Slower. But it WORKS.
#
# YOUR SETUP:
# AnimateDiff pipeline uses enable_model_cpu_offload().
# It moves the UNet to CPU while VAE is on GPU, and vice versa.
# Total VRAM needed at any point: max(UNet, VAE) instead of UNet + VAE.
#
# =============================================================================

import torch
import torch.nn as nn

print("=" * 60)
print("MANUAL DEVICE SWITCHING")
print("=" * 60)
#
# The basic operation: .to("cpu") to offload, .to("cuda") to bring back.

class BigModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1024, 1024)

    def forward(self, x):
        return self.layer(x)

model_A = BigModel()   # represents UNet
model_B = BigModel()   # represents VAE

# Initially both on CPU
print(f"model_A device: {next(model_A.parameters()).device}")
print(f"model_B device: {next(model_B.parameters()).device}")

if torch.cuda.is_available():
    # Move A to GPU for processing
    model_A = model_A.to("cuda")
    print(f"\nAfter move: model_A on {next(model_A.parameters()).device}")

    # Offload A, bring B to GPU
    model_A = model_A.to("cpu")
    model_B = model_B.to("cuda")
    print(f"After swap: model_A on {next(model_A.parameters()).device}")
    print(f"            model_B on {next(model_B.parameters()).device}")

    # Move both back to CPU for cleanup
    model_A = model_A.cpu()
    model_B = model_B.cpu()
    torch.cuda.empty_cache()
else:
    print("(No CUDA — simulating the pattern)")

print()
print("=" * 60)
print("VRAM PEAK COMPARISON")
print("=" * 60)
#
# AnimateDiff components and their approximate sizes (FP16):
# UNet (SD 1.5 + motion module): ~3.5 GB
# VAE (AutoencoderKL):           ~0.3 GB
# CLIP text encoder:             ~0.6 GB
# Scheduler, misc:               ~0.1 GB
#
# Without CPU offload: 3.5 + 0.3 + 0.6 + 0.1 = 4.5 GB (minimum if no activations)
# With activations during 50-step denoising: could reach 7-8+ GB
# With CPU offload:   peak = max at any step (usually just the UNet = ~3.5 GB active at once)

components = {
    "UNet (SD 1.5 + motion module)": 3.5,
    "VAE (AutoencoderKL)":           0.3,
    "CLIP text encoder":             0.6,
    "Scheduler + misc":              0.1,
}

total = sum(components.values())
print("AnimateDiff component sizes (FP16, approximate):")
for name, gb in components.items():
    bar = "█" * int(gb * 10)
    print(f"  {name:<35}: {gb:.1f} GB  {bar}")
print(f"\n  Total: {total:.1f} GB  (all at once, no offload)")
print(f"  Your VRAM: 8 GB")
print(f"  Fits without offload? {'Yes' if total <= 8 else 'Only if activations are small'}")
print()
print("  With enable_model_cpu_offload():")
print("  Peak VRAM ≈ max(UNet, VAE, CLIP) = 3.5 GB")
print("  Activation memory also matters but offloading helps significantly.")

print()
print("=" * 60)
print("SEQUENTIAL CPU OFFLOAD vs MODEL CPU OFFLOAD")
print("=" * 60)
#
# enable_model_cpu_offload():
#   Each TOP-LEVEL module moved to CPU when not needed.
#   e.g. UNet on CPU while VAE runs, then UNet moves to GPU for denoising.
#
# enable_sequential_cpu_offload():
#   Each SUB-MODULE (layer by layer) moved to CPU when not needed.
#   Much more aggressive. Even slower. But allows running models > 10 GB on 8 GB VRAM.
#   Can be 5-10x slower than no offload.

print("enable_model_cpu_offload():     module-level offload. ~2x slower.")
print("enable_sequential_cpu_offload(): layer-level offload.  ~5-10x slower.")
print()
print("Your LocalVideoStudio setup:")
print("  pipeline.enable_model_cpu_offload()  # in studio_pipeline.py")
print("  pipeline.enable_vae_slicing()         # decode one frame at a time")
print("  torch_dtype=torch.float16             # FP16 for all weights")
print()
print("This is the standard setup for AnimateDiff on 8GB VRAM. Well configured.")

print()
print("=" * 60)
print("TIMING: THE COST OF OFFLOADING")
print("=" * 60)

if torch.cuda.is_available():
    import time

    model = BigModel().cuda()
    x     = torch.randn(32, 1024, device="cuda")

    # Time without offload
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()
    t_no_offload = (time.time() - start) * 1000

    # Time with one offload cycle per forward pass (simulate)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        model.cpu()
        model.cuda()
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()
    t_with_offload = (time.time() - start) * 1000

    print(f"100 forward passes without offload: {t_no_offload:.0f} ms")
    print(f"100 forward passes with offload:    {t_with_offload:.0f} ms")
    print(f"Offload overhead: {t_with_offload / t_no_offload:.1f}x slower")
    model.cpu()
    torch.cuda.empty_cache()
else:
    print("(No CUDA — skipping timing)")

# =============================================================================
# EXPECTED OUTPUT:
# Total: 4.5 GB (all at once, no offload)
# 100 forward passes with offload: ~10x slower
# =============================================================================
