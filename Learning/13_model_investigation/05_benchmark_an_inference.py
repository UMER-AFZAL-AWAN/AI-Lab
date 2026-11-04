# =============================================================================
# 05 — BENCHMARK AN INFERENCE: MEASURE EVERYTHING
# =============================================================================
#
# LIKE YOU ARE 5:
# When you try a new model, you want to know:
# - How long does it take to load?
# - How long does one generation take?
# - How much of my GPU did it use?
# - How much RAM did it use?
# - What shapes went in and came out?
# This file is a reusable measuring tape for any model.
#
# C# ANALOGY:
# Like a Stopwatch + PerformanceCounter + a custom logger.
# Start() before each step, Stop() after, record your metrics in a dict.
#
# RULE: Measure every new model/checkpoint/setting before using it in production.
# torch.cuda.max_memory_allocated() tells you the PEAK — the highest VRAM
# reached during that run. If it's close to 8 GB, you're at risk of OOM errors.
#
# =============================================================================

import torch
import torch.nn as nn
import time
import os
import sys
import gc

print("=" * 60)
print("BENCHMARK TOOLKIT: REUSABLE FUNCTIONS")
print("=" * 60)

# ---------------------------------------------------------------------------
# 1. Timer
# ---------------------------------------------------------------------------

class Timer:
    """Context manager and manual timer. Accumulates seconds."""
    def __init__(self, name=""):
        self.name   = name
        self.elapsed = 0.0
        self._start  = None

    def __enter__(self):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.elapsed = time.perf_counter() - self._start
        print(f"  ⏱  {self.name}: {self.elapsed:.3f} s")

# ---------------------------------------------------------------------------
# 2. VRAM snapshot helpers
# ---------------------------------------------------------------------------

def reset_vram_peak():
    """Reset the high-water mark so next run starts clean."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def vram_peak_gb():
    """Peak VRAM allocated (GB) since last reset."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0

def vram_current_gb():
    """VRAM currently held (GB)."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0

def vram_reserved_gb():
    """VRAM reserved by PyTorch's caching allocator (GB).
    Reserved >= allocated. The gap is the allocator's free cache."""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1e9
    return 0.0

# ---------------------------------------------------------------------------
# 3. RAM (CPU) usage
# ---------------------------------------------------------------------------

def ram_used_gb():
    """Current process RSS memory in GB."""
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / 1e9
    except ImportError:
        return -1.0  # psutil not installed

# ---------------------------------------------------------------------------
# 4. Tensor shape inspector
# ---------------------------------------------------------------------------

def inspect_tensor(t, name=""):
    if isinstance(t, torch.Tensor):
        print(f"  {name}: shape={list(t.shape)}, dtype={t.dtype}, device={t.device}")
    elif isinstance(t, (list, tuple)):
        for i, item in enumerate(t):
            inspect_tensor(item, f"{name}[{i}]")
    else:
        print(f"  {name}: {type(t).__name__}")

# ---------------------------------------------------------------------------
# 5. Full benchmark wrapper
# ---------------------------------------------------------------------------

def benchmark_model(
    model,
    input_tensor,
    n_warmup=2,
    n_measure=5,
    label="model",
):
    """
    Runs `model(input_tensor)` n_warmup+n_measure times.
    Reports:
      - Warmup time (first run, includes CUDA JIT compilation)
      - Average inference time over n_measure runs
      - Peak VRAM
      - Input / output shapes
    """
    device = next(model.parameters()).device
    x      = input_tensor.to(device)

    print(f"\n{'='*50}")
    print(f"BENCHMARK: {label}")
    print(f"{'='*50}")
    inspect_tensor(x, "input")

    # Warmup (don't measure these — CUDA kernels compile on first run)
    print(f"\n  Warmup ({n_warmup} runs)...")
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(x)

    # Peak VRAM during inference
    reset_vram_peak()
    times = []
    output = None
    for i in range(n_measure):
        with Timer(f"run {i+1}") as t:
            with torch.no_grad():
                output = model(x)
        times.append(t.elapsed)

    avg_ms = (sum(times) / len(times)) * 1000
    peak   = vram_peak_gb()
    ram    = ram_used_gb()

    print(f"\n  Results:")
    print(f"    Avg inference time : {avg_ms:.1f} ms")
    print(f"    Min / Max          : {min(times)*1000:.1f} / {max(times)*1000:.1f} ms")
    print(f"    Peak VRAM          : {peak:.3f} GB")
    print(f"    Current VRAM       : {vram_current_gb():.3f} GB")
    print(f"    Reserved VRAM      : {vram_reserved_gb():.3f} GB")
    if ram >= 0:
        print(f"    RAM (process RSS)  : {ram:.2f} GB")
    inspect_tensor(output, "output")

    return {"avg_ms": avg_ms, "peak_vram_gb": peak, "output": output}


# ---------------------------------------------------------------------------
# DEMO: benchmark a tiny CNN (stand-in for a real model)
# ---------------------------------------------------------------------------

class TinyDenoisingUNet(nn.Module):
    """
    Minimal stand-in for a real UNet denoiser.
    Real AnimateDiff UNet: ~860M params, ~2.1 GB at FP16.
    This one: ~200K params for demonstration.
    """
    def __init__(self, in_ch=4, hidden=32):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden * 2, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(hidden * 2, hidden, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(hidden, in_ch, 3, padding=1),
        )

    def forward(self, x):
        return self.up(self.down(x))


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")
print(f"Initial VRAM: {vram_current_gb():.3f} GB")
print(f"Initial RAM:  {ram_used_gb():.2f} GB")

# ---------------------------------------------------------------------------
# STEP 1: Measure model load time
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 1: LOAD TIME")
print("=" * 60)

reset_vram_peak()
with Timer("model load"):
    model = TinyDenoisingUNet().to(device)

load_vram = vram_peak_gb()
param_count = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {param_count:,} ({param_count/1e6:.2f}M)")
print(f"  VRAM after load: {vram_current_gb():.3f} GB")
print(f"  Peak during load: {load_vram:.3f} GB")

# Parameter count → estimated VRAM at different precisions
print("\n  Estimated VRAM at each dtype (this model):")
for dtype, bpp in [("fp32", 4), ("fp16", 2), ("int8", 1), ("int4", 0.5)]:
    est = param_count * bpp / 1e9
    fits = "✓" if est < 7.5 else "✗"
    print(f"    {dtype:6}: {est:.4f} GB  {fits}")

# ---------------------------------------------------------------------------
# STEP 2: Benchmark inference at FP32
# ---------------------------------------------------------------------------

latent = torch.randn(1, 4, 64, 64)  # 1 × 4ch × 64×64 latent (like SD)

results_fp32 = benchmark_model(model, latent, label="TinyUNet FP32")

# ---------------------------------------------------------------------------
# STEP 3: Benchmark at FP16
# ---------------------------------------------------------------------------

model_fp16 = TinyDenoisingUNet().half().to(device)
latent_fp16 = latent.half()

results_fp16 = benchmark_model(model_fp16, latent_fp16, label="TinyUNet FP16")

# ---------------------------------------------------------------------------
# STEP 4: Compare
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("COMPARISON TABLE")
print("=" * 60)
print(f"{'dtype':<10} {'avg_ms':>10} {'peak_vram':>12}")
print("-" * 35)
print(f"{'FP32':<10} {results_fp32['avg_ms']:>10.1f} {results_fp32['peak_vram_gb']:>12.4f} GB")
print(f"{'FP16':<10} {results_fp16['avg_ms']:>10.1f} {results_fp16['peak_vram_gb']:>12.4f} GB")

# ---------------------------------------------------------------------------
# STEP 5: Real-model load pattern (documentation)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("REAL MODEL MEASUREMENT PATTERN (documentation, not runnable here)")
print("=" * 60)
print("""
# After loading a real model:

import torch
from diffusers import AnimateDiffPipeline

torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()

pipe = AnimateDiffPipeline.from_pretrained("...", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

load_time = time.perf_counter() - t0
load_vram = torch.cuda.max_memory_allocated() / 1e9
print(f"Load: {load_time:.1f}s, peak VRAM: {load_vram:.2f} GB")

# Single inference
torch.cuda.reset_peak_memory_stats()
t1 = time.perf_counter()

out = pipe(prompt="a cat", num_frames=16, num_inference_steps=20)

infer_time = time.perf_counter() - t1
infer_vram = torch.cuda.max_memory_allocated() / 1e9
print(f"Inference: {infer_time:.1f}s, peak VRAM: {infer_vram:.2f} GB")

# Log it! Store these numbers so you can compare checkpoints.
results = {
    "model": "animatediff-v1-5-2",
    "dtype": "fp16",
    "load_s": load_time,
    "infer_s": infer_time,
    "peak_vram_load_gb": load_vram,
    "peak_vram_infer_gb": infer_vram,
    "num_frames": 16,
    "steps": 20,
}
import json
with open("benchmark_log.json", "a") as f:
    f.write(json.dumps(results) + "\\n")
""")

print("=" * 60)
print("THE 8 GB RULE FOR YOUR RTX 5060")
print("=" * 60)
print("""
Peak VRAM during inference:
  < 6.5 GB  → SAFE (comfortable headroom)
  6.5–7.5 GB → TIGHT (might OOM at higher res or more frames)
  > 7.5 GB  → DANGER (OOM likely at any variation)
  > 8.0 GB  → IMMEDIATE OOM

If peak > 7.5 GB, try:
  1. Reduce num_frames (16→8) or resolution
  2. Enable vae_slicing (processes one frame at a time in VAE)
  3. Enable sequential_cpu_offload (even more aggressive than model_cpu_offload)
  4. Use INT8 quantisation (halves weight VRAM)
""")

# Clean up
del model, model_fp16
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# =============================================================================
# EXPECTED OUTPUT:
# Device: cuda (or cpu)
# STEP 1: LOAD TIME
#   Parameters: ~200,000
#   VRAM after load: ~0.001 GB (tiny model)
# BENCHMARK: TinyUNet FP32
#   Avg inference time: ~X ms
#   Peak VRAM: ~0.002 GB
# BENCHMARK: TinyUNet FP16
#   Avg inference time: ~Y ms (Y < X — FP16 is faster)
# =============================================================================
