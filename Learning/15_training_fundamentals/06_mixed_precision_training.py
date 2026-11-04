# =============================================================================
# 15_training_fundamentals / 06 — MIXED PRECISION TRAINING
# =============================================================================
#
# LIKE YOU ARE 5:
# FP32 (full precision) is like writing every measurement to 10 decimal places.
# FP16 (half precision) is like writing to 5 decimal places — less accurate,
# but takes half the space and runs twice as fast on modern GPUs.
# Mixed precision = use FP16 where it's safe, FP32 where it matters.
# Your model trains 2-3x faster and uses half the VRAM.
#
# C# ANALOGY:
# Like using float vs double in C#.
# float for fast intermediate calculations.
# double for accumulating the final result (to avoid rounding errors).
# autocast = the compiler deciding automatically which one to use.
#
# =============================================================================

import torch
import torch.nn as nn
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

print()
print("=" * 60)
print("THE VRAM SAVINGS")
print("=" * 60)
print("""
FP32: 4 bytes per parameter
FP16: 2 bytes per parameter  (2× VRAM savings for weights)
BF16: 2 bytes per parameter  (same size, different range — better for LLMs)

For AnimateDiff UNet (860M parameters):
  FP32: 860M × 4 = 3.44 GB
  FP16: 860M × 2 = 1.72 GB  ← this is why we use torch_dtype=torch.float16

BUT: training in pure FP16 is unstable:
  - Gradients can become 0 or ∞ due to limited range
  - Very small weight updates underflow to 0 and are lost

FIX: Mixed precision training
  - Forward pass: FP16 (fast, low memory)
  - Loss scaling:  multiply loss by a large number so gradients don't underflow
  - Backward pass: FP16 (fast)
  - Optimizer step: FP32 (safe, keeps precision for weight updates)
  - Master weights: stored in FP32
""")

print("=" * 60)
print("AUTOCAST + GRADSCALER: THE STANDARD RECIPE")
print("=" * 60)
print("""
Two tools:
  torch.autocast("cuda"):  runs the forward pass in FP16/BF16 automatically
  torch.cuda.amp.GradScaler(): scales loss up → prevents gradient underflow
                               automatically adjusts scale factor

The 5-line recipe:
  scaler = GradScaler()
  with autocast("cuda"):
      output = model(input)
      loss   = criterion(output, target)
  scaler.scale(loss).backward()
  scaler.step(optimizer)
  scaler.update()
""")

class TinyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10),
        )
    def forward(self, x):
        return self.net(x)

model     = TinyConvNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
x         = torch.randn(16, 3, 64, 64, device=device)
targets   = torch.randint(0, 10, (16,), device=device)

# ─── FP32 training (baseline) ───────────────────────────────────────────────
t0 = time.perf_counter()
for _ in range(10):
    optimizer.zero_grad()
    out  = model(x)
    loss = criterion(out, targets)
    loss.backward()
    optimizer.step()
time_fp32 = time.perf_counter() - t0
vram_fp32 = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
print(f"FP32 training (10 steps): {time_fp32:.3f}s,  VRAM peak: {vram_fp32:.1f} MB")

# ─── Mixed precision training ───────────────────────────────────────────────
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    scaler = torch.cuda.amp.GradScaler()   # loss scaler

    t0 = time.perf_counter()
    for _ in range(10):
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out  = model(x)
            loss = criterion(out, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    time_amp = time.perf_counter() - t0
    vram_amp = torch.cuda.max_memory_allocated() / 1e6
    print(f"AMP  training (10 steps): {time_amp:.3f}s,  VRAM peak: {vram_amp:.1f} MB")
    print(f"AMP speedup: {time_fp32/time_amp:.1f}x  VRAM savings: {(vram_fp32-vram_amp)/vram_fp32*100:.0f}%")
else:
    print("(AMP demo skipped — no CUDA device)")

print()
print("=" * 60)
print("BF16 vs FP16 — WHICH ONE TO USE")
print("=" * 60)
print("""
Both are 16-bit (2 bytes each), but different internal format:

         | Exponent bits | Mantissa bits | Range          | Precision
─────────────────────────────────────────────────────────────────────
FP32     |     8         |     23        | ±3.4e38        | High
FP16     |     5         |     10        | ±65504         | Medium
BF16     |     8         |      7        | ±3.4e38 (!)    | Lower

BF16 has the SAME RANGE as FP32 → no overflow/underflow during training
    but LESS precision (fewer mantissa bits)

FP16 has limited range → can overflow (∞ for large values, 0 for tiny values)
    → needs GradScaler to compensate

Rule:
  Nvidia Ampere (A100, 3090) and newer → use BF16 (simpler, safer)
  Your RTX 5060 → supports BF16 (Ada Lovelace architecture)
  Older GPUs (V100, 2080) → use FP16 + GradScaler

For inference (not training): both work fine.
  AnimateDiff: torch_dtype=torch.float16
  Wan / LLMs: torch_dtype=torch.bfloat16   ← Wan specifically uses BF16
""")

# dtype sizes
for name, dt in [("float32", torch.float32), ("float16", torch.float16), ("bfloat16", torch.bfloat16)]:
    t = torch.zeros(1, dtype=dt)
    print(f"  {name}: {t.element_size()} bytes/element")

print()
print("=" * 60)
print("THE LOSS SCALER EXPLAINED")
print("=" * 60)
print("""
Problem: gradient of 1e-7 in FP16 → rounds to 0.0 (underflow → lost update)

GradScaler fix:
  1. Scale loss UP: loss_scaled = loss × scale_factor  (scale_factor ~= 2^16)
  2. Backward on loss_scaled → gradients are 2^16× larger → don't underflow
  3. Before optimizer step: unscale gradients (divide by scale_factor)
  4. If any gradient is inf/nan: SKIP this optimizer step (don't corrupt weights)
  5. If N consecutive steps are clean: increase scale_factor (×2)
  6. If a step was skipped: decrease scale_factor (÷2)

This is all automatic. You just call:
  scaler.scale(loss).backward()
  scaler.step(optimizer)   ← automatically unscales and skips if inf
  scaler.update()          ← adjusts scale_factor
""")

if torch.cuda.is_available():
    scaler = torch.cuda.amp.GradScaler()
    print(f"GradScaler initial scale: {scaler.get_scale():.0f}")
    # After several successful steps it would increase; after overflow it decreases

print()
print("=" * 60)
print("AUTOCAST IN INFERENCE (what diffusers uses)")
print("=" * 60)
print("""
For inference, you don't need GradScaler — just autocast:

pipe = AnimateDiffPipeline.from_pretrained(..., torch_dtype=torch.float16)
# ^ loads weights directly in FP16 — different from autocast (which keeps FP32 weights)

OR manually:
with torch.autocast("cuda", dtype=torch.float16):
    output = model(input_fp32)
# autocast converts FP32 input to FP16 for the forward pass automatically

The difference:
  torch_dtype=torch.float16  → weights stored in FP16 (saves VRAM permanently)
  autocast                   → weights in FP32, computation in FP16 (saves compute)
  For inference, torch_dtype is preferred (also saves VRAM).
""")

# =============================================================================
# EXPECTED OUTPUT:
# FP32: some time and VRAM usage
# AMP: faster time, less VRAM (on CUDA)
# BF16/FP16: 2 bytes each vs FP32's 4 bytes
# =============================================================================
