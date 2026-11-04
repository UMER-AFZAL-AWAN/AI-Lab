# =============================================================================
# 10_lora_and_fine_tuning / 01 — WHAT IS LORA?
# =============================================================================
#
# LIKE YOU ARE 5:
# A LoRA is a tiny sticky note you attach to a big model.
# Instead of rewriting the whole book (fine-tuning all 860M UNet params),
# you write a 2-page summary that says "when you see X, do Y differently."
# The book stays the same. Only the sticky note changes.
# Cost: ~20 MB instead of ~2 GB. Training time: hours instead of weeks.
#
# C# ANALOGY:
# Like a decorator pattern. The base class (UNet) stays sealed.
# You add a wrapper (LoRA) that intercepts specific method calls
# and nudges the output slightly — without changing the original assembly.
#
# HOW IT WORKS MATHEMATICALLY:
# A weight matrix W has shape (out, in).
# Full fine-tuning: update W directly → out×in params to train.
# LoRA: instead, train two small matrices A (out, rank) and B (rank, in).
#   New weight = W + scale × (A @ B)
#   where rank << out, rank << in  (usually 4–64)
#
# Example: W is (1280, 1280) → 1.6M params to train.
# LoRA rank=16: A=(1280,16) + B=(16,1280) → 41K params. 40x smaller.
#
# =============================================================================

import torch
import torch.nn as nn
import math

print("=" * 60)
print("THE LORA MATH")
print("=" * 60)

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that adds a LoRA branch.
    The original weight W is frozen. Only A and B are trained.
    """
    def __init__(self, in_features, out_features, rank=4, alpha=16):
        super().__init__()
        # Original weight — FROZEN during LoRA training
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.01
        )
        self.weight.requires_grad = False   # freeze

        # LoRA branch — these are trained
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * (1 / math.sqrt(rank)))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))  # init to zero → LoRA starts as identity

        # scale: alpha / rank. Alpha controls overall LoRA strength.
        # Keeping alpha=rank gives scale=1.0. Larger alpha = stronger effect.
        self.scale = alpha / rank

    def forward(self, x):
        # Base output (frozen weights)
        base = x @ self.weight.T
        # LoRA delta: x @ A.T @ B.T (two small matmuls)
        lora_delta = x @ self.lora_A.T @ self.lora_B.T
        return base + self.scale * lora_delta

    def lora_param_count(self):
        return self.lora_A.numel() + self.lora_B.numel()

    def total_param_count(self):
        return self.weight.numel() + self.lora_param_count()


# Demo
in_f, out_f, rank = 256, 256, 16

layer     = LoRALinear(in_f, out_f, rank=rank, alpha=rank)
full_params = out_f * in_f
lora_params = layer.lora_param_count()

print(f"Weight matrix size:    {in_f} × {out_f} = {full_params:,} params")
print(f"LoRA params (rank={rank}): {lora_params:,} params")
print(f"Compression ratio:     {full_params/lora_params:.1f}x fewer trainable params")
print()

x   = torch.randn(2, 10, in_f)   # batch=2, seq=10, features=256
out = layer(x)
print(f"Input shape:  {list(x.shape)}")
print(f"Output shape: {list(out.shape)}")

# Count trainable vs frozen
trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
frozen    = sum(p.numel() for p in layer.parameters() if not p.requires_grad)
print(f"Trainable params: {trainable:,}  (the LoRA A and B matrices)")
print(f"Frozen params:    {frozen:,}     (the original weight W)")

print()
print("=" * 60)
print("RANK: THE KEY HYPERPARAMETER")
print("=" * 60)

print(f"{'rank':>6} | {'LoRA params':>12} | {'% of full':>10} | {'typical use'}")
print("-" * 55)
for r in [1, 4, 8, 16, 32, 64, 128]:
    p = r * in_f + out_f * r
    pct = 100 * p / full_params
    use = {1: "style transfer (subtle)", 4: "concept (default)",
           8: "style (common)", 16: "detail (common)",
           32: "face likeness", 64: "structural change", 128: "near fine-tune"}.get(r, "")
    print(f"  {r:>4}   | {p:>12,} | {pct:>9.1f}% | {use}")

print()
print("RULE OF THUMB:")
print("  rank=4   → subtle style shift (LoRA for art style, lighting)")
print("  rank=16  → character/concept (most popular default)")
print("  rank=64+ → architectural changes (dangerous, may overfit quickly)")

print()
print("=" * 60)
print("WHERE LORA IS APPLIED IN A DIFFUSION MODEL")
print("=" * 60)
print("""
In AnimateDiff / Stable Diffusion UNet, LoRA is typically applied to:
  - Self-attention: Q, K, V, Output projection matrices
  - Cross-attention: Q, K, V, Output projection matrices
  - Sometimes: FFN up/down projections

NOT usually applied to:
  - Convolution layers (LoRA works on Linear; conv LoRA exists but rare)
  - Normalization layer weights
  - VAE (the latent codec stays fixed)
  - Text encoder (sometimes fine-tuned separately as "text LoRA")

Typical UNet LoRA covers ~144 Linear layers (for SD 1.5).
At rank=16: 144 × 2 × (16 × 768 + 768 × 16) ≈ 56M params
Full UNet:  860M params
LoRA size:  ~6% of full model → ~40 MB file vs 1.7 GB
""")

print("=" * 60)
print("LORA FILE FORMAT (.safetensors)")
print("=" * 60)
print("""
A LoRA file is a small .safetensors with keys like:
  lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight
  lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight
  ...

The naming maps to UNet layers. diffusers handles the key matching.
"lora_down" = matrix A (rank × in_features)
"lora_up"   = matrix B (out_features × rank)

In diffusers, loading a LoRA:
  pipe.load_lora_weights("path/to/lora.safetensors")
  pipe.set_adapters(["my_lora"], adapter_weights=[0.8])  # 0.0–1.0 strength

Unloading:
  pipe.unload_lora_weights()
""")

# =============================================================================
# EXPECTED OUTPUT:
# Weight matrix size:    256 × 256 = 65,536 params
# LoRA params (rank=16): 8,192 params
# Compression ratio:     8.0x fewer trainable params
# Input shape:  [2, 10, 256]
# Output shape: [2, 10, 256]
# Trainable params: 8,192
# Frozen params:    65,536
# =============================================================================
