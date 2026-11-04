# =============================================================================
# 15_training_fundamentals / 05 — REGULARIZATION
# =============================================================================
#
# LIKE YOU ARE 5:
# Regularization = rules that stop the model from cheating.
# Without rules, the model memorises the training data
# (like a student who memorises answers without understanding).
# Regularization forces the model to actually learn patterns, not answers.
#
# C# ANALOGY:
# Like coding standards + unit tests that prevent over-engineering.
# Dropout = randomly delete methods during code review — forces redundancy.
# Weight decay = penalise huge classes — forces simplicity.
# Batch norm = standardise inputs at every layer — prevents internal drift.
#
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

print("=" * 60)
print("DROPOUT — randomly zero out neurons during training")
print("=" * 60)
print("""
During training: each neuron is set to 0 with probability p.
  → Network must learn redundant representations
  → Can't rely on any single neuron → more robust features

During evaluation: all neurons active, outputs scaled by (1-p).
  → Deterministic: same input → same output

Hyperparameter: p = dropout rate
  p=0.0 → no dropout (off)
  p=0.1 → mild (common for CNNs, residual networks)
  p=0.5 → aggressive (classic LSTM/Dense layers)

Where used in modern models:
  Transformer: usually p=0.0–0.1 (most modern models use very little dropout)
  Dense classification head: p=0.5
  AnimateDiff motion module: p=0.1
""")

x = torch.ones(1, 10)   # 10 neurons, all active
drop50  = nn.Dropout(p=0.5)
drop10  = nn.Dropout(p=0.1)

drop50.train(); drop10.train()
print("Training mode (neurons randomly zeroed):")
print(f"  p=0.5: {drop50(x).tolist()}")
print(f"  p=0.1: {drop10(x).tolist()}")

drop50.eval(); drop10.eval()
print("Eval mode (all neurons active):")
print(f"  p=0.5: {drop50(x).tolist()}  (all 1.0)")
print(f"  p=0.1: {drop10(x).tolist()}  (all 1.0)")

print()
print("=" * 60)
print("WEIGHT DECAY (L2 REGULARISATION) — penalise large weights")
print("=" * 60)
print("""
Adds a penalty to the loss: L_total = L_task + λ × Σ w²

Effect: optimiser pushes all weights toward zero.
  Large weights = complex, overfit-prone model → penalised heavily.
  Small weights = simpler model → lower penalty.
  → Prevents any single weight from becoming too dominant.

Implementation in PyTorch:
  AdamW(params, weight_decay=0.01)   ← 0.01 is a common default

Typical values:
  0.0    → no regularisation
  0.01   → moderate (standard for most tasks)
  0.1    → strong (may underfit if too high)

L1 vs L2:
  L2 (weight_decay): penalises w², pushes weights small but non-zero
  L1: penalises |w|, pushes weights to exactly zero (feature selection)
  L2 is more common in deep learning.
""")

# Demonstrate: weight decay slows weight growth
def train_with_wd(wd, steps=50):
    w = nn.Linear(1, 1, bias=False)
    nn.init.ones_(w.weight)
    opt = torch.optim.AdamW(w.parameters(), lr=0.1, weight_decay=wd)
    x  = torch.ones(1, 1)
    target = torch.tensor([[10.0]])   # target is very large → weights want to grow
    for _ in range(steps):
        opt.zero_grad()
        loss = F.mse_loss(w(x), target)
        loss.backward()
        opt.step()
    return w.weight.item()

for wd in [0.0, 0.01, 0.1, 1.0]:
    final_w = train_with_wd(wd)
    print(f"  weight_decay={wd:4}: final weight = {final_w:.4f}  {'← kept small' if wd > 0 else '← unrestricted'}")

print()
print("=" * 60)
print("BATCH NORMALISATION — stabilise layer inputs during training")
print("=" * 60)
print("""
Problem: as weights change, the distribution of activations shifts layer by layer.
  Layer 3's input distribution changes every step because Layers 1 & 2 changed.
  This is called "internal covariate shift."
  → Deep networks train slowly / unstably.

Fix (Batch Norm, 2015): after each layer, normalise the activations.
  For each feature j:
    x̂_j = (x_j - μ_j) / (σ_j + ε)
    y_j  = γ_j × x̂_j + β_j         ← learnable scale and shift

  μ and σ computed over the BATCH at training time.
  At eval time: use running mean/std accumulated during training.

Effect:
  Stabilises training → can use higher learning rates → faster convergence
  Slight regularisation effect (similar to dropout)

Where used:
  CNNs: very common (after every conv)
  Transformers: replaced by LayerNorm (see below)
""")

# BatchNorm demo
bn = nn.BatchNorm2d(4)   # 4 channels
x  = torch.randn(8, 4, 16, 16)   # batch=8, channels=4, 16×16

bn.train()
out_train = bn(x)
print(f"Input  mean: {x.mean():.4f}  std: {x.std():.4f}")
print(f"BN out mean: {out_train.mean():.4f}  std: {out_train.std():.4f}  (normalised → ~0, ~1)")

print()
print("=" * 60)
print("LAYER NORMALISATION — the Transformer alternative to BatchNorm")
print("=" * 60)
print("""
Batch Norm normalises ACROSS the batch dimension.
  Problem: at inference with batch_size=1, stats are unreliable.
  Problem: doesn't work for variable-length sequences (NLP).

Layer Norm normalises ACROSS the feature dimension for EACH sample.
  x̂ = (x - μ) / (σ + ε)   ← μ and σ computed per-sample, per-layer
  y = γ × x̂ + β

No batch-size dependency → works for batch=1, sequences of any length.
That's why every Transformer (GPT, BERT, CLIP, AnimateDiff UNet) uses LayerNorm.
""")

ln = nn.LayerNorm(64)   # normalise last dim
x  = torch.randn(2, 10, 64)   # batch=2, seq=10, features=64

out_ln = ln(x)
print(f"LayerNorm input  mean: {x.mean():.4f}  std: {x.std():.4f}")
print(f"LayerNorm output mean: {out_ln.mean():.4f}  std: {out_ln.std():.4f}")
# Each (seq, feature) slice is normalised independently:
print(f"Per-sample per-token mean: {out_ln[0, 0].mean():.4f}  std: {out_ln[0, 0].std():.4f}")

print()
print("=" * 60)
print("DATA AUGMENTATION — regularise through diversity")
print("=" * 60)
print("""
Artificially expand the training set by transforming existing samples.
  → Model sees more variation → generalises better
  → Effectively: regularisation through data diversity

Common augmentations for images:
  RandomHorizontalFlip:   flip 50% of images horizontally
  RandomRotation(15):     rotate by up to 15 degrees
  ColorJitter:            random brightness/contrast/saturation
  RandomCrop:             crop a random subregion
  Normalize:              subtract mean, divide by std (not augmentation but required)

For videos:
  Apply the same spatial transform consistently across frames
  Random temporal crop: select a random 16-frame window from a longer clip
  Temporal flip: reverse frame order (sometimes)

In diffusion training:
  Random horizontal flip of the training images
  Random cropping (then resize to target size)
  NOT color augmentation (can confuse the colour semantics)
""")

from torchvision import transforms

augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

print("Augmentation pipeline:")
for t in augment.transforms:
    print(f"  {t.__class__.__name__}")

print()
print("=" * 60)
print("QUICK REFERENCE: WHICH REGULARISER WHEN")
print("=" * 60)
print("""
Regulariser      | When to use
───────────────────────────────────────────────────────────
Dropout          | Dense layers, MLPs; use p=0.1–0.3 for Transformers
Weight decay     | Almost always (AdamW weight_decay=0.01)
BatchNorm        | CNNs, residual networks
LayerNorm        | Transformers (always), anything sequential
Data augmentation| When dataset is small or model overfits
Early stopping   | Monitor val loss; stop when it starts rising
Gradient clip    | RNNs, Transformers with long sequences (max_norm=1.0)
""")

# =============================================================================
# EXPECTED OUTPUT:
# weight_decay=0.0 → final weight grows large
# weight_decay=0.1 → final weight kept much smaller
# LayerNorm: per-token mean≈0, std≈1
# =============================================================================
