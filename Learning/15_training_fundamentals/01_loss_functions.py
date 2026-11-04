# =============================================================================
# 15_training_fundamentals / 01 — LOSS FUNCTIONS
# =============================================================================
#
# LIKE YOU ARE 5:
# The loss function is a scorecard. After the model makes a prediction,
# the scorecard says "you were THIS wrong." Training tries to make that number
# as small as possible. Different tasks need different scorecards.
#
# C# ANALOGY:
# Like a scoring interface: ILossFunction.Score(predicted, actual) → float
# Different implementations for different tasks — MSE for regression,
# CrossEntropy for classification, etc.
#
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("MSE — MEAN SQUARED ERROR  (regression, denoising)")
print("=" * 60)
#
# MSE = mean( (y_pred - y_true)² )
# Used for: image reconstruction, noise prediction in diffusion models,
#           depth estimation, any continuous-valued output.
#
# Squaring: larger errors are penalised more.  Error of 2 → penalty of 4.
# Error of 0.1 → penalty of 0.01.  So MSE cares a LOT about big mistakes.

y_true = torch.tensor([3.0, 5.0, 2.0, 7.0])
y_pred = torch.tensor([2.5, 5.5, 2.0, 6.0])

mse_manual = ((y_pred - y_true)**2).mean()
mse_torch  = F.mse_loss(y_pred, y_true)
print(f"y_true: {y_true.tolist()}")
print(f"y_pred: {y_pred.tolist()}")
print(f"MSE (manual): {mse_manual:.4f}")
print(f"MSE (torch):  {mse_torch:.4f}")
print()
print("In diffusion models: loss = MSE(noise_pred, actual_noise)")
print("  → model learns to predict the exact noise that was added each step")

print()
print("=" * 60)
print("MAE — MEAN ABSOLUTE ERROR  (robust to outliers)")
print("=" * 60)
#
# MAE = mean( |y_pred - y_true| )
# Same as MSE but without squaring → less sensitive to large errors.
# Use when dataset has outliers you don't want to overweight.

mae = F.l1_loss(y_pred, y_true)
print(f"MAE: {mae:.4f}  (errors: {(y_pred-y_true).abs().tolist()})")
print("MSE vs MAE: MSE punishes large errors harder. MAE treats them equally.")

print()
print("=" * 60)
print("BCE — BINARY CROSS-ENTROPY  (yes/no decisions)")
print("=" * 60)
#
# BCE = -[ y·log(ŷ) + (1-y)·log(1-ŷ) ]
# Used when output is a probability of ONE class (0=no, 1=yes).
# Examples: "is this image real or fake?" (GAN discriminator)
#           "does this pixel belong to the foreground?" (segmentation)

y_binary  = torch.tensor([1.0, 0.0, 1.0, 1.0])   # true labels (0 or 1)
y_pred_01 = torch.tensor([0.9, 0.1, 0.8, 0.3])    # predicted probabilities

bce = F.binary_cross_entropy(y_pred_01, y_binary)
print(f"y_true: {y_binary.tolist()}")
print(f"y_pred: {y_pred_01.tolist()}")
print(f"BCE: {bce:.4f}")
print("  Last prediction (true=1, pred=0.3): high loss because model was wrong")
print("  First prediction (true=1, pred=0.9): low loss because model was right")

print()
print("=" * 60)
print("CROSS-ENTROPY — multi-class classification")
print("=" * 60)
#
# Used when there are N mutually exclusive classes.
# nn.CrossEntropyLoss = Softmax + NegativeLogLikelihood in one step.
# Input: raw logits (not softmaxed). PyTorch handles the softmax internally.

logits = torch.tensor([
    [2.0, 1.0, 0.5],   # model thinks class 0 is most likely
    [0.1, 3.0, 0.2],   # model thinks class 1 is most likely
    [0.1, 0.1, 4.0],   # model thinks class 2 is most likely
])
targets = torch.tensor([0, 1, 2])   # correct classes

ce_loss = F.cross_entropy(logits, targets)
print(f"Logits: {logits.tolist()}")
print(f"Targets: {targets.tolist()}")
print(f"Cross-entropy loss: {ce_loss:.4f}  (low because predictions are correct)")

# Wrong predictions → higher loss
wrong_targets = torch.tensor([2, 0, 1])   # all wrong
ce_wrong = F.cross_entropy(logits, wrong_targets)
print(f"Cross-entropy (wrong targets): {ce_wrong:.4f}  (high because all wrong)")

print()
print("=" * 60)
print("HUBER LOSS — MSE + MAE hybrid  (used in RLHF, some diffusion variants)")
print("=" * 60)
#
# Behaves like MSE for small errors, like MAE for large errors.
# Robust to outliers while still being smooth near zero.

y_with_outlier = torch.tensor([1.0, 1.0, 1.0, 20.0])   # outlier at index 3
y_zero         = torch.zeros(4)

mse_outlier   = F.mse_loss(y_with_outlier, y_zero)
mae_outlier   = F.l1_loss(y_with_outlier, y_zero)
huber_outlier = F.huber_loss(y_with_outlier, y_zero)
print(f"With outlier (20.0 vs 0.0):")
print(f"  MSE: {mse_outlier:.2f}  ← dominated by outlier (20²=400 / 4 = 100)")
print(f"  MAE: {mae_outlier:.2f}  ← less dominated")
print(f"  Huber: {huber_outlier:.2f}  ← compromise")

print()
print("=" * 60)
print("COMBINED LOSSES — common in generative models")
print("=" * 60)
print("""
Most real models combine multiple losses:

VAE:
  loss = reconstruction_loss + β × KL_loss
  reconstruction_loss: how well does the decoded output match input?
  KL_loss: how close is the latent distribution to N(0,I)?

GAN discriminator:
  loss_D = BCE(real_pred, 1) + BCE(fake_pred, 0)
  → "predict 1 for real images, 0 for fake images"

GAN generator:
  loss_G = BCE(fake_pred, 1)   # fool the discriminator

Diffusion model:
  loss = MSE(predicted_noise, actual_noise)
  (simple! just noise prediction MSE at each timestep)

LoRA fine-tuning (SFT):
  loss = cross_entropy(predicted_token_logits, target_tokens)
  → standard language modelling loss on the target text
""")

# Demonstrate combined loss
recon = torch.randn(4, 3, 8, 8)      # reconstructed
orig  = torch.randn(4, 3, 8, 8)      # original
mu    = torch.randn(4, 16)           # VAE latent mean
log_v = torch.randn(4, 16)           # VAE latent log-variance

recon_loss = F.mse_loss(recon, orig)
kl_loss    = -0.5 * torch.mean(1 + log_v - mu.pow(2) - log_v.exp())
beta       = 0.001
total_loss = recon_loss + beta * kl_loss

print(f"VAE combined loss demo:")
print(f"  Reconstruction loss: {recon_loss:.4f}")
print(f"  KL loss:             {kl_loss:.4f}")
print(f"  β × KL:              {beta * kl_loss:.6f}")
print(f"  Total:               {total_loss:.4f}")

print()
print("=" * 60)
print("QUICK REFERENCE: WHICH LOSS FOR WHICH TASK")
print("=" * 60)
print("""
Task                           | Loss function
──────────────────────────────────────────────────────────
Image reconstruction (VAE)     | MSE (+ KL)
Noise prediction (diffusion)   | MSE
Classification (N classes)     | CrossEntropy
Binary classification (yes/no) | BCE
GAN discriminator              | BCE
Image quality (perceptual)     | VGG perceptual loss (not in PyTorch core)
Text generation (LLM)          | CrossEntropy on token logits
Video frame prediction         | MSE + perceptual loss
LoRA fine-tuning               | CrossEntropy (language modelling)
Regression (continuous output) | MSE or MAE or Huber
""")

# =============================================================================
# EXPECTED OUTPUT:
# MSE: ~0.19
# BCE: ~0.3 (last item has high individual loss)
# Cross-entropy (correct): ~0.08  (all predictions correct)
# Cross-entropy (wrong):   ~3.4   (all predictions wrong)
# =============================================================================
