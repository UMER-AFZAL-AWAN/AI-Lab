# =============================================================================
# 00_math_foundations / 03 — PROBABILITY AND STATISTICS FOR AI
# =============================================================================
#
# LIKE YOU ARE 5:
# AI models don't say "the answer IS this."
# They say "the answer is probably this, with 87% confidence."
# Probability is the language AI uses to express uncertainty.
# This file teaches the parts of probability that show up everywhere in AI.
#
# C# ANALOGY:
# Like System.Random, but instead of uniform random numbers,
# you work with weighted distributions (some outcomes are more likely).
# softmax() = turning raw scores into a valid probability distribution.
# sampling = Random.NextWeighted(probabilities).
#
# =============================================================================

import torch
import torch.nn.functional as F
import numpy as np

print("=" * 60)
print("PROBABILITY DISTRIBUTIONS")
print("=" * 60)
print("""
A probability distribution assigns a probability to every possible outcome.
Two kinds:
  Discrete: countable outcomes (which word comes next? → vocabulary of 50k words)
  Continuous: infinite outcomes (what pixel value? → any float in [0, 1])

Rules:
  All probabilities ≥ 0
  All probabilities sum to 1 (discrete) or integrate to 1 (continuous)
""")

# Discrete: next-word distribution (tiny vocabulary)
vocab = ["cat", "dog", "plays", "piano", "runs"]
raw_logits = torch.tensor([2.0, 1.0, 3.0, 1.5, 0.5])  # raw scores from a language model
probs = F.softmax(raw_logits, dim=0)                    # convert to probabilities

print("Next-word distribution:")
for word, logit, prob in zip(vocab, raw_logits, probs):
    bar = "█" * int(prob * 30)
    print(f"  {word:8} logit={logit:.1f}  P={prob:.3f}  {bar}")
print(f"  Sum of probabilities: {probs.sum().item():.4f}  (must = 1.0)")

print()
print("=" * 60)
print("SOFTMAX — the universal probability converter")
print("=" * 60)
print("""
softmax(x)_i = exp(xᵢ) / Σ exp(xⱼ)

Takes ANY vector of real numbers (logits) → valid probability distribution.
Key properties:
  - All outputs in (0, 1)
  - Outputs sum to 1
  - Larger input → larger probability (order preserved)
  - Temperature: softmax(x/T)
      T→0 : spiky (winner takes all, like argmax)
      T→∞ : flat (uniform, maximum uncertainty)
""")

logits = torch.tensor([1.0, 2.0, 3.0])
for temp in [0.1, 1.0, 5.0]:
    p = F.softmax(logits / temp, dim=0)
    print(f"  T={temp:3}: {[f'{v:.3f}' for v in p.tolist()]}  (entropy={(-p*p.log()).sum():.3f})")

print()
print("=" * 60)
print("GAUSSIAN (NORMAL) DISTRIBUTION — the bell curve")
print("=" * 60)
print("""
Most important continuous distribution in AI.
  N(μ, σ²): mean μ, variance σ², standard deviation σ

Where you see it:
  - Weight initialisation: weights ~ N(0, 0.02²)  (small random values)
  - Noise in diffusion:    ε ~ N(0, I)  (pure Gaussian noise)
  - VAE latent:            z ~ N(μ, σ²)  (reparameterisation trick)
  - Gradient noise:        mini-batch gradients ≈ N(true_grad, σ²)
""")

torch.manual_seed(42)
# Sampling from a Gaussian
mu, sigma = 2.0, 0.5
samples = torch.normal(mean=mu, std=sigma, size=(10000,))
print(f"N({mu}, {sigma}²): mean={samples.mean():.3f}, std={samples.std():.3f}  (should be ≈{mu}, {sigma})")

# Reparameterisation trick (used in VAE — see 03_autoencoder_vae/03_vae.py)
# Instead of sampling z ~ N(μ, σ²) directly (not differentiable),
# write z = μ + σ·ε where ε ~ N(0, 1)  (ε is the random part, μ and σ are learnable)
eps = torch.randn(1000)
z   = mu + sigma * eps      # reparameterised sample
print(f"Reparameterised: mean={z.mean():.3f}, std={z.std():.3f}  (same distribution, but differentiable!)")

print()
print("=" * 60)
print("KL DIVERGENCE — measuring difference between distributions")
print("=" * 60)
print("""
KL(P || Q) = Σ P(x) · log(P(x) / Q(x))

Measures: how different is distribution Q from "true" distribution P?
  KL = 0   → P and Q are identical
  KL > 0   → Q is different from P (never negative)
  KL is NOT symmetric: KL(P||Q) ≠ KL(Q||P)

Where you see it in AI:
  VAE loss:     KL(q(z|x) || N(0,I))  → pushes latent toward a standard Gaussian
  LLM training: KL(human probs || model probs)  → language modelling objective
  RLHF:         KL(policy || reference_policy)  → keeps fine-tuned model close to base
""")

# KL between two discrete distributions
P = torch.tensor([0.4, 0.3, 0.2, 0.1])   # true distribution
Q1 = torch.tensor([0.4, 0.3, 0.2, 0.1])  # same as P
Q2 = torch.tensor([0.1, 0.2, 0.3, 0.4])  # reversed

kl_same = F.kl_div(Q1.log(), P, reduction='sum')
kl_diff = F.kl_div(Q2.log(), P, reduction='sum')
print(f"KL(P || P) = {kl_same:.6f}  (identical → 0)")
print(f"KL(P || Q_reversed) = {kl_diff:.4f}  (very different → large)")

# VAE closed-form KL (for unit Gaussian prior):
mu_vae     = torch.tensor([0.5, -0.3])
log_var    = torch.tensor([0.1, -0.5])
kl_vae     = -0.5 * torch.sum(1 + log_var - mu_vae.pow(2) - log_var.exp())
print(f"\nVAE KL (latent vs N(0,I)): {kl_vae:.4f}")
print("  (this is what the VAE loss penalises — see 03_autoencoder_vae/03_vae.py)")

print()
print("=" * 60)
print("ENTROPY — measuring uncertainty")
print("=" * 60)
print("""
H(P) = -Σ P(x) · log P(x)

High entropy = high uncertainty (flat distribution, could be anything)
Low entropy  = low uncertainty (spiky distribution, one clear winner)

Maximum entropy = uniform distribution (all equally likely)
Minimum entropy = deterministic (one outcome has probability 1)
""")

def entropy(p):
    p = p.clamp(min=1e-9)
    return -(p * p.log()).sum().item()

uniform   = torch.ones(4) / 4          # maximum uncertainty
spiky     = torch.tensor([0.97, 0.01, 0.01, 0.01])  # near-certain
very_flat = torch.ones(100) / 100

print(f"Uniform over 4:  H = {entropy(uniform):.4f}")
print(f"Near-certain:    H = {entropy(spiky):.4f}")
print(f"Uniform over 100: H = {entropy(very_flat):.4f}  (more options → higher max entropy)")

print()
print("=" * 60)
print("CROSS-ENTROPY LOSS — the standard classification loss")
print("=" * 60)
print("""
Cross-entropy loss between true labels and predicted probabilities:
  L = -Σ y_true · log(y_pred)

For one-hot labels (one class is correct):
  L = -log(predicted_probability_of_correct_class)

Intuition: how surprised was the model by the correct answer?
  Predicted 0.99 for the correct class → L = -log(0.99) ≈ 0.01  (not surprised → low loss)
  Predicted 0.01 for the correct class → L = -log(0.01) ≈ 4.6   (very surprised → high loss)
""")

logits_cls = torch.tensor([[2.0, 1.0, 0.1],   # batch item 1: logits for 3 classes
                            [0.5, 2.5, 0.1]])  # batch item 2
labels     = torch.tensor([0, 1])              # correct class indices

loss = F.cross_entropy(logits_cls, labels)
print(f"Logits: {logits_cls.tolist()}")
print(f"Labels: {labels.tolist()}  (class 0 for item 1, class 1 for item 2)")
print(f"Cross-entropy loss: {loss.item():.4f}")

# Manual verification for item 1
p_item1 = F.softmax(logits_cls[0], dim=0)
manual_loss_1 = -torch.log(p_item1[0]).item()   # correct class is 0
print(f"Manual: -log(P(class=0)) = -log({p_item1[0]:.4f}) = {manual_loss_1:.4f}")

print()
print("=" * 60)
print("BAYES' THEOREM — updating beliefs with evidence")
print("=" * 60)
print("""
P(A|B) = P(B|A) × P(A) / P(B)

  P(A):    prior — what you believed before seeing B
  P(B|A):  likelihood — how probable is B if A is true
  P(A|B):  posterior — updated belief after seeing B

Example: "Is this image a cat?"
  P(cat) = 0.3           (prior: 30% of images are cats)
  P(fur | cat) = 0.9     (90% of cat images have visible fur)
  P(fur) = 0.5           (50% of all images have visible fur)

  P(cat | fur) = 0.9 × 0.3 / 0.5 = 0.54
  → seeing fur raises cat probability from 30% to 54%

In deep learning: Bayesian thinking shows up in:
  - Probabilistic models (VAE, diffusion)
  - Uncertainty estimation
  - Bayesian neural networks (rarely used in practice, but conceptually important)
  - Dropout at test time ≈ approximate Bayesian inference
""")
p_cat = 0.3; p_fur_given_cat = 0.9; p_fur = 0.5
p_cat_given_fur = p_fur_given_cat * p_cat / p_fur
print(f"P(cat | fur) = {p_fur_given_cat} × {p_cat} / {p_fur} = {p_cat_given_fur:.2f}")

# =============================================================================
# EXPECTED OUTPUT:
# Softmax T=0.1: [~0, ~0, ~1.0]  (spiky)
# Softmax T=5.0: [~0.3, ~0.3, ~0.4] (flat)
# KL(P || P) = 0.000000
# Uniform over 4: H = 1.3863
# Cross-entropy loss: ~0.4
# P(cat | fur) = 0.54
# =============================================================================
