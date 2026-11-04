# =============================================================================
# 16_classical_ml / 01 — LINEAR & LOGISTIC REGRESSION
# =============================================================================
#
# LIKE YOU ARE 5:
# Linear regression: draw the best straight line through points.
#   "Height increases 2cm per year" — that's linear regression.
# Logistic regression: same idea but the output is a probability (0 to 1).
#   "What's the chance this email is spam?" — that's logistic regression.
#
# C# ANALOGY:
# Linear regression = a linear function: f(x) = m*x + b
# Logistic regression = that function wrapped in a Sigmoid:
#   f(x) = 1 / (1 + exp(-(m*x + b)))
#   Outputs range [0, 1] instead of (-∞, +∞)
#
# WHY THESE MATTER FOR AI/ML:
# - Every neural network is stacked logistic regressions (roughly)
# - Understanding these FROM SCRATCH builds the intuition for backprop
# - Logistic regression is still used heavily in industry (fast, explainable)
#
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import math

torch.manual_seed(42)

print("=" * 60)
print("PART 1: LINEAR REGRESSION FROM SCRATCH")
print("=" * 60)
print("""
Goal: given feature x, predict output y.
      y_pred = w*x + b   (a line through 2D space)

Loss: MSE = (1/N) × Σ (y_pred_i - y_true_i)²
      "how far off are we, on average?"

Training: gradient descent — nudge w and b downhill on the loss surface.
""")

# ── Generate data: y = 2x + 1 + noise ──────────────────────────────────────
N = 100
X = torch.linspace(-3, 3, N).unsqueeze(1)   # shape (100, 1)
y = 2 * X + 1 + 0.3 * torch.randn(N, 1)     # true: w=2, b=1, with noise

# ── Manual implementation (no nn.Linear) ────────────────────────────────────
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

lr = 0.05
for step in range(200):
    y_pred = w * X + b
    loss   = ((y_pred - y) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_()
    b.grad.zero_()

print(f"Manual GD result:  w={w.item():.4f} (true: 2.0),  b={b.item():.4f} (true: 1.0)")

# ── PyTorch nn.Linear ────────────────────────────────────────────────────────
model = nn.Linear(1, 1)
opt   = optim.SGD(model.parameters(), lr=0.05)
crit  = nn.MSELoss()

for _ in range(200):
    opt.zero_grad()
    crit(model(X), y).backward()
    opt.step()

w_nn = model.weight.item()
b_nn = model.bias.item()
print(f"nn.Linear result:  w={w_nn:.4f} (true: 2.0),  b={b_nn:.4f} (true: 1.0)")

# ── Multiple linear regression ───────────────────────────────────────────────
print()
print("Multiple linear regression: y = w1*x1 + w2*x2 + b")
X2    = torch.randn(200, 2)
y2    = 3*X2[:, 0:1] - 2*X2[:, 1:2] + 0.5 + 0.1*torch.randn(200, 1)

model2 = nn.Linear(2, 1)
opt2   = optim.AdamW(model2.parameters(), lr=1e-2)
for _ in range(500):
    opt2.zero_grad()
    nn.MSELoss()(model2(X2), y2).backward()
    opt2.step()

w1, w2 = model2.weight[0].tolist()
b2     = model2.bias.item()
print(f"  Learned: w1={w1:.3f} (true: 3), w2={w2:.3f} (true: -2), b={b2:.3f} (true: 0.5)")

print()
print("=" * 60)
print("PART 2: SIGMOID — THE BRIDGE TO CLASSIFICATION")
print("=" * 60)
print("""
Problem with linear regression for classification:
  y = w*x + b can output any value (-∞, +∞)
  But a probability must be in [0, 1]

Fix: squash the output through Sigmoid:
  σ(z) = 1 / (1 + e^(-z))
  → z → -∞: σ → 0
  → z →  0: σ = 0.5
  → z → +∞: σ → 1

Decision rule: if σ(w*x + b) > 0.5 → class 1, else → class 0
  Equivalently: if w*x + b > 0 → class 1  (the decision boundary)
""")

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))

for z in [-3, -1, 0, 1, 3]:
    prob = sigmoid(z)
    bar  = "█" * int(prob * 20)
    print(f"  z={z:+d}  →  σ={prob:.3f}  {bar}")

print()
print("=" * 60)
print("PART 3: LOGISTIC REGRESSION FROM SCRATCH")
print("=" * 60)

# ── Binary classification data: two clusters ────────────────────────────────
# Class 0: centred at (-2, -2)
# Class 1: centred at (+2, +2)
X_cls = torch.cat([
    torch.randn(100, 2) + torch.tensor([-2.0, -2.0]),   # class 0
    torch.randn(100, 2) + torch.tensor([ 2.0,  2.0]),   # class 1
], dim=0)
y_cls = torch.cat([torch.zeros(100), torch.ones(100)])  # labels 0/1

# ── Manual logistic regression ───────────────────────────────────────────────
w_lr = torch.zeros(2, requires_grad=True)
b_lr = torch.tensor(0.0, requires_grad=True)
opt_lr = optim.SGD([w_lr, b_lr], lr=0.1)

print(f"{'Step':>6} | {'Loss':>8} | {'Accuracy':>10}")
for step in range(300):
    opt_lr.zero_grad()
    logits   = X_cls @ w_lr + b_lr          # (200,)
    probs    = torch.sigmoid(logits)         # (200,) in [0,1]
    # Binary cross-entropy:  -(y·log(p) + (1-y)·log(1-p))
    loss     = nn.BCELoss()(probs, y_cls)
    loss.backward()
    opt_lr.step()
    if step % 100 == 99:
        preds    = (probs > 0.5).float()
        accuracy = (preds == y_cls).float().mean().item()
        print(f"  {step+1:>4}  | {loss.item():>8.4f} | {accuracy*100:>9.1f}%")

print()
print(f"Learned weights: w={w_lr.tolist()}")
print(f"Decision boundary: {w_lr[0].item():.2f}*x1 + {w_lr[1].item():.2f}*x2 + {b_lr.item():.2f} = 0")

print()
print("=" * 60)
print("PART 4: MULTI-CLASS (SOFTMAX REGRESSION)")
print("=" * 60)
print("""
Logistic regression for 2 classes → Softmax regression for K classes.

Instead of 1 output → K outputs (one "score" per class)
Softmax converts scores to probabilities:
  softmax(z_i) = e^z_i / Σ_j e^z_j

Loss: CrossEntropyLoss  (the standard for classification)
  = -log(probability of the correct class)

nn.CrossEntropyLoss in PyTorch combines LogSoftmax + NLLLoss.
You pass raw logits (scores), NOT probabilities.
""")

X_mc = torch.randn(300, 4)   # 300 samples, 4 features
# 3 classes: label = argmax of a linear transform + noise
W_true = torch.randn(4, 3)
y_mc   = (X_mc @ W_true + 0.1*torch.randn(300, 3)).argmax(dim=1)

model_mc = nn.Linear(4, 3)   # 4 input features, 3 classes
opt_mc   = optim.AdamW(model_mc.parameters(), lr=1e-2)
crit_mc  = nn.CrossEntropyLoss()   # expects raw logits, not softmax output

for step in range(300):
    opt_mc.zero_grad()
    logits = model_mc(X_mc)       # shape (300, 3) — raw scores, not probabilities
    loss   = crit_mc(logits, y_mc)
    loss.backward()
    opt_mc.step()

acc = (model_mc(X_mc).argmax(1) == y_mc).float().mean().item()
print(f"3-class softmax regression accuracy: {acc*100:.1f}%")
print()
print("Note: probabilities from logits:")
example_logits = torch.tensor([2.0, 0.5, -1.0])
probs = torch.softmax(example_logits, dim=0)
print(f"  logits={example_logits.tolist()}  → softmax={[f'{p:.3f}' for p in probs.tolist()]}")
print(f"  sum = {probs.sum().item():.4f}  (always sums to 1.0)")

print()
print("=" * 60)
print("QUICK REFERENCE")
print("=" * 60)
print("""
Task                  | Output activation | Loss
──────────────────────────────────────────────────────────
Regression            | None (linear)     | MSELoss
Binary classification | Sigmoid           | BCELoss (or BCEWithLogitsLoss)
Multi-class classif.  | Softmax           | CrossEntropyLoss (pass raw logits)

BCEWithLogitsLoss = Sigmoid + BCELoss combined (more numerically stable)
CrossEntropyLoss  = Softmax + NLLLoss combined  (always use this — not Softmax + CE manually)
""")

# =============================================================================
# EXPECTED OUTPUT:
# Manual GD result: w≈2.0, b≈1.0
# Multi-class accuracy: ~80-95%
# Softmax output sums to 1.0
# =============================================================================
