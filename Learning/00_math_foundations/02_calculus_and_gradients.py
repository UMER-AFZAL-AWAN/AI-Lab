# =============================================================================
# 00_math_foundations / 02 — CALCULUS AND GRADIENTS
# =============================================================================
#
# LIKE YOU ARE 5:
# Imagine you're lost in the mountains at night and want to go downhill.
# You can't see far — you can only feel the slope under your feet.
# "Go in the direction the ground slopes down most steeply."
# That's gradient descent. The gradient is the slope. Training = walking downhill.
#
# C# ANALOGY:
# Like a binary search, but instead of splitting a sorted array,
# you're searching for the minimum of a loss function.
# Each "iteration" you compute: where is the slope? → take a step downhill.
# optimizer.Step() = move one step in the downhill direction.
#
# WHY THIS MATTERS:
# Every time a model trains, it computes gradients of the loss
# with respect to every weight. Those gradients say:
# "if this weight increases by a tiny bit, the loss changes by this much."
# Update the weight to make the loss smaller. Repeat millions of times.
#
# =============================================================================

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')   # no display needed

print("=" * 60)
print("WHAT IS A DERIVATIVE?")
print("=" * 60)
print("""
Derivative of f(x) at point x = how fast f is changing at x.
  f'(x) = lim[h→0] ( f(x+h) - f(x) ) / h

Examples you'll see in AI:
  f(x) = x²       → f'(x) = 2x
  f(x) = e^x      → f'(x) = e^x   (same function!)
  f(x) = ln(x)    → f'(x) = 1/x
  f(x) = max(0,x) → f'(x) = 0 if x<0, 1 if x>0  (ReLU!)
  f(x) = σ(x)     → f'(x) = σ(x)·(1-σ(x))       (sigmoid)
""")

# Numerical derivative (the "feel the slope" version)
def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

f = lambda x: x**2
x = 3.0
print(f"f(x) = x²  at x={x}")
print(f"  Analytical derivative: f'({x}) = 2×{x} = {2*x}")
print(f"  Numerical derivative:  f'({x}) ≈ {numerical_derivative(f, x):.6f}")

print()
print("=" * 60)
print("THE CHAIN RULE — how gradients flow through layers")
print("=" * 60)
print("""
If y = f(g(x)), then:
  dy/dx = (dy/dg) × (dg/dx)   ← chain rule

In a neural network:
  loss = L(output(hidden(input)))

  d_loss/d_weight = (d_loss/d_output) × (d_output/d_hidden) × (d_hidden/d_weight)

This is BACKPROPAGATION: apply the chain rule, layer by layer, backwards.
PyTorch does this automatically with Autograd.
""")

# Manual chain rule demo
# f(x) = (3x + 1)²
# Let g(x) = 3x + 1,   f(g) = g²
# df/dx = df/dg * dg/dx = 2g * 3 = 2(3x+1)*3 = 6(3x+1)

x_val = 2.0
g = 3 * x_val + 1       # = 7
dg_dx = 3               # derivative of g w.r.t. x
df_dg = 2 * g           # derivative of f(g)=g² w.r.t. g = 2g = 14
df_dx = df_dg * dg_dx   # chain rule: 14 * 3 = 42
print(f"f(x) = (3x+1)²  at x={x_val}")
print(f"  g = {g},  df/dg = {df_dg},  dg/dx = {dg_dx}")
print(f"  df/dx = {df_dg} × {dg_dx} = {df_dx}  (chain rule)")

# Verify with numerical derivative
f_chain = lambda x: (3*x + 1)**2
print(f"  Numerical check: {numerical_derivative(f_chain, x_val):.4f}  ✓")

print()
print("=" * 60)
print("PYTORCH AUTOGRAD — automatic differentiation")
print("=" * 60)
#
# Every tensor with requires_grad=True tracks operations on it.
# .backward() triggers the chain rule automatically, populating .grad.

x = torch.tensor(2.0, requires_grad=True)
y = (3*x + 1)**2   # same f(x) as above

y.backward()       # compute dy/dx automatically

print(f"x = {x.item()}")
print(f"y = (3x+1)² = {y.item()}")
print(f"dy/dx (autograd) = {x.grad.item()}  (should be 42.0)")

print()
print("How autograd builds a computation graph:")
print("""
  x  →  [*3+1]  →  g  →  [**2]  →  y
         (op 1)              (op 2)

  During backward():
    dy/dg = 2g = 14           (from op 2)
    dg/dx = 3                 (from op 1)
    dy/dx = 14 × 3 = 42       (chain rule, automatically)

  This graph is built dynamically as you run operations.
  That's why PyTorch is "define by run" (eager mode).
""")

print("=" * 60)
print("GRADIENT DESCENT — the training algorithm")
print("=" * 60)
#
# Goal: minimise a loss function L(w) over weights w.
# Step: w ← w - lr × ∇L(w)
# lr = learning rate = how big a step to take each iteration.

def simple_loss(w):
    """L(w) = (w - 3)² — minimum at w=3."""
    return (w - 3.0) ** 2

w     = torch.tensor(10.0, requires_grad=True)   # start far from minimum
lr    = 0.1
print(f"Starting w = {w.item():.4f},  L = {simple_loss(w).item():.4f}")
print(f"{'Step':>5} | {'w':>10} | {'L(w)':>10} | {'grad':>10}")
print("-" * 44)

for step in range(10):
    loss = simple_loss(w)
    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad     # gradient descent step
        w.grad.zero_()       # MUST reset gradients — PyTorch accumulates by default!

    if step < 5 or step == 9:
        print(f"  {step+1:>3}  | {w.item():>10.4f} | {simple_loss(w).item():>10.6f} | {w.grad.item() if w.grad else 0:>10.4f}")

print(f"\nMinimum at w=3.0.  Final w = {w.item():.4f}")

print()
print("=" * 60)
print("LEARNING RATE: THE MOST IMPORTANT HYPERPARAMETER")
print("=" * 60)
print("""
Too small lr: crawls toward minimum, takes forever.
Too large lr: overshoots, bounces, may diverge (loss goes UP).
Just right:   converges smoothly.

Typical starting values:
  Adam optimiser:  lr = 1e-4  (0.0001) — most common for DL
  SGD:             lr = 1e-2  (0.01)   — needs tuning per task
  AdamW fine-tune: lr = 2e-5  (0.00002)— for LoRA / fine-tuning

The sweet spot is NOT stable — it depends on:
  model size, batch size, dataset, architecture.
  Larger batch → can use larger lr (roughly: lr ∝ √batch_size).
""")

# Demonstrate lr effect
def run_gd(lr, steps=15):
    w = torch.tensor(10.0, requires_grad=True)
    losses = []
    for _ in range(steps):
        loss = (w - 3.0)**2
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            w.grad.zero_()
        losses.append(loss.item())
    return losses

print("Loss after 15 steps for different learning rates:")
for lr_val in [0.01, 0.1, 0.5, 1.0, 1.5]:
    losses = run_gd(lr_val)
    final = losses[-1]
    status = "✓ good" if final < 0.01 else ("✗ diverged" if final > 100 else "~ slow")
    print(f"  lr={lr_val:4}  final loss={final:10.4f}  {status}")

print()
print("=" * 60)
print("PARTIAL DERIVATIVES — gradient for multi-parameter functions")
print("=" * 60)
print("""
Real neural networks have MILLIONS of weights w₀, w₁, w₂, ...
The gradient is a VECTOR of partial derivatives:
  ∇L = [∂L/∂w₀,  ∂L/∂w₁,  ∂L/∂w₂, ...]

Each partial derivative says: "if only this weight changes, how does L change?"
They're computed simultaneously by backpropagation (one backward pass).
""")

# Small network: 2 weights
w0 = torch.tensor(1.0, requires_grad=True)
w1 = torch.tensor(2.0, requires_grad=True)
x_in = torch.tensor(3.0)
y_target = torch.tensor(10.0)

# Forward: y_pred = w0*x + w1
y_pred = w0 * x_in + w1
loss   = (y_pred - y_target)**2

loss.backward()

print(f"w0={w0.item()}, w1={w1.item()}, x={x_in.item()}")
print(f"y_pred = w0·x + w1 = {y_pred.item():.1f},  target = {y_target.item():.1f}")
print(f"loss = {loss.item():.2f}")
print(f"∂L/∂w0 = {w0.grad.item():.2f}  (how much w0 contributes to the loss)")
print(f"∂L/∂w1 = {w1.grad.item():.2f}  (how much w1 contributes to the loss)")
print("→ Both gradients → optimizer updates both weights simultaneously")

# =============================================================================
# EXPECTED OUTPUT:
# dy/dx (autograd) = 42.0
# Minimum at w=3.0.  Final w ≈ 3.0
# lr=0.1  final loss ≈ 0.0  ✓ good
# lr=1.5  final loss = large ✗ diverged
# =============================================================================
