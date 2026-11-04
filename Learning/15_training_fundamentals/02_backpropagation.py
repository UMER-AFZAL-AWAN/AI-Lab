# =============================================================================
# 15_training_fundamentals / 02 — BACKPROPAGATION
# =============================================================================
#
# LIKE YOU ARE 5:
# After the model makes a wrong prediction, we need to figure out
# WHO IS TO BLAME — which weights caused the mistake?
# Backpropagation is the blame-assignment algorithm.
# It starts at the loss (the final score) and walks BACKWARDS through the
# network, telling each weight "you caused THIS MUCH of the mistake."
#
# C# ANALOGY:
# Like a stack trace, but for responsibility.
# Forward pass = calling methods down the call stack.
# Backward pass = walking back up the stack, assigning blame at each frame.
# .grad on each parameter = "this parameter's share of the total loss."
#
# =============================================================================

import torch
import torch.nn as nn

print("=" * 60)
print("FORWARD AND BACKWARD PASS SIDE BY SIDE")
print("=" * 60)

# A tiny 2-layer network
# input → Linear(2→3) → ReLU → Linear(3→1) → loss

torch.manual_seed(0)
w1 = torch.randn(3, 2, requires_grad=True)   # first layer weights
b1 = torch.zeros(3,    requires_grad=True)
w2 = torch.randn(1, 3, requires_grad=True)   # second layer weights
b2 = torch.zeros(1,    requires_grad=True)

x      = torch.tensor([[1.0, 2.0]])   # input (1 sample, 2 features)
target = torch.tensor([[5.0]])         # desired output

# --- FORWARD PASS ---
h1     = x @ w1.T + b1          # (1, 3)
h1_act = torch.relu(h1)         # (1, 3)
out    = h1_act @ w2.T + b2     # (1, 1)
loss   = ((out - target)**2).mean()

print("FORWARD:")
print(f"  Input x:      {x.tolist()}")
print(f"  h1 = x@W1+b1: {h1.tolist()}")
print(f"  relu(h1):     {h1_act.tolist()}")
print(f"  out = h@W2+b2:{out.tolist()}")
print(f"  loss = MSE:   {loss.item():.4f}")

# --- BACKWARD PASS ---
loss.backward()   # PyTorch computes all gradients automatically

print("\nBACKWARD (gradients ∂loss/∂weight):")
print(f"  w2.grad: {w2.grad.tolist()}")
print(f"  b2.grad: {b2.grad.tolist()}")
print(f"  w1.grad:\n{w1.grad.tolist()}")
print(f"  b1.grad: {b1.grad.tolist()}")
print()
print("Each .grad value is: 'if this weight increases by 1, loss changes by this much'")
print("Negative grad → increasing the weight DECREASES loss → we should INCREASE it")
print("Positive grad → increasing the weight INCREASES loss → we should DECREASE it")

print()
print("=" * 60)
print("THE GRADIENT ACCUMULATION TRAP")
print("=" * 60)
#
# CRITICAL: PyTorch ADDS gradients each time .backward() is called.
# It does NOT clear the previous gradients automatically.
# You MUST call .zero_grad() or zero the gradients before each backward pass.

w = torch.tensor(2.0, requires_grad=True)

for step in range(3):
    loss_i = (w - 5.0)**2
    loss_i.backward()   # grad accumulates!
    print(f"  Step {step+1}: w.grad = {w.grad.item():.1f}  (should be {2*(w.item()-5):.1f} for ONE step)")

print(f"After 3 steps WITHOUT zeroing: w.grad = {w.grad.item()}")
print(f"Expected for one step:         {2*(w.item()-5.0):.1f}")
print("→ WRONG. Gradient is 3× too large because it accumulated.")

# Fix: zero_grad before each backward
print()
w = torch.tensor(2.0, requires_grad=True)
for step in range(3):
    if w.grad is not None:
        w.grad.zero_()      # ← MUST DO THIS
    loss_i = (w - 5.0)**2
    loss_i.backward()
    print(f"  Step {step+1} (with zero_grad): w.grad = {w.grad.item():.1f}  ✓")

print()
print("=" * 60)
print("GRADIENT FLOW THROUGH COMMON OPERATIONS")
print("=" * 60)
print("""
Operation          | Forward          | Backward (gradient)
──────────────────────────────────────────────────────────────
y = x + c          | add constant     | dy/dx = 1  (passes through unchanged)
y = c * x          | scale            | dy/dx = c
y = x²             | square           | dy/dx = 2x
y = max(0, x) ReLU | threshold at 0   | dy/dx = 0 if x<0, 1 if x>0
y = sigmoid(x)     | S-curve          | dy/dx = σ(x)·(1-σ(x))
y = softmax(x)_i   | exponentiate+norm| Jacobian (more complex)
y = LayerNorm(x)   | normalise        | propagates through normalisation
y = x @ W          | matmul           | dL/dW = x.T @ dL/dy
                   |                  | dL/dx = dL/dy @ W.T
""")

# Demonstrate: gradient of ReLU is 0 for negative inputs
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
y = torch.relu(x)
y.sum().backward()
print("ReLU gradient:")
for xi, yi, gi in zip(x, y, x.grad):
    print(f"  x={xi.item():5.1f} → relu={yi.item():.1f} → grad={gi.item():.1f}  {'(dead neuron)' if gi==0 else ''}")

print()
print("=" * 60)
print("DETACH AND NO_GRAD — stopping gradient flow")
print("=" * 60)
print("""
Sometimes you want to compute something but NOT backpropagate through it.

torch.no_grad():
  Context manager — no computation graph built inside.
  Used for inference (saves memory, faster).

x.detach():
  Creates a new tensor with the same values but NO gradient history.
  Used when: computing a "target" that shouldn't be updated
             (e.g., target network in DQN, EMA model).
""")

# torch.no_grad() demo
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
with torch.no_grad():
    y = x * 2   # no grad tracked
print(f"y.requires_grad (inside no_grad): {y.requires_grad}")  # False

# .detach() demo
z = x * 3          # tracks grad
z_detached = z.detach()
print(f"z.requires_grad:          {z.requires_grad}")         # True
print(f"z_detached.requires_grad: {z_detached.requires_grad}") # False
print("→ z_detached can be used as a target without affecting gradients")

print()
print("=" * 60)
print("GRADIENT CLIPPING — preventing explosions")
print("=" * 60)
print("""
Deep networks can produce HUGE gradients — especially with long sequences.
If gradients are too large, the weight update overshoots and the model diverges.

Fix: clip gradients to a maximum norm before the optimizer step.
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  → if ||grad|| > 1.0, scale all gradients down so ||grad|| = 1.0

Standard values:
  Transformers/LLMs: max_norm = 1.0
  CNNs: rarely needed (but 1.0 is safe)
  RNNs: max_norm = 5.0 (RNNs are especially prone to explosion)
""")

model = nn.Linear(4, 2)
x = torch.randn(1, 4)
y = torch.randn(1, 2)
loss = nn.MSELoss()(model(x), y)
loss.backward()

before_clip = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
after_clip  = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
print(f"Gradient norm before clip: {before_clip:.4f}")
print(f"Gradient norm after clip:  {after_clip:.4f}  (≤ 0.1)")

# =============================================================================
# EXPECTED OUTPUT:
# loss = MSE: some positive number
# Step 1 (with zero_grad): w.grad = -6.0  ✓
# ReLU gradient: x=-2.0 → grad=0.0 (dead neuron), x=1.0 → grad=1.0
# z_detached.requires_grad: False
# =============================================================================
