# =============================================================================
# 15_training_fundamentals / 03 — OPTIMIZERS AND LEARNING RATE SCHEDULES
# =============================================================================
#
# LIKE YOU ARE 5:
# After backprop tells every weight "you caused this much mistake",
# the optimizer decides HOW to fix the weights.
# "Walk downhill" (gradient descent) is the basic idea.
# But smart optimizers remember where they've been walking (momentum)
# and step carefully in directions that have been consistent (adaptive lr).
#
# C# ANALOGY:
# Like different sorting algorithms — all try to minimise the same thing,
# but with different strategies.
# SGD = bubble sort (simple, works, slow).
# Adam = quicksort (smart, adapts, usually faster).
#
# =============================================================================

import torch
import torch.nn as nn

print("=" * 60)
print("SGD — STOCHASTIC GRADIENT DESCENT (the original)")
print("=" * 60)
print("""
w ← w - lr × grad

"Stochastic" = uses a random mini-batch, not the whole dataset.
  → each step is noisy (not the exact gradient of the full dataset)
  → but much faster per step, and the noise can help escape local minima

Hyperparameters:
  lr = 0.01–0.1  (needs tuning; much more sensitive than Adam)
  momentum = 0.9  (adds "velocity": keeps moving in the same direction)
  weight_decay = 1e-4  (L2 regularisation penalty, see 05_regularization.py)
""")

torch.manual_seed(42)
# Single parameter optimisation demo — minimise L(w) = (w-3)²
def make_param():
    return torch.tensor(10.0, requires_grad=True)

def run_optimizer(opt_class, opt_kwargs, steps=30, label=""):
    w = make_param()
    opt = opt_class([w], **opt_kwargs)
    history = []
    for _ in range(steps):
        opt.zero_grad()
        loss = (w - 3.0)**2
        loss.backward()
        opt.step()
        history.append(loss.item())
    return history, w.item()

histories = {}
for name, cls, kw in [
    ("SGD lr=0.1",      torch.optim.SGD,  {"lr": 0.1}),
    ("SGD+mom lr=0.1",  torch.optim.SGD,  {"lr": 0.1, "momentum": 0.9}),
    ("Adam lr=0.1",     torch.optim.Adam, {"lr": 0.1}),
    ("AdamW lr=0.1",    torch.optim.AdamW,{"lr": 0.1, "weight_decay": 0.01}),
]:
    hist, final_w = run_optimizer(cls, kw, steps=30)
    histories[name] = hist
    print(f"  {name:22}  final w={final_w:.4f},  final loss={hist[-1]:.6f}")

print()
print("=" * 60)
print("ADAM — ADAPTIVE MOMENT ESTIMATION (default for DL)")
print("=" * 60)
print("""
Adam tracks TWO running averages for each weight:
  m = β₁·m + (1-β₁)·grad          ← 1st moment: smoothed gradient (momentum)
  v = β₂·v + (1-β₂)·grad²         ← 2nd moment: smoothed gradient² (variance)

  update = lr × m̂ / (√v̂ + ε)      ← m̂, v̂ are bias-corrected

What this achieves:
  - Momentum (m): steps are smooth — doesn't jitter in noisy gradients
  - Adaptive lr (v): weights that rarely get large gradients get a bigger step;
                     weights that always get large gradients get a smaller step
  - Per-weight lr: each weight has its own effective learning rate

Defaults (almost never change these):
  β₁ = 0.9      (momentum decay — higher = smoother)
  β₂ = 0.999    (variance decay — higher = more memory)
  ε  = 1e-8     (numerical stability, prevent divide-by-zero)

Typical lr values:
  Training from scratch:    lr = 1e-3
  Fine-tuning a model:      lr = 1e-4 to 1e-5
  LoRA fine-tuning:         lr = 2e-4
  LLM pre-training:         lr = 3e-4 (then cosine-decayed)
""")

print("=" * 60)
print("ADAMW — ADAM WITH DECOUPLED WEIGHT DECAY (modern standard)")
print("=" * 60)
print("""
Bug in Adam: weight_decay in Adam is not true L2 regularisation —
it gets scaled by the adaptive lr, making it inconsistent across weights.

Fix (AdamW): apply weight decay DIRECTLY to weights, not through the gradient.
  w ← w × (1 - lr·λ) - lr × m̂/(√v̂+ε)   ← weight decay applied separately

In practice: use AdamW for almost everything. It's Adam but correct.
  torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
""")

print("=" * 60)
print("LEARNING RATE SCHEDULES — changing lr during training")
print("=" * 60)
print("""
Why: start high (fast learning), end low (fine-grained convergence).
Or: warmup first (let Adam's momentum statistics stabilise), then decay.

── Cosine Annealing ────────────────────────────────────────
lr(t) = lr_min + 0.5·(lr_max - lr_min)·(1 + cos(π·t/T))
Gradually decreases lr following a cosine curve. Standard for LLMs.

── Linear Warmup ───────────────────────────────────────────
lr(t) = lr_max × (t / warmup_steps)   for t < warmup_steps
Then decay. Prevents big gradient steps at the start before Adam warms up.

── Step Decay ──────────────────────────────────────────────
lr drops by a factor (e.g. ×0.1) every N epochs. Simple, common for CNNs.

── OneCycleLR ──────────────────────────────────────────────
Ramps up lr, then decays. Very fast convergence. Good for fine-tuning.
""")

# Demonstrate schedulers
model = nn.Linear(4, 2)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Cosine annealing
scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
lrs_cos = []
for _ in range(20):
    lrs_cos.append(optimizer.param_groups[0]['lr'])
    optimizer.step()
    scheduler_cos.step()

# Reset
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler_step = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
lrs_step = []
for _ in range(20):
    lrs_step.append(optimizer.param_groups[0]['lr'])
    optimizer.step()
    scheduler_step.step()

print("Cosine schedule (20 steps):  ", [f"{lr:.5f}" for lr in lrs_cos[::4]])
print("Step schedule (÷2 every 5):  ", [f"{lr:.5f}" for lr in lrs_step[::4]])

print()
print("=" * 60)
print("LINEAR WARMUP + COSINE DECAY (standard LLM/diffusion recipe)")
print("=" * 60)

def get_lr_with_warmup_cosine(step, warmup_steps, total_steps, lr_max, lr_min=0):
    """The standard warmup+cosine schedule used in most modern models."""
    import math
    if step < warmup_steps:
        return lr_max * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))

total, warmup = 100, 10
lrs_warmup = [get_lr_with_warmup_cosine(s, warmup, total, 1e-3) for s in range(total)]
print("Warmup+Cosine schedule (every 10 steps):")
for i in range(0, 100, 10):
    bar = "█" * int(lrs_warmup[i] * 30000)
    print(f"  step {i:3d}: lr={lrs_warmup[i]:.5f}  {bar}")

print()
print("=" * 60)
print("QUICK REFERENCE: WHICH OPTIMIZER FOR WHAT")
print("=" * 60)
print("""
Task                           | Optimizer         | lr
──────────────────────────────────────────────────────────
Training a CNN from scratch    | AdamW             | 1e-3
Training a Transformer         | AdamW + warmup    | 3e-4
Fine-tuning (LoRA)             | AdamW             | 2e-4
Fine-tuning (full)             | AdamW             | 1e-5
Diffusion model training       | AdamW             | 1e-4
GAN training                   | Adam (no wd)      | 2e-4
Classic CV (ResNet, etc.)      | SGD + momentum    | 0.1 (with decay)

Rule: when unsure, use AdamW with lr=1e-4.
      Then try 10× higher and 10× lower to find the sweet spot.
""")

# =============================================================================
# EXPECTED OUTPUT:
# Adam lr=0.1   final w≈3.0, final loss≈0.0
# Cosine schedule: starts at 1e-3, decays to ~0
# =============================================================================
