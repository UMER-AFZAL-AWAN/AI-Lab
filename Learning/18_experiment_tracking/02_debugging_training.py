# =============================================================================
# 18_experiment_tracking / 02 — DEBUGGING TRAINING
# =============================================================================
#
# LIKE YOU ARE 5:
# When a machine learning model doesn't train properly, it's like
# a student who keeps getting worse at maths despite studying.
# Something is wrong. This file shows how to diagnose what.
#
# C# ANALOGY:
# Like debugging a production service:
#   - Logs (TensorBoard) = Application Insights
#   - Gradient checking = unit test for the backward pass
#   - NaN detection = exception monitoring
#   - Loss explosion = memory leak detector
#
# =============================================================================

import torch
import torch.nn as nn
import math

torch.manual_seed(42)

print("=" * 60)
print("PART 1: THE THREE FAILURE MODES")
print("=" * 60)
print("""
Training failure shows up as one of three patterns:

1. LOSS DOESN'T DECREASE
   → Learning rate too low, or model is broken (all weights → 0)
   → Check: is loss better than random chance?

2. LOSS EXPLODES (NaN or ∞)
   → Learning rate too high, gradient explosion, bad data
   → Check: gradient norms, any NaN in data

3. LOSS DECREASES THEN PLATEAUS
   → Learning rate too high (bouncing over minimum)
   → Data issue (mislabelled, normalisation wrong)
   → Model capacity too low
""")

print("=" * 60)
print("PART 2: SANITY CHECKS BEFORE TRAINING")
print("=" * 60)
print("""
Run these BEFORE starting a full training run:
""")

class TinyClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )
    def forward(self, x):
        return self.net(x)

model = TinyClassifier(8, 3)

# ── Sanity check 1: Can it overfit a single batch? ───────────────────────────
print("Sanity check 1: Can the model overfit 1 batch?")
print("(If NO → the model or loss function is broken)")
X_one = torch.randn(4, 8)
y_one = torch.tensor([0, 1, 2, 0])
opt   = torch.optim.Adam(model.parameters(), lr=1e-2)
crit  = nn.CrossEntropyLoss()

for step in range(100):
    opt.zero_grad()
    loss = crit(model(X_one), y_one)
    loss.backward()
    opt.step()
    if step % 25 == 0:
        print(f"  step {step:3d}: loss={loss.item():.4f}")

final_loss = loss.item()
print(f"  Final: loss={final_loss:.6f}")
if final_loss < 0.01:
    print("  ✓ Overfit successful — model and loss function are working")
else:
    print("  ✗ Could not overfit — check model architecture or loss function")

print()
print("Sanity check 2: Output at random init (before any training)")
model_fresh = TinyClassifier(8, 3)
with torch.no_grad():
    logits = model_fresh(torch.randn(100, 8))
    probs  = torch.softmax(logits, dim=-1)
    print(f"  Random init mean probs per class: {probs.mean(dim=0).tolist()}")
    print(f"  Expected ≈ [0.33, 0.33, 0.33] (uniform across 3 classes)")

print()
print("=" * 60)
print("PART 3: GRADIENT CHECKING")
print("=" * 60)
print("""
Gradients flow backward through every layer.
If any layer has a bug, the gradient stops or becomes wrong.

Common gradient bugs:
  - Used non-differentiable operation (argmax, numpy, in-place on leaf)
  - Forgot .backward() or used wrong loss variable
  - Detached a tensor accidentally
  - Gradient vanishing: very deep net with no residual connections

How to check:
  1. Print gradient norms per parameter
  2. Check any parameter has grad=None (broken backprop)
  3. Watch for near-zero gradients (vanishing) or huge ones (exploding)
""")

model_check = TinyClassifier(8, 3)
opt_check   = torch.optim.Adam(model_check.parameters(), lr=1e-3)

X_chk = torch.randn(16, 8)
y_chk = torch.randint(0, 3, (16,))

opt_check.zero_grad()
loss_chk = nn.CrossEntropyLoss()(model_check(X_chk), y_chk)
loss_chk.backward()

print("Gradient norms per parameter:")
any_none = False
for name, param in model_check.named_parameters():
    if param.grad is None:
        print(f"  {name}: NONE ← broken! gradient not reaching this layer")
        any_none = True
    else:
        norm = param.grad.norm().item()
        status = "OK" if 1e-6 < norm < 100 else ("tiny (vanishing?)" if norm < 1e-6 else "HUGE (exploding!)")
        print(f"  {name}: grad_norm={norm:.4f}  [{status}]")

if not any_none:
    print("  ✓ All parameters received gradients")

print()
print("=" * 60)
print("PART 4: NAN DETECTION")
print("=" * 60)
print("""
NaN (Not a Number) in loss or gradients means something went wrong:
  - Division by zero
  - log(0) in CrossEntropyLoss with zero probability
  - sqrt(-x) due to negative values
  - Overflow: FP16 value exceeds max (65504) → ∞

How NaN spreads: once one NaN appears, it multiplies through the entire
backward pass. After one bad step, every weight becomes NaN → model is ruined.

Prevention:
  1. Clip gradients: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  2. Use GradScaler with FP16 (see 15_training_fundamentals/06)
  3. Detect early with hooks
""")

def check_for_nan(loss, model, step):
    """Call this every step or every N steps to catch NaNs early."""
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"  ✗ Step {step}: loss is {loss.item()} — training corrupted!")
        return True

    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"  ✗ Step {step}: NaN gradient in {name}")
                return True
            if torch.isinf(param.grad).any():
                print(f"  ✗ Step {step}: Inf gradient in {name}")
                return True
    return False

# Demonstrate: inject NaN to see detection
print("NaN detection demo:")
model_nan = TinyClassifier(8, 3)
opt_nan   = torch.optim.Adam(model_nan.parameters(), lr=1e-3)

for step in range(5):
    opt_nan.zero_grad()
    X_bad = torch.randn(4, 8)
    if step == 2:
        X_bad[0, 0] = float('nan')   # bad sample in data
    logits = model_nan(X_bad)
    loss   = nn.CrossEntropyLoss()(logits, torch.randint(0, 3, (4,)))
    if check_for_nan(loss, model_nan, step):
        print(f"  → Stopping at step {step} (NaN detected)")
        break
    loss.backward()
    opt_nan.step()
    print(f"  step {step}: loss={loss.item():.4f} (OK)")

print()
print("=" * 60)
print("PART 5: GRADIENT CLIPPING")
print("=" * 60)
print("""
Gradient clipping: if any gradient is too large, scale them all down.
  clip_grad_norm_(params, max_norm=1.0)
  → Computes total gradient norm across all params
  → If > max_norm: scales all gradients so total norm = max_norm

Where to put it in the training loop:
  loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  optimizer.step()

Typical values:
  NLP/Transformers: max_norm=1.0
  Diffusion models: max_norm=1.0
  RNNs: max_norm=0.25 (more unstable)
""")

model_clip = TinyClassifier(8, 3)
opt_clip   = torch.optim.Adam(model_clip.parameters(), lr=1e-3)

for step in range(3):
    opt_clip.zero_grad()
    loss = nn.CrossEntropyLoss()(model_clip(torch.randn(16, 8)), torch.randint(0, 3, (16,)))
    loss.backward()

    # Measure before clipping
    total_norm_before = sum(p.grad.norm()**2 for p in model_clip.parameters() if p.grad is not None) ** 0.5

    clip_val = torch.nn.utils.clip_grad_norm_(model_clip.parameters(), max_norm=0.1)

    total_norm_after = sum(p.grad.norm()**2 for p in model_clip.parameters() if p.grad is not None) ** 0.5

    opt_clip.step()
    print(f"  step {step}: grad_norm {total_norm_before.item():.4f} → {total_norm_after.item():.4f} (clipped)")

print()
print("=" * 60)
print("PART 6: DIAGNOSING LOSS CURVES")
print("=" * 60)
print("""
Pattern: train loss going down, val loss going UP early (epoch 2-5)
  → Severe overfitting. Try: smaller model, dropout, weight decay, more data.

Pattern: both losses plateau at a high value
  → Learning rate too low? Model too small? Wrong normalisation?

Pattern: loss goes to exactly 0 immediately
  → Data leakage (val set is identical to train set)

Pattern: loss oscillates wildly
  → Learning rate too high. Halve it.

Pattern: loss starts OK, then suddenly jumps to NaN at step N
  → Bad data at that point in the epoch, or gradient explosion.
  → Add gradient clipping; check the data batch that caused it.

Pattern: loss decreases but model outputs garbage
  → Loss function mismatch (e.g., MSE for classification)
  → Label encoding wrong (0-indexed vs 1-indexed)
  → Output not in expected range (forgot sigmoid / softmax)

The "overfit one batch" test from Part 2 rules out model/loss bugs.
""")

print("=" * 60)
print("QUICK REFERENCE: DEBUGGING CHECKLIST")
print("=" * 60)
print("""
Before training:
  ☐ Can the model overfit a single batch?
  ☐ Does the initial loss match theoretical value?
     (Binary cross-entropy on N classes: loss ≈ ln(N) = ln(3) ≈ 1.09)
  ☐ Are gradients flowing to all parameters?
  ☐ Is input data properly normalised?

During training:
  ☐ Is train loss decreasing?
  ☐ Any NaN in loss or gradients?
  ☐ Gradient norms staying stable (not exploding or vanishing)?
  ☐ Are val and train loss tracking together?

If stuck:
  ☐ Halve the learning rate
  ☐ Double the batch size
  ☐ Remove all augmentation temporarily
  ☐ Switch to Adam if using SGD
  ☐ Reduce model complexity
  ☐ Print 5 input samples and their labels — look at them
""")

# =============================================================================
# EXPECTED OUTPUT:
# Overfit test: loss approaching 0.0
# Gradient norms printed for each layer
# NaN detected at step 2
# Gradient clipping: norm after <= 0.1
# =============================================================================
