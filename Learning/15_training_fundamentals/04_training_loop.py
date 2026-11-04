# =============================================================================
# 15_training_fundamentals / 04 — THE TRAINING LOOP
# =============================================================================
#
# LIKE YOU ARE 5:
# Training a neural network is like learning to shoot free throws.
# 1. Take a shot (forward pass).
# 2. See how far you missed (compute loss).
# 3. Figure out what you did wrong (backward pass / gradients).
# 4. Adjust your posture for next time (optimizer step).
# 5. Repeat thousands of times.
# After enough repetitions, you get good.
#
# C# ANALOGY:
# Like a main game loop: while (!done) { update(); render(); }
# forward() = render (what does the model produce?)
# loss.backward() = evaluate (how wrong was it?)
# optimizer.step() = update (fix the weights)
#
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

print("=" * 60)
print("TRAIN MODE vs EVAL MODE")
print("=" * 60)
print("""
Two behaviours differ between training and evaluation:
  Dropout:    training → randomly zero out neurons  |  eval → use all neurons
  BatchNorm:  training → compute stats from batch   |  eval → use running stats

ALWAYS set mode before running:
  model.train()  ← before the training loop
  model.eval()   ← before validation / inference

Forgetting model.eval() is a very common bug:
  Dropout at inference → results are RANDOM each call
  BatchNorm at inference → stats computed on 1-sample batches → wrong
""")

class ModelWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)
        self.drop = nn.Dropout(p=0.5)   # 50% chance of zeroing any neuron
    def forward(self, x):
        return self.fc(self.drop(x))

model = ModelWithDropout()
x = torch.ones(1, 4)

model.train()
out_train = [model(x).item() for _ in range(5)]
model.eval()
out_eval  = [model(x).item() for _ in range(5)]

print(f"model.train() outputs (varies each call):  {[f'{v:.3f}' for v in out_train]}")
print(f"model.eval() outputs  (same each call):    {[f'{v:.3f}' for v in out_eval]}")

print()
print("=" * 60)
print("THE COMPLETE TRAINING LOOP")
print("=" * 60)

# ─── Dataset ───────────────────────────────────────────────────────────────
# Synthetic regression: learn y = 2x₁ + 3x₂ + noise
N = 500
X = torch.randn(N, 2)
y_true = 2*X[:, 0] + 3*X[:, 1]
y = y_true + 0.1 * torch.randn(N)   # add small noise

# Split: 80% train, 20% validation
split = int(0.8 * N)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

train_ds = TensorDataset(X_train, y_train)
val_ds   = TensorDataset(X_val,   y_val)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=32, shuffle=False)

# ─── Model ─────────────────────────────────────────────────────────────────
class SimpleRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

model     = SimpleRegressor()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# ─── Training loop ─────────────────────────────────────────────────────────
print(f"{'Epoch':>5} | {'Train Loss':>12} | {'Val Loss':>12} | {'Status'}")
print("-" * 50)

for epoch in range(1, 11):
    # ── TRAIN ─────────────────────────────────────────────────────────────
    model.train()          # ← IMPORTANT
    train_loss = 0.0
    for X_batch, y_batch in train_dl:
        optimizer.zero_grad()          # 1. clear old gradients
        pred = model(X_batch)          # 2. forward pass
        loss = criterion(pred, y_batch) # 3. compute loss
        loss.backward()                # 4. backward pass (compute gradients)
        optimizer.step()               # 5. update weights
        train_loss += loss.item() * len(X_batch)
    train_loss /= len(train_ds)

    # ── VALIDATE ───────────────────────────────────────────────────────────
    model.eval()           # ← IMPORTANT
    val_loss = 0.0
    with torch.no_grad():  # ← saves memory and computation (no grad graph)
        for X_batch, y_batch in val_dl:
            pred = model(X_batch)
            val_loss += criterion(pred, y_batch).item() * len(X_batch)
    val_loss /= len(val_ds)

    status = "✓" if val_loss < 0.05 else "training..."
    print(f"  {epoch:>3}  | {train_loss:>12.4f} | {val_loss:>12.4f} | {status}")

print()
print("=" * 60)
print("OVERFITTING AND UNDERFITTING")
print("=" * 60)
print("""
Underfitting (high train loss AND high val loss):
  → Model too simple (too few layers/neurons)
  → Not trained long enough
  → Fix: larger model, more epochs

Overfitting (low train loss BUT high val loss):
  → Model memorised training data, doesn't generalise
  → Fix: more data, dropout, weight decay, early stopping

The loss curves tell the story:
  Both go down:            underfitting → still improving
  Train down, val down:    healthy training
  Train down, val plateaus: start of overfitting
  Train down, val UP:      overfitting — stop here (early stopping)
""")

# Demonstrate overfitting on a tiny dataset
X_tiny = torch.randn(20, 2)
y_tiny = 2*X_tiny[:, 0] + 3*X_tiny[:, 1]

class BigModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

big_model = BigModel()
opt_big   = optim.AdamW(big_model.parameters(), lr=1e-3)

for epoch in range(200):
    big_model.train()
    opt_big.zero_grad()
    pred = big_model(X_tiny)
    loss = criterion(pred, y_tiny)
    loss.backward()
    opt_big.step()

big_model.eval()
with torch.no_grad():
    train_loss_big = criterion(big_model(X_tiny), y_tiny).item()
    val_loss_big   = criterion(big_model(X_val), y_val).item()

print(f"Big model on tiny dataset:")
print(f"  Train loss: {train_loss_big:.4f}  (very low — memorised)")
print(f"  Val loss:   {val_loss_big:.4f}  (much higher — doesn't generalise)")
print(f"  → classic overfitting: train≈0 but val is high")

print()
print("=" * 60)
print("EPOCH vs STEP vs ITERATION")
print("=" * 60)
print(f"""
Dataset size: {N} samples, batch_size=32

  Step / Iteration = one forward+backward pass on ONE batch
    = one call to optimizer.step()
    = {N}÷32 ≈ {N//32} steps per epoch

  Epoch = one full pass through the ENTIRE training dataset
    = {N//32} steps
    = after one epoch, every training sample has been seen once

  10 epochs = {10*(N//32)} steps total

When reading papers:
  "trained for 100k steps" = 100,000 optimizer.step() calls
  "trained for 90 epochs"  = depends on dataset size
  Both mean the same type of thing, different units.
""")

# =============================================================================
# EXPECTED OUTPUT:
# Epoch 1: Train Loss ~0.5+, Val Loss ~0.5+
# Epoch 10: Train Loss ~0.01, Val Loss ~0.01  (model converges)
# Overfitting demo: train~0.0001, val~1.5+
# =============================================================================
