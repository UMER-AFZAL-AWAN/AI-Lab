# =============================================================================
# 18_experiment_tracking / 01 — TENSORBOARD AND LOGGING
# =============================================================================
#
# LIKE YOU ARE 5:
# When you train a model, you want to watch it learn over time.
# "Is the loss going down? Is it going down smoothly?"
# TensorBoard is a web dashboard that shows live training graphs.
# You log numbers during training, TensorBoard draws the charts.
#
# C# ANALOGY:
# Like Application Insights or Serilog with a dashboard:
#   logger.LogMetric("loss", value, step=step_number)
# TensorBoard = the monitoring dashboard
# SummaryWriter = the logger client
#
# =============================================================================

import torch
import torch.nn as nn
import os

print("=" * 60)
print("PART 1: SummaryWriter — logging to TensorBoard")
print("=" * 60)
print("""
SummaryWriter writes event files that TensorBoard reads.

Basic usage:
  from torch.utils.tensorboard import SummaryWriter
  writer = SummaryWriter("runs/experiment_name")

  # Log a scalar (loss, accuracy, lr)
  writer.add_scalar("Loss/train", loss.item(), global_step=step)

  # Log images (what does the model see?)
  writer.add_image("Sample/input", image_tensor, step)

  # View: tensorboard --logdir runs/

  writer.close()
""")

# Simulate training and log metrics (runs without TensorBoard installed —
# installs it only if present, skips gracefully if not)
def simulate_training_with_logging():
    """
    Train a tiny model and log metrics.
    Uses TensorBoard if available, otherwise prints to console.
    """
    # Build a minimal model and dataset
    torch.manual_seed(42)
    model     = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    X = torch.randn(200, 4)
    y = X[:, 0] * 2 + X[:, 1] - 1 + 0.1 * torch.randn(200, 1)

    log_dir = "runs/demo_experiment"

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir)
        use_tb = True
        print(f"  TensorBoard: logging to {log_dir}")
        print(f"  → Run: tensorboard --logdir runs/ then open http://localhost:6006")
    except ImportError:
        writer = None
        use_tb = False
        print("  (TensorBoard not installed — printing metrics only)")
        print("  Install: pip install tensorboard")

    for step in range(100):
        idx    = torch.randperm(200)[:32]
        Xb, yb = X[idx], y[idx]

        optimizer.zero_grad()
        pred = model(Xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            # Validation
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X), y).item()
            model.train()

            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  step {step:3d}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}, lr={lr_now:.5f}")

            if use_tb and writer:
                writer.add_scalar("Loss/train", loss.item(), step)
                writer.add_scalar("Loss/val",   val_loss,    step)
                writer.add_scalar("LR",         lr_now,      step)

    if use_tb and writer:
        # Log the model graph (optional — shows the architecture)
        dummy_input = torch.randn(1, 4)
        try:
            writer.add_graph(model, dummy_input)
        except Exception:
            pass
        writer.close()
        print(f"  Logs saved to {log_dir}/")

simulate_training_with_logging()

print()
print("=" * 60)
print("PART 2: LOGGING IMAGES AND HISTOGRAMS")
print("=" * 60)
print("""
For image models, log the actual images at each checkpoint:
  - What does the model generate at step 1000?
  - What does it generate at step 10000?
  - Visually check if quality improves.

TensorBoard image logging:
  # Single image: tensor shape (C, H, W), values [0, 1]
  writer.add_image("Generated/sample", gen_image, step)

  # Multiple images in a grid:
  from torchvision.utils import make_grid
  grid = make_grid(images, nrow=4)   # (C, H, W) grid image
  writer.add_image("Generated/grid", grid, step)

  # Video (for AnimateDiff):
  # shape: (N, T, C, H, W), values [0, 1]
  writer.add_video("Generated/video", video, step, fps=8)
""")

# Demonstrate make_grid (no TensorBoard needed)
from torchvision.utils import make_grid

fake_images = torch.rand(8, 3, 64, 64)   # 8 generated images
grid        = make_grid(fake_images, nrow=4, normalize=False)
print(f"8 images (3×64×64) → grid shape: {grid.shape}")
print(f"Grid is (3, 2*64, 4*64) = one image with 2 rows of 4")

print()
print("Histogram logging (understand weight distributions):")
print("""
  # Track how weights change across training
  for name, param in model.named_parameters():
      writer.add_histogram(f"Weights/{name}", param, step)
      if param.grad is not None:
          writer.add_histogram(f"Gradients/{name}", param.grad, step)

  Healthy model: weights spread out (not all near 0 = dead neurons)
                 gradients consistent in scale (not exploding/vanishing)
""")

print("=" * 60)
print("PART 3: ALTERNATIVE LOGGERS")
print("=" * 60)
print("""
TensorBoard: local, built-in with PyTorch
Weights & Biases (wandb): cloud-based, more features, free for individuals
  pip install wandb
  wandb login  # one-time setup

  import wandb
  wandb.init(project="my-project", name="experiment-1")
  wandb.log({"loss": loss.item(), "step": step})
  wandb.log({"image": wandb.Image(tensor)})

MLflow: open source, can self-host
  import mlflow
  mlflow.log_metric("loss", loss.item(), step=step)
  mlflow.log_artifact("model.pth")

Comparison:
  TensorBoard:  simple, local, already installed with PyTorch
  wandb:        best UI, compare multiple runs, cloud sync
  MLflow:       enterprise, model registry, artifacts tracking
""")

print("=" * 60)
print("PART 4: WHAT TO LOG")
print("=" * 60)
print("""
Scalars (every N steps):
  train_loss       ← primary signal — should decrease
  val_loss         ← gap vs train_loss reveals overfitting
  learning_rate    ← confirm schedule is applied
  grad_norm        ← should be <1.0; spikes = instability

Images (every few hundred steps):
  input_sample     ← what the model sees (after normalisation)
  generated_sample ← what the model outputs
  For diffusion: at multiple denoising steps

Histograms (every few hundred steps):
  model weights    ← should spread out, not collapse to 0
  activations      ← layer outputs; dead neurons show up here
  gradients        ← scale should be similar across layers

Hyperparameters (once per run):
  writer.add_hparams(
      {"lr": lr, "batch_size": bs, "dropout": p},
      {"hparam/best_val_loss": best_val_loss}
  )
  → Lets you compare runs in TensorBoard's HParams tab.

Good rule: if you can't see it in TensorBoard, you're flying blind.
""")

print("=" * 60)
print("QUICK REFERENCE: COMMON COMMANDS")
print("=" * 60)
print("""
# Start TensorBoard
tensorboard --logdir runs/

# Multiple experiment dirs
tensorboard --logdir exp1:runs/exp1,exp2:runs/exp2

# Clean up old logs
rm -rf runs/   # or: for f in runs/*: rm -rf $f

# In Jupyter notebook:
%load_ext tensorboard
%tensorboard --logdir runs/
""")

# =============================================================================
# EXPECTED OUTPUT:
# Training steps with decreasing loss (step 0→90)
# Grid shape: torch.Size([3, 128, 256])
# No TensorBoard errors if not installed (graceful skip)
# =============================================================================
