# =============================================================================
# 19_computer_vision / 03 — CLASSIFICATION FINE-TUNING
# =============================================================================
#
# LIKE YOU ARE 5:
# A pretrained ResNet already learned to see: edges, textures, shapes, objects.
# Fine-tuning = "you already know how to look; now learn to recognise MY objects."
# We replace the last layer (which says "cat vs dog" for ImageNet)
# with a new layer that says "my_class_1 vs my_class_2".
# Then we train, but only lightly — the pretrained features are already good.
#
# C# ANALOGY:
# Like inheriting from a base class and overriding ONE method.
# Base class = pretrained model (learned from 1.2M images)
# Override = replace final Linear(1000) with Linear(num_my_classes)
# Fine-tune = run .train() on a small dataset with your data
#
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet18_Weights, ViT_B_16_Weights
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

print("=" * 60)
print("PART 1: LOADING A PRETRAINED MODEL")
print("=" * 60)
print("""
torchvision.models has 80+ pretrained models.
Every model comes with a 'weights' enum that bundles:
  - The pretrained weights
  - The transforms used at training time
  - The ImageNet class names

Always use the matching transforms — using different normalisation
causes the model to see out-of-distribution inputs → worse accuracy.
""")

# Load ResNet-18 pretrained on ImageNet
weights = ResNet18_Weights.DEFAULT
model   = models.resnet18(weights=weights)
transform = weights.transforms()   # official inference transforms

print(f"ResNet-18 pretrained:")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Final layer: {model.fc}")

# See what we'd get on imagenet
model.eval()
dummy = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    logits = model(dummy)
print(f"  Output shape (ImageNet 1000 classes): {logits.shape}")

print()
print("=" * 60)
print("PART 2: REPLACING THE CLASSIFICATION HEAD")
print("=" * 60)
print("""
Two strategies:

FEATURE EXTRACTION (freeze all, train only new head):
  + Fastest; tiny dataset (100-500 images per class)
  + Good when ImageNet features are relevant
  - Less accuracy than full fine-tuning

FULL FINE-TUNING (train everything):
  + Best accuracy with enough data (1000+ per class)
  - Slower; needs more data to avoid catastrophic forgetting
  - Use a low learning rate (1e-4) so we don't destroy pretrained features

GRADUAL UNFREEZING (best of both):
  1. Freeze all, train new head for a few epochs
  2. Unfreeze last block, train with low lr
  3. Unfreeze everything, train with very low lr
""")

NUM_CLASSES = 5   # our custom task has 5 classes

# ── Strategy 1: Feature extraction ──────────────────────────────────────────
model_freeze = models.resnet18(weights=weights)

# Freeze all parameters
for param in model_freeze.parameters():
    param.requires_grad = False

# Replace final layer (only this will be trained)
in_features = model_freeze.fc.in_features   # 512 for ResNet-18
model_freeze.fc = nn.Linear(in_features, NUM_CLASSES)   # new trainable layer

trainable = sum(p.numel() for p in model_freeze.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model_freeze.parameters())
print(f"Feature extraction:")
print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
print(f"  New head:  {model_freeze.fc}")

# ── Strategy 2: Full fine-tuning ─────────────────────────────────────────────
model_full = models.resnet18(weights=weights)
model_full.fc = nn.Linear(in_features, NUM_CLASSES)

# Different learning rates: very low for pretrained body, normal for new head
optimizer_full = optim.AdamW([
    {"params": [p for n, p in model_full.named_parameters() if "fc" not in n],
     "lr": 1e-5},     # pretrained body — very low lr
    {"params": model_full.fc.parameters(),
     "lr": 1e-3},     # new head — normal lr
], weight_decay=0.01)

trainable_full = sum(p.numel() for p in model_full.parameters() if p.requires_grad)
print(f"\nFull fine-tuning:")
print(f"  Trainable: {trainable_full:,} / {total:,} ({100*trainable_full/total:.1f}%)")
print(f"  Body lr: 1e-5, Head lr: 1e-3")

print()
print("=" * 60)
print("PART 3: RUNNING THE FINE-TUNING LOOP")
print("=" * 60)

# Synthetic dataset (replace with your ImageFolder dataset)
# Shape: (N, 3, 224, 224) — standard ImageNet size
X_train = torch.randn(100, 3, 224, 224)
y_train = torch.randint(0, NUM_CLASSES, (100,))
X_val   = torch.randn(20,  3, 224, 224)
y_val   = torch.randint(0, NUM_CLASSES, (20,))

train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
val_dl   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=16)

criterion = nn.CrossEntropyLoss()
model_freeze.train()

print(f"{'Epoch':>5} | {'Train Loss':>12} | {'Train Acc':>10} | {'Val Acc':>10}")
print("-" * 48)

for epoch in range(1, 4):
    model_freeze.train()
    total_loss = correct = seen = 0
    for Xb, yb in train_dl:
        logits = model_freeze(Xb)
        loss   = criterion(logits, yb)
        optimizer_full.zero_grad()
        loss.backward()
        optimizer_full.step()
        total_loss += loss.item() * len(yb)
        correct    += (logits.argmax(1) == yb).sum().item()
        seen       += len(yb)

    model_freeze.eval()
    val_correct = val_seen = 0
    with torch.no_grad():
        for Xb, yb in val_dl:
            preds = model_freeze(Xb).argmax(1)
            val_correct += (preds == yb).sum().item()
            val_seen    += len(yb)

    print(f"  {epoch:3d}  | {total_loss/seen:>12.4f} | "
          f"{100*correct/seen:>9.1f}% | {100*val_correct/val_seen:>9.1f}%")

print()
print("=" * 60)
print("PART 4: VISION TRANSFORMER (ViT) FINE-TUNING")
print("=" * 60)
print("""
ViT (Vision Transformer) replaces CNN with a Transformer for images.
  1. Split image into 16×16 patches → 196 patches for 224×224 image
  2. Each patch → linear embedding → sequence of 196 tokens
  3. Run through Transformer encoder (12 layers for ViT-B/16)
  4. Use [CLS] token for classification

ViT tends to outperform ResNet on ImageNet at the same parameter count,
but needs more data to train well from scratch.
Fine-tuning works the same way as ResNet.
""")

vit_weights = ViT_B_16_Weights.DEFAULT
model_vit   = models.vit_b_16(weights=vit_weights)
# Replace the head (ViT uses model.heads.head, not model.fc)
vit_in = model_vit.heads.head.in_features
model_vit.heads.head = nn.Linear(vit_in, NUM_CLASSES)

print(f"ViT-B/16 fine-tuning:")
print(f"  Total parameters: {sum(p.numel() for p in model_vit.parameters()):,}")
print(f"  Original head: Linear({vit_in}, 1000)")
print(f"  New head:      Linear({vit_in}, {NUM_CLASSES})")

# Forward pass to verify shape
model_vit.eval()
with torch.no_grad():
    out_vit = model_vit(torch.randn(2, 3, 224, 224))
print(f"  Output shape: {out_vit.shape}  ({NUM_CLASSES} classes, correct)")

print()
print("=" * 60)
print("PART 5: PRACTICAL TIPS")
print("=" * 60)
print("""
Data requirements:
  Feature extraction: 100-500 images per class (fast, works well)
  Full fine-tuning:   1000+ images per class

Augmentation (always use for training):
  RandomHorizontalFlip, RandomResizedCrop, ColorJitter
  Consider CutMix / MixUp for stronger regularisation

Learning rate:
  New head: 1e-3 to 1e-4
  Pretrained body: 10x to 100x lower (1e-5 to 1e-6)
  Use OneCycleLR for fast convergence

Saving and loading:
  torch.save(model.state_dict(), "model.pth")
  model.load_state_dict(torch.load("model.pth"))

HuggingFace alternative (simpler API):
  from transformers import AutoModelForImageClassification, AutoFeatureExtractor
  model = AutoModelForImageClassification.from_pretrained(
      "microsoft/resnet-18",
      num_labels=NUM_CLASSES,
      ignore_mismatched_sizes=True,  # ignore the head size mismatch
  )
  # trainer = Trainer(model=model, ...)  — handles everything
""")

print("=" * 60)
print("QUICK REFERENCE: PRETRAINED MODELS FOR FINE-TUNING")
print("=" * 60)
print("""
Model          | Accuracy | Params | Speed  | Fine-tune access
───────────────────────────────────────────────────────────────
ResNet-18      | 69.8%    | 11M    | Fast   | model.fc
ResNet-50      | 76.1%    | 25M    | Medium | model.fc
EfficientNet-B0| 77.7%    | 5M     | Fast   | model.classifier[1]
ViT-B/16       | 81.1%    | 86M    | Slow   | model.heads.head
ViT-L/16       | 85.1%    | 307M   | Slow   | model.heads.head
ConvNeXt-Base  | 84.1%    | 89M    | Medium | model.classifier[2]

All available via: torchvision.models.<name>(weights=..._Weights.DEFAULT)
""")

# =============================================================================
# EXPECTED OUTPUT:
# Feature extraction: ~513K trainable / 11M total (4.6%)
# Training loop: 3 epochs on synthetic data
# ViT output shape: (2, 5)
# =============================================================================
