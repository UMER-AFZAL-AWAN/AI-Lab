# =============================================================================
# 15_training_fundamentals / 07 — DATASET AND DATALOADER
# =============================================================================
#
# LIKE YOU ARE 5:
# Imagine you have a library of 100,000 books (your training data).
# You can't read all 100,000 at once — your desk (GPU memory) is too small.
# DataLoader is the librarian: they fetch 32 books at a time (a batch),
# let you read them, then swap them for the next 32 (without you asking).
# They also shuffle the order so you don't always read the same books first.
#
# C# ANALOGY:
# Dataset = IEnumerable<Sample>  (defines how to get one item by index)
# DataLoader = IAsyncEnumerable<Batch>  (batches items, shuffles, runs workers)
# __getitem__ = indexer: dataset[42]  → returns sample at index 42
#
# =============================================================================

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
from pathlib import Path

print("=" * 60)
print("PART 1: WRITING A CUSTOM DATASET")
print("=" * 60)
print("""
Every PyTorch Dataset must implement:
  __len__(self)         → how many samples total?
  __getitem__(self, idx) → return one sample by index

That's it. The rest (batching, shuffling, parallel loading) is DataLoader's job.
""")

# ─── Example 1: Simple tensor dataset ───────────────────────────────────────
class RegressionDataset(Dataset):
    """
    Dataset for synthetic regression: y = 2x₁ + 3x₂ + noise
    This is the simplest possible custom Dataset.
    """
    def __init__(self, n_samples=1000):
        self.X = torch.randn(n_samples, 2)
        self.y = 2 * self.X[:, 0] + 3 * self.X[:, 1] + 0.1 * torch.randn(n_samples)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

ds = RegressionDataset(1000)
x0, y0 = ds[0]
print(f"Dataset size: {len(ds)}")
print(f"Sample 0: x={x0.tolist()}, y={y0.item():.4f}")

# ─── Example 2: Image dataset (reads from folder) ───────────────────────────
class FolderImageDataset(Dataset):
    """
    Loads images from:
      root/
        class_A/
          img001.jpg
          img002.jpg
        class_B/
          img001.jpg
          ...

    Real usage: torchvision.datasets.ImageFolder does exactly this.
    This version shows the internals.
    """
    def __init__(self, root, transform=None):
        self.root      = Path(root)
        self.transform = transform
        self.samples   = []   # list of (path, label_idx)
        self.classes   = []

        if not self.root.exists():
            print(f"  (Note: {root} doesn't exist — this is a documentation example)")
            return

        for label_idx, class_dir in enumerate(sorted(self.root.iterdir())):
            if class_dir.is_dir():
                self.classes.append(class_dir.name)
                for img_path in class_dir.glob("*.jpg"):
                    self.samples.append((img_path, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

print()
print("FolderImageDataset structure (documentation):")
print("""
  my_dataset/
    cats/   img001.jpg  img002.jpg  ...  → label 0
    dogs/   img001.jpg  img002.jpg  ...  → label 1
    birds/  img001.jpg  ...              → label 2

  ds = FolderImageDataset("my_dataset", transform=transforms.Compose([...]))
  img_tensor, label = ds[42]   → (3, H, W) tensor, int label
""")

# ─── Example 3: Video frame dataset ─────────────────────────────────────────
class VideoFrameDataset(Dataset):
    """
    Loads a clip of T consecutive frames from a folder of extracted frames.
    Used for training AnimateDiff-style models.
    """
    def __init__(self, frames_dir, num_frames=16, transform=None):
        self.frames_dir  = Path(frames_dir)
        self.num_frames  = num_frames
        self.transform   = transform
        # In a real dataset, build a list of (video_folder, start_frame) tuples
        self.clips = []   # would be populated by scanning frames_dir

    def __len__(self):
        return len(self.clips) if self.clips else 0

    def __getitem__(self, idx):
        video_folder, start = self.clips[idx]
        frames = []
        for i in range(self.num_frames):
            from PIL import Image
            img = Image.open(video_folder / f"frame_{start+i:06d}.jpg")
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        # Stack: list of (C, H, W) → (T, C, H, W) → transpose to (C, T, H, W)
        return torch.stack(frames).permute(1, 0, 2, 3)

print("VideoFrameDataset output shape: (C, T, H, W) = (3, 16, H, W)")
print("This feeds directly into AnimateDiff's motion module.")

print()
print("=" * 60)
print("PART 2: DATALOADER — BATCHING AND PARALLELISM")
print("=" * 60)

ds_train = RegressionDataset(800)
ds_val   = RegressionDataset(200)

# Basic DataLoader
dl = DataLoader(
    ds_train,
    batch_size  = 32,       # how many samples per batch
    shuffle     = True,     # randomise order each epoch (train=True, val=False)
    num_workers = 0,        # workers for parallel data loading (0 = main thread)
    pin_memory  = False,    # if True, copy batch to pinned CPU memory before GPU transfer
    drop_last   = False,    # if True, drop incomplete final batch
)

print(f"Dataset size: {len(ds_train)}")
print(f"Batch size:   32")
print(f"Batches per epoch: {len(dl)}")

# Inspect one batch
X_batch, y_batch = next(iter(dl))
print(f"Batch X shape: {X_batch.shape}  (32 samples × 2 features)")
print(f"Batch y shape: {y_batch.shape}  (32 targets)")

print()
print("=" * 60)
print("NUM_WORKERS — parallel data loading")
print("=" * 60)
print("""
num_workers=0: loading runs on the main Python thread.
  → Slow for image loading (I/O blocks training)

num_workers=4: 4 background workers prefetch batches while GPU trains.
  → ~4× faster data loading; hides I/O latency

Rule of thumb: num_workers = number of CPU cores / 2
  (not more than 8 for typical workloads)

Gotchas on Windows:
  Requires if __name__ == '__main__': guard in scripts
  → not needed in Jupyter notebooks
  → not needed when num_workers=0

For diffusion model training (image datasets):
  num_workers=4 is a good starting point
  Set pin_memory=True if you have a GPU (speeds up CPU→GPU transfer)
""")

print()
print("=" * 60)
print("TRAIN / VAL / TEST SPLIT")
print("=" * 60)

full_ds = RegressionDataset(1000)
n_total = len(full_ds)
n_train = int(0.7 * n_total)   # 70%
n_val   = int(0.15 * n_total)  # 15%
n_test  = n_total - n_train - n_val  # 15%

train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test])
print(f"Total: {n_total}  →  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
print()
print("Usage:")
print("  Train set: fit the model (optimizer sees these samples)")
print("  Val set:   monitor during training (tune hyperparameters)")
print("  Test set:  final evaluation ONCE after training (never tune on this)")

print()
print("=" * 60)
print("COLLATE_FN — custom batch assembly")
print("=" * 60)
print("""
Default: DataLoader stacks tensors along dim=0 (all same shape required).
Custom collate_fn: handles variable-length sequences, mixed types.

Example: NLP batches where sentences have different lengths.
""")

def variable_length_collate(batch):
    """Stack items into a padded batch for variable-length sequences."""
    texts, labels = zip(*batch)
    max_len = max(len(t) for t in texts)
    padded  = torch.zeros(len(texts), max_len)
    for i, t in enumerate(texts):
        padded[i, :len(t)] = t
    return padded, torch.tensor(labels)

# Demo
batch_items = [
    (torch.tensor([1.0, 2.0, 3.0]), 0),
    (torch.tensor([4.0, 5.0]), 1),
    (torch.tensor([6.0, 7.0, 8.0, 9.0]), 0),
]
padded_batch, batch_labels = variable_length_collate(batch_items)
print(f"Variable lengths [3, 2, 4] → padded tensor: {padded_batch.shape}")
print(f"Padded:\n{padded_batch}")

print()
print("=" * 60)
print("QUICK REFERENCE: DATALOADER SETTINGS")
print("=" * 60)
print("""
Setting           | Train          | Validation / Inference
────────────────────────────────────────────────────────────
shuffle           | True           | False
num_workers       | 4+             | 2+
pin_memory        | True (GPU)     | True (GPU)
drop_last         | True (stable)  | False (need all samples)
batch_size        | 16–64          | can be 2× larger (no grad)
""")

# =============================================================================
# EXPECTED OUTPUT:
# Dataset size: 1000
# Batches per epoch: 25  (800 ÷ 32)
# Batch X shape: torch.Size([32, 2])
# Split: 700 / 150 / 150
# Padded tensor: torch.Size([3, 4]) with zeros for shorter sequences
# =============================================================================
