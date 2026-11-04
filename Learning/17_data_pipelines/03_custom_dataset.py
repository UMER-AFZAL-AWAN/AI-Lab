# =============================================================================
# 17_data_pipelines / 03 — CUSTOM DATASET CLASS
# =============================================================================
#
# LIKE YOU ARE 5:
# A Dataset is a recipe card that tells PyTorch how to get one item.
# You write: "to get item #42, open this folder, load this file, process it."
# DataLoader then fetches items in batches using that recipe.
#
# C# ANALOGY:
# Dataset = implementing IList<T> for your data source
# __len__  = Count property
# __getitem__ = indexer this[int index]
# DataLoader = IAsyncEnumerable<Batch> that calls your indexer in background threads
#
# =============================================================================

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import os

torch.manual_seed(42)

print("=" * 60)
print("PART 1: ANATOMY OF A DATASET CLASS")
print("=" * 60)
print("""
Minimum required: __len__ and __getitem__

class MyDataset(Dataset):
    def __init__(self, ...):   # set up file paths, indices, transforms
    def __len__(self):         # → int: total number of samples
    def __getitem__(self, idx):# → sample at index idx

Optional:
    def __getitems__(self, indices): # batch-aware fetching (faster)

DataLoader calls __len__ to know how many batches to make,
then calls __getitem__(idx) for random indices in each batch.
""")

# ── Example 1: In-memory tensor dataset ─────────────────────────────────────
class TensorDataset(Dataset):
    """Wraps tensors as a Dataset (this is built-in as torch.utils.data.TensorDataset)."""
    def __init__(self, *tensors):
        assert all(tensors[0].shape[0] == t.shape[0] for t in tensors), "size mismatch"
        self.tensors = tensors

    def __len__(self):
        return self.tensors[0].shape[0]

    def __getitem__(self, idx):
        return tuple(t[idx] for t in self.tensors)

X = torch.randn(100, 4)
y = torch.randint(0, 3, (100,))
ds = TensorDataset(X, y)
x0, y0 = ds[0]
print(f"TensorDataset: len={len(ds)}, sample shapes: {x0.shape}, {y0.shape}")

print()
print("=" * 60)
print("PART 2: IMAGE CLASSIFICATION DATASET")
print("=" * 60)

class ImageClassificationDataset(Dataset):
    """
    Loads images from:
      root/
        class_name_0/   img001.jpg  img002.jpg  ...
        class_name_1/   img001.jpg  ...

    Returns (image_tensor, class_index).
    This is what torchvision.datasets.ImageFolder does.
    Showing the internals so you know what it does.
    """
    def __init__(self, root, transform=None):
        self.root      = Path(root)
        self.transform = transform
        self.samples   = []   # (path, label)
        self.classes   = []
        self.class_to_idx = {}

        if not self.root.exists():
            print(f"  (Note: {root} not found — documentation example)")
            return

        class_dirs = sorted(d for d in self.root.iterdir() if d.is_dir())
        for label_idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            self.classes.append(class_name)
            self.class_to_idx[class_name] = label_idx

            for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp'):
                for img_path in class_dir.glob(ext):
                    self.samples.append((img_path, label_idx))

        print(f"  Found {len(self.classes)} classes, {len(self.samples)} total images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

ds_img = ImageClassificationDataset("data/train")  # will print "not found" note

print()
print("=" * 60)
print("PART 3: VIDEO CLIP DATASET (for AnimateDiff fine-tuning)")
print("=" * 60)

class VideoClipDataset(Dataset):
    """
    Dataset for video generation training.

    Directory structure:
      data/
        video_001/
          frames/
            frame_000001.jpg
            frame_000002.jpg
            ...
          caption.txt    (or metadata.json)
        video_002/
          ...

    Returns (video_tensor, caption) pairs.
    video_tensor shape: (C, T, H, W) = (3, 16, 512, 512)
    """
    def __init__(self, root, num_frames=16, frame_size=512, transform=None):
        self.root       = Path(root)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.transform  = transform
        self.clips      = []   # list of (frames_dir, caption)

        if not self.root.exists():
            print(f"  (Note: {root} not found — documentation example)")
            # Create a fake entry for docstring purposes
            return

        for video_dir in sorted(self.root.iterdir()):
            if not video_dir.is_dir():
                continue
            frames_dir = video_dir / "frames"
            caption_path = video_dir / "caption.txt"
            if not frames_dir.exists():
                continue
            caption = ""
            if caption_path.exists():
                caption = caption_path.read_text().strip()
            self.clips.append((frames_dir, caption))

        print(f"  Found {len(self.clips)} video clips")

    def __len__(self):
        return len(self.clips)

    def _load_frames(self, frames_dir):
        """Load num_frames evenly spaced from the frames directory."""
        from PIL import Image
        import numpy as np

        all_frames = sorted(frames_dir.glob("frame_*.jpg"))
        total      = len(all_frames)
        indices    = np.linspace(0, total - 1, self.num_frames, dtype=int)

        frames = []
        for idx in indices:
            img = Image.open(all_frames[idx]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            else:
                from torchvision import transforms
                img = transforms.Compose([
                    transforms.Resize((self.frame_size, self.frame_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5]*3, [0.5]*3),
                ])(img)
            frames.append(img)

        return torch.stack(frames).permute(1, 0, 2, 3)   # (C, T, H, W)

    def __getitem__(self, idx):
        frames_dir, caption = self.clips[idx]
        video_tensor = self._load_frames(frames_dir)
        return video_tensor, caption

ds_vid = VideoClipDataset("data/videos")

print()
print("=" * 60)
print("PART 4: DATASETS WITH METADATA (JSON)")
print("=" * 60)

class MetadataDataset(Dataset):
    """
    Many real datasets use a JSON/CSV metadata file:
      [
        {"path": "images/001.jpg", "label": 0, "caption": "a cat"},
        {"path": "images/002.jpg", "label": 1, "caption": "a dog"},
        ...
      ]

    This pattern is flexible: add any per-sample fields you need.
    """
    def __init__(self, metadata_path, transform=None):
        self.transform = transform
        self.samples   = []

        if not Path(metadata_path).exists():
            # Simulate a metadata file for the demo
            self.samples = [
                {"path": f"images/{i:04d}.jpg", "label": i % 3, "caption": f"sample {i}"}
                for i in range(50)
            ]
            print(f"  (Simulated {len(self.samples)} samples)")
            return

        with open(metadata_path) as f:
            self.samples = json.load(f)
        print(f"  Loaded {len(self.samples)} samples from {metadata_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item    = self.samples[idx]
        label   = item["label"]
        caption = item.get("caption", "")

        # In a real dataset: load the image from item["path"]
        # For this demo: return a fake tensor
        fake_image = torch.randn(3, 64, 64)
        return fake_image, label, caption

ds_meta = MetadataDataset("data/metadata.json")
img, label, caption = ds_meta[0]
print(f"Sample: img={img.shape}, label={label}, caption='{caption}'")

print()
print("=" * 60)
print("PART 5: COLLATE_FN FOR VARIABLE-LENGTH DATA")
print("=" * 60)
print("""
Default DataLoader expects all samples in a batch to be the same shape.
When they differ (variable-length captions, different video lengths):
write a custom collate_fn.
""")

def video_caption_collate(batch):
    """
    Custom collate for (video_tensor, caption) batches.
    Videos: same shape → stack normally.
    Captions: strings → keep as list.
    """
    videos   = torch.stack([item[0] for item in batch])   # (B, C, T, H, W)
    captions = [item[1] for item in batch]                # list of strings
    return videos, captions

# Test DataLoader with collate_fn
class FakeVideoDataset(Dataset):
    def __len__(self): return 20
    def __getitem__(self, idx):
        return torch.randn(3, 16, 64, 64), f"caption {idx}"

dl = DataLoader(
    FakeVideoDataset(),
    batch_size=4,
    shuffle=True,
    collate_fn=video_caption_collate,
)

videos_b, captions_b = next(iter(dl))
print(f"Batched videos:   {videos_b.shape}  (B, C, T, H, W)")
print(f"Batched captions: {captions_b}")

print()
print("=" * 60)
print("QUICK REFERENCE: DATASET PATTERNS")
print("=" * 60)
print("""
Pattern                | When to use
────────────────────────────────────────────────────────────
TensorDataset          | Small datasets that fit in RAM
ImageFolder            | Classification datasets in class subdirs
Custom Dataset         | Any custom data format
MetadataDataset (JSON) | Multi-modal data (image+caption+label)
VideoClipDataset       | Video generation training
Streaming Dataset      | Datasets too large for disk (HuggingFace)

Performance tips:
  num_workers=4+   → parallel loading (images/videos are I/O bound)
  pin_memory=True  → faster CPU→GPU transfers
  persistent_workers=True  → keep workers alive between epochs (saves fork overhead)
  prefetch_factor=2  → fetch 2 batches ahead
""")

# =============================================================================
# EXPECTED OUTPUT:
# TensorDataset: len=100, shapes (4,), ()
# MetadataDataset: 50 simulated samples
# Batched videos: (4, 3, 16, 64, 64)
# No errors (all runs with synthetic/simulated data)
# =============================================================================
