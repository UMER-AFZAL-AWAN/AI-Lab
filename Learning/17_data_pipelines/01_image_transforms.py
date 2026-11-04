# =============================================================================
# 17_data_pipelines / 01 — IMAGE TRANSFORMS AND AUGMENTATION
# =============================================================================
#
# LIKE YOU ARE 5:
# Before feeding an image to a neural network, we must prepare it:
# 1. Resize to the size the model expects (e.g., 512×512)
# 2. Convert to numbers (a tensor of pixel values)
# 3. Normalise (shift the numbers to a range the model understands)
# 4. Augment (randomly flip/rotate during training to prevent overfitting)
#
# C# ANALOGY:
# Like an ASP.NET middleware pipeline.
# Each Transform is a middleware: input → transform → output.
# transforms.Compose is the pipeline builder.
# transforms.Compose([A, B, C])(image) = C(B(A(image)))
#
# =============================================================================

import torch
from torchvision import transforms
from PIL import Image
import io, struct, zlib

print("=" * 60)
print("PART 1: THE STANDARD PIPELINE")
print("=" * 60)
print("""
Most image models expect this input pipeline:

  PIL Image (H×W×3, uint8 0-255)
      ↓  Resize(512)
  PIL Image (512×512×3)
      ↓  ToTensor()
  torch.Tensor (3×512×512, float32 0.0-1.0)
      ↓  Normalize(mean, std)
  torch.Tensor (3×512×512, float32, normalised)

The model sees the normalised tensor.
The normalisation constants come from the dataset the model was pretrained on.
""")

IMAGENET_MEAN = [0.485, 0.456, 0.406]   # RGB means from ImageNet
IMAGENET_STD  = [0.229, 0.224, 0.225]   # RGB stds from ImageNet

# Standard eval/inference pipeline
inference_pipeline = transforms.Compose([
    transforms.Resize(256),                      # resize shorter edge to 256
    transforms.CenterCrop(224),                  # crop 224×224 from centre
    transforms.ToTensor(),                       # PIL → Tensor, 0-1 range
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),  # normalise
])

# Standard training pipeline (with augmentation)
training_pipeline = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # random crop+resize
    transforms.RandomHorizontalFlip(p=0.5),               # flip 50% of images
    transforms.ColorJitter(
        brightness=0.2,   # random brightness ±20%
        contrast=0.2,     # random contrast ±20%
        saturation=0.1,   # slight saturation change
        hue=0.05,         # very slight hue shift
    ),
    transforms.RandomGrayscale(p=0.05),   # occasionally drop to grayscale
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

print("Inference pipeline:")
for t in inference_pipeline.transforms:
    print(f"  {t.__class__.__name__}")

print("\nTraining pipeline (with augmentation):")
for t in training_pipeline.transforms:
    print(f"  {t.__class__.__name__}")

print()
print("=" * 60)
print("PART 2: NORMALISATION VALUES EXPLAINED")
print("=" * 60)
print("""
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

Before: pixel values in [0.0, 1.0]  (after ToTensor)
After:  pixel values ≈ [-2.0, +2.0]  (standardised)

Formula for each channel:  output = (input - mean) / std

Why these specific values?
  Measured from the entire ImageNet dataset (1.2M images).
  Using these for any ImageNet-pretrained model gives correct results.
  Using different values → model sees out-of-distribution inputs → worse performance.

For diffusion models (Stable Diffusion, AnimateDiff):
  Input is normalised to [-1, 1]:
  Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

  To reverse:
  original = (normalised + 1) / 2   → [0, 1]
  original = (original * 255).clamp(0, 255).byte()  → PIL image
""")

# Demo without an actual image
def show_normalise_effect(pixel_value, mean, std):
    after = (pixel_value - mean) / std
    return after

print("Normalisation demo (single pixel, red channel):")
for pv in [0.0, 0.5, 1.0]:
    after = show_normalise_effect(pv, IMAGENET_MEAN[0], IMAGENET_STD[0])
    print(f"  pixel={pv:.1f}  →  normalised={after:.3f}")

print()
print("Diffusion model normalisation (-1 to 1):")
for pv in [0.0, 0.5, 1.0]:
    after = (pv - 0.5) / 0.5
    print(f"  pixel={pv:.1f}  →  normalised={after:.1f}")

print()
print("=" * 60)
print("PART 3: VIDEO TRANSFORMS")
print("=" * 60)
print("""
For video, we need spatially CONSISTENT transforms across frames.
Randomly flipping frame 1 but not frame 2 would create incoherent videos.

Pattern: generate random parameters ONCE, apply to ALL frames.
""")

class ConsistentVideoTransform:
    """
    Applies the same random transform to all frames of a video.
    This ensures temporal consistency.
    """
    def __init__(self, size=512, p_flip=0.5):
        self.size    = size
        self.p_flip  = p_flip
        self.resize  = transforms.Resize(size)
        self.to_norm = transforms.Normalize([0.5]*3, [0.5]*3)

    def __call__(self, frames):
        """
        frames: list of PIL Images or (T, C, H, W) tensor
        Returns: (C, T, H, W) tensor normalised to [-1, 1]
        """
        # Decide random parameters ONCE for the whole video
        do_flip = torch.rand(1).item() < self.p_flip

        processed = []
        for frame in frames:
            # Apply spatial transforms consistently
            if do_flip:
                frame = transforms.functional.hflip(frame)
            # Deterministic transforms
            frame = transforms.functional.resize(frame, [self.size, self.size])
            frame = transforms.functional.to_tensor(frame)   # 0-1
            frame = self.to_norm(frame)                      # -1 to 1
            processed.append(frame)

        # Stack: list of (C, H, W) → (T, C, H, W) → (C, T, H, W)
        return torch.stack(processed).permute(1, 0, 2, 3)

print("ConsistentVideoTransform:")
print("  - Decides flip ONCE per video (not per frame)")
print("  - Output shape: (C, T, H, W) = (3, T, 512, 512)")

print()
print("=" * 60)
print("PART 4: TENSOR OPERATIONS ON IMAGES")
print("=" * 60)

# Create a synthetic "image" tensor
img = torch.rand(3, 64, 64)   # (C, H, W)

print(f"Image tensor shape: {img.shape}  (C, H, W)")
print(f"Value range: [{img.min():.3f}, {img.max():.3f}]")

# Channel operations
r, g, b = img[0], img[1], img[2]   # each is (H, W)
grayscale = 0.299*r + 0.587*g + 0.114*b   # luminance formula
print(f"Grayscale shape: {grayscale.shape}")

# Batch operations
batch = torch.rand(4, 3, 64, 64)   # (B, C, H, W)
batch_norm = transforms.Normalize([0.5]*3, [0.5]*3)(batch[0])  # apply to one
print(f"Batch shape: {batch.shape}")

# How to un-normalise a tensor for display (diffusion model convention)
normalised = torch.rand(3, 64, 64) * 2 - 1   # fake [-1, 1] tensor
unnorm = ((normalised + 1) / 2).clamp(0, 1)   # back to [0, 1]
pixels = (unnorm * 255).to(torch.uint8)        # to uint8 for PIL
print(f"Un-normalised for display: dtype={pixels.dtype}, range=[{pixels.min()}, {pixels.max()}]")

print()
print("=" * 60)
print("QUICK REFERENCE: COMMON SIZES AND NORMALISATION")
print("=" * 60)
print("""
Model                 | Input size     | Normalisation
──────────────────────────────────────────────────────────────────
ResNet / ViT          | 224×224        | ImageNet mean/std
CLIP (ViT-L/14)       | 224×224        | ImageNet mean/std
Stable Diffusion      | 512×512 (VAE)  | [-1, 1]  (mean=0.5, std=0.5)
SDXL                  | 1024×1024      | [-1, 1]
AnimateDiff           | 512×512 frames | [-1, 1]
Wan 2.1               | 480×832        | [-1, 1] + BF16
LLaVA / VLMs          | 336×336        | CLIP normalisation

Note: "input size" for diffusion models is the LATENT size ÷ 8.
  Stable Diffusion 512×512 image → 64×64 latent (the UNet's input).
""")

# =============================================================================
# EXPECTED OUTPUT:
# Normalisation demo: 0.0→-2.117, 0.5→0.0, 1.0→2.249 (approx)
# Diffusion: 0.0→-1.0, 0.5→0.0, 1.0→1.0
# No errors (all operations use synthetic tensors)
# =============================================================================
