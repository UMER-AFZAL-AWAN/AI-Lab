# =============================================================================
# 19_computer_vision / 02 — SEGMENTATION AND SAM
# =============================================================================
#
# LIKE YOU ARE 5:
# Detection draws a rectangle AROUND the cat.
# Segmentation colours EXACTLY the pixels that are cat.
# Like using a paint-by-number template: each pixel gets a colour/label.
#
# SAM (Segment Anything Model): click on a cat → SAM selects exactly that cat.
# "Tell me where you want to cut, and I'll do it precisely."
#
# C# ANALOGY:
# Detection = List<Rectangle> for objects
# Segmentation = int[,] mask where each int is a class label
#   (like a 2D array: mask[y, x] = class_id for that pixel)
#
# =============================================================================

import torch
import torch.nn as nn

print("=" * 60)
print("PART 1: TYPES OF SEGMENTATION")
print("=" * 60)
print("""
SEMANTIC SEGMENTATION:
  Assign a class to EVERY pixel in the image.
  Output: mask of shape (H, W) where each value is a class ID.

  Example: in a street photo:
    road pixels → class 0
    car pixels  → class 1
    sky pixels  → class 2
    person pixels → class 3

  Limitation: "car" vs "that specific car" — can't distinguish instances.

INSTANCE SEGMENTATION:
  Detect individual INSTANCES and produce a mask PER instance.
  Output: list of (class_id, mask) pairs.

  Example: "left car" → mask_1, "right car" → mask_2 (separately)
  Used in Mask R-CNN.

PANOPTIC SEGMENTATION:
  Combines semantic + instance.
  Every pixel has a class. "Things" (countable objects) also have an instance ID.
  "Stuff" (sky, road, grass) only has semantic class.

WHAT SAM DOES:
  Given a prompt (point, box, text), SAM outputs a binary mask
  for exactly the object you clicked on.
  It's NOT doing full scene segmentation — it's interactive.
""")

print("=" * 60)
print("PART 2: SEMANTIC SEGMENTATION ARCHITECTURES")
print("=" * 60)
print("""
Fully Convolutional Network (FCN, 2015):
  Replace the final Linear classifier with a Conv layer.
  Upsample back to original resolution using deconvolutions.
  Output: (num_classes, H, W) logits → argmax → segmentation mask.

U-Net (2015, medical imaging → diffusion models):
  Encoder: downsample (like any CNN), builds a compressed representation
  Decoder: upsample back to full resolution
  Skip connections: concatenate encoder features to decoder at each scale

  Why skip connections?
    The encoder loses spatial details while compressing.
    Skips bring back the spatial details for precise pixel labelling.

  Why is this relevant to you?
    The UNet in Stable Diffusion / AnimateDiff has the SAME encoder-decoder
    structure with skip connections. It just also has cross-attention.

DeepLab v3+:
  Uses dilated (atrous) convolutions to maintain resolution.
  ASPP (Atrous Spatial Pyramid Pooling): look at multiple scales at once.
""")

# Minimal U-Net (the architecture behind Stable Diffusion)
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
        )
    def forward(self, x): return self.conv(x)

class TinyUNet(nn.Module):
    """
    Minimal U-Net for segmentation.
    Real UNets have more levels and more channels.
    SD/AnimateDiff UNet additionally has cross-attention at each level.
    """
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        # Encoder (downsample)
        self.enc1 = UNetBlock(in_channels, 32)
        self.enc2 = UNetBlock(32, 64)
        self.pool = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = UNetBlock(64, 128)
        # Decoder (upsample)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)   # upsample ×2
        self.dec2 = UNetBlock(128, 64)   # 128 = 64 (up) + 64 (skip)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = UNetBlock(64, 32)    # 64 = 32 (up) + 32 (skip)
        # Output
        self.head = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)        # (B, 32, H, W)
        e2 = self.enc2(self.pool(e1))  # (B, 64, H/2, W/2)
        # Bottleneck
        bn = self.bottleneck(self.pool(e2))  # (B, 128, H/4, W/4)
        # Decoder with skip connections
        d2 = self.dec2(torch.cat([self.up2(bn), e2], dim=1))  # (B, 64, H/2, W/2)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # (B, 32, H, W)
        return self.head(d1)   # (B, num_classes, H, W)

model = TinyUNet(in_channels=3, num_classes=5)
x     = torch.randn(2, 3, 128, 128)   # batch=2, RGB, 128×128
out   = model(x)
print(f"U-Net input:  {x.shape}")
print(f"U-Net output: {out.shape}  (B, num_classes, H, W) — same H,W as input")
print(f"Argmax mask:  {out.argmax(dim=1).shape}  (B, H, W) — class per pixel")

print()
print("Loss for segmentation: CrossEntropyLoss with 2D inputs:")
mask_gt = torch.randint(0, 5, (2, 128, 128))   # ground-truth labels per pixel
loss    = nn.CrossEntropyLoss()(out, mask_gt)
print(f"  CrossEntropyLoss(logits, mask_gt) = {loss.item():.4f}")
print(f"  logits: {out.shape}, labels: {mask_gt.shape}")

print()
print("=" * 60)
print("PART 3: SAM — SEGMENT ANYTHING MODEL")
print("=" * 60)
print("""
SAM (Meta, 2023):
  - Trained on 1 billion masks across 11 million images
  - Works on ANY image, ANY object
  - Interactive: prompt with a point, a box, or text

Architecture:
  Image Encoder:    ViT-H (heavy, runs ONCE per image)
  Prompt Encoder:   encodes clicks, boxes, text masks
  Mask Decoder:     lightweight transformer, runs per prompt

Three prompting modes:
  1. Point prompt:   click on an object → its mask
  2. Box prompt:     draw a rectangle → mask inside it
  3. Automatic mode: SAM finds ALL objects automatically (grid of points)

Usage:
  pip install segment-anything
  from segment_anything import sam_model_registry, SamPredictor

  sam   = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
  pred  = SamPredictor(sam)
  pred.set_image(image_rgb)   # run encoder once (slow ~2s)

  masks, scores, logits = pred.predict(
      point_coords=[[500, 300]],   # x, y in pixels
      point_labels=[1],             # 1=foreground, 0=background
  )
  # masks: (N, H, W) boolean arrays

SAM 2 (2024):
  Extends SAM to VIDEO: prompt one frame, SAM tracks the mask through the video.
  pip install sam2

  Huge use case: video editing (remove background from moving objects)
""")

print("=" * 60)
print("PART 4: GROUNDED SAM")
print("=" * 60)
print("""
Combining Grounding DINO (text → detection) + SAM (box → mask):

  You say: "Segment all cats in this image"

  Step 1: Grounding DINO: "cat" → bounding boxes for each cat
  Step 2: SAM: each bounding box → precise mask

  Result: labelled masks for all cats, without manual clicking.

Practical code:
  from groundingdino.util.inference import load_model, predict as gdino_predict
  from segment_anything import SamPredictor

  # Detect objects with DINO
  boxes, scores, labels = gdino_predict(image, "cat . dog")

  # Segment each detection with SAM
  predictor.set_image(image)
  for box in boxes:
      masks, _, _ = predictor.predict(box=box)
      # masks[0] is the best mask for this detection
""")

print("=" * 60)
print("QUICK REFERENCE: SEGMENTATION TOOLS")
print("=" * 60)
print("""
Library/Model         | Task                    | pip install
──────────────────────────────────────────────────────────────────
torchvision           | Semantic (DeepLab, FCN)  | torchvision
transformers          | SegFormer, MaskFormer    | transformers
segment-anything      | SAM (interactive)        | segment-anything
sam2                  | SAM2 (video)             | sam2
ultralytics           | YOLOv8 instance seg      | ultralytics
mmsegmentation        | Research models          | mmsegmentation

HuggingFace quick start:
  from transformers import pipeline
  seg = pipeline("image-segmentation", model="facebook/sam-vit-base")
  result = seg("photo.jpg")
  # result: list of {"label": "cat", "score": 0.98, "mask": PIL Image}
""")

# =============================================================================
# EXPECTED OUTPUT:
# U-Net output: torch.Size([2, 5, 128, 128])  — same spatial dims as input
# Argmax mask: torch.Size([2, 128, 128])
# CrossEntropyLoss computed without error
# =============================================================================
