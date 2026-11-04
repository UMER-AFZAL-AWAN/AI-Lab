# =============================================================================
# 06 — VISION TRANSFORMER (ViT): ATTENTION FOR IMAGES
# =============================================================================
#
# LIKE YOU ARE 5:
# Normal CNN: looks at a small window (3x3 pixels) one piece at a time.
# ViT: cuts the image into 16x16 "patches", treats each patch as a TOKEN,
#      then runs a full Transformer (every patch attends to every other patch).
#
# Example: a 224x224 image with 16x16 patches
#   = (224/16) × (224/16) = 14 × 14 = 196 patches
#   Each patch becomes a 768-dim embedding.
#   Input to Transformer: 196 tokens of 768 dimensions.
#   This is like a 196-word sentence where each "word" is a patch.
#
# WHY ViT MATTERS:
# CLIP (used in AnimateDiff for text encoding and image encoding) has a ViT image encoder.
# CLIP ViT-L/14 uses 14x14 patches and 24 Transformer layers.
# It was trained to match images and text descriptions.
# "openai/clip-vit-large-patch14" is in your .venv (from diffusers).
#
# =============================================================================

import torch
import torch.nn as nn

print("=" * 60)
print("PATCH EMBEDDING: IMAGE -> SEQUENCE OF TOKENS")
print("=" * 60)
#
# Step 1: Cut the image into non-overlapping 16x16 patches.
# Step 2: Flatten each patch into a vector (16×16×3 = 768 for RGB patches).
# Step 3: Project to embed_dim with a linear layer (or Conv2d with same kernel & stride).

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size  = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        # Conv2d with kernel=patch_size, stride=patch_size is equivalent to
        # "flatten each patch and project to embed_dim"
        self.projection  = nn.Conv2d(in_channels, embed_dim,
                                      kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.projection(x)        # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)              # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)        # (B, num_patches, embed_dim)
        return x

# Standard ViT-B: 224x224 image, 16x16 patches, 768 embed_dim
img_size, patch_size, embed_dim = 224, 16, 768
patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)

x = torch.randn(2, 3, 224, 224)   # batch=2, RGB, 224x224
patches = patch_embed(x)
num_patches = (img_size // patch_size) ** 2
print(f"Image:         {x.shape}")
print(f"Patch tokens:  {patches.shape}  (batch, {num_patches} patches, {embed_dim}-dim)")
print(f"Each patch = {patch_size}×{patch_size}×3 = {patch_size*patch_size*3} pixels -> projected to {embed_dim}")

print()
print("=" * 60)
print("THE CLS TOKEN (Classification Token)")
print("=" * 60)
#
# BERT / ViT prepend one special [CLS] (classification) token to the sequence.
# After all Transformer blocks, the CLS token's output is used for prediction.
# Because it attends to all other tokens, it accumulates a global representation.
# In CLIP, this is the image embedding used for image-text matching.

cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
cls_tokens = cls_token.expand(2, -1, -1)   # broadcast to batch=2
x_with_cls = torch.cat([cls_tokens, patches], dim=1)
print(f"CLS token: (1, 1, {embed_dim})")
print(f"Patches:   {patches.shape}")
print(f"Combined:  {x_with_cls.shape}  (CLS + 196 patches = 197 tokens)")
print("The CLS token at position 0 aggregates the whole image after all attention blocks.")

print()
print("=" * 60)
print("TINY VIT (a runnable mini-version)")
print("=" * 60)

class TinyViT(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=3,
                 embed_dim=64, num_heads=4, num_layers=4, num_classes=10):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                batch_first=True, norm_first=True,
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        patches = self.patch_embed(x)
        cls     = self.cls_token.expand(B, -1, -1)
        x       = torch.cat([cls, patches], dim=1) + self.pos_embed

        for block in self.blocks:
            x = block(x)

        cls_out = self.norm(x[:, 0])    # CLS token output
        return self.head(cls_out)

vit    = TinyViT()
imgs   = torch.randn(4, 3, 64, 64)
logits = vit(imgs)
print(f"Input:   {imgs.shape}")
print(f"Output:  {logits.shape}  (4 images, 10 class scores)")
params = sum(p.numel() for p in vit.parameters())
print(f"Params:  {params:,}")

print()
print("=" * 60)
print("CLIP ViT-L/14: WHAT ANIMATEDIFF USES")
print("=" * 60)
print("Image encoder: ViT-L/14 (large, patch=14)")
print("  - Input:  224×224 RGB image")
print("  - Patches: (224/14)×(224/14) = 16×16 = 256 patches + 1 CLS = 257 tokens")
print("  - embed_dim: 1024")
print("  - num_layers: 24")
print("  - Output: (1, 257, 1024) per image, or just the CLS token (1, 1024)")
print()
print("Text encoder: Transformer with 12 layers, embed_dim=768")
print("  - Output: (1, 77, 768)")
print()
print("Training: contrastive — text and image of the same scene should have")
print("  similar embeddings. 'a cat' and a photo of a cat -> close vectors.")

# =============================================================================
# EXPECTED OUTPUT:
# Patch tokens:  torch.Size([2, 196, 768])
# Combined:  torch.Size([2, 197, 768])
# Output:  torch.Size([4, 10])
# =============================================================================
