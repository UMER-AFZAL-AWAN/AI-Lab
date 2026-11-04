# =============================================================================
# 05 — LATENT DIFFUSION: WHY WORK IN LATENT SPACE
# =============================================================================
#
# LIKE YOU ARE 5:
# Running the denoising 50 times in PIXEL space means:
#   50 × UNet(512×512×3) = enormous computation
#
# Running it in LATENT space means:
#   50 × UNet(64×64×4) = 48x fewer numbers, way faster
#
# The trick: compress first (VAE encode), denoise many times (diffusion),
#            then decompress once (VAE decode).
#
# This is the core insight of Latent Diffusion Models (LDM),
# which became Stable Diffusion.
#
# =============================================================================

import torch
import torch.nn as nn

print("=" * 60)
print("PIXEL DIFFUSION vs LATENT DIFFUSION")
print("=" * 60)

# Sizes at play
H_px, W_px = 512, 512
C_px = 3   # RGB
H_lat, W_lat = H_px // 8, W_px // 8   # 64x64
C_lat = 4  # latent channels

pixel_size  = C_px  * H_px  * W_px
latent_size = C_lat * H_lat * W_lat

print(f"Pixel  UNet input:  ({C_px}, {H_px}, {W_px}) = {pixel_size:,} values per batch item")
print(f"Latent UNet input:  ({C_lat}, {H_lat}, {W_lat}) = {latent_size:,} values per batch item")
print(f"Latent is {pixel_size // latent_size}x smaller")
print()

num_steps = 50
print(f"With {num_steps} denoising steps:")
print(f"  Pixel diffusion:  {num_steps} × {pixel_size:,} = {num_steps * pixel_size:,} operations (per sample)")
print(f"  Latent diffusion: {num_steps} × {latent_size:,} = {num_steps * latent_size:,} operations")
print(f"  Latent is ~{(num_steps * pixel_size) // (num_steps * latent_size)}x cheaper for the diffusion part")

print()
print("=" * 60)
print("THE THREE-STAGE PIPELINE")
print("=" * 60)
#
# Stage 1: VAE ENCODE (once, at the start)
#   pixels (B, 3, 512, 512) -> latents (B, 4, 64, 64)
#
# Stage 2: DIFFUSION LOOP (N times, expensive but on small tensors)
#   latents (B, 4, 64, 64) -> UNet -> denoised latents (B, 4, 64, 64)
#
# Stage 3: VAE DECODE (once, at the end)
#   latents (B, 4, 64, 64) -> pixels (B, 3, 512, 512)

class FakeVAE(nn.Module):
    """Stand-in for the real VAE — shows shapes only."""
    def encode(self, pixels):
        B, C, H, W = pixels.shape
        # 8x spatial compression, 3 channels -> 4 channels
        return torch.randn(B, 4, H // 8, W // 8)

    def decode(self, latents):
        B, C, H, W = latents.shape
        # reverse: 4 channels -> 3 channels, 8x upscale
        return torch.randn(B, 3, H * 8, W * 8)

class FakeUNet(nn.Module):
    """Stand-in for the real UNet — shows shapes only."""
    def forward(self, latents, t, text_embed):
        return torch.randn_like(latents)  # predicts noise, same shape

vae  = FakeVAE()
unet = FakeUNet()

# Stage 1: Encode
pixels = torch.randn(1, 3, 512, 512)
print(f"Input pixels: {pixels.shape}")
latents = vae.encode(pixels) * 0.18215   # scale by factor
print(f"Encoded latents: {latents.shape}  (8x smaller in each spatial dim)")

# Stage 2: Diffuse (50 steps in latent space)
num_steps = 50
noise = torch.randn_like(latents)
text_embed = torch.randn(1, 77, 768)

for step in range(num_steps):
    t = torch.tensor([num_steps - step])   # simplified: just counts down
    with torch.no_grad():
        eps = unet(latents, t, text_embed)
    # In reality: scheduler.step(eps, t, latents) for proper denoising
    latents = latents - 0.01 * eps   # simplified update (not mathematically correct)

print(f"After {num_steps} denoising steps: {latents.shape}")

# Stage 3: Decode
pixels_out = vae.decode(latents / 0.18215)
print(f"Decoded pixels: {pixels_out.shape}")

print()
print("=" * 60)
print("THE UNET ARCHITECTURE (conceptual)")
print("=" * 60)
#
# The UNet in Stable Diffusion is NOT a simple Conv UNet.
# It is a Conv-Attention UNet (Transformer-augmented UNet):
#
# ENCODER: (downsampling path)
#   ResBlock -> Attention -> ResBlock -> Attention -> Downsample
#   ResBlock -> Attention -> ResBlock -> Attention -> Downsample
#   ...
#
# BOTTLENECK:
#   ResBlock -> Attention -> ResBlock
#
# DECODER: (upsampling path, with skip connections from encoder!)
#   Upsample -> ResBlock -> Attention -> ResBlock -> Attention
#   Upsample -> ResBlock -> Attention -> ResBlock -> Attention
#   ...
#
# SKIP CONNECTIONS (from encoder to decoder):
#   Same spatial resolution feature maps from encoder are added to decoder.
#   These carry fine-grained structure through the network.
#
# TIMESTEP EMBEDDING:
#   The timestep t is encoded as a 320-dim vector (sinusoidal + linear layers).
#   This vector is added to each ResBlock output: tells the UNet "how noisy is this?"
#
# TEXT EMBEDDING:
#   The 77×768 text embedding is used as K and V in cross-attention in every Transformer block.
#   The UNet's spatial features are Q. So every spatial position "attends to" text tokens.

print("""
UNet structure (simplified):
  Input latent (B, 4, 64, 64)
      |
  [DownBlock 0]  → 64→64, attention  → save skip
  [DownBlock 1]  → 64→32, attention  → save skip
  [DownBlock 2]  → 32→16, attention  → save skip
      |
  [MidBlock]     → 16×16, ResBlock + attention + ResBlock
      |
  [UpBlock 2]    → concat(skip from DownBlock 2) → upsample → 16→32
  [UpBlock 1]    → concat(skip from DownBlock 1) → upsample → 32→64
  [UpBlock 0]    → concat(skip from DownBlock 0) → 64→64
      |
  Output noise (B, 4, 64, 64)
""")

# =============================================================================
# EXPECTED OUTPUT:
# Input pixels: torch.Size([1, 3, 512, 512])
# Encoded latents: torch.Size([1, 4, 64, 64])
# Decoded pixels: torch.Size([1, 3, 512, 512])
# =============================================================================
