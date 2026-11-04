# =============================================================================
# 05 — THE VAE IN ANIMATEDIFF (how it fits together)
# =============================================================================
#
# LIKE YOU ARE 5:
# AnimateDiff generates a VIDEO by working in 4D latent space:
#   - Instead of one 4x64x64 image latent, it has 16 of them (one per frame).
#   - The shape is (1, 4, 16, 64, 64):
#     batch=1, channels=4, frames=16, height=64, width=64
#
# STEPS TO GENERATE A VIDEO:
# 1. Start: pure random noise in latent space (16 frames of noise).
# 2. Denoise: the UNet + motion adapter progressively remove noise.
#             The text prompt guides WHICH direction to denoise.
#             (50 steps of denoising)
# 3. Decode: the VAE decoder turns each latent frame into a 512x512 RGB frame.
# 4. Result: 16 frames of video.
#
# The VAE is used ONCE at the start (encode) and ONCE at the end (decode).
# Everything expensive (the 50 denoising steps) happens in latent space.
#
# =============================================================================

import torch
import torch.nn as nn

print("=" * 60)
print("THE VAE ENCODE/DECODE INTERFACE IN DIFFUSERS")
print("=" * 60)
#
# In HuggingFace diffusers, the VAE is an AutoencoderKL model.
# It has two useful methods:
#   vae.encode(pixel_frames)  -> latent distribution
#   vae.decode(latent_frames) -> pixel frames

print("""
# How the VAE is used in AnimateDiff (from diffusers pipeline):

# 1. ENCODE (only needed if you are doing img2img or inpainting):
latents = vae.encode(pixel_frames).latent_dist.sample()
latents = latents * vae.config.scaling_factor   # scale to unit variance

# 2. DECODE (always at the end):
pixel_frames = vae.decode(latents / vae.config.scaling_factor).sample
""")

print("=" * 60)
print("SIMULATING THE VAE SHAPES")
print("=" * 60)
#
# Real VAE:  512x512 pixel -> 64x64 latent  (8x spatial compression)
# Real VAE:  3-channel RGB -> 4-channel latent
#
# Let's trace the shapes without loading the actual model.

batch      = 1
num_frames = 16
H_pixel, W_pixel = 512, 512
H_latent, W_latent = H_pixel // 8, W_pixel // 8
latent_channels = 4

print(f"Pixel space: (B={batch}, C=3, T={num_frames}, H={H_pixel}, W={W_pixel})")
print(f"             = {batch * 3 * num_frames * H_pixel * W_pixel:,} total values")
print()
print(f"Latent space: (B={batch}, C={latent_channels}, T={num_frames}, H={H_latent}, W={W_latent})")
print(f"             = {batch * latent_channels * num_frames * H_latent * W_latent:,} total values")
print(f"Compression: {(3 * H_pixel * W_pixel) / (latent_channels * H_latent * W_latent):.0f}x per frame")

print()
print("=" * 60)
print("THE SCALING FACTOR")
print("=" * 60)
#
# The VAE latent space has an arbitrary scale (the values could be any range).
# The diffusion model expects latents with ~unit variance (std ≈ 1).
# vae.config.scaling_factor (typically 0.18215 for SD 1.5) normalises this.
# When encoding:  latents = raw_latents * 0.18215
# When decoding:  raw_latents = latents / 0.18215

scaling_factor = 0.18215
print(f"Scaling factor: {scaling_factor}")
print("Why: raw VAE latents have std >> 1.")
print("Multiplying by 0.18215 brings std close to 1.")
print("The diffusion model was trained expecting std ≈ 1.")

# Simulate
raw_latents = torch.randn(1, 4, 16, 64, 64) * 5.0   # simulate raw encoder output
print(f"\nRaw latent std:    {raw_latents.std().item():.2f}  (too large)")
scaled = raw_latents * scaling_factor
print(f"Scaled latent std: {scaled.std().item():.2f}  (closer to 1 ✓)")

print()
print("=" * 60)
print("VRAM COST: VAE DECODE FOR 16 FRAMES")
print("=" * 60)
#
# The decode step is surprisingly expensive.
# AnimateDiff uses vae_slicing to process one frame at a time
# instead of all 16 at once.

# Without slicing: all 16 frames at once
latent_full = torch.randn(1, 4, 16, 64, 64)
vram_full = latent_full.element_size() * latent_full.nelement() / 1024**2
print(f"All 16 latents at once: {vram_full:.1f} MB (just for the latents)")

# With slicing: one frame at a time
latent_one = torch.randn(1, 4, 1, 64, 64)
vram_one = latent_one.element_size() * latent_one.nelement() / 1024**2
print(f"One latent at a time:   {vram_one:.1f} MB")
print()
print("vae.enable_vae_slicing() makes the pipeline decode one frame at a time,")
print("reducing peak VRAM. Used in your AnimateDiff setup (8GB VRAM RTX 5060).")
print()
print("enable_model_cpu_offload() moves the VAE to CPU when not in use,")
print("freeing GPU VRAM for the UNet during the 50 denoising steps.")

# =============================================================================
# EXPECTED OUTPUT:
# Compression: 48x per frame
# Scaled latent std: ~1.0
# All 16 latents at once: 0.5 MB (just for the latents; actual model activations much larger)
# =============================================================================
