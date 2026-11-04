# =============================================================================
# 06 — ANIMATEDIFF: MAKING THE IMAGES MOVE
# =============================================================================
#
# LIKE YOU ARE 5:
# Stable Diffusion makes one image. AnimateDiff makes 16 images at once
# that look like a smooth video.
#
# The trick: add a MOTION MODULE (a Transformer that looks across time)
# to the existing UNet. The spatial layers stay the same (they know what
# things look like). The motion layers are new (they learn how things MOVE).
#
# TRAINING:
# - Start with the pre-trained SD 1.5 UNet weights (frozen).
# - Add the motion module layers (initially random).
# - Train ONLY the motion module weights on video data.
# - The spatial weights already know "what a cat looks like".
#   The motion weights learn "how cats move".
#
# INFERENCE:
# - Input: 16 frames of pure noise, text prompt
# - Output: 16 frames of denoised latents
# - Decode all 16 with VAE -> 16 RGB frames -> video
#
# =============================================================================

import torch
import torch.nn as nn

print("=" * 60)
print("THE 5D TENSOR: (B, C, T, H, W)")
print("=" * 60)
#
# Normal image in SD: (B, C, H, W) — 4D
# AnimateDiff video:  (B, C, T, H, W) — 5D (T = num_frames)
#
# The spatial UNet layers process each frame independently:
#   (B, C, T, H, W) -> reshape to (B*T, C, H, W) -> apply UNet layer -> reshape back
#
# The motion module layers process across frames:
#   (B, C, T, H, W) -> reshape to (B*H*W, T, C) -> apply Transformer -> reshape back
#   Each spatial position attends to itself across all 16 time steps.

B, C, T, H, W = 1, 4, 16, 8, 8   # small latents for demo

latents_5d = torch.randn(B, C, T, H, W)
print(f"5D latent: {latents_5d.shape}  (batch, channels, frames, height, width)")

# How spatial layers process it
def process_spatial(x5d, spatial_layer):
    """Apply a 2D spatial layer to each frame independently."""
    B, C, T, H, W = x5d.shape
    # Merge batch and time
    x4d = x5d.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
    out4d = spatial_layer(x4d)
    # Restore time dimension
    out5d = out4d.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
    return out5d

spatial_conv = nn.Conv2d(C, C, kernel_size=3, padding=1)
out = process_spatial(latents_5d, spatial_conv)
print(f"After spatial layer: {out.shape}  (each frame processed independently)")

print()
print("=" * 60)
print("THE MOTION MODULE")
print("=" * 60)
#
# The motion module is a Transformer that runs along the TIME dimension.
# For each spatial position (h, w), it has a sequence of T=16 latent vectors.
# The Transformer (self-attention) lets each frame "look at" all other frames.
# This is what makes the output look like a smooth video instead of 16 random images.

class TemporalTransformer(nn.Module):
    """
    Simplified motion module: applies self-attention across the T dimension.
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x5d):
        B, C, T, H, W = x5d.shape
        # Reshape: each spatial position is a sequence of T feature vectors
        # (B, C, T, H, W) -> (B*H*W, T, C)
        x = x5d.permute(0, 3, 4, 2, 1).contiguous()  # (B, H, W, T, C)
        x = x.view(B * H * W, T, C)
        # Self-attention across T (each frame attends to all other frames)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)       # residual + norm
        # Reshape back
        x = x.view(B, H, W, T, C).permute(0, 4, 3, 1, 2).contiguous()
        return x

motion = TemporalTransformer(channels=C, num_heads=2)
out_motion = motion(latents_5d)
print(f"Input:  {latents_5d.shape}")
print(f"Output: {out_motion.shape}  (frames now attend to each other)")

params = sum(p.numel() for p in motion.parameters())
print(f"Motion module parameters: {params:,}")
print("(In real AnimateDiff motion modules: ~300M for the full UNet's motion layers)")

print()
print("=" * 60)
print("PING-PONG LOOP: HOW INJECT_SCENE USES ANIMATEDIFF")
print("=" * 60)
#
# Your LocalVideoStudio pipeline generates 16 frames with AnimateDiff.
# To make a seamless loop, it generates a "ping-pong" sequence:
# forward frames: [0,1,2,3,4,5,6,7] + reversed middle: [6,5,4,3,2,1]
# Total: 14 frames, but they loop smoothly because last=first.

num_frames = 8
forward = list(range(num_frames))
middle  = list(range(num_frames - 2, 0, -1))  # exclude first and last
pingpong = forward + middle
print(f"Forward:   {forward}")
print(f"Middle:    {middle}")
print(f"Ping-pong: {pingpong}")
print(f"Length:    {len(pingpong)} frames")
print()
print("First frame == last frame? ", end="")
print(pingpong[0] == pingpong[-1])  # True if even

print()
print("=" * 60)
print("VRAM USAGE FOR ANIMATEDIFF")
print("=" * 60)
#
# Real AnimateDiff on 8GB VRAM requires:
# - enable_model_cpu_offload(): move UNet to CPU when VAE is running
# - enable_vae_slicing(): decode one frame at a time (not all 16)
# Your pipeline already does both.

latent_all = torch.randn(1, 4, 16, 64, 64)
latent_one = torch.randn(1, 4, 1, 64, 64)
print(f"All 16 latents: {latent_all.shape}  = {latent_all.numel() * 4 / 1024**2:.1f} MB (FP32)")
print(f"One latent:     {latent_one.shape}   = {latent_one.numel() * 4 / 1024**2:.2f} MB (FP32)")
print("VAE slicing: decodes one at a time -> 16x less VRAM for the decode step")

# =============================================================================
# EXPECTED OUTPUT:
# 5D latent: torch.Size([1, 4, 16, 8, 8])
# After spatial layer: torch.Size([1, 4, 16, 8, 8])
# Ping-pong: [0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1]
# =============================================================================
