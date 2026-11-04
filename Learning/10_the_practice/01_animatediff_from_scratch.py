# =============================================================================
# 10_the_practice / 01 — ANIMATEDIFF FROM SCRATCH (no magic pipe())
# =============================================================================
#
# LIKE YOU ARE 5:
# Every time you call pipe("a cat playing piano"), magic happens.
# This file tears the magic apart and shows EVERY STEP by name.
# After this file, pipe() is not magic — it's just your code, condensed.
#
# C# ANALOGY:
# Like reading the disassembled IL of a framework method.
# pipe() = AnimateDiffPipeline.GenerateAsync() under the hood.
# This file writes that method out manually so you can see each call.
#
# WHAT THIS COVERS:
#   Step 1  — Encode the text prompt into embeddings
#   Step 2  — Encode the negative prompt
#   Step 3  — Generate a random starting latent
#   Step 4  — Set up the scheduler's timestep list
#   Step 5  — The denoising loop (the core)
#     5a    —   Classifier-Free Guidance (CFG) batching
#     5b    —   UNet forward pass (predict noise)
#     5c    —   CFG: combine unconditional + conditional predictions
#     5d    —   Scheduler step: update latent
#   Step 6  — Decode the final latent with the VAE
#   Step 7  — Convert tensors to PIL Images
#
# =============================================================================

import torch
import torch.nn as nn
import math

print("=" * 60)
print("THE FULL ANIMATEDIFF INFERENCE LOOP — ANNOTATED")
print("=" * 60)
print("""
# ─── Prerequisites ──────────────────────────────────────────────────────────
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_gif
import torch

device     = "cuda"
dtype      = torch.float16
num_frames = 16
height     = 512
width      = 512
prompt     = "a cat playing piano, cinematic lighting"
neg_prompt = "blurry, ugly, watermark"
steps      = 25
cfg_scale  = 7.5
seed       = 42

# Load components (same as normal usage)
adapter   = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
pipe      = AnimateDiffPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                motion_adapter=adapter,
                torch_dtype=dtype
            )
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()
""")

print("""
# ─── STEP 1: Encode the text prompt ─────────────────────────────────────────
# Tokenise → 77 token IDs → CLIP text encoder → (1, 77, 768)
text_input = pipe.tokenizer(
    prompt,
    padding="max_length",
    max_length=pipe.tokenizer.model_max_length,   # 77
    truncation=True,
    return_tensors="pt",
)
with torch.no_grad():
    text_embeddings = pipe.text_encoder(text_input.input_ids.to(device))[0]
# text_embeddings: (1, 77, 768)

# ─── STEP 2: Encode the negative prompt ─────────────────────────────────────
# Same shape, represents "what to avoid"
uncond_input = pipe.tokenizer(
    neg_prompt,
    padding="max_length",
    max_length=pipe.tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)
with torch.no_grad():
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]
# uncond_embeddings: (1, 77, 768)

# CFG works by doubling the batch:
# [unconditional, conditional] processed in one forward pass each step
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
# text_embeddings: (2, 77, 768)
""")

print("""
# ─── STEP 3: Generate the starting latent ───────────────────────────────────
# Instead of starting from the image, start from pure noise.
# Shape: (B, C, T, H/8, W/8) — the latent is 8x smaller than pixel space
#         1  4  16  64   64   for 512×512, 16 frames

generator = torch.Generator(device=device).manual_seed(seed)
latents = torch.randn(
    (1, pipe.unet.config.in_channels, num_frames, height // 8, width // 8),
    generator=generator,
    device=device,
    dtype=dtype,
)

# Scale to the scheduler's starting sigma (DDIMScheduler: scale = 1.0)
latents = latents * pipe.scheduler.init_noise_sigma
# latents: (1, 4, 16, 64, 64)
""")

print("""
# ─── STEP 4: Set up the scheduler's timestep list ───────────────────────────
# The scheduler decides which noise levels (timesteps) to visit.
# DDIM with 25 steps picks 25 evenly spaced values from 0–999.
pipe.scheduler.set_timesteps(steps, device=device)
timesteps = pipe.scheduler.timesteps
# timesteps: tensor([999, 958, 917, ..., 41, 0])  — 25 values, descending
""")

print("""
# ─── STEP 5: THE DENOISING LOOP ─────────────────────────────────────────────
for i, t in enumerate(timesteps):

    # 5a — CFG: double the latent batch for unconditional + conditional
    latent_model_input = torch.cat([latents] * 2)
    # latent_model_input: (2, 4, 16, 64, 64)

    # Apply scheduler scaling (some schedulers rescale the latent at each step)
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

    # 5b — UNet forward pass: predict the noise at this timestep
    with torch.no_grad():
        noise_pred = pipe.unet(
            latent_model_input,       # (2, 4, 16, 64, 64) — the noisy latent
            t,                         # current timestep (a scalar)
            encoder_hidden_states=text_embeddings,  # (2, 77, 768) — text conditioning
        ).sample
    # noise_pred: (2, 4, 16, 64, 64)  — predicted noise for each batch item

    # 5c — CFG: split and combine unconditional + conditional predictions
    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
    # noise_pred: (1, 4, 16, 64, 64)  — the CFG-adjusted noise prediction

    # 5d — Scheduler step: use the predicted noise to update the latent
    latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
    # latents: (1, 4, 16, 64, 64)  — slightly less noisy than before

# After 25 steps: latents ≈ the clean video latent
""")

print("""
# ─── STEP 6: Decode the latent with the VAE ─────────────────────────────────
# The VAE's decode() takes a latent and produces a pixel-space image/video.
# For video: process one frame at a time (enable_vae_slicing() does this automatically)

latents = 1 / pipe.vae.config.scaling_factor * latents
# Undo the latent scaling (SD uses scaling_factor=0.18215 to normalise the latent space)

# Decode T frames individually (what vae_slicing does internally)
video_frames = []
for frame_idx in range(num_frames):
    frame_latent = latents[:, :, frame_idx, :, :]   # (1, 4, 64, 64)
    with torch.no_grad():
        frame_pixels = pipe.vae.decode(frame_latent).sample   # (1, 3, 512, 512)
    video_frames.append(frame_pixels)

video = torch.cat(video_frames, dim=0)   # (16, 3, 512, 512)
""")

print("""
# ─── STEP 7: Convert to PIL Images ──────────────────────────────────────────
# Tensor values are in [-1, 1]. PIL needs [0, 255] uint8.
video = (video / 2 + 0.5).clamp(0, 1)   # → [0, 1]
video = (video * 255).to(torch.uint8)    # → [0, 255]
video = video.cpu().numpy()              # → numpy array

from PIL import Image
pil_frames = [Image.fromarray(frame.transpose(1, 2, 0)) for frame in video]
# pil_frames: list of 16 PIL Images, each 512×512 RGB

# Save as GIF
export_to_gif(pil_frames, "output.gif")
""")

print("=" * 60)
print("RUNNABLE MINI DEMO (stand-ins, no model download)")
print("=" * 60)

# Simulate the full loop with tiny tensors
torch.manual_seed(42)

B, C, T, H, W = 1, 4, 4, 8, 8   # tiny: 4 frames, 8×8 latents
D_text = 16    # text embedding dim (real: 768)
N_text = 5     # text tokens (real: 77)
steps  = 3

class TinyUNet(nn.Module):
    """Stand-in for the full UNet. Just adds a small learned perturbation."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(C, C, 1)  # 1×1×1 conv — no spatial processing
    def forward(self, x, t, encoder_hidden_states=None):
        return type('obj', (object,), {'sample': self.conv(x)})()

class TinyVAEDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.ConvTranspose2d(C, 3, 1)  # C latent channels → 3 RGB
    def forward(self, z):
        return type('obj', (object,), {'sample': torch.tanh(self.up(z))})()

unet    = TinyUNet()
decoder = TinyVAEDecoder()

# Step 3: starting latent
latents = torch.randn(B, C, T, H, W)
print(f"Step 3 — starting latent: {list(latents.shape)}")

# Step 4: timesteps
timesteps_sim = torch.linspace(999, 0, steps).long()
print(f"Step 4 — timesteps: {timesteps_sim.tolist()}")

# Step 5: denoising loop
for i, t in enumerate(timesteps_sim):
    # CFG batch
    latent_in = torch.cat([latents] * 2)                    # (2B, C, T, H, W)
    text_cond = torch.randn(B * 2, N_text, D_text)

    with torch.no_grad():
        noise_pred = unet(latent_in, t, encoder_hidden_states=text_cond).sample

    noise_uncond, noise_cond = noise_pred.chunk(2)
    cfg = 7.5
    noise_pred = noise_uncond + cfg * (noise_cond - noise_uncond)

    # Simplified scheduler step (Euler: x_t-1 = x_t - noise * dt)
    dt = 1.0 / steps
    latents = latents - noise_pred * dt
    print(f"  Step 5 ({i+1}/{steps}) t={t.item():4d}: latent mean={latents.mean():.4f}")

# Step 6: decode
pil_frames = []
for f in range(T):
    frame_latent = latents[:, :, f, :, :]                   # (1, C, H, W)
    with torch.no_grad():
        pixels = decoder(frame_latent).sample                # (1, 3, H, W)
    pixels = (pixels / 2 + 0.5).clamp(0, 1)
    pixels = (pixels * 255).byte().squeeze(0).permute(1, 2, 0).numpy()
    pil_frames.append(pixels.shape)   # just show the shape in this demo

print(f"Step 6 — decoded {T} frames, each shape: {pil_frames[0]}")
print(f"Step 7 — convert to PIL → save as GIF")
print()
print("All 7 steps complete. This is exactly what pipe() does inside.")

# =============================================================================
# EXPECTED OUTPUT:
# Step 3 — starting latent: [1, 4, 4, 8, 8]
# Step 4 — timesteps: [999, 499, 0]
# Step 5 (1/3) t= 999: latent mean=...
# Step 5 (2/3) t= 499: latent mean=...
# Step 5 (3/3) t=   0: latent mean=...
# Step 6 — decoded 4 frames, each shape: (8, 8, 3)
# All 7 steps complete. This is exactly what pipe() does inside.
# =============================================================================
