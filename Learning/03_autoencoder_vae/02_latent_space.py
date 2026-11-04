# =============================================================================
# 02 — LATENT SPACE: THE COMPRESSED WORLD
# =============================================================================
#
# LIKE YOU ARE 5:
# Imagine a map of the world.
# Every country is in a specific place on the map.
# France and Spain are CLOSE because they are similar (European neighbours).
# France and Japan are FAR because they are very different.
#
# In a trained autoencoder, the latent space is like a map of the data.
# Images of the number "3" cluster in one region.
# Images of the number "7" cluster in another region.
# Blurry images cluster near clear images of the same thing.
#
# This map is LEARNED — the network figures out the best map by itself.
#
# WHY THIS MATTERS FOR AI VIDEO:
# AnimateDiff and Stable Diffusion generate images by starting at a random
# point in latent space and slowly moving to a valid image point.
# The latent space of their VAE is a 4-channel 64x64 grid.
# Adding noise = moving randomly. Denoising = moving toward a valid image.
# The TEXT PROMPT steers which direction "valid" means.
#
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("WHAT LIVES IN LATENT SPACE")
print("=" * 60)

# Imagine we have a trained autoencoder.
# Two similar images should produce nearby latent codes.
# We simulate this with two "similar" and one "different" input.

torch.manual_seed(42)

class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 2),   # 2D latent for easy visualisation
        )
    def forward(self, x): return self.fc(x)

enc = TinyEncoder()

# Two "similar" inputs (small difference) and one "different" input
img_A     = torch.tensor([[1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.]])
img_A_noisy = img_A + 0.05 * torch.randn_like(img_A)
img_B     = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.]])

z_A       = enc(img_A)
z_A_noisy = enc(img_A_noisy)
z_B       = enc(img_B)

dist_similar  = (z_A - z_A_noisy).norm().item()
dist_different= (z_A - z_B).norm().item()
print(f"Latent of A:         {z_A.detach().numpy().round(2)}")
print(f"Latent of A+noise:   {z_A_noisy.detach().numpy().round(2)}")
print(f"Latent of B:         {z_B.detach().numpy().round(2)}")
print(f"Distance A vs A+noise: {dist_similar:.4f}  (should be small)")
print(f"Distance A vs B:       {dist_different:.4f}  (should be larger)")

print()
print("=" * 60)
print("INTERPOLATION IN LATENT SPACE")
print("=" * 60)
#
# Because similar things are close in latent space,
# you can INTERPOLATE between two codes and get something
# that is "between" the two inputs.
#
# E.g. interpolate between "cat" and "dog" to get a creature
# that is halfway between them.
# Or between "frame 0" and "frame 10" to get frames 1-9.
# RIFE does this with optical flow instead of latent interpolation,
# but Stable Diffusion Video and AnimateDiff use latent interpolation.

class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 8), nn.ReLU(),
            nn.Linear(8, 16), nn.Sigmoid(),
        )
    def forward(self, z): return self.fc(z)

dec = TinyDecoder()

# Interpolate from z_A to z_B in 5 steps
print("Interpolating from z_A to z_B in latent space:")
for i, t in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
    z_interp = (1 - t) * z_A + t * z_B
    out = dec(z_interp.detach())
    print(f"  t={t:.2f}  z={z_interp.detach().numpy().round(2)}  "
          f"output[:4]={out.detach().numpy()[0, :4].round(2)}")

print()
print("=" * 60)
print("THE SHAPE OF SD/ANIMATEDIFF LATENT SPACE")
print("=" * 60)
#
# In Stable Diffusion (and therefore AnimateDiff):
# Input image: (1, 3, 512, 512) — RGB, 512x512 pixels
# Latent code: (1, 4, 64, 64)   — 4 channels, 64x64
#
# The 8x spatial compression comes from the VAE encoder.
# The 4 channels are abstract "features" learned by the VAE.
# They are NOT RGB. They encode structure, texture, colour
# in a form the diffusion model can work with efficiently.
#
# 512x512x3 = 786,432 values (pixel space)
# 64x64x4   =  16,384 values (latent space)
# That is a 48x compression.
# Running the diffusion model 50 times costs 50 × 16,384 ops
# instead of 50 × 786,432 ops. Much cheaper.

pixel_space  = 512 * 512 * 3
latent_space = 64  * 64  * 4
print(f"Pixel space:  {pixel_space:,} values  (3 × 512 × 512)")
print(f"Latent space: {latent_space:,} values  (4 × 64 × 64)")
print(f"Compression:  {pixel_space / latent_space:.0f}x")
print()
print("AnimateDiff generates 16 frames at once:")
print(f"  16 latent frames: {16 * latent_space:,} values")
print(f"  16 pixel  frames: {16 * pixel_space:,} values")
print(f"  Still {pixel_space // latent_space}x cheaper to run diffusion in latent space.")

# =============================================================================
# EXPECTED OUTPUT:
# Distance A vs A+noise: ~0.02  (should be small)
# Distance A vs B:       ~0.5   (should be larger)
# Compression:  48x
# =============================================================================
