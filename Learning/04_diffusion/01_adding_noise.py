# =============================================================================
# 01 — ADDING NOISE: THE FORWARD PROCESS
# =============================================================================
#
# LIKE YOU ARE 5:
# Imagine you have a photo of a cat.
# You take a bucket of sand and throw a tiny pinch on it. Still looks like a cat.
# You throw another tiny pinch. Still a cat.
# You keep going. 1000 pinches later, it looks like a completely sandy mess —
# indistinguishable from random sand.
#
# That is the FORWARD PROCESS (or diffusion process).
# Start with a real image. Add a tiny bit of random noise T times.
# After T steps: pure noise. The original image is gone.
#
# A DIFFUSION MODEL learns the REVERSE: given the sandy mess,
# can you figure out what was under it and remove the sand grain by grain?
#
# KEY INSIGHT: You can jump from step 0 to step t in ONE calculation.
# No need to simulate all T steps. This makes training efficient.
# Formula: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
#
# =============================================================================

import torch
import torch.nn as nn

print("=" * 60)
print("THE NOISE SCHEDULE")
print("=" * 60)
#
# The noise schedule controls HOW MUCH noise is added at each step.
# "Linear" schedule: start = 0.0001, end = 0.02 (small -> bigger jumps)
# "Cosine" schedule: smoother, works better for small images.
#
# beta_t  = amount of noise added at step t
# alpha_t = 1 - beta_t  (how much of the signal remains)
# alpha_bar_t = product of all alphas from 1 to t (cumulative signal remaining)

T = 1000  # total number of noise steps

# Linear beta schedule (SD 1.5 uses this)
beta_start = 1e-4
beta_end   = 2e-2
betas      = torch.linspace(beta_start, beta_end, T)   # shape (T,)
alphas     = 1.0 - betas                                # shape (T,)
alpha_bars = torch.cumprod(alphas, dim=0)               # shape (T,)

print(f"beta[0]   = {betas[0].item():.5f}   (tiny noise at start)")
print(f"beta[999] = {betas[-1].item():.5f}  (larger noise near end)")
print(f"alpha_bar[0]   = {alpha_bars[0].item():.4f}  (almost all signal at t=1)")
print(f"alpha_bar[499] = {alpha_bars[499].item():.4f}  (half signal at t=500)")
print(f"alpha_bar[999] = {alpha_bars[-1].item():.6f} (almost no signal at t=1000)")

print()
print("=" * 60)
print("ADDING NOISE AT TIMESTEP t (the closed-form shortcut)")
print("=" * 60)

def add_noise(x0, t, alpha_bars):
    """
    Given a clean image x0 and timestep t, compute x_t directly.
    No need to simulate t steps one at a time.
    """
    abar = alpha_bars[t].view(-1, 1, 1, 1)   # shape (B, 1, 1, 1) for broadcasting
    noise = torch.randn_like(x0)
    x_t = abar.sqrt() * x0 + (1 - abar).sqrt() * noise
    return x_t, noise

# Simulate a batch of 4 latent frames (as AnimateDiff would see them)
# Shape: (B=4, C=4, H=8, W=8) — small latents for demo
x0 = torch.randn(4, 4, 8, 8)

# Add noise at different timesteps
for t_val in [0, 100, 500, 999]:
    t = torch.tensor([t_val] * 4)   # batch of same timestep
    xt, noise = add_noise(x0, t, alpha_bars)
    signal_ratio = alpha_bars[t_val].item()
    print(f"t={t_val:4d}  alpha_bar={signal_ratio:.4f}  "
          f"x0 mean={x0.mean().item():.3f}  "
          f"xt mean={xt.mean().item():.3f}  "
          f"xt std={xt.std().item():.3f}")

print()
print("At t=0:   xt ≈ x0  (clean image)")
print("At t=999: xt ≈ noise  (pure noise, std ≈ 1)")

print()
print("=" * 60)
print("WHAT THE UNet LEARNS")
print("=" * 60)
#
# Given a noisy image xt at timestep t,
# the UNet predicts the noise eps that was added.
# Then: x0_estimate = (xt - sqrt(1-alpha_bar) * eps_predicted) / sqrt(alpha_bar)
#
# The UNet takes BOTH xt AND t as input.
# It needs to know "how noisy is this?" to know how much to remove.
# t is encoded as a sinusoidal embedding (like word position in Transformers).

# Simple noise predictor (conceptual, not architecturally accurate)
class TinyNoisePredictor(nn.Module):
    def __init__(self, channels=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels + 1, 32, 3, padding=1),  # +1 for timestep channel
            nn.ReLU(),
            nn.Conv2d(32, channels, 3, padding=1),
        )

    def forward(self, xt, t):
        B = xt.shape[0]
        # Encode timestep as a constant channel (simplified)
        t_channel = (t.float() / 1000).view(B, 1, 1, 1).expand(B, 1, *xt.shape[2:])
        x_with_t  = torch.cat([xt, t_channel], dim=1)
        return self.net(x_with_t)

predictor = TinyNoisePredictor(channels=4)
t = torch.tensor([500, 100, 800, 200])
xt_batch, true_noise = add_noise(x0, t, alpha_bars)
pred_noise = predictor(xt_batch, t)

print(f"Noisy input:      {xt_batch.shape}")
print(f"Timestep:         {t}")
print(f"Predicted noise:  {pred_noise.shape}  (same shape as input)")
print(f"True noise:       {true_noise.shape}")
print()
print("Loss = MSE(pred_noise, true_noise)")
print("That's it. Train for a long time. The model learns to denoise.")

loss = torch.nn.functional.mse_loss(pred_noise, true_noise)
print(f"Initial loss (random model): {loss.item():.4f}")

# =============================================================================
# EXPECTED OUTPUT:
# beta[0]   = 0.00010   (tiny noise at start)
# beta[999] = 0.02000  (larger noise near end)
# alpha_bar[999] = ~0.00  (almost no signal at t=1000)
# At t=999: xt ≈ noise  (pure noise, std ≈ 1)
# =============================================================================
