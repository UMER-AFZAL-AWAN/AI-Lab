# =============================================================================
# 02 — REMOVING NOISE: THE REVERSE PROCESS
# =============================================================================
#
# LIKE YOU ARE 5:
# You have a pile of pure sand (noise).
# You have a smart machine (the UNet) that you trained earlier.
# You show the machine the sandy pile and ask: "what sand grain was added LAST?"
# It looks and says: "I think this grain was added last" and removes it.
# Now you have a slightly-less-sandy pile.
# You ask again. It removes another grain.
# 50 to 1000 times later, you can see the cat under the sand.
#
# That is DENOISING. The model removes one "layer" of noise at each step.
#
# SAMPLING ALGORITHMS:
# Different algorithms for how to do the denoising step:
# - DDPM: original. Needs 1000 steps. Slow.
# - DDIM: deterministic. Can use 20-50 steps. Fast. Used in AnimateDiff.
# - DPM++: mathematical shortcut. 15-20 steps.
#
# =============================================================================

import torch
import torch.nn as nn

# Reuse the noise schedule from 01_adding_noise.py
T = 1000
betas      = torch.linspace(1e-4, 2e-2, T)
alphas     = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

print("=" * 60)
print("ONE DENOISING STEP (DDPM)")
print("=" * 60)
#
# Given: x_t (noisy at step t), predicted noise eps_theta, timestep t
# Compute: x_{t-1} (slightly less noisy)
#
# Formula (simplified):
# x0_pred = (x_t - sqrt(1-alpha_bar_t) * eps_theta) / sqrt(alpha_bar_t)
# x_{t-1} = sqrt(alpha_bar_{t-1}) * x0_pred + sqrt(1-alpha_bar_{t-1}) * eps_theta
# (plus a small random term for stochasticity, which we skip here)

def ddpm_step(xt, t, eps_predicted, alpha_bars):
    """One DDPM reverse step: go from x_t to x_{t-1}."""
    abar_t  = alpha_bars[t].view(-1, 1, 1, 1)
    abar_tm1= alpha_bars[t - 1].view(-1, 1, 1, 1) if t > 0 else torch.ones_like(abar_t)

    # Predict x0 from xt and predicted noise
    x0_pred = (xt - (1 - abar_t).sqrt() * eps_predicted) / abar_t.sqrt()
    x0_pred = x0_pred.clamp(-1, 1)   # keep in reasonable range

    # Reconstruct x_{t-1}
    xtm1 = abar_tm1.sqrt() * x0_pred + (1 - abar_tm1).sqrt() * eps_predicted
    return xtm1

# Simulate: start with pure noise, run one step
torch.manual_seed(0)
xt = torch.randn(2, 4, 8, 8)       # pure noise (like x_T)
t  = torch.tensor([500, 500])       # at timestep 500

# A "perfect" predictor would know the original noise exactly.
# Here we use the noisy input as a stand-in (not trained, just demo).
eps_pred = torch.randn_like(xt)   # fake: in reality comes from UNet

xtm1 = ddpm_step(xt, 500, eps_pred, alpha_bars)
print(f"x_t shape:   {xt.shape}   (noisy at t=500)")
print(f"x_t-1 shape: {xtm1.shape}  (slightly less noisy)")
print(f"x_t std:   {xt.std().item():.4f}")
print(f"x_t-1 std: {xtm1.std().item():.4f}  (should differ slightly)")

print()
print("=" * 60)
print("DDIM SAMPLING (deterministic, 50 steps instead of 1000)")
print("=" * 60)
#
# DDIM (Denoising Diffusion Implicit Models) skips steps.
# Instead of going t=1000, 999, 998 ... 1,
# it jumps: t=1000, 980, 960 ... 20, 0  (every 20 steps).
# Same formula but on a subset of timesteps.
# Because it's deterministic (no random term), same seed = same image.

num_inference_steps = 10  # 50 in real use, 10 for demo
timesteps = torch.linspace(T - 1, 0, num_inference_steps, dtype=torch.long)
print(f"DDIM timesteps ({num_inference_steps} steps):")
print(f"  {timesteps.tolist()}")
print()

# Fake UNet that always predicts zero noise (for demo — a trained model would predict real noise)
class FakeUNet(nn.Module):
    def forward(self, xt, t):
        return torch.zeros_like(xt)   # pretend there is no noise

unet = FakeUNet()

# Denoising loop (conceptual)
x = torch.randn(1, 4, 8, 8)   # start: pure noise
print("Denoising loop:")
for step_idx, t_val in enumerate(timesteps):
    t_batch = t_val.unsqueeze(0)
    with torch.no_grad():
        eps = unet(x, t_batch)         # predict noise
    if t_val > 0:
        x = ddpm_step(x, t_val.item(), eps, alpha_bars)
    print(f"  step {step_idx+1:2d}  t={t_val.item():4d}  x_std={x.std().item():.4f}")

print(f"\nFinal output shape: {x.shape}  (a latent that VAE decodes to pixels)")

print()
print("=" * 60)
print("WHAT GUIDANCE DOES (CFG preview)")
print("=" * 60)
#
# In REAL AnimateDiff, the UNet runs TWICE per denoising step:
# - Once with the text prompt ("a cat playing piano")
# - Once without (empty prompt "")
# Then blends them: eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
# guidance_scale = 7.5 means "follow the prompt 7.5x more than random"
# Higher = more prompt-faithful but less diverse. Lower = more random.

eps_cond   = torch.randn(1, 4, 8, 8)
eps_uncond = torch.randn(1, 4, 8, 8)
guidance_scale = 7.5

eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
print(f"eps_uncond std: {eps_uncond.std().item():.3f}")
print(f"eps_cond std:   {eps_cond.std().item():.3f}")
print(f"eps_guided std: {eps_guided.std().item():.3f}  (amplified — more committed to prompt)")
print()
print(f"guidance_scale={guidance_scale}: each step steered strongly toward the prompt.")
print("AnimateDiff default is 7.5. Your pipeline likely uses this same value.")

# =============================================================================
# EXPECTED OUTPUT:
# DDIM timesteps (10 steps): [999, 888, 777, 666, 555, ...]
# eps_guided std: ~higher than cond or uncond alone
# =============================================================================
