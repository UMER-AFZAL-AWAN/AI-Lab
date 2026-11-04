# =============================================================================
# 03 — SCHEDULER: THE TRAFFIC CONTROLLER
# =============================================================================
#
# LIKE YOU ARE 5:
# The noise schedule is like a recipe book that says:
#   "At step 1, add a tiny pinch. At step 500, add a bigger pinch."
# The SCHEDULER is the assistant who reads the recipe book
# and tells the model: "right now you're at step 500, here's what to do."
#
# In diffusers, the scheduler:
# 1. Sets the timesteps (which steps to run)
# 2. Scales the latent before feeding it to the UNet
# 3. Does the denoising step computation (all the alpha_bar maths)
#
# Common schedulers:
# - DDPMScheduler:    original, 1000 steps
# - DDIMScheduler:    deterministic, 20-50 steps
# - EulerDiscreteScheduler: fast, 20 steps, good quality
# - DPMSolverMultistepScheduler: 15-20 steps, very good quality
#
# =============================================================================

import torch
print("=" * 60)
print("USING A DIFFUSERS SCHEDULER (without loading a model)")
print("=" * 60)

try:
    from diffusers import DDIMScheduler

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        beta_schedule="linear",
    )
    print(f"Scheduler type: {type(scheduler).__name__}")
    print(f"Num train timesteps: {scheduler.config.num_train_timesteps}")

    # Set how many steps for inference
    scheduler.set_timesteps(num_inference_steps=20)
    print(f"Inference timesteps (20 steps): {scheduler.timesteps[:5].tolist()} ... {scheduler.timesteps[-5:].tolist()}")

    # The scheduler knows all the alpha_bar values
    abar = scheduler.alphas_cumprod
    print(f"\nalphas_cumprod shape: {abar.shape}")
    print(f"alpha_bar[0]:   {abar[0].item():.4f}   (clean image)")
    print(f"alpha_bar[500]: {abar[499].item():.4f}  (half noisy)")
    print(f"alpha_bar[999]: {abar[999].item():.6f} (pure noise)")

    print()
    print("One denoising step with scheduler.step():")
    # Simulate
    latents    = torch.randn(1, 4, 8, 8)
    t          = scheduler.timesteps[0]   # first (most noisy) step
    model_output = torch.randn_like(latents)  # fake UNet noise prediction

    result = scheduler.step(model_output, t, latents)
    print(f"  t={t.item()}  input={latents.shape}  output={result.prev_sample.shape}")
    print(f"  Input std:  {latents.std().item():.4f}")
    print(f"  Output std: {result.prev_sample.std().item():.4f}  (slightly less noisy)")

except ImportError:
    print("diffusers not found. Run: pip install diffusers")

print()
print("=" * 60)
print("MANUAL SCHEDULER (what diffusers does internally)")
print("=" * 60)
#
# Here is the core of what a scheduler does, with no library:

class SimpleScheduler:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=2e-2):
        self.T         = T
        betas          = torch.linspace(beta_start, beta_end, T)
        alphas         = 1.0 - betas
        self.alpha_bars= torch.cumprod(alphas, dim=0)
        self.betas     = betas

    def set_timesteps(self, num_inference_steps):
        self.timesteps = torch.linspace(self.T - 1, 0, num_inference_steps, dtype=torch.long)

    def add_noise(self, x0, noise, t):
        abar = self.alpha_bars[t].view(-1, 1, 1, 1)
        return abar.sqrt() * x0 + (1 - abar).sqrt() * noise

    def step(self, eps_pred, t, xt):
        abar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        t_prev = max(t - self.T // len(self.timesteps), 0)
        abar_prev = self.alpha_bars[t_prev].view(-1, 1, 1, 1)

        # DDIM deterministic step
        x0_pred = (xt - (1 - abar_t).sqrt() * eps_pred) / abar_t.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)
        return abar_prev.sqrt() * x0_pred + (1 - abar_prev).sqrt() * eps_pred

sched = SimpleScheduler()
sched.set_timesteps(10)
print(f"Manual scheduler timesteps: {sched.timesteps.tolist()}")

x = torch.randn(1, 4, 8, 8)
for i, t in enumerate(sched.timesteps):
    eps = torch.randn_like(x) * 0.1   # fake: trained model would predict real noise
    x = sched.step(eps, t.item(), x)
    if i < 3 or i == 9:
        print(f"  Step {i+1:2d}  t={t.item():4d}  std={x.std().item():.4f}")

print(f"\nFinal latent shape: {x.shape}")

# =============================================================================
# EXPECTED OUTPUT:
# Scheduler type: DDIMScheduler
# Inference timesteps (20 steps): [999, 949, 899, ...] ... [49, 0]
# =============================================================================
