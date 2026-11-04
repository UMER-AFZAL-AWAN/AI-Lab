# =============================================================================
# 04_diffusion / 07 — DiT AND FLOW MATCHING
# =============================================================================
#
# LIKE YOU ARE 5:
# AnimateDiff uses a UNet as its brain (a CNN with skip connections).
# Wan and MiniMax-H3 use a DiT — a Transformer as the brain instead.
# Instead of building rooms (UNet blocks), you do attention across patches.
#
# FLOW MATCHING:
# Regular diffusion (DDPM) adds noise with a random schedule.
# Flow Matching says: just draw a straight line from noise → clean image.
# It's simpler math, trains faster, needs fewer denoising steps.
# Wan2.1 uses Flow Matching. That's why it uses FlowMatchEulerDiscreteScheduler.
#
# WHY IT MATTERS FOR YOUR GOAL:
# MiniMax-H3 and Wan are DiT-based. If you switch from AnimateDiff to Wan,
# - The "unet" key in model_index.json becomes "transformer"
# - The scheduler changes from DDIM → FlowMatchEuler
# - The input format changes (may be 3D patches, not 2D+time)
#
# =============================================================================

import torch
import torch.nn as nn
import math

print("=" * 60)
print("PART 1: THE UNET DENOISER (what AnimateDiff uses)")
print("=" * 60)
print("""
UNet architecture for denoising:

Input: noisy latent  (B, C, T, H, W) ← for video; 4-16 channels
  │
  ├─ DownBlock 0  ← conv + cross-attn + motion module
  ├─ DownBlock 1  ← conv + cross-attn + motion module
  ├─ DownBlock 2  ← conv + cross-attn + motion module
  │                                     (gets smaller spatially)
  ├─ MiddleBlock  ← cross-attn
  │
  ├─ UpBlock 2    ← conv + cross-attn + motion module  (skip connection from Down 2)
  ├─ UpBlock 1    ← conv + cross-attn + motion module  (skip connection from Down 1)
  ├─ UpBlock 0    ← conv + cross-attn + motion module  (skip connection from Down 0)
  │
Output: predicted noise  (same shape as input)

KEY: The UNet's hourglass shape means:
  - Early layers see fine details (high resolution)
  - Middle layers see global structure (low resolution)
  - Skip connections combine both
""")

print("=" * 60)
print("PART 2: THE DiT DENOISER (what Wan / MiniMax-H3 use)")
print("=" * 60)
print("""
DiT = Diffusion Transformer

Input: noisy latent  (B, C, T, H, W)
  │
  ├─ Patchify: split into 3D patches
  │             (B, num_patches, patch_dim)
  │             e.g. 16×16 spatial patches × T temporal patches
  │
  ├─ Positional embedding (learned or RoPE)
  │
  ├─ N × DiT blocks:
  │     AdaLayerNorm (conditioning on timestep + text)
  │     Self-attention across all patches
  │     Cross-attention to text embedding
  │     MLP feedforward
  │
  ├─ Unpatchify: reconstruct spatial/temporal layout
  │
Output: predicted noise  (same shape as input)

KEY: Every patch attends to every other patch in every block.
  - No hourglass shape, no skip connections
  - Longer range understanding (a patch at frame 1 can "see" frame 16)
  - More VRAM because attention over T×H/p×W/p tokens (larger sequence)
  - Better quality, especially for long videos and complex motion
""")

print("=" * 60)
print("PART 3: SIDE-BY-SIDE CODE DEMO")
print("=" * 60)

# ---------------------------------------------------------------------------
# Toy UNet block
# ---------------------------------------------------------------------------
class ToyUNetBlock(nn.Module):
    """
    One level of a UNet: conv down (halves spatial), conv up (doubles).
    Real UNet also has cross-attention and residual connections.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.up   = nn.ConvTranspose2d(out_ch, in_ch, 2, stride=2)

    def forward(self, x):
        return self.up(self.down(x))   # down then up → same size

# ---------------------------------------------------------------------------
# Toy DiT block
# ---------------------------------------------------------------------------
class ToyDiTBlock(nn.Module):
    """
    One DiT block: patchify → self-attention → unpatchify.
    Real DiT also has: AdaLayerNorm (timestep conditioning), cross-attention.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.norm   = nn.LayerNorm(embed_dim)
        self.attn   = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn    = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        # x: (B, N, D)  — B batches, N patches, D dimensions
        h      = self.norm(x)
        attn_out, _ = self.attn(h, h, h)
        x      = x + attn_out
        x      = x + self.ffn(self.norm(x))
        return x


def patchify(x, patch_size):
    """
    Split a spatial feature map into patches.
    x: (B, C, H, W)
    Returns: (B, N, C*patch_size*patch_size)  where N = (H/p)*(W/p)
    """
    B, C, H, W = x.shape
    p = patch_size
    assert H % p == 0 and W % p == 0, "H and W must be divisible by patch_size"
    # Rearrange into patches
    x = x.reshape(B, C, H // p, p, W // p, p)
    x = x.permute(0, 2, 4, 1, 3, 5)         # (B, H/p, W/p, C, p, p)
    x = x.reshape(B, (H // p) * (W // p), C * p * p)
    return x

def unpatchify(x, patch_size, C, H, W):
    """Inverse of patchify."""
    B  = x.shape[0]
    p  = patch_size
    nh, nw = H // p, W // p
    x = x.reshape(B, nh, nw, C, p, p)
    x = x.permute(0, 3, 1, 4, 2, 5)         # (B, C, H/p, p, W/p, p)
    x = x.reshape(B, C, H, W)
    return x


# Demo
B, C, H, W   = 1, 4, 16, 16
patch_size   = 4
embed_dim    = C * patch_size * patch_size   # 4*4*4 = 64

x = torch.randn(B, C, H, W)
print(f"Input latent: {list(x.shape)}")

patches = patchify(x, patch_size)
print(f"After patchify: {list(patches.shape)}")   # (1, 16, 64)

# Project patches to embed_dim (in real DiT, this is a linear layer)
proj  = nn.Linear(embed_dim, embed_dim)
h     = proj(patches)                             # (1, 16, 64)

dit_block = ToyDiTBlock(embed_dim, num_heads=4)
h         = dit_block(h)
print(f"After DiT block: {list(h.shape)}")        # (1, 16, 64)

# Project back and unpatchify
h_proj = nn.Linear(embed_dim, embed_dim)
h      = h_proj(h)
out    = unpatchify(h, patch_size, C, H, W)
print(f"After unpatchify: {list(out.shape)}")     # (1, 4, 16, 16) — same as input

print()
print("=" * 60)
print("PART 4: DDPM vs FLOW MATCHING")
print("=" * 60)
print("""
── DDPM (what AnimateDiff, SD 1.x use) ───────────────────

Forward: q(x_t | x_0) = sqrt(ᾱ_t)*x_0 + sqrt(1-ᾱ_t)*ε
  - ᾱ_t follows a cosine/linear schedule
  - At T=1000, x_T ≈ pure Gaussian noise
  - The curve from x_0 → x_T is NOT a straight line

Reverse: predict ε (the noise) → subtract it → repeat
  - Needs 50-1000 steps for good quality (without DDIM acceleration)

── Flow Matching (what Wan 2.1, SD3, Flux use) ──────────

Forward: x_t = (1-t)*x_0 + t*ε    t ∈ [0, 1]
  - That IS a straight line from clean (t=0) to noise (t=1)
  - No special schedule needed — just linear interpolation

Reverse: predict the vector field v = dx/dt = ε - x_0
  - Then Euler-step: x_{t-dt} = x_t - v*dt
  - Needs only 20-30 steps because the path is straight

ADVANTAGE:
  - Simpler math (ODE instead of SDE)
  - Faster training convergence
  - Fewer inference steps needed
  - Generalises to arbitrary source/target distributions
""")

print("=" * 60)
print("PART 5: FLOW MATCHING STEP DEMO")
print("=" * 60)

def add_noise_flow_matching(x0, t):
    """
    Flow Matching forward: straight line from x0 (clean) to noise.
    t ∈ [0, 1]. At t=0 → x0 (clean). At t=1 → pure noise.
    """
    noise = torch.randn_like(x0)
    x_t   = (1 - t) * x0 + t * noise
    # True velocity field: v = dx/dt = noise - x0
    v_true = noise - x0
    return x_t, v_true, noise

def euler_step_flow_matching(x_t, v_pred, dt):
    """One Euler denoising step: move x_t along predicted velocity."""
    return x_t - v_pred * dt

x0  = torch.randn(1, 4, 8, 8)   # clean latent (stand-in)

# Forward: add noise at t=0.5
t   = 0.5
x_t, v_true, noise = add_noise_flow_matching(x0, t)
print(f"Clean x0:      mean={x0.mean():.4f}")
print(f"Noisy x_t:     mean={x_t.mean():.4f}")
print(f"True velocity: mean={v_true.mean():.4f}")

# Imagine a perfect predictor (toy: use the true v)
v_pred = v_true   # in real life, this is what the DiT outputs

# Denoising step: go from t=0.5 to t=0.0 in one big step
dt   = 0.5
x_recovered = euler_step_flow_matching(x_t, v_pred, dt)
error = (x_recovered - x0).abs().mean().item()
print(f"Recovered x0: error={error:.6f} (should be ~0 with perfect predictor)")

print()
print("=" * 60)
print("WHAT THIS MEANS FOR LOCALVIDEOSTUDIO")
print("=" * 60)
print("""
Current stack (AnimateDiff):
  model_index.json → "unet": UNet2DConditionModel
  Scheduler        → DDIMScheduler
  Steps needed     → 20-25 (with DDIM)
  Architecture     → CNN + temporal attention

Future upgrade (Wan 2.1):
  model_index.json → "transformer": WanTransformer3DModel  ← DiT!
  Scheduler        → FlowMatchEulerDiscreteScheduler       ← Flow Matching!
  Steps needed     → 10-20
  Architecture     → 3D Transformer (patches of video)

VRAM IMPACT (1.3B Wan model):
  Weights: 1.3B × 2 bytes = 2.6 GB (FP16)
  Peak:    ~4-5 GB with model_cpu_offload + vae_slicing
  → Should fit on your 8 GB RTX 5060!
""")

# =============================================================================
# EXPECTED OUTPUT:
# Input latent: [1, 4, 16, 16]
# After patchify: [1, 16, 64]
# After DiT block: [1, 16, 64]
# After unpatchify: [1, 4, 16, 16]
# Recovered x0: error=0.000000 (perfect predictor)
# =============================================================================
