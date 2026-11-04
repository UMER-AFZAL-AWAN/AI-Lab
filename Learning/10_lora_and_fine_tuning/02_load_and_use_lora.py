# =============================================================================
# 10_lora_and_fine_tuning / 02 — LOADING AND USING A LORA
# =============================================================================
#
# LIKE YOU ARE 5:
# You have a base model (AnimateDiff / Wan).
# Someone trained a LoRA that makes it draw in a studio-ghibli style.
# You snap the LoRA onto your model, run generation, snap it off.
# The base model never changes — you can swap LoRAs freely.
#
# C# ANALOGY:
# Like loading a plugin at runtime.
# pipe.load_lora_weights() = Assembly.LoadFrom("my_lora.dll")
# pipe.set_adapters(weight=0.8) = plugin.Scale(0.8)
# pipe.unload_lora_weights() = AppDomain.Unload(plugin)
#
# =============================================================================

import torch
print("=" * 60)
print("PATTERN 1: LOAD A LORA FROM HUGGING FACE")
print("=" * 60)
print("""
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
pipe = AnimateDiffPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    motion_adapter=adapter,
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

# ─── Load a LoRA from HuggingFace ───────────────────────────────────────────
# Loads the LoRA weights into the UNet's attention layers.
# Does NOT change the original weights — adds the delta on top.
pipe.load_lora_weights(
    "CiroN2022/toy-face",    # HF repo containing a .safetensors LoRA
    weight_name="toy_face_sdxl.safetensors",  # specific file (optional if only one)
    adapter_name="toy_face",  # name you give this LoRA (for multi-LoRA use)
)

# Generate
out = pipe(
    prompt="a cat playing piano, toy face style",
    negative_prompt="blurry, ugly",
    num_frames=16,
    num_inference_steps=25,
    guidance_scale=7.5,
    generator=torch.Generator().manual_seed(42),
)
frames = out.frames[0]   # list of 16 PIL Images

# ─── Adjust strength without reloading ──────────────────────────────────────
pipe.set_adapters(["toy_face"], adapter_weights=[0.6])   # 0.0 = off, 1.0 = full
# Generate again with 60% strength...

# ─── Unload when done ───────────────────────────────────────────────────────
pipe.unload_lora_weights()
# UNet is now exactly as it was before load_lora_weights().
""")

print("=" * 60)
print("PATTERN 2: LOAD FROM A LOCAL FILE")
print("=" * 60)
print("""
# Downloaded from civitai.com or trained yourself
pipe.load_lora_weights(
    "C:/models/loras/",          # folder path
    weight_name="ghibli_v2.safetensors",
    adapter_name="ghibli",
)

# or just a full path to the file:
pipe.load_lora_weights("C:/models/loras/ghibli_v2.safetensors")
""")

print("=" * 60)
print("PATTERN 3: MULTIPLE LORAS AT ONCE (composition)")
print("=" * 60)
print("""
# Load two LoRAs and blend them
pipe.load_lora_weights("path/lora_style.safetensors",   adapter_name="style")
pipe.load_lora_weights("path/lora_lighting.safetensors", adapter_name="lighting")

# Apply both — style at 80%, lighting at 50%
pipe.set_adapters(["style", "lighting"], adapter_weights=[0.8, 0.5])

out = pipe(prompt="...", ...)

# They are ADDITIVE: final delta = 0.8*(A_style@B_style) + 0.5*(A_lighting@B_lighting)
""")

print("=" * 60)
print("PATTERN 4: FUSE LORA INTO WEIGHTS (for speed)")
print("=" * 60)
print("""
# After loading, you can bake the LoRA into W permanently:
pipe.fuse_lora(lora_scale=1.0)
# Now W += scale * (A @ B) is computed once and stored in W.
# Slightly faster at inference (one less matmul per layer).
# BUT: you can no longer unload the LoRA or change its scale.

# Unfuse (reverts the bake, but only if the original W was saved — diffusers does this):
pipe.unfuse_lora()
""")

print("=" * 60)
print("WHAT LORA WEIGHT (SCALE) TO USE")
print("=" * 60)

print("""
Scale 0.0  → LoRA has no effect (same as not loading it)
Scale 0.5  → subtle influence; good starting point for style LoRAs
Scale 0.8  → strong influence; typical for concept/character LoRAs
Scale 1.0  → full strength; can overpower the prompt at high CFG
Scale 1.2+ → may cause artifacts, saturated colours, or anatomical issues

Rule: start at 0.7, adjust in 0.1 steps while watching quality.
If the style is too strong (colours saturated, faces warped) → lower the scale.
If the style doesn't show → raise the scale OR lower the CFG (guidance_scale).
""")

print("=" * 60)
print("DEMO: SIMULATING LORA FORWARD PASS")
print("=" * 60)

# Simulate what diffusers does internally when a LoRA is loaded
import torch.nn as nn
import math

class SimulatedLoRAUNetLayer(nn.Module):
    """
    Simulates one attention projection layer with a LoRA attached.
    diffusers installs LoRA as a hook/wrapper on existing nn.Linear layers.
    """
    def __init__(self, dim, rank=16, alpha=16):
        super().__init__()
        self.base  = nn.Linear(dim, dim, bias=False)
        self.lora_A = nn.Linear(dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.lora_B.weight)   # LoRA starts as identity (zero delta)
        self.scale = alpha / rank
        self._lora_active = True

    def forward(self, x, lora_weight=1.0):
        base_out = self.base(x)
        if self._lora_active:
            lora_out = self.lora_B(self.lora_A(x))
            return base_out + lora_weight * self.scale * lora_out
        return base_out


dim   = 64
layer = SimulatedLoRAUNetLayer(dim, rank=8)
x     = torch.randn(1, 10, dim)

with torch.no_grad():
    out_no_lora = layer(x, lora_weight=0.0)
    out_full    = layer(x, lora_weight=1.0)
    out_half    = layer(x, lora_weight=0.5)

# Initially B=0, so all outputs should be identical (zero delta)
diff_full = (out_full - out_no_lora).abs().max().item()
print(f"Freshly initialised LoRA delta (should be 0): {diff_full:.6f}")
print(f"Output shape: {list(out_full.shape)}")

# Simulate "trained LoRA" by giving B non-zero values
with torch.no_grad():
    layer.lora_B.weight.fill_(0.01)
    out_trained_full = layer(x, lora_weight=1.0)
    out_trained_half = layer(x, lora_weight=0.5)

diff_trained = (out_trained_full - out_no_lora).abs().mean().item()
diff_half_v  = (out_trained_half - out_no_lora).abs().mean().item()
print(f"After training LoRA — mean delta at scale=1.0: {diff_trained:.4f}")
print(f"After training LoRA — mean delta at scale=0.5: {diff_half_v:.4f}")
print(f"Half-scale is {diff_trained/diff_half_v:.1f}x smaller (linear scaling confirmed)")

print()
print("=" * 60)
print("WHERE TO FIND LORAS FOR YOUR MODELS")
print("=" * 60)
print("""
For SD 1.5 / AnimateDiff base:
  civitai.com → filter by "SD 1.5", type "LoRA"
  huggingface.co/models?other=lora&pipeline_tag=text-to-image

For Wan2.1:
  huggingface.co → search "wan2.1 lora" (newer, fewer available)

WHAT TO CHECK before downloading:
  1. Base model match: "SD 1.5" LoRA won't work on Wan
  2. File format: .safetensors preferred over .pt (security)
  3. Rank: listed in model card; rank=16 is most common
  4. Trigger word: some LoRAs require a specific word in the prompt
     e.g. "ohwx man" for a face LoRA

SAFETY:
  .safetensors LoRAs are safe (no code execution).
  .pt/.ckpt LoRAs can contain arbitrary Python — only use from trusted sources.
""")

# =============================================================================
# EXPECTED OUTPUT:
# Freshly initialised LoRA delta (should be 0): 0.000000
# Output shape: [1, 10, 64]
# After training LoRA — mean delta at scale=1.0: ~0.0064
# After training LoRA — mean delta at scale=0.5: ~0.0032
# Half-scale is 2.0x smaller (linear scaling confirmed)
# =============================================================================
