# =============================================================================
# 14_hf_libraries / 01 — DIFFUSERS LIBRARY
# =============================================================================
#
# LIKE YOU ARE 5:
# "diffusers" is a LEGO kit that includes pre-built versions of every popular
# diffusion model. You snap the pieces together.
# The main pieces are: Pipeline, Scheduler, UNet (or Transformer), VAE, text encoder.
#
# C# ANALOGY:
# Like a NuGet package (Microsoft.ML) that ships pre-built ML pipelines.
# DiffusionPipeline = your top-level orchestrator.
# Each component = an injectable dependency (think IScheduler, IEncoder).
# Swapping the scheduler = calling services.Replace<IScheduler, DDIMScheduler>().
#
# =============================================================================

import torch
print("=" * 60)
print("WHAT IS IN THE DIFFUSERS PACKAGE")
print("=" * 60)
print("""
pip install diffusers accelerate transformers

diffusers contains:
  PIPELINES     → orchestrate everything end-to-end
  MODELS        → individual neural nets (UNet, VAE, DiT, ...)
  SCHEDULERS    → noise schedule + denoising step math
  UTILS         → image I/O, callbacks, memory helpers

── Pipelines ──────────────────────────────────────────
  DiffusionPipeline          ← auto-detects class from model_index.json
  StableDiffusionPipeline    ← SD 1.x/2.x, text-to-image
  StableDiffusionXLPipeline  ← SDXL
  AnimateDiffPipeline        ← text-to-video (you've used this)
  WanPipeline                ← Wan2.1 (added in diffusers 0.33+)
  CogVideoXPipeline          ← CogVideoX

── Models ─────────────────────────────────────────────
  UNet2DConditionModel       ← the denoiser for SD/AnimateDiff
  AutoencoderKL              ← VAE encoder/decoder
  WanTransformer3DModel      ← DiT-based denoiser for Wan
  MotionAdapter              ← AnimateDiff temporal motion module

── Schedulers ─────────────────────────────────────────
  DDIMScheduler              ← deterministic, 20–50 steps
  DPMSolverMultistepScheduler ← fast, 15–25 steps, good quality
  UniPCMultistepScheduler    ← very fast, 10–20 steps
  EulerAncestralDiscreteScheduler ← stochastic, more varied outputs
  FlowMatchEulerDiscreteScheduler ← for Flow Matching models (Wan, SD3)
""")

print("=" * 60)
print("THE PIPELINE CALL PATTERN")
print("=" * 60)
print("""
# Every diffusers pipeline follows the same call signature:
output = pipe(
    prompt               = "a cat playing piano",   # text guidance
    negative_prompt      = "blurry, ugly",           # what to avoid
    num_inference_steps  = 20,                       # denoising steps
    guidance_scale       = 7.5,                      # CFG strength
    # --- video-specific ---
    num_frames           = 16,
    # --- output size ---
    height               = 512,
    width                = 512,
    # --- reproducibility ---
    generator            = torch.Generator().manual_seed(42),
)
frames = output.frames[0]   # for video pipelines
image  = output.images[0]   # for image pipelines

# The pipeline automatically:
#   1. Encodes the prompt with the text encoder
#   2. Generates a noisy latent
#   3. Loops N denoising steps (the scheduler controls each step)
#   4. Decodes the final latent with the VAE
""")

print("=" * 60)
print("SWAPPING THE SCHEDULER (the most useful trick)")
print("=" * 60)
print("""
# Schedulers are INTERCHANGEABLE. You can load a model with its default
# scheduler and swap it for a faster one.

from diffusers import AnimateDiffPipeline, DPMSolverMultistepScheduler

pipe = AnimateDiffPipeline.from_pretrained("...", torch_dtype=torch.float16)

# Default scheduler (from model_index.json) — may be DDIM (50 steps)
print(type(pipe.scheduler).__name__)   # e.g. DDIMScheduler

# Swap to DPM++ which needs only 15-20 steps for comparable quality
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,            # keeps all other settings the same
    algorithm_type="dpmsolver++",
)
print(type(pipe.scheduler).__name__)   # DPMSolverMultistepScheduler

# Now generate with fewer steps:
out = pipe(prompt="...", num_inference_steps=15)   # was 50 before
# Same quality, 3x faster.
""")

print("=" * 60)
print("ACCESSING INDIVIDUAL COMPONENTS")
print("=" * 60)
print("""
# A pipeline is just a container. You can access each component:

pipe.unet           # the denoising UNet (or transformer for DiT models)
pipe.vae            # AutoencoderKL
pipe.text_encoder   # CLIPTextModel or T5
pipe.tokenizer      # CLIPTokenizer or T5Tokenizer
pipe.scheduler      # the active scheduler

# Example: check UNet parameter count
unet_params = sum(p.numel() for p in pipe.unet.parameters())
print(f"UNet params: {unet_params/1e6:.0f}M")

# Example: encode text manually (what the pipeline does internally)
tokens = pipe.tokenizer(
    "a cat playing piano",
    padding="max_length",
    max_length=77,
    return_tensors="pt"
)
with torch.no_grad():
    text_embeds = pipe.text_encoder(tokens.input_ids)[0]
print(f"Text embedding shape: {text_embeds.shape}")  # (1, 77, 768)
""")

print("=" * 60)
print("MEMORY HELPERS (critical for 8 GB VRAM)")
print("=" * 60)
print("""
# From your LocalVideoStudio code — these are diffusers built-ins:

pipe.enable_model_cpu_offload()
# Moves each component to GPU only when needed, then back to CPU.
# Slower than keeping everything on GPU, but uses ~1-2 GB VRAM instead of full model.

pipe.enable_sequential_cpu_offload()
# Even more aggressive: offloads LAYER by LAYER.
# Very slow, but minimum possible VRAM. Good for exploring huge models.

pipe.enable_vae_slicing()
# Processes the VAE decode one frame at a time instead of all at once.
# Halves the VAE VRAM spike during decode. Almost no speed penalty.

pipe.enable_vae_tiling()
# Splits large images into tiles for VAE decode. For high-resolution images.

# RULE: Always use these on 8 GB VRAM:
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()
""")

print("=" * 60)
print("WHAT CONNECT TO LOCALVIDEOSTUDIO")
print("=" * 60)
print("""
In your LocalVideoStudio code you use:
  AnimateDiffPipeline.from_pretrained(...)   → diffusers.AnimateDiffPipeline
  pipe.enable_model_cpu_offload()            → diffusers built-in
  pipe.enable_vae_slicing()                  → diffusers built-in
  DDIMScheduler                              → diffusers.DDIMScheduler
  output.frames[0]                           → pipeline output format

Future upgrade path:
  AnimateDiff UNet → Wan2.1 Transformer (DiT-based, better quality)
  DDIMScheduler → FlowMatchEulerDiscreteScheduler (for Wan)
  → see 04_diffusion/07_dit_and_flow_matching.py for why
""")

# =============================================================================
# EXPECTED OUTPUT: (all printed documentation — no tensor ops)
# =============================================================================
