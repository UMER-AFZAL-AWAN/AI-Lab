# =============================================================================
# 04 — LOAD AND RUN: FROM_PRETRAINED PATTERNS
# =============================================================================
#
# LIKE YOU ARE 5:
# "from_pretrained" is a universal remote-control that works for most models.
# But sometimes the model needs its own special remote (custom code).
# This file teaches you which remote to use and when.
#
# =============================================================================

import torch
import torch.nn as nn
print("=" * 60)
print("PATTERN 1: STANDARD DIFFUSION PIPELINE (AnimateDiff, SD, SDXL)")
print("=" * 60)
print("""
# When model_index.json uses only standard diffusers classes:

from diffusers import DiffusionPipeline, AnimateDiffPipeline
import torch

# Simple form — detects pipeline class from model_index.json automatically
pipe = DiffusionPipeline.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-2",
    torch_dtype=torch.float16,
)

# Or with explicit class (better: you know what you're getting)
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
adapter   = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5",
                                           subfolder="scheduler",
                                           clip_sample=False, timestep_spacing="linspace",
                                           beta_schedule="linear", steps_offset=1)
pipe = AnimateDiffPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    motion_adapter=adapter,
    scheduler=scheduler,
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

out    = pipe(prompt="a cat playing piano", num_frames=16, num_inference_steps=25)
frames = out.frames[0]   # list of 16 PIL Images
""")

print("=" * 60)
print("PATTERN 2: NEWER MODEL, NEEDS RECENT DIFFUSERS")
print("=" * 60)
print("""
# Wan2.1, CogVideoX, Mochi etc. were added in diffusers >= 0.31.
# Symptom: ImportError: cannot import 'WanPipeline' from 'diffusers'
# Fix: pip install --upgrade diffusers

pip install --upgrade diffusers   # or: pip install diffusers>=0.33.0

from diffusers import WanPipeline
import torch

pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B",
    torch_dtype=torch.bfloat16,   # Wan uses bfloat16
)
pipe.enable_model_cpu_offload()

out    = pipe(prompt="a panda eating bamboo", num_frames=16, num_inference_steps=20)
frames = out.frames[0]
""")

print("=" * 60)
print("PATTERN 3: CUSTOM CODE (trust_remote_code)")
print("=" * 60)
print("""
# When the model_index.json points to classes in repo .py files:
#   "transformer": [".", "CustomDiTModel"]   <- '.' means 'in this repo'

pipe = DiffusionPipeline.from_pretrained(
    "owner/custom-model",
    trust_remote_code=True,   # downloads and runs their .py files
    torch_dtype=torch.float16,
)

# BEFORE RUNNING: review their code at:
#   https://huggingface.co/owner/custom-model/tree/main
# Look at modeling_*.py and pipeline_*.py for suspicious calls.
""")

print("=" * 60)
print("PATTERN 4: WEIGHTS ONLY, MANUAL LOAD (no pipeline)")
print("=" * 60)
print("""
# Some models ship only model.safetensors + config.json — no model_index.json.
# You load the architecture class directly.

from diffusers import UNet2DConditionModel
from safetensors.torch import load_file

# Method A: from_pretrained with the right class
unet = UNet2DConditionModel.from_pretrained(
    "owner/model",
    torch_dtype=torch.float16,
)

# Method B: build architecture from config, load weights manually
import json

with open("config.json") as f:
    cfg = json.load(f)
# Build the model from config params (each class has from_config classmethod)
unet = UNet2DConditionModel(**{k: v for k, v in cfg.items() if not k.startswith('_')})
state = load_file("model.safetensors")
unet.load_state_dict(state, strict=False)   # strict=False if keys differ slightly
""")

print("=" * 60)
print("device_map: AUTOMATIC LAYER PLACEMENT")
print("=" * 60)
print("""
# device_map distributes model layers across GPU and CPU automatically.
# Requires: pip install accelerate

from transformers import AutoModelForCausalLM

# 'auto': fills GPU first, spills overflow to CPU
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    device_map="auto",
    torch_dtype=torch.float16,
)

# 'balanced': distributes evenly across all GPUs (multi-GPU)
# 'sequential': fills GPU 0 first, then GPU 1, etc.
# 'cpu': loads entire model on CPU (inference only, very slow)
# A dict: manual mapping {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 'cpu'}

# After loading, check where everything ended up:
print(model.hf_device_map)
# -> {'model.embed_tokens': 0, 'model.layers.0': 0, ..., 'model.layers.28': 'cpu', ...}

# IMPORTANT: works with transformers models. For diffusers use enable_model_cpu_offload() instead.
""")

print("=" * 60)
print("WHEN GENERIC PIPELINE FAILS: DECISION TREE")
print("=" * 60)
print("""
DiffusionPipeline.from_pretrained("owner/model") fails with...

ImportError: "Cannot find class X"
  → Check diffusers version: pip show diffusers
  → pip install --upgrade diffusers
  → If still fails: use trust_remote_code=True after reviewing their .py files

AttributeError: "object has no attribute Y"
  → Usually a version mismatch; upgrade diffusers or transformers

RuntimeError: "size mismatch for layer Z"
  → Wrong loader class for these weights; read config.json to find the right class

CUDA out of memory
  → Model too large; see 03_choosing_a_checkpoint.py
  → Try: pipe.enable_sequential_cpu_offload() (slower but lower VRAM peak)
  → Or: load in INT8 with load_in_8bit=True (transformers models only)

ValueError: "trust_remote_code required"
  → Review their repo .py files, then add trust_remote_code=True
""")

# =============================================================================
# EXPECTED OUTPUT:
# (Entirely printed documentation — no tensor ops in this file)
# This file is a reference card, not a demo. Run it to print the patterns.
# =============================================================================
