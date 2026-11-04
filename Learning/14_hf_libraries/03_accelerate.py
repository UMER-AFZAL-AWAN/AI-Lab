# =============================================================================
# 14_hf_libraries / 03 — ACCELERATE LIBRARY
# =============================================================================
#
# LIKE YOU ARE 5:
# "accelerate" is the smart packer that decides which layer of a model
# goes on your GPU and which goes on CPU RAM — automatically.
# Without it, big models crash with "CUDA out of memory" because PyTorch
# just puts EVERYTHING on the GPU with no questions asked.
#
# C# ANALOGY:
# Like a dependency injection container that decides which services run
# in-process (fast, in VRAM) and which are out-of-process (slower, in RAM).
# device_map="auto" = IoC container.AutoResolve() — it figures out placement.
#
# =============================================================================

import torch
import torch.nn as nn
print("=" * 60)
print("WHAT ACCELERATE GIVES YOU")
print("=" * 60)
print("""
pip install accelerate

accelerate provides:
  1. device_map        — automatic layer-to-device placement
  2. init_empty_weights — load architecture without allocating VRAM
  3. load_checkpoint_and_dispatch — stream weights to the right device
  4. Hooks             — HookCaller that moves activations device-to-device
  5. Training utils    — mixed precision, gradient accumulation, multi-GPU
                         (not needed for inference; this file covers inference)

For diffusers models, you DON'T call accelerate directly.
diffusers wraps it in:
  pipe.enable_model_cpu_offload()       → uses accelerate under the hood
  pipe.enable_sequential_cpu_offload()  → uses accelerate under the hood
""")

print("=" * 60)
print("HOW device_map='auto' WORKS")
print("=" * 60)
print("""
# device_map is a dict: { layer_name → device }
# "auto" builds this dict for you, filling GPU first, spilling to CPU.

from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    device_map="auto",          # magic happens here
    torch_dtype=torch.float16,
)

# Check where each layer landed:
print(model.hf_device_map)
# → {
#     'model.embed_tokens':    0,     (GPU 0)
#     'model.layers.0':        0,     (GPU 0)
#     ...
#     'model.layers.20':       0,     (GPU 0 — VRAM fills up)
#     'model.layers.21':       'cpu', (overflows to CPU)
#     ...
#     'lm_head':               'cpu',
#   }

# When you call model(input_ids), accelerate's hooks move activations
# automatically as they cross device boundaries.
# You just call model() normally — the multi-device plumbing is hidden.
""")

print("=" * 60)
print("THE FOUR device_map VALUES")
print("=" * 60)
print("""
device_map="auto"
  Fill GPU(s) first, then CPU. Best for inference.

device_map="balanced"
  Distribute evenly across all GPUs. Good for multi-GPU training.

device_map="sequential"
  Fill GPU 0 → GPU 1 → CPU in that order. Deterministic placement.

device_map="cpu"
  Entire model on CPU. Very slow, but useful to test without GPU.

device_map={"layer.0": 0, "layer.1": "cpu"}
  Manual: you specify exactly where each part goes.
  Use when auto wastes VRAM on less-used layers.
""")

print("=" * 60)
print("THE LOW-LEVEL PATTERN: init_empty_weights + dispatch")
print("=" * 60)
print("""
# This is what from_pretrained does internally.
# You almost never need this directly — but knowing it demystifies OOM errors.

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM

# Step 1: Build the architecture with ZERO VRAM usage
config = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B")
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
# At this point: model exists as Python objects, but all weights are
# "meta tensors" — they have shape and dtype but NO memory allocated.
# torch.cuda.memory_allocated() is still 0!

# Step 2: Stream weights from disk to the right device
model = load_checkpoint_and_dispatch(
    model,
    checkpoint="path/to/model.safetensors",  # or a folder with shards
    device_map="auto",
    dtype=torch.float16,
)
# Now weights flow from disk → CPU → GPU one shard at a time.
# Peak RAM usage = one shard, not the entire model.
""")

print("=" * 60)
print("DIFFUSERS WRAPPERS: WHAT THEY DO UNDER THE HOOD")
print("=" * 60)
print("""
pipe.enable_model_cpu_offload()
  ↓
  accelerate installs CpuOffload hooks on each top-level component
  (unet, vae, text_encoder, motion_adapter).
  When component A finishes, it moves back to CPU.
  When component B starts, it moves to GPU.
  VRAM at any moment ≈ size of one component (usually the UNet = 2 GB).
  Total pipeline VRAM: ~2-3 GB instead of ~4-5 GB (all on GPU).

pipe.enable_sequential_cpu_offload()
  ↓
  accelerate installs hooks at the LAYER level, not the component level.
  Each nn.Module moves to GPU just before its forward() call,
  then immediately back to CPU.
  VRAM at any moment ≈ size of one layer (a few MB).
  Very slow (many small transfers), but minimum possible VRAM.
  Use when enable_model_cpu_offload() still OOMs.
""")

print("=" * 60)
print("DEMO: MANUAL CPU OFFLOAD WITH HOOKS (tiny model)")
print("=" * 60)

class TinyBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
    def forward(self, x):
        return torch.relu(self.fc(x))

class TinyPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TinyBlock(16)
        self.decoder = TinyBlock(16)

    def forward(self, x):
        h = self.encoder(x)
        return self.decoder(h)


device = "cuda" if torch.cuda.is_available() else "cpu"
pipe   = TinyPipeline()

# Without offload: everything stays on GPU
pipe.to(device)
x = torch.randn(1, 16).to(device)
print(f"Output shape: {pipe(x).shape}")

# With manual "offload" (illustrates the concept):
# In diffusers this is done by accelerate hooks, but conceptually:
def run_with_manual_offload(pipe, x, cpu, gpu):
    # encoder: move to GPU, run, move back to CPU
    pipe.encoder.to(gpu)
    h = pipe.encoder(x.to(gpu))
    pipe.encoder.to(cpu)

    # decoder: move to GPU, run, move back to CPU
    pipe.decoder.to(gpu)
    out = pipe.decoder(h)
    pipe.decoder.to(cpu)
    return out

cpu_device = torch.device("cpu")
gpu_device = torch.device(device)
pipe.to(cpu_device)  # start everything on CPU

x_cpu = torch.randn(1, 16)
out = run_with_manual_offload(pipe, x_cpu, cpu_device, gpu_device)
print(f"Manual offload output shape: {out.shape}")
print(f"Peak VRAM (tiny): {torch.cuda.max_memory_allocated()/1e6:.1f} MB"
      if torch.cuda.is_available() else "CUDA not available")

print()
print("=" * 60)
print("QUICK REFERENCE: WHICH HELPER TO USE")
print("=" * 60)
print("""
Situation                               → Use
──────────────────────────────────────────────────────────
Model fits in VRAM comfortably          → .to('cuda') directly
Model is tight (AnimateDiff on 8 GB)    → enable_model_cpu_offload()
Model almost fits but OOMs sometimes    → enable_model_cpu_offload()
                                           + enable_vae_slicing()
Model is too big (Wan 14B, Llama 70B)   → enable_sequential_cpu_offload()
                                           (slow but works)
Transformers LLM too large              → device_map="auto" in from_pretrained
Multi-GPU setup                         → device_map="balanced"
""")

# =============================================================================
# EXPECTED OUTPUT:
# Output shape: torch.Size([1, 16])
# Manual offload output shape: torch.Size([1, 16])
# Peak VRAM (tiny): ~0.1 MB
# =============================================================================
