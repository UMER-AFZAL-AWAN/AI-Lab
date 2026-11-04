# =============================================================================
# 01 — ARCHITECTURE vs WEIGHTS vs PIPELINE vs FORMAT
# =============================================================================
#
# LIKE YOU ARE 5:
# ARCHITECTURE = the blueprint of a building (walls, windows, layout).
# WEIGHTS      = the furniture inside the building (learned, specific).
# PIPELINE     = how you use the building (walk in, go upstairs, turn left).
# FRAMEWORK    = the construction tools (hammer, crane, scaffolding).
# FORMAT       = the box the furniture was delivered in (.safetensors, .gguf).
#
# These are FOUR DIFFERENT THINGS that people often confuse.
#
# EXAMPLE:
# Architecture: "Transformer decoder with 32 layers, embed_dim=4096"
# Weights:      the 7 billion parameter values, trained on web data
# Framework:    PyTorch (runs the architecture on GPU)
# Pipeline:     HuggingFace transformers pipeline("text-generation", model="llama3")
#               (tokenises input, calls model, decodes output, loops until done)
# Format:       the .safetensors file on your disk
#
# Same ARCHITECTURE, DIFFERENT WEIGHTS:
# "Llama 3 8B" architecture + base training weights = Llama 3 8B
# Same architecture + fine-tuned on code = CodeLlama
# Same architecture + fine-tuned on instructions = Llama 3 8B Instruct
#
# =============================================================================

import torch
import torch.nn as nn

print("=" * 60)
print("THE SAME ARCHITECTURE, DIFFERENT WEIGHTS")
print("=" * 60)

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=100, embed_dim=16, num_heads=2, num_layers=2):
        super().__init__()
        self.emb    = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=64,
                                        batch_first=True, norm_first=True)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        for b in self.blocks: x = b(x)
        return self.head(x[:, -1])  # predict next token from last position

# Architecture: same class definition
arch_v1 = TinyTransformer()   # random weights (like before training)
arch_v2 = TinyTransformer()   # random weights (a "different model" with same arch)

# Give v2 completely different weights
for p in arch_v2.parameters():
    nn.init.constant_(p, 0.001)   # all weights = 0.001

ids = torch.tensor([[1, 2, 3, 4, 5]])
out_v1 = arch_v1(ids)
out_v2 = arch_v2(ids)
print("Same architecture, different weights:")
print(f"  v1 output (random weights): {out_v1.detach().numpy().round(3)[:5]}")
print(f"  v2 output (0.001 weights):  {out_v2.detach().numpy().round(3)[:5]}")
print("Different outputs — the weights ARE the learned knowledge.")

print()
print("=" * 60)
print("FRAMEWORK vs LIBRARY vs RUNTIME")
print("=" * 60)
print("""
FRAMEWORK (the foundation):
  PyTorch          — defines tensors, autograd, nn.Module. The base of everything.
  TensorFlow/Keras — Google's alternative framework.

LIBRARY (built on top of the framework):
  HuggingFace Transformers — 1000+ pretrained models. Easy load/use/fine-tune.
  HuggingFace Diffusers    — diffusion pipelines (Stable Diffusion, AnimateDiff).
  timm                     — vision models (ResNets, ViTs, etc.)
  bitsandbytes             — quantisation (INT4/INT8) on top of PyTorch.

RUNTIME (can run models WITHOUT Python):
  ONNX Runtime             — runs .onnx models in C++, C#, Java, etc.
  llama.cpp                — runs .gguf LLMs efficiently on CPU.
  TensorRT                 — NVIDIA's optimised runtime for GPU inference.
  OpenVINO                 — Intel's runtime for CPU/NPU.

FORMAT (the saved file):
  .safetensors, .pt, .onnx, .gguf  — see 08/05_checkpoint_formats.py
""")

print("=" * 60)
print("THE ANIMATEDIFF STACK (what runs when you generate a video)")
print("=" * 60)
print("""
Your call in studio_pipeline.py:
  pipeline(prompt=..., num_frames=16)

Layer 1: Python call
  pipeline.__call__() in diffusers/pipelines/animatediff/pipeline_animatediff.py

Layer 2: HuggingFace Diffusers pipeline logic
  - Encodes prompt with CLIP tokeniser + CLIPTextModel
  - Runs DDIM scheduler loop (50 steps)
  - Calls UNet at each step with text embedding and noisy latent
  - Calls VAE decode at the end

Layer 3: PyTorch nn.Module
  - UNet2DConditionMotionModel.forward()
  - AutoencoderKL.decode()
  - CLIPTextModel.forward()

Layer 4: PyTorch CUDA kernel dispatch
  - nn.Conv2d  -> cuDNN convolution kernel
  - nn.Linear  -> cuBLAS matrix multiply
  - nn.MultiheadAttention -> flash attention (if available)

Layer 5: CUDA driver
  - Schedules kernel execution on RTX 5060 GPU
  - Copies tensors between CPU RAM and GPU VRAM

Layer 6: Hardware
  - RTX 5060 Tensor Cores execute the matrix multiplications
""")

print("=" * 60)
print("WHERE YOUR CODE SITS IN THE STACK")
print("=" * 60)
print("""
C:\\AI-Lab\\LocalVideoStudio\\Inference\\pipeline\\studio_pipeline.py
  ← YOUR CODE (top of the stack)
  Uses: diffusers AnimateDiff pipeline

diffusers (HuggingFace)
  Uses: PyTorch nn.Module

PyTorch
  Uses: CUDA kernels, cuDNN, cuBLAS

CUDA / GPU Driver
  Uses: RTX 5060 hardware

Each layer hides complexity. You call pipeline(prompt="...") and
50 layers of optimised math happen underneath.
""")

# =============================================================================
# EXPECTED OUTPUT:
# Different outputs — the weights ARE the learned knowledge.
# (The rest is printed documentation)
# =============================================================================
