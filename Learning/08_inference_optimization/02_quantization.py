# =============================================================================
# 02 — QUANTISATION: CRAMMING MORE INTO LESS MEMORY
# =============================================================================
#
# LIKE YOU ARE 5:
# Imagine a list of prices: 1.99, 3.49, 2.75, 5.00, ...
# You could store each price in 32 bits (very precise).
# Or you could say "the cheapest is 1.99 and the most expensive is 5.00,
# let me just store a number from 0-255 that represents where in that range it falls."
# 1.99 = 0, 3.00 ≈ 128, 5.00 = 255.
# That's 8 bits (1 byte) instead of 32 bits (4 bytes). 4x smaller.
# To get back the original: price = 1.99 + (n / 255) * (5.00 - 1.99)
# A tiny bit of accuracy lost, but 4x less memory.
#
# QUANTISATION for neural networks: store weights as INT8 or INT4 instead of FP16.
# Compute: dequantise on the fly, multiply, re-quantise.
# Modern GPUs have INT8 matrix multiply instructions (much faster than FP16).
#
# BITSANDBYTES: the library in your .venv that enables 4-bit and 8-bit quantisation.
# Used for running Llama-class LLMs on 8GB VRAM.
#
# =============================================================================

import torch
import torch.nn as nn
import numpy as np

print("=" * 60)
print("MANUAL QUANTISATION (8-bit)")
print("=" * 60)

def quantise_to_int8(tensor):
    """Quantise a float tensor to int8 and return (quantised, scale, zero_point)."""
    tmin = tensor.min().item()
    tmax = tensor.max().item()
    # Map [tmin, tmax] to [0, 255]
    scale      = (tmax - tmin) / 255.0
    zero_point = -round(tmin / scale)   # offset so tmin maps to 0
    quantised  = torch.clamp(torch.round(tensor / scale + zero_point), 0, 255).to(torch.uint8)
    return quantised, scale, zero_point

def dequantise_from_int8(quantised, scale, zero_point):
    """Reverse: int8 -> float."""
    return (quantised.float() - zero_point) * scale

# A weight tensor (like a Linear layer's weights)
weights = torch.randn(4, 4)
print(f"Original weights:\n{weights.round(decimals=3)}")

q, scale, zp = quantise_to_int8(weights)
print(f"\nQuantised (uint8, range 0-255):\n{q}")
print(f"Scale: {scale:.6f},  Zero point: {zp}")

dq = dequantise_from_int8(q, scale, zp)
print(f"\nDequantised (approximate original):\n{dq.round(decimals=3)}")

error = (weights - dq).abs()
print(f"\nMax error:  {error.max().item():.6f}")
print(f"Mean error: {error.mean().item():.6f}")
print("(Small error — quantisation is lossy but the error is usually acceptable)")

print()
print("=" * 60)
print("MEMORY SAVING")
print("=" * 60)

size = (1024, 1024)
t_fp32  = torch.randn(*size, dtype=torch.float32)
t_fp16  = t_fp32.half()
t_int8  = torch.randint(0, 255, size, dtype=torch.uint8)   # quantised weights
t_int4  = torch.randint(0, 15,  size, dtype=torch.uint8)   # 4-bit (packed 2 per byte)

mb = lambda t: t.element_size() * t.nelement() / 1024**2
print(f"1024×1024 tensor:")
print(f"  FP32:  {mb(t_fp32):.2f} MB")
print(f"  FP16:  {mb(t_fp16):.2f} MB")
print(f"  INT8:  {mb(t_int8):.2f} MB  (4x smaller than FP32)")
print(f"  INT4:  {mb(t_int4)/2:.2f} MB  (8x smaller than FP32, packed)")

print()
print("=" * 60)
print("USING BITSANDBYTES FOR REAL QUANTISATION")
print("=" * 60)

try:
    import bitsandbytes as bnb

    # 8-bit linear layer (drop-in replacement for nn.Linear)
    linear_fp16 = nn.Linear(512, 256, bias=False, dtype=torch.float16)
    linear_int8 = bnb.nn.Linear8bitLt(512, 256, has_fp16_weights=False, threshold=6.0)

    params_fp16 = sum(p.numel() * p.element_size() for p in linear_fp16.parameters())
    params_int8 = sum(p.numel() * p.element_size() for p in linear_int8.parameters())
    print(f"nn.Linear FP16:          {params_fp16 / 1024:.1f} KB")
    print(f"bnb.Linear8bitLt INT8:   {params_int8 / 1024:.1f} KB  (half the memory)")
    print("(In practice the INT8 layer needs weights loaded and quantised from a checkpoint)")

except ImportError:
    print("(bitsandbytes not fully usable here — CUDA required for INT8 ops)")
    print("How to use it in your code:")
    print()
    print("  import bitsandbytes as bnb")
    print("  # Replace nn.Linear with bnb.nn.Linear8bitLt in model definition")
    print("  linear = bnb.nn.Linear8bitLt(512, 256, has_fp16_weights=False)")
    print()
    print("  # Or load a Llama model in 4-bit automatically with HuggingFace:")
    print("  from transformers import BitsAndBytesConfig, AutoModelForCausalLM")
    print("  config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4')")
    print("  model  = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B',")
    print("                                                  quantization_config=config)")

print()
print("=" * 60)
print("WHICH QUANTISATION TO USE")
print("=" * 60)
print("""
Scenario                       Recommended         VRAM save
─────────────────────────────────────────────────────────────
AnimateDiff generation          FP16 (default)       2x vs FP32
LLM inference (Llama 7-13B)     INT4 (bitsandbytes)  8x vs FP32
LLM inference (quality first)   INT8 (bitsandbytes)  4x vs FP32
LLM fine-tuning (LoRA)          BF16 + INT8 base     4x vs FP32
Production deployment           ONNX + INT8 runtime  4x vs FP32
""")
print("Your RTX 5060 with 8GB: INT4 for LLMs, FP16 for image/video models.")

# =============================================================================
# EXPECTED OUTPUT:
# Max error:  ~0.01  (small quantisation error)
# INT8:  1.00 MB  (4x smaller than FP32)
# =============================================================================
