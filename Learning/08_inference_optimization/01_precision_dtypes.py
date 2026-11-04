# =============================================================================
# 01 — PRECISION AND DTYPES: THE VRAM BUDGET
# =============================================================================
#
# LIKE YOU ARE 5:
# Numbers in a computer need storage.
# A very precise number (like 3.14159265) needs more storage than a rough one (3.1).
# AI models use different precisions depending on what they need:
#
# FP32 (float32): 32 bits = 4 bytes.  VERY precise. Standard for training.
# FP16 (float16): 16 bits = 2 bytes.  Half the memory. Fast on modern GPUs.
# BF16 (bfloat16): 16 bits = 2 bytes. Same exponent range as FP32, less mantissa.
#                  Better for training than FP16 (doesn't overflow as easily).
# INT8:            8 bits = 1 byte.   For quantised inference (see 02_quantization.py).
# INT4:            4 bits = 0.5 bytes. Very aggressive quantisation.
#
# VRAM RULE (from ChatGPT's message):
# FP32:  4 GB per 1B parameters
# FP16:  2 GB per 1B parameters
# INT8:  1 GB per 1B parameters
# INT4: 0.5 GB per 1B parameters
#
# YOUR RTX 5060: 8GB VRAM
# - AnimateDiff: ~2.5B parameters total -> ~5GB at FP16 -> fits!
# - Llama 3 8B at INT4: 8B × 0.5 GB/B = 4GB -> fits!
# - Llama 3 8B at FP16: 8B × 2 GB/B = 16GB -> does NOT fit!
#
# =============================================================================

import torch
import numpy as np

print("=" * 60)
print("PYTORCH DTYPES")
print("=" * 60)

dtypes = {
    "float32 (FP32)": torch.float32,
    "float16 (FP16)": torch.float16,
    "bfloat16 (BF16)": torch.bfloat16,
    "int8 (INT8)":   torch.int8,
    "int4 (INT4)":   None,   # no native int4 in standard PyTorch
}

print(f"{'dtype':<20}  {'bytes':>6}  {'min':>12}  {'max':>12}")
print("-" * 55)
for name, dt in dtypes.items():
    if dt is None:
        print(f"{name:<20}  {'0.5':>6}  (packed as int8 by bitsandbytes)")
        continue
    t  = torch.tensor(1.0, dtype=dt) if dt.is_floating_point else torch.tensor(1, dtype=dt)
    nb = t.element_size()
    if dt.is_floating_point:
        info = torch.finfo(dt)
        print(f"{name:<20}  {nb:>6}  {info.min:>12.2e}  {info.max:>12.2e}")
    else:
        info = torch.iinfo(dt)
        print(f"{name:<20}  {nb:>6}  {info.min:>12}  {info.max:>12}")

print()
print("=" * 60)
print("VRAM CALCULATION")
print("=" * 60)

def vram_gb(num_params_B, dtype="fp16"):
    """Estimate VRAM needed for a model with num_params_B billion parameters."""
    bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}
    return num_params_B * bytes_per_param[dtype]

your_vram = 8   # GB (RTX 5060)

models = [
    ("AnimateDiff UNet",   2.5),
    ("Llama 3 8B",         8.0),
    ("Llama 3 70B",       70.0),
    ("GPT-2 Small",        0.117),
    ("CLIP ViT-L/14",      0.428),
]

print(f"{'Model':<22}  {'Params (B)':>10}  {'FP32 (GB)':>10}  {'FP16 (GB)':>10}  {'INT8 (GB)':>10}  {'INT4 (GB)':>10}  {'Fits @8GB FP16?':>15}")
print("-" * 95)
for name, params in models:
    fp32 = vram_gb(params, "fp32")
    fp16 = vram_gb(params, "fp16")
    int8 = vram_gb(params, "int8")
    int4 = vram_gb(params, "int4")
    fits = "✓" if fp16 <= your_vram else "✗ (use INT4)"
    print(f"{name:<22}  {params:>10.3f}  {fp32:>10.1f}  {fp16:>10.1f}  {int8:>10.1f}  {int4:>10.1f}  {fits:>15}")

print()
print("=" * 60)
print("CREATING TENSORS IN DIFFERENT DTYPES")
print("=" * 60)

x_fp32 = torch.randn(1000, 1000, dtype=torch.float32)
x_fp16 = x_fp32.half()   # .half() = convert to float16
x_bf16 = x_fp32.to(torch.bfloat16)

mem_fp32 = x_fp32.element_size() * x_fp32.nelement() / 1024**2
mem_fp16 = x_fp16.element_size() * x_fp16.nelement() / 1024**2
print(f"FP32 tensor (1000x1000): {mem_fp32:.2f} MB")
print(f"FP16 tensor (1000x1000): {mem_fp16:.2f} MB  (half the memory)")
print(f"BF16 tensor (1000x1000): {x_bf16.element_size() * x_bf16.nelement() / 1024**2:.2f} MB")

print()
print("=" * 60)
print("FP16 vs BF16: WHICH TO USE AND WHEN")
print("=" * 60)

# FP16 has limited exponent range -> can overflow (become inf) or underflow (become 0)
# BF16 has same exponent range as FP32 -> no overflow
# FP16 has more mantissa bits -> slightly more precise for small numbers

# Example: BF16 handles large values better
large_val = torch.tensor(65504.0)   # max FP16 value
large_fp16= large_val.half()
large_bf16= large_val.to(torch.bfloat16)
overflow  = (large_val * 2).half()   # this overflows FP16!
print(f"Value 65504 in FP16: {large_fp16.item()}")
print(f"Value 65504 in BF16: {large_bf16.item()}")
print(f"65504 * 2 in FP16:   {overflow.item()}  <- OVERFLOW (inf)!")
print(f"65504 * 2 in BF16:   {(large_val * 2).to(torch.bfloat16).item()}")
print()
print("AnimateDiff uses FP16 inference with the pipeline's torch_dtype=torch.float16.")
print("BF16 would also work if your GPU supports it (Ampere/Ada = yes; Turing = no).")
print("RTX 5060 (Ada Lovelace) supports BF16.")

# =============================================================================
# EXPECTED OUTPUT:
# FP32 tensor (1000x1000): 3.81 MB
# FP16 tensor (1000x1000): 1.91 MB
# 65504 * 2 in FP16: inf  <- OVERFLOW
# =============================================================================
