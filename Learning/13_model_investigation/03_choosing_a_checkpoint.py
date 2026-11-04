# =============================================================================
# 03 — CHOOSING A CHECKPOINT: CAN IT FIT IN 8 GB VRAM?
# =============================================================================
#
# LIKE YOU ARE 5:
# You want to run a giant robot (a large model) in your small garage (8 GB VRAM).
# Step 1: measure the robot.
# Step 2: measure the garage.
# Step 3: decide if it fits — and if not, can you fold the robot (quantise) to fit?
#
# REAL CASE: 10Eros-Max / MiniMax-H3 (from your investigation)
# FP16:  ~40 GB  → does NOT fit in 8 GB
# INT8:  ~21 GB  → does NOT fit in 8 GB
# W4A8:  ~14 GB  → does NOT fit in 8 GB  (W=4-bit weights, A=8-bit activations)
# INT4:  ~10 GB  → barely doesn't fit (activations push it over 8 GB)
# → CONCLUSION: too large for your 8 GB. Need A100/H100 or cloud.
#
# FILE SIZE ≠ INFERENCE VRAM:
# A 14 GB .safetensors file needs MORE than 14 GB VRAM because:
#   - Activation tensors during inference (can be 2-5x the weight size for large models)
#   - VAE, text encoder, scheduler all loaded alongside
#   - CUDA framework overhead (~300 MB)
#
# =============================================================================

import torch

print("=" * 60)
print("THE VRAM BUDGET FORMULA")
print("=" * 60)
#
# VRAM_needed ≈ weights + activations + other_components + overhead
#
# weights:      num_params × bytes_per_param
# activations:  highly variable; rough rule: 0.5-2x the weight memory for inference
#               (larger for Transformers with large context, smaller for CNNs)
# other:        VAE (~0.3 GB), text_encoder (~0.6-5 GB), scheduler (~0 GB)
# overhead:     CUDA context ~0.3 GB

def estimate_vram(
    model_params_B,     # model parameters in billions
    dtype="fp16",       # weight precision
    activation_ratio=0.5,  # activations as fraction of weight memory
    vae_gb=0.3,
    text_encoder_gb=0.6,
    overhead_gb=0.3,
):
    bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5, "w4a8": 0.5}
    weight_gb   = model_params_B * bytes_per_param[dtype]
    # Activation dtype is usually fp16 regardless of weight dtype
    activation_gb = model_params_B * 2 * activation_ratio
    total = weight_gb + activation_gb + vae_gb + text_encoder_gb + overhead_gb
    return weight_gb, activation_gb, total

your_vram = 8.0

print(f"{'Model':<25} {'dtype':<8} {'Weights GB':>11} {'Activ. GB':>10} {'Total GB':>10} {'Fits 8GB?':>10}")
print("-" * 80)

scenarios = [
    ("10Eros-Max/H3",    5.0, "fp16"),
    ("10Eros-Max/H3",    5.0, "int8"),
    ("10Eros-Max/H3",    5.0, "int4"),
    ("Wan 1.3B",         1.3, "fp16"),
    ("Wan 1.3B",         1.3, "int8"),
    ("Wan 14B",         14.0, "fp16"),
    ("Wan 14B",         14.0, "int4"),
    ("AnimateDiff UNet", 0.86,"fp16"),
    ("SD 1.5 UNet",      0.86,"fp16"),
    ("Llama 3 8B",       8.0, "int4"),
    ("Llama 3 8B",       8.0, "fp16"),
]

for name, params, dtype in scenarios:
    w, a, total = estimate_vram(params, dtype)
    fits = "✓ FITS" if total <= your_vram else ("~ OFFLOAD" if total <= your_vram * 2 else "✗ NO")
    print(f"  {name:<23} {dtype:<8} {w:>11.1f} {a:>10.1f} {total:>10.1f} {fits:>10}")

print()
print("NOTE: activation_ratio=0.5 is a rough estimate. DiT models with long sequences")
print("have MUCH larger activation memory than CNN UNets.")
print("The only way to know exactly: run it and check torch.cuda.max_memory_allocated().")

print()
print("=" * 60)
print("THE FOUR CATEGORIES")
print("=" * 60)

categories = {
    "FITS (< 7 GB total)": [
        "AnimateDiff + SD 1.5    — your current setup, works great",
        "Wan 1.3B at FP16       — small video model, should run",
        "Llama 3 8B at INT4     — LLM, 4-5 GB weights + activations",
    ],
    "FITS WITH OFFLOAD (7-16 GB total, cpu_offload needed)": [
        "Wan 14B at INT4        — possible but very slow (~10 min/clip)",
        "SDXL at FP16           — ~6.5 GB, tight but workable",
        "CogVideoX-2B at FP16   — ~5 GB, possible",
    ],
    "EXPERIMENT ONLY (16-32 GB, cloud or offload + patience)": [
        "10Eros-Max at INT4     — ~10 GB weights + activations overflow",
        "MiniMax-H3 at INT4     — same",
        "Wan 14B at FP16        — 28+ GB, cloud only",
    ],
    "NO (> 32 GB)": [
        "10Eros-Max at FP16     — 40 GB, need A100/H100",
        "GPT-4 (est. 1.8T MoE) — cloud API only",
        "Sora-class models      — unknown, likely 100+ GB",
    ],
}

for category, examples in categories.items():
    print(f"\n{category}:")
    for ex in examples:
        print(f"    • {ex}")

print()
print("=" * 60)
print("READING THE REAL SIZES FROM A REPO")
print("=" * 60)

print("""
# Method 1: check Files tab on huggingface.co manually
# Sum up all .safetensors file sizes = weights on disk = minimum VRAM at same dtype

# Method 2: use huggingface_hub API
from huggingface_hub import HfApi, ModelCard

api   = HfApi()
files = api.list_repo_files("Wan-AI/Wan2.1-T2V-14B")
safetensors_files = [f for f in files if f.endswith(".safetensors")]

# Get sizes (requires downloading the index JSON)
import requests
index_url = "https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/raw/main/model.safetensors.index.json"
index = requests.get(index_url).json()
total_bytes = index.get("metadata", {}).get("total_size", 0)
print(f"Total weights: {total_bytes / 1e9:.1f} GB")

# Method 3: check model card for quantised variants
# Most popular models have community GGUF/AWQ/GPTQ variants with smaller sizes
# Search: huggingface.co/models?search=wan2.1 gguf
""")

print()
print("RECOMMENDATION FOR YOUR 10EROS-MAX / H3 INVESTIGATION:")
print("  File size on disk:   ~40 GB (FP16), ~20 GB (INT8)")
print("  Inference VRAM need: >40 GB (FP16), >20 GB (INT8)")
print("  Your VRAM:            8 GB")
print("  Decision:             Cloud (Colab A100 or RunPod) or wait for")
print("                        a community INT4 GGUF variant (if the community makes one).")

# =============================================================================
# EXPECTED OUTPUT:
# AnimateDiff at FP16:  2.1 GB weights + 1.1 GB activations = ~4.3 GB  ✓ FITS
# Llama 3 8B at INT4:   4.0 GB weights + 2.0 GB activations = ~7.2 GB  ✓ FITS
# 10Eros-Max at FP16:  10.0 GB weights + 5.0 GB activations = ~16.2 GB ✗ NO
# =============================================================================
