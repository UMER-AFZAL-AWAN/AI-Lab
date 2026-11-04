# =============================================================================
# 04 — MEMORY BUDGET: UNDERSTANDING YOUR 8GB VRAM
# =============================================================================
#
# LIKE YOU ARE 5:
# Your GPU VRAM is like a bag you bring to school.
# Every model, every tensor, every intermediate result takes up space.
# If you overfill the bag, you get "CUDA out of memory" error.
# This file teaches you to understand and manage what's in the bag.
#
# =============================================================================

import torch
import torch.nn as nn

print("=" * 60)
print("CHECKING CURRENT VRAM USAGE")
print("=" * 60)

if not torch.cuda.is_available():
    print("(No CUDA device — showing formulas and calculations)")
else:
    total  = torch.cuda.get_device_properties(0).total_memory / 1024**3
    reserved  = torch.cuda.memory_reserved(0) / 1024**3
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    free_in_reserved = (reserved - allocated)
    free_total = total - reserved

    print(f"GPU:              {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM:       {total:.2f} GB")
    print(f"Reserved by PyTorch: {reserved:.2f} GB  (claimed from OS)")
    print(f"Allocated tensors:   {allocated:.4f} GB  (actually used)")
    print(f"Free (not reserved): {free_total:.2f} GB")
    print()
    print("Note: PyTorch reserves memory in chunks to avoid fragmentation.")
    print("'Reserved' > 'Allocated' by design.")
    print("empty_cache() releases unreserved allocated memory back to the OS.")

print()
print("=" * 60)
print("WHAT TAKES VRAM DURING INFERENCE")
print("=" * 60)
#
# 1. MODEL WEIGHTS: the parameters of the loaded model.
# 2. ACTIVATIONS: intermediate results during the forward pass.
#    For a UNet, this is proportional to batch_size × seq_len × channels.
# 3. FRAMEWORK OVERHEAD: CUDA context, kernel cache (~200-500MB typically).
# 4. OTHER TENSORS: the input latents, output tensors, scheduler state.

print("Inference memory breakdown:")
print()
print("1. Weights (static, loaded once):")
print("   = num_params × bytes_per_param")
print("   AnimateDiff UNet at FP16: ~3.5B × 2 = 7.0 GB")
print("   Wait — 7GB just for weights? No: SD 1.5 UNet = ~860M params, motion = ~200M")
print("   Total UNet: ~1.06B × 2 bytes = ~2.1 GB at FP16")
print()
print("2. Activations (temporary, peak during forward pass):")
print("   = batch_size × channels × H × W × bytes × num_active_layers")
print("   For UNet at 64×64 latent, batch=1, 16 frames:")
print("   = 1 × 16 × 1280 × 64 × 64 × 2 bytes = ~1.6 GB (approximate)")
print()
print("3. CUDA framework: ~300 MB")
print()
print("Total estimate: 2.1 + 1.6 + 0.3 = ~4 GB without cpu_offload")
print("With VAE + CLIP loaded too: +0.3 + 0.6 = ~5 GB")
print("Peak with activations: could spike to 7-8 GB without cpu_offload.")
print("Your cpu_offload + vae_slicing keeps peak under 6 GB. Smart setup.")

print()
print("=" * 60)
print("MEASURING PEAK VRAM DURING AN OPERATION")
print("=" * 60)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Simulate a large operation
    a = torch.randn(2000, 2000, device="cuda", dtype=torch.float16)
    b = torch.randn(2000, 2000, device="cuda", dtype=torch.float16)
    c = a @ b   # matrix multiply
    torch.cuda.synchronize()

    peak_mb = torch.cuda.max_memory_allocated(0) / 1024**2
    curr_mb = torch.cuda.memory_allocated(0) / 1024**2
    print(f"Peak VRAM during 2000×2000 FP16 matmul: {peak_mb:.1f} MB")
    print(f"Current VRAM (tensors still alive):      {curr_mb:.1f} MB")

    del a, b, c
    torch.cuda.empty_cache()
    after_mb = torch.cuda.memory_allocated(0) / 1024**2
    print(f"After del + empty_cache:                 {after_mb:.1f} MB")
    print()
    print("Tip: torch.cuda.reset_peak_memory_stats() resets the peak counter.")
    print("     Use it before an operation to measure just that operation's peak.")
else:
    print("(Showing the pattern for when CUDA is available:)")
    print()
    print("  torch.cuda.empty_cache()")
    print("  torch.cuda.reset_peak_memory_stats()")
    print("  # ... run your operation ...")
    print("  peak_mb = torch.cuda.max_memory_allocated(0) / 1024**2")
    print("  print(f'Peak: {peak_mb:.1f} MB')")

print()
print("=" * 60)
print("THE VRAM BUDGET TABLE (from ChatGPT's message)")
print("=" * 60)
print()
print(f"{'Precision':<12} {'VRAM per 1B params':>20} {'Example: 7B model':>20}")
print("-" * 55)
for prec, bpp, example in [
    ("FP32",   "4 GB",   "28 GB"),
    ("FP16",   "2 GB",   "14 GB"),
    ("INT8",   "1 GB",   " 7 GB"),
    ("INT4",   "0.5 GB", " 3.5 GB"),
]:
    print(f"{prec:<12} {bpp:>20} {example:>20}")

print()
print("Your RTX 5060 (8 GB VRAM):")
print("  Max model size at INT4:  8 GB / 0.5 GB/B = 16B params (minus overhead)")
print("  Max model size at FP16:  8 GB / 2.0 GB/B =  4B params (minus overhead)")
print("  With cpu_offload at FP16: can run models up to ~12-16B params (slowly)")

# =============================================================================
# EXPECTED OUTPUT:
# Total VRAM: 8.0 GB  (or similar on your RTX 5060)
# Peak VRAM during 2000×2000 matmul: ~30 MB
# =============================================================================
