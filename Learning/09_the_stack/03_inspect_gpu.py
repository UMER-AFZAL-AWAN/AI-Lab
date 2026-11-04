# =============================================================================
# 03 — INSPECTING YOUR GPU AND VRAM
# =============================================================================
#
# LIKE YOU ARE 5:
# Before you start cooking, you check: how big is the pot?
# How much butter do I have left?
# Is the stove on?
#
# Before running AI models, you check:
# Is CUDA available?
# How much VRAM do I have?
# How much is free right now?
# Which operations are fast on this GPU?
#
# =============================================================================

import torch

print("=" * 60)
print("IS CUDA AVAILABLE?")
print("=" * 60)

print(f"CUDA available:    {torch.cuda.is_available()}")
print(f"CUDA version:      {torch.version.cuda}")
print(f"PyTorch version:   {torch.__version__}")
print(f"Number of GPUs:    {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print()
    print("=" * 60)
    print("GPU DETAILS")
    print("=" * 60)

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / 1024**3
        print(f"GPU {i}: {props.name}")
        print(f"  Total VRAM:    {total_gb:.2f} GB")
        print(f"  CUDA Compute:  {props.major}.{props.minor}  (8.9 = Ada Lovelace/RTX 4000-5000 series)")
        print(f"  Multiprocessors: {props.multi_processor_count}")
        print(f"  Max threads per MP: {props.max_threads_per_multi_processor}")
        print(f"  Max threads per block: {props.max_threads_per_block}")
        print()

        # Compute capability determines available features:
        cc = f"{props.major}.{props.minor}"
        features = {
            "7.0+": "Volta — Tensor Cores (FP16/INT8)",
            "7.5+": "Turing — INT4 Tensor Cores, Ray Tracing",
            "8.0+": "Ampere — TF32, BF16 Tensor Cores, A100-class",
            "8.9+": "Ada Lovelace — FP8, 4th gen Tensor Cores, RTX 4000/5000",
        }
        print(f"  Your compute capability {cc} supports:")
        for threshold, feature in features.items():
            major_t, minor_t = map(int, threshold[:-1].split("."))
            if props.major > major_t or (props.major == major_t and props.minor >= minor_t):
                print(f"    ✓ {feature}")

    print()
    print("=" * 60)
    print("CURRENT VRAM STATUS")
    print("=" * 60)

    total     = torch.cuda.get_device_properties(0).total_memory
    reserved  = torch.cuda.memory_reserved(0)
    allocated = torch.cuda.memory_allocated(0)
    free      = total - reserved

    print(f"Total:     {total  /1024**3:.3f} GB")
    print(f"Reserved:  {reserved /1024**3:.3f} GB  (held by PyTorch caching allocator)")
    print(f"Allocated: {allocated/1024**3:.4f} GB  (actual tensors)")
    print(f"Free:      {free    /1024**3:.3f} GB  (available to claim)")
    print()
    print("Note: 'Free' may be a little less than 'Total' due to OS overhead (~200 MB).")

    # Allocate something and watch the memory
    print()
    print("Allocating a 1GB tensor and watching:")
    t = torch.randn(256 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)  # ~1 GB
    after_alloc = torch.cuda.memory_allocated(0)
    print(f"  Allocated now: {after_alloc/1024**3:.3f} GB  (increased by ~1 GB)")

    del t
    torch.cuda.empty_cache()
    after_free = torch.cuda.memory_allocated(0)
    print(f"  After del+empty_cache: {after_free/1024**3:.4f} GB  (returned)")

    print()
    print("=" * 60)
    print("TENSOR CORE SPEEDUP TEST")
    print("=" * 60)

    import time

    def time_matmul(size, dtype, n=10):
        a = torch.randn(size, size, device="cuda", dtype=dtype)
        b = torch.randn(size, size, device="cuda", dtype=dtype)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n):
            c = a @ b
        torch.cuda.synchronize()
        ms = (time.time() - start) / n * 1000
        del a, b, c
        torch.cuda.empty_cache()
        return ms

    size = 2048
    ms_fp32 = time_matmul(size, torch.float32)
    ms_fp16 = time_matmul(size, torch.float16)
    ms_bf16 = time_matmul(size, torch.bfloat16)
    print(f"Matrix multiply {size}×{size}:")
    print(f"  FP32: {ms_fp32:.2f} ms")
    print(f"  FP16: {ms_fp16:.2f} ms  ({ms_fp32/ms_fp16:.1f}x speedup via Tensor Cores)")
    print(f"  BF16: {ms_bf16:.2f} ms  ({ms_fp32/ms_bf16:.1f}x speedup via Tensor Cores)")
    print()
    print("FP16 and BF16 use Tensor Cores — dedicated hardware for fast matmul.")
    print("AnimateDiff uses FP16 to get this speedup on every UNet operation.")

else:
    print()
    print("No CUDA device. Expected output on your RTX 5060:")
    print("  GPU 0: NVIDIA GeForce RTX 5060 Laptop GPU")
    print("  Total VRAM:    8.00 GB")
    print("  CUDA Compute:  8.9  (Ada Lovelace)")
    print("  Tensor Cores:  4th generation (FP16, BF16, FP8, INT8, INT4)")
    print()
    print("FP16 matmul should be 10-20x faster than FP32 on Tensor Cores.")

# =============================================================================
# EXPECTED OUTPUT (on your RTX 5060):
# GPU 0: NVIDIA GeForce RTX 5060 Laptop GPU
# Total VRAM: ~8.0 GB
# FP16 speedup via Tensor Cores: 5-15x faster than FP32
# =============================================================================
