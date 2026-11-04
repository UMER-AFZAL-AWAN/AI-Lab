# =============================================================================
# 03 — GPU AND CUDA
# =============================================================================
#
# LIKE YOU ARE 5:
# Your CPU is a very smart person who can do complex tasks one at a time.
# Your GPU is a stadium full of simple workers who all do the same simple task
# at exactly the same moment.
#
# Matrix multiplication (which is ALL neural networks do) fits the GPU perfectly.
# A 1280x720 image has 921,600 pixels. The GPU can process all of them at once.
# The CPU would do them one (or a few) at a time.
#
# CUDA is NVIDIA's system for talking to the GPU from code.
# ".to('cuda')" moves a tensor from your RAM to your GPU's VRAM.
# Once a tensor is on the GPU, all operations on it run on the GPU.
#
# C# ANALOGY:
# It is like marshalling data between managed heap and GPU memory.
# The move itself costs time (like a P/Invoke call).
# But once it is there, parallel operations are extremely fast.
#
# =============================================================================

import torch
import time

print("=" * 60)
print("IS CUDA AVAILABLE?")
print("=" * 60)

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU name:       {torch.cuda.get_device_name(0)}")
print(f"VRAM total:     {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# How much VRAM is currently free?
reserved  = torch.cuda.memory_reserved(0)  / 1024**3
allocated = torch.cuda.memory_allocated(0) / 1024**3
free      = torch.cuda.get_device_properties(0).total_memory / 1024**3 - reserved
print(f"VRAM reserved:  {reserved:.2f} GB")
print(f"VRAM allocated: {allocated:.2f} GB")
print(f"VRAM free:      {free:.2f} GB (approx)")

print()
print("=" * 60)
print("MOVING TENSORS TO THE GPU")
print("=" * 60)

# Create a tensor on CPU (default)
x_cpu = torch.randn(1000, 1000)
print(f"x_cpu device: {x_cpu.device}")  # cpu

# Move it to GPU
x_gpu = x_cpu.to("cuda")
print(f"x_gpu device: {x_gpu.device}")  # cuda:0

# Shorthand: create directly on GPU
y_gpu = torch.randn(1000, 1000, device="cuda")
print(f"y_gpu device: {y_gpu.device}")

# Operations between tensors must be on the SAME device
# This would crash: x_cpu + x_gpu  <-- one on CPU, one on GPU
# Always move both to GPU first.

print()
print("=" * 60)
print("CPU vs GPU SPEED COMPARISON")
print("=" * 60)

size = 4096
a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)
a_gpu = a_cpu.to("cuda")
b_gpu = b_cpu.to("cuda")

# Warm up the GPU (first call has overhead from setup)
_ = a_gpu @ b_gpu
torch.cuda.synchronize()  # wait for GPU to finish

# CPU timing
start = time.time()
for _ in range(5):
    c_cpu = a_cpu @ b_cpu
cpu_ms = (time.time() - start) / 5 * 1000

# GPU timing
torch.cuda.synchronize()
start = time.time()
for _ in range(5):
    c_gpu = a_gpu @ b_gpu
torch.cuda.synchronize()  # IMPORTANT: GPU is async, must sync before measuring time
gpu_ms = (time.time() - start) / 5 * 1000

print(f"Matrix multiply {size}x{size}:")
print(f"  CPU: {cpu_ms:.1f} ms per call")
print(f"  GPU: {gpu_ms:.1f} ms per call")
print(f"  GPU is ~{cpu_ms/gpu_ms:.0f}x faster")

print()
print("=" * 60)
print("MEMORY AFTER ALLOCATIONS")
print("=" * 60)

# Check VRAM now that we have tensors on GPU
allocated_now = torch.cuda.memory_allocated(0) / 1024**2
print(f"VRAM now in use: {allocated_now:.1f} MB")

# Free GPU memory
del a_gpu, b_gpu, c_gpu, x_gpu, y_gpu
torch.cuda.empty_cache()  # release cached memory back to the OS
allocated_after = torch.cuda.memory_allocated(0) / 1024**2
print(f"VRAM after cleanup: {allocated_after:.1f} MB")

print()
print("=" * 60)
print("FP16 ON GPU  (how AnimateDiff saves VRAM)")
print("=" * 60)
#
# Most AI models use FP16 (half precision) during inference on GPU.
# FP32 (default) = 4 bytes per number.
# FP16           = 2 bytes per number.
# Same computation, half the VRAM.
#
# Your RTX 5060 also has Tensor Cores which run FP16 MUCH faster than FP32.

a_fp32 = torch.randn(2000, 2000, device="cuda", dtype=torch.float32)
a_fp16 = torch.randn(2000, 2000, device="cuda", dtype=torch.float16)

mem_fp32 = a_fp32.element_size() * a_fp32.nelement() / 1024**2
mem_fp16 = a_fp16.element_size() * a_fp16.nelement() / 1024**2
print(f"2000x2000 FP32: {mem_fp32:.1f} MB")
print(f"2000x2000 FP16: {mem_fp16:.1f} MB  (half the memory)")

# =============================================================================
# EXPECTED OUTPUT:
# CUDA available: True
# GPU name:       NVIDIA GeForce RTX 5060 Laptop GPU
# VRAM total:     8.0 GB
# GPU is ~100x faster  (or more, depending on the task)
# 2000x2000 FP32: 15.3 MB
# 2000x2000 FP16: 7.6 MB
# =============================================================================
