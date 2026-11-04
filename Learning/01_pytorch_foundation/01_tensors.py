# =============================================================================
# 01 — WHAT IS A TENSOR
# =============================================================================
#
# LIKE YOU ARE 5:
# A number is just one thing: 7.
# A list of numbers is a row: [1, 2, 3].
# A grid of numbers is a table: [[1,2,3],[4,5,6]].
# A tensor is ALL of these — 1D, 2D, 3D, 4D, however many dimensions you need.
#
# In AI, almost everything is a tensor:
#   A greyscale image (28x28 pixels)     -> a 2D tensor, shape [28, 28]
#   An RGB image (3 channels, 720x1280)  -> a 3D tensor, shape [3, 720, 1280]
#   A batch of 8 RGB video frames        -> a 4D tensor, shape [8, 3, 720, 1280]
#
# "Shape" is just the size of each dimension.
# Shape [8, 3, 720, 1280] means: 8 frames, 3 colour channels, 720 rows, 1280 cols.
#
# C# ANALOGY:
# A tensor is like a multi-dimensional array: float[,,,,].
# PyTorch's tensor has the same idea but runs on the GPU and supports automatic
# differentiation (computing gradients for training).
#
# =============================================================================

import torch

print("=" * 60)
print("SCALARS, VECTORS, MATRICES, TENSORS")
print("=" * 60)

# A scalar — one number, 0 dimensions
scalar = torch.tensor(7.0)
print(f"Scalar:  {scalar}  shape={scalar.shape}  ndim={scalar.ndim}")
# shape=torch.Size([]) means no dimensions — it is just a number

# A vector — a list of numbers, 1 dimension
vector = torch.tensor([1.0, 2.0, 3.0])
print(f"Vector:  {vector}  shape={vector.shape}  ndim={vector.ndim}")

# A matrix — a grid of numbers, 2 dimensions
matrix = torch.tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])
print(f"Matrix:\n{matrix}  shape={matrix.shape}  ndim={matrix.ndim}")

# A 3D tensor — like an RGB image (channels, height, width)
image_tensor = torch.zeros(3, 4, 4)  # 3 channels, 4x4 pixels, all zeros
print(f"\n3D tensor (fake image): shape={image_tensor.shape}")

# A 4D tensor — a batch of images (batch, channels, height, width)
batch = torch.zeros(8, 3, 720, 1280)  # 8 frames of 1280x720 RGB
print(f"4D tensor (batch of frames): shape={batch.shape}")
# How much memory does this take?
bytes_used = batch.element_size() * batch.nelement()
print(f"Memory: {bytes_used / 1024 / 1024:.1f} MB  (float32 = 4 bytes per number)")

print()
print("=" * 60)
print("DATA TYPES (dtype)")
print("=" * 60)
#
# Every tensor stores numbers in a specific precision.
# More precision = more memory = slower = more accurate.
# Less precision = less memory = faster = less accurate.
#
# FP32 = 32-bit float. 4 bytes per number. Standard training precision.
# FP16 = 16-bit float. 2 bytes per number. Fits twice as many numbers in VRAM.
# BF16 = 16-bit brain float. Same size as FP16 but better range (used in LLMs).
# INT8 = 8-bit integer. 1 byte. Used in quantized inference.
# INT4 = 4-bit integer. 0.5 bytes. Heavily quantized, some quality loss.

fp32 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
fp16 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
int8 = torch.tensor([1, 2, 3],       dtype=torch.int8)

print(f"FP32: {fp32}  bytes/element={fp32.element_size()}")
print(f"FP16: {fp16}  bytes/element={fp16.element_size()}")
print(f"INT8: {int8}  bytes/element={int8.element_size()}")

print()
print("=" * 60)
print("CREATING TENSORS")
print("=" * 60)

# All zeros — often used to initialise output buffers
zeros = torch.zeros(2, 3)
print(f"zeros:\n{zeros}")

# All ones — sometimes used as masks
ones = torch.ones(2, 3)
print(f"ones:\n{ones}")

# Random values between 0 and 1 — used to initialise weights
rand = torch.rand(2, 3)
print(f"rand:\n{rand}")

# Random values from normal distribution (mean=0, std=1) — common for weight init
randn = torch.randn(2, 3)
print(f"randn:\n{randn}")

print()
print("=" * 60)
print("SHAPE OPERATIONS")
print("=" * 60)
#
# Reshaping is constant in deep learning.
# The DATA does not change — only how it is interpreted changes.
# C# analogy: Span<float>.Slice() — same memory, different view.

x = torch.arange(12, dtype=torch.float32)  # [0, 1, 2, ..., 11]
print(f"Original: {x}  shape={x.shape}")

x_2d = x.reshape(3, 4)   # view as 3 rows, 4 columns
print(f"Reshaped to (3,4):\n{x_2d}")

x_4d = x.reshape(1, 1, 3, 4)  # batch=1, channel=1, h=3, w=4
print(f"Reshaped to (1,1,3,4): shape={x_4d.shape}")

# squeeze() removes dimensions of size 1
x_squeezed = x_4d.squeeze()
print(f"Squeezed: shape={x_squeezed.shape}")

# unsqueeze(dim) adds a dimension of size 1 at position dim
# Used constantly when adding a batch dimension
x_with_batch = x_2d.unsqueeze(0)  # adds batch dim at front
print(f"Unsqueeze(0): shape={x_with_batch.shape}  (added batch dim)")

# =============================================================================
# EXPECTED OUTPUT:
# Scalar:  7.0  shape=torch.Size([])  ndim=0
# Vector:  tensor([1., 2., 3.])  shape=torch.Size([3])  ndim=1
# Matrix: ... shape=torch.Size([2, 3])
# 3D tensor (fake image): shape=torch.Size([3, 4, 4])
# 4D tensor (batch of frames): shape=torch.Size([8, 3, 720, 1280])
# Memory: 211.0 MB
# FP32: ... bytes/element=4
# FP16: ... bytes/element=2
# INT8: ... bytes/element=1
# =============================================================================
