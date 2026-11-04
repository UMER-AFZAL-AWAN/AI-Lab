# =============================================================================
# 02 — TENSOR OPERATIONS
# =============================================================================
#
# LIKE YOU ARE 5:
# Adding numbers: 3 + 4 = 7.
# Adding lists:   [1,2,3] + [4,5,6] = [5,7,9].  (element by element)
# Adding grids:   same idea, but with 2D, 3D, 4D grids.
# That is ALL tensor operations — apply math to every element, element-by-element.
#
# The special trick is called BROADCASTING:
# If the shapes don't match exactly, PyTorch stretches the smaller one
# to match the bigger one (if it makes sense).
#
# =============================================================================

import torch

print("=" * 60)
print("ELEMENT-WISE OPERATIONS")
print("=" * 60)

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(f"a = {a}")
print(f"b = {b}")
print(f"a + b = {a + b}")         # [5, 7, 9]
print(f"a * b = {a * b}")         # [4, 10, 18]  (NOT dot product, element-wise)
print(f"a ** 2 = {a ** 2}")       # [1, 4, 9]
print(f"a.sqrt() = {a.sqrt()}")   # [1, 1.41, 1.73]

print()
print("=" * 60)
print("MATRIX MULTIPLICATION  (the core of neural networks)")
print("=" * 60)
#
# In a neural network, every layer does one thing:
#   output = weights @ input + bias
#
# @ is matrix multiplication (dot product between rows and columns).
# C# analogy: multiplying two matrices — row dot col for each output cell.
#
# Shape rule: (A, B) @ (B, C) = (A, C)
# The inner dimensions must match. The outer dimensions give the output shape.

W = torch.tensor([[1.0, 0.0],
                  [0.0, 2.0],
                  [3.0, 1.0]])  # shape (3, 2)

x = torch.tensor([[5.0],
                  [2.0]])       # shape (2, 1) — column vector

out = W @ x  # shape (3, 1)
print(f"W shape: {W.shape},  x shape: {x.shape}")
print(f"W @ x:\n{out}  shape: {out.shape}")
# Row 0: 1*5 + 0*2 = 5
# Row 1: 0*5 + 2*2 = 4
# Row 2: 3*5 + 1*2 = 17

print()
print("=" * 60)
print("BROADCASTING")
print("=" * 60)
#
# Broadcasting: PyTorch automatically expands smaller tensors to match larger ones.
# Rule: dimensions are aligned from the right. A size-1 dimension is stretched.
#
# Example: an image [3, 4, 4] plus a colour shift [3, 1, 1]
# The [3, 1, 1] gets stretched to [3, 4, 4] automatically.

image = torch.ones(3, 4, 4)       # 3 channels, 4x4 pixels, all ones
shift = torch.tensor([[[0.1]],    # red channel   +0.1
                       [[0.2]],   # green channel +0.2
                       [[0.3]]])  # blue channel  +0.3
# shift shape: (3, 1, 1)

result = image + shift             # broadcasts shift to (3, 4, 4)
print(f"image shape: {image.shape},  shift shape: {shift.shape}")
print(f"result shape: {result.shape}")
print(f"result[0,0,0]={result[0,0,0]:.1f}  (red:   1.0 + 0.1 = 1.1)")
print(f"result[1,0,0]={result[1,0,0]:.1f}  (green: 1.0 + 0.2 = 1.2)")
print(f"result[2,0,0]={result[2,0,0]:.1f}  (blue:  1.0 + 0.3 = 1.3)")

print()
print("=" * 60)
print("REDUCTION OPERATIONS")
print("=" * 60)
#
# Reduction: collapse one or more dimensions into a single number.
# Used constantly: computing loss, measuring stats, normalising.

x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
print(f"x:\n{x}")
print(f"x.sum()        = {x.sum()}")            # sum of all 6 numbers
print(f"x.sum(dim=0)   = {x.sum(dim=0)}")       # sum across rows -> [5, 7, 9]
print(f"x.sum(dim=1)   = {x.sum(dim=1)}")       # sum across cols -> [6, 15]
print(f"x.mean()       = {x.mean()}")
print(f"x.max()        = {x.max()}")
print(f"x.argmax()     = {x.argmax()}")         # index of max (flattened)

print()
print("=" * 60)
print("ACTIVATION FUNCTIONS  (the non-linearity)")
print("=" * 60)
#
# LIKE YOU ARE 5:
# After a matrix multiply, the output can be any number (negative, huge, tiny).
# An activation function squashes that into a useful range, and adds non-linearity.
# Without activation functions, stacking 100 layers = same as 1 layer.
#
# ReLU: if x < 0, output 0. If x > 0, output x. Simple and fast.
# Sigmoid: squashes any number to 0..1. Used for probabilities/masks.
# Softmax: squashes a list to probabilities that sum to 1. Used for classification.

import torch.nn.functional as F

x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
print(f"x:                {x}")
print(f"ReLU(x):          {F.relu(x)}")
print(f"Sigmoid(x):       {F.sigmoid(x).round(decimals=2)}")

logits = torch.tensor([2.0, 1.0, 0.5])  # raw scores (before softmax)
probs  = F.softmax(logits, dim=0)
print(f"Softmax([2,1,0.5]): {probs.round(decimals=3)}  sum={probs.sum():.1f}")
# Largest logit gets the highest probability. Sum = 1.0.

# =============================================================================
# EXPECTED OUTPUT (excerpt):
# a + b = tensor([5., 7., 9.])
# W @ x: tensor([[ 5.], [ 4.], [17.]])  shape: torch.Size([3, 1])
# result[0,0,0]=1.1  (red:   1.0 + 0.1 = 1.1)
# Softmax([2,1,0.5]): tensor([0.576, 0.212, 0.133])  sum=1.0
# =============================================================================
