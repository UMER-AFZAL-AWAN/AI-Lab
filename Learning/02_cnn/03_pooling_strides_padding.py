# =============================================================================
# 03 — POOLING, STRIDES, AND PADDING
# =============================================================================
#
# LIKE YOU ARE 5:
# Imagine you have a 100x100 pixel photo.
# If you apply convolution and keep all 100x100 pixels at every layer,
# the model gets HUGE and slow.
# You need a way to make the feature maps SMALLER as you go deeper.
# That lets the deeper layers "see more of the image at once".
#
# Three tools for this:
#
# 1. STRIDE:  Skip pixels while sliding the filter.
#             stride=2 means jump 2 pixels each step -> output is half the size.
#
# 2. MAX POOLING: Take a 2x2 window, keep only the BIGGEST value. Repeat across
#                 the whole feature map. Output is half the size.
#                 "If ANY pixel in this region had a strong edge, the region DID
#                  have an edge." Information survives, location becomes rough.
#
# 3. PADDING: Add zeros around the border of the input before applying a filter.
#             padding=1 with kernel=3 makes the output the SAME size as input.
#             Without it, every conv layer would shrink the spatial size.
#
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("PADDING: KEEPING SPATIAL SIZE")
print("=" * 60)

x = torch.randn(1, 1, 5, 5)
conv_no_pad = nn.Conv2d(1, 1, kernel_size=3, padding=0)
conv_pad    = nn.Conv2d(1, 1, kernel_size=3, padding=1)

out_no_pad = conv_no_pad(x)
out_pad    = conv_pad(x)

print(f"Input:            {x.shape}  (5x5)")
print(f"padding=0 output: {out_no_pad.shape}  (3x3 — shrank!)")
print(f"padding=1 output: {out_pad.shape}    (5x5 — same size)")
print()
print("Why padding=1 works for kernel=3:")
print("  Zeros added on each side add 1 pixel border.")
print("  3x3 filter centered on a border pixel now has valid neighbours.")
print("  Formula: padding = (kernel_size - 1) // 2  keeps size")

print()
print("=" * 60)
print("STRIDE: DOWNSAMPLING WHILE CONVOLVING")
print("=" * 60)

x = torch.randn(1, 3, 64, 64)
conv_s1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
conv_s2 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
conv_s4 = nn.Conv2d(3, 16, kernel_size=3, stride=4, padding=1)

print(f"Input: {x.shape}")
print(f"stride=1: {conv_s1(x).shape}  (no downsampling)")
print(f"stride=2: {conv_s2(x).shape}  (half size)")
print(f"stride=4: {conv_s4(x).shape}  (quarter size)")
print()
print("RIFE's IFBlock uses stride=2 in the first two conv layers.")
print("This makes the network process a quarter-resolution version first,")
print("then refines at higher resolution later.")

print()
print("=" * 60)
print("MAX POOLING: DOWNSAMPLING BY KEEPING THE WINNER")
print("=" * 60)

x = torch.tensor([[[[1., 3., 2., 4.],
                     [5., 6., 1., 2.],
                     [3., 1., 4., 7.],
                     [2., 2., 5., 6.]]]])
print(f"Input (4x4):\n{x[0,0]}")

pool = nn.MaxPool2d(kernel_size=2, stride=2)
out  = pool(x)
print(f"\nMaxPool2d(2,2) output (2x2):\n{out[0,0]}")
# Top-left 2x2: max(1,3,5,6) = 6
# Top-right 2x2: max(2,4,1,2) = 4
# Bottom-left: max(3,1,2,2) = 3
# Bottom-right: max(4,7,5,6) = 7

print()
print("Average pooling — takes the mean instead of max:")
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
out_avg  = avg_pool(x)
print(f"AvgPool2d output:\n{out_avg[0,0]}")

print()
print("=" * 60)
print("COMBINING: A REAL DOWNSAMPLING BLOCK")
print("=" * 60)
#
# Most modern CNNs use stride=2 convolution instead of MaxPool + conv.
# It is slightly more expressive (the network LEARNS how to downsample).
# ResNet and RIFE both do this.

class DownsampleBlock(nn.Module):
    """Halves H and W, doubles channels."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))

x = torch.randn(1, 3, 128, 128)
b1 = DownsampleBlock(3,  16)
b2 = DownsampleBlock(16, 32)
b3 = DownsampleBlock(32, 64)

h = x
for i, block in enumerate([b1, b2, b3], 1):
    h = block(h)
    print(f"After block {i}: {h.shape}  (spatial halved, channels doubled)")

# =============================================================================
# EXPECTED OUTPUT:
# padding=0 output: torch.Size([1, 1, 3, 3])  (3x3 — shrank!)
# padding=1 output: torch.Size([1, 1, 5, 5])  (5x5 — same size)
# After block 1: torch.Size([1, 16, 64, 64])
# After block 2: torch.Size([1, 32, 32, 32])
# After block 3: torch.Size([1, 64, 16, 16])
# =============================================================================
