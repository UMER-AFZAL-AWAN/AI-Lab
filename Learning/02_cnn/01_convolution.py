# =============================================================================
# 01 — CONVOLUTION: HOW A FILTER READS AN IMAGE
# =============================================================================
#
# LIKE YOU ARE 5:
# Imagine you have a small magnifying glass that is 3x3 pixels.
# You place it at the top-left corner of the image, look at the 9 pixels under it,
# multiply each by a number written on the glass, add them all up — that is ONE output.
# Then you slide the glass one pixel to the right and repeat.
# Then down one row. Then right again. Until you have covered the whole image.
# The result is a new grid of numbers — a FEATURE MAP.
#
# The numbers written on the glass are the FILTER (or KERNEL).
# Different filters detect different things:
#   - A filter that is [-1,0,1] in one row detects VERTICAL EDGES.
#   - A filter that is all 1/9 (blur) makes the image BLURRY.
#   - A filter that sharpens detects fine DETAILS.
#
# In a neural network, the filter values are LEARNED (not hand-designed).
# The network figures out which filters are most useful for the task.
# RIFE uses many stacked convolutions to detect motion between frames.
#
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("MANUAL 2D CONVOLUTION")
print("=" * 60)

# A tiny 5x5 greyscale image (1 channel)
# (batch=1, channels=1, height=5, width=5)
image = torch.tensor([[[[0., 0., 0., 0., 0.],
                         [0., 1., 1., 1., 0.],
                         [0., 1., 1., 1., 0.],
                         [0., 1., 1., 1., 0.],
                         [0., 0., 0., 0., 0.]]]])
print(f"Input image (5x5):\n{image[0,0]}")

# A 3x3 edge-detection filter (vertical edges)
# Left column = -1, right column = +1, middle = 0
# Fires strongly where the image changes from dark to light horizontally
edge_filter = torch.tensor([[[[-1., 0., 1.],
                                [-1., 0., 1.],
                                [-1., 0., 1.]]]])
print(f"\nEdge detection filter:\n{edge_filter[0,0]}")

# Apply convolution
output = F.conv2d(image, edge_filter, padding=0)
print(f"\nConvolution output (3x3):\n{output[0,0]}")
# Strong positive = light-on-right edge. Strong negative = light-on-left edge.
# Near-zero = no edge.

print()
print("=" * 60)
print("nn.Conv2d — THE LAYER VERSION")
print("=" * 60)
#
# nn.Conv2d(in_channels, out_channels, kernel_size)
#   in_channels:  how many channels the input has  (1 for greyscale, 3 for RGB)
#   out_channels: how many different filters to apply (each gives one feature map)
#   kernel_size:  size of each filter (3 = 3x3)
#
# It LEARNS the filter values. You don't hand-design them.
# A Conv2d with 64 output channels = 64 different learned filters.

conv_layer = nn.Conv2d(
    in_channels=1,
    out_channels=4,    # 4 different filters
    kernel_size=3,
    padding=1,         # pad input edges so output is same size as input
)
print(f"Conv2d: {conv_layer}")
print(f"Filter weights shape: {conv_layer.weight.shape}")
# (out_channels=4, in_channels=1, kernel_h=3, kernel_w=3)
# = 4 filters, each 1x3x3

rgb_image = torch.randn(1, 3, 32, 32)  # batch=1, RGB, 32x32
conv_rgb = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
out = conv_rgb(rgb_image)
print(f"\nInput:  shape={rgb_image.shape}")
print(f"Output: shape={out.shape}")
# (1, 16, 32, 32): same spatial size (because padding=1), but 16 feature maps

print()
print("=" * 60)
print("STRIDE — CONTROLLING HOW FAR THE FILTER SLIDES")
print("=" * 60)
#
# stride=1: filter moves 1 pixel at a time. Output ≈ same size as input.
# stride=2: filter moves 2 pixels at a time. Output is half the size.
# Using stride=2 instead of a separate pooling layer is common in modern CNNs.
# RIFE uses stride=2 convolutions to process images at lower resolution first.

x = torch.randn(1, 3, 64, 64)
conv_s1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
conv_s2 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
out_s1 = conv_s1(x)
out_s2 = conv_s2(x)
print(f"Input:        {x.shape}")
print(f"stride=1 out: {out_s1.shape}  (same spatial size)")
print(f"stride=2 out: {out_s2.shape}  (halved spatial size)")

print()
print("=" * 60)
print("HOW RIFE USES CONVOLUTION (from IFNet.py)")
print("=" * 60)
#
# In IFNet.py the conv() function is:
#   nn.Conv2d(in, out, 3, 1, 1) followed by nn.PReLU(out)
# PReLU is a learnable ReLU variant — it has one learned parameter per channel.
# The IFBlock stacks 8 of these conv layers in sequence.
# The model feeds two video frames (concatenated as 6 channels) through these layers
# to predict optical flow — where each pixel moved.

# Simulate the first layer of IFBlock (from IFNet.py lines 7-18)
def conv_rife(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True),
        nn.PReLU(out_planes),
    )

# IFBlock.conv0: takes 2 stacked RGB frames (6 channels), outputs 120 channels
# with stride=2 twice to work at 1/4 resolution
frame_A = torch.randn(1, 3, 720, 1280)
frame_B = torch.randn(1, 3, 720, 1280)
stacked = torch.cat([frame_A, frame_B], dim=1)  # concat on channel dim -> 6 channels

conv_down1 = conv_rife(6, 60)    # 6 -> 60 channels, stride=2 (halves resolution)
# (For simplicity we use stride=1 here — real code uses stride=2)
feat = conv_down1(stacked)
print(f"Two frames stacked: {stacked.shape}")
print(f"After first conv:   {feat.shape}")

# =============================================================================
# EXPECTED OUTPUT:
# Input image (5x5):
#   tensor([[0., 0., 0., 0., 0.], [0., 1., 1., 1., 0.], ...])
# Convolution output (3x3): (shows edge responses)
# stride=2 out: torch.Size([1, 16, 32, 32])  (halved)
# =============================================================================
