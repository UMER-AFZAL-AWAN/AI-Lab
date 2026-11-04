# =============================================================================
# 04 — RESIDUAL CONNECTIONS (Skip Connections)
# =============================================================================
#
# LIKE YOU ARE 5:
# Imagine you have to teach someone to draw a cat.
# Instead of saying "draw a cat from scratch", you say:
#   "Here is a rough cat. Just fix the bits that are wrong."
# It is MUCH easier to fix small mistakes than to create from nothing.
#
# A residual connection is exactly that.
# output = layer(x) + x
# Instead of "learn to produce the perfect output",
# the layer learns "learn what CHANGE to make to the input".
# That change is called the RESIDUAL.
#
# WHY IT MATTERS:
# Without residual connections, stacking >10 layers made training worse
# (gradients vanished, the network stopped learning).
# ResNet (2015) added skip connections and went to 152 layers.
# It won ImageNet by a large margin.
#
# RIFE's IFNet has residual connections in its U-Net decoder.
# Transformers (attention blocks) use residual connections everywhere.
#
# =============================================================================

import torch
import torch.nn as nn

print("=" * 60)
print("THE SIMPLEST RESIDUAL BLOCK")
print("=" * 60)

class ResidualBlock(nn.Module):
    """
    output = x + Conv(ReLU(Conv(x)))
    The 'x +' part is the skip connection.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu  = nn.ReLU()

    def forward(self, x):
        residual = x                        # save the original input
        out = self.relu(self.conv1(x))      # first transformation
        out = self.conv2(out)               # second transformation (no relu yet)
        out = out + residual                # ADD the original input back (skip!)
        out = self.relu(out)                # relu after the addition
        return out

x = torch.randn(1, 64, 32, 32)
block = ResidualBlock(channels=64)
out   = block(x)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {out.shape}  (same — residual blocks preserve shape)")
print(f"Input and output have the same shape: {x.shape == out.shape}")

print()
print("=" * 60)
print("WHAT THE SKIP CONNECTION DOES TO GRADIENTS")
print("=" * 60)
#
# During training, gradients flow BACKWARDS through the network.
# Without skip connections: gradient × weight × weight × weight × ... → near zero
# With skip connections: gradient can skip over layers via the + path
#
# The gradient of (x + f(x)) with respect to x = 1 + df/dx
# That +1 means gradients NEVER go below 1 on the skip path.
# Deep networks can train reliably.

# Demonstrate: gradient is non-zero at input after one forward+backward pass
x = torch.randn(1, 64, 32, 32, requires_grad=True)
block2 = ResidualBlock(64)
out = block2(x)
loss = out.sum()
loss.backward()
print(f"Gradient at input after backward: mean abs = {x.grad.abs().mean():.6f}")
print("(Non-zero gradient means the skip path kept gradient flowing)")

print()
print("=" * 60)
print("BOTTLENECK RESIDUAL BLOCK (used in ResNet-50+)")
print("=" * 60)
#
# To reduce computation, the full block is replaced by:
# 1x1 conv to SHRINK channels (fewer multiply-adds in the 3x3)
# 3x3 conv (now cheaper because fewer channels)
# 1x1 conv to EXPAND back to original channels
# This is called a "bottleneck" — squeezes then expands.

class BottleneckBlock(nn.Module):
    def __init__(self, channels, bottleneck=4):
        super().__init__()
        inner = channels // bottleneck
        self.shrink  = nn.Conv2d(channels, inner,   kernel_size=1)  # 1x1
        self.conv    = nn.Conv2d(inner,    inner,   kernel_size=3, padding=1)
        self.expand  = nn.Conv2d(inner,    channels, kernel_size=1)  # 1x1
        self.relu    = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.shrink(x))
        out = self.relu(self.conv(out))
        out = self.expand(out)
        return self.relu(out + x)

x = torch.randn(1, 256, 16, 16)
bb = BottleneckBlock(256, bottleneck=4)
out = bb(x)

params_full       = sum(p.numel() for p in ResidualBlock(256).parameters())
params_bottleneck = sum(p.numel() for p in bb.parameters())
print(f"Full residual block (256 ch):       {params_full:,} parameters")
print(f"Bottleneck block (256 ch, /4):      {params_bottleneck:,} parameters")
print(f"Bottleneck uses {params_bottleneck/params_full*100:.0f}% of the parameters")

print()
print("=" * 60)
print("STACKING RESIDUAL BLOCKS (like ResNet)")
print("=" * 60)

class TinyResNet(nn.Module):
    def __init__(self, num_blocks=4, channels=32):
        super().__init__()
        self.stem   = nn.Conv2d(3, channels, kernel_size=7, stride=2, padding=3)
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.pool   = nn.AdaptiveAvgPool2d(1)   # collapse H,W to 1x1
        self.fc     = nn.Linear(channels, 10)   # 10 class classifier

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

net   = TinyResNet(num_blocks=4, channels=32)
img   = torch.randn(2, 3, 64, 64)  # batch of 2
logits= net(img)
print(f"Input:   {img.shape}")
print(f"Output:  {logits.shape}  (batch=2, 10 class scores)")
total = sum(p.numel() for p in net.parameters())
print(f"Total parameters: {total:,}")

# =============================================================================
# EXPECTED OUTPUT:
# Input shape:  torch.Size([1, 64, 32, 32])
# Output shape: torch.Size([1, 64, 32, 32])  (same — residual blocks preserve shape)
# Full residual block (256 ch):       1,180,160 parameters
# Bottleneck uses 25% of the parameters
# =============================================================================
