# =============================================================================
# 02 — FEATURE MAPS: WHAT THE CNN SEES
# =============================================================================
#
# LIKE YOU ARE 5:
# When you look at a photo of a cat, you first notice the EARS (pointy shapes),
# then the EYES (round dark circles), then the FUR (texture).
# Each of those is a different "feature". You look for them in different parts
# of the image.
#
# A CNN does the same:
#   - Layer 1 filters detect: edges, corners, blobs of colour
#   - Layer 2 filters detect: eyes, noses, wheels (combinations of edges)
#   - Layer 3 filters detect: faces, cars (combinations of eyes and noses)
#
# The result of applying ONE filter to the whole image is called a FEATURE MAP.
# If you have 64 filters you get 64 feature maps — each one highlighting
# a different thing the network is "looking for".
#
# In RIFE, the feature maps after the first few IFBlock layers represent
# "where things are moving" — not pixels, but motion patterns.
#
# =============================================================================

import torch
import torch.nn as nn

print("=" * 60)
print("ONE LAYER: INPUT CHANNELS -> FEATURE MAPS")
print("=" * 60)

# An RGB image: 3 channels, 32x32 pixels
x = torch.randn(1, 3, 32, 32)

# A Conv2d with 16 output channels = 16 filters = 16 feature maps
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
out  = conv(x)

print(f"Input:        channels={x.shape[1]}, H={x.shape[2]}, W={x.shape[3]}")
print(f"Output:       channels={out.shape[1]}, H={out.shape[2]}, W={out.shape[3]}")
print(f"              16 feature maps, each is 32x32")
print(f"Filter bank:  {conv.weight.shape}  (16 filters × 3-channel × 3×3)")

print()
print("=" * 60)
print("STACKING LAYERS: GOING DEEPER")
print("=" * 60)
#
# Every layer takes the FEATURE MAPS from the previous layer as its input.
# The first layer works on pixel channels (R, G, B).
# The second layer works on EDGE feature maps.
# The third layer works on CORNER feature maps (combinations of edges).
# And so on. Each layer "looks at" increasingly abstract patterns.

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # (3 channels) -> 16 feature maps -> 32 feature maps -> 64 feature maps
        self.conv1 = nn.Conv2d(3,  16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu  = nn.ReLU()

    def forward(self, x):
        print(f"  Input:   {x.shape}")
        x = self.relu(self.conv1(x))
        print(f"  conv1:   {x.shape}  — 16 edge/colour feature maps")
        x = self.relu(self.conv2(x))
        print(f"  conv2:   {x.shape}  — 32 corner/texture feature maps")
        x = self.relu(self.conv3(x))
        print(f"  conv3:   {x.shape}  — 64 object-part feature maps")
        return x

net = SmallCNN()
img = torch.randn(1, 3, 32, 32)
print("Forward pass through 3-layer CNN:")
out = net(img)

total = sum(p.numel() for p in net.parameters())
print(f"\nTotal parameters: {total:,}")
print("Note: spatial size stays 32x32 because we used padding=1")
print("The only thing that grew was the number of feature maps (depth)")

print()
print("=" * 60)
print("WHAT FEATURE MAPS LOOK LIKE (values)")
print("=" * 60)

# After training, feature maps have HIGH values where the filter MATCHED
# and LOW (near-zero) values where it did not match.
# A filter that detects "vertical edge" will have high values only
# at pixels where there IS a vertical edge in that region.

# Simulate an image with a sharp left edge
edge_image = torch.zeros(1, 1, 8, 8)
edge_image[:, :, :, 4:] = 1.0   # right half is bright, left is dark
print("Image (1 = bright, 0 = dark):")
print(edge_image[0, 0])

# Vertical edge filter (Sobel-like): detects left-to-right brightness change
vertical_filter = torch.tensor([[[[0., 0., 0.],
                                   [-1., 0., 1.],
                                   [0., 0., 0.]]]])

import torch.nn.functional as F
response = F.conv2d(edge_image, vertical_filter, padding=1)
print("\nFilter response (high = edge detected here):")
print(response[0, 0].round(decimals=1))
# Should be ~0 everywhere except the column where bright meets dark

# =============================================================================
# EXPECTED OUTPUT:
# Input:        channels=3, H=32, W=32
# Output:       channels=16, H=32, W=32
#   16 feature maps, each is 32x32
# conv1:   torch.Size([1, 16, 32, 32])  — 16 edge/colour feature maps
# conv2:   torch.Size([1, 32, 32, 32])  — 32 corner/texture feature maps
# conv3:   torch.Size([1, 64, 32, 32])  — 64 object-part feature maps
# =============================================================================
