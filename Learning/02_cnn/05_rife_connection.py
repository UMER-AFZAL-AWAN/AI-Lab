# =============================================================================
# 05 — RIFE: HOW IFNet USES ALL OF THIS
# =============================================================================
#
# LIKE YOU ARE 5:
# RIFE takes two video frames (before and after) and creates the frame that
# should appear BETWEEN them. Like filling in a missing puzzle piece.
#
# The network's job: "which pixels moved, and how far?"
# That answer is called OPTICAL FLOW. If a ball moved from pixel (100,200)
# to pixel (110,205), the flow at (100,200) is (10, 5).
# Once you know the flow, you can "warp" (stretch/move) each frame toward
# the middle and blend them together.
#
# RIFE'S ARCHITECTURE (IFNet):
#   Three IFBlocks run in sequence, each at a different scale:
#     Block 0: 1/4 resolution — rough flow estimate (big motions)
#     Block 1: 1/2 resolution — refine the estimate (medium motions)
#     Block 2: full resolution — final refinement (fine details)
#   This is called a COARSE-TO-FINE or PYRAMID approach.
#   Each block ADDS its correction to the previous block's output.
#   That is exactly a residual connection — applied at the WHOLE-BLOCK level.
#
# =============================================================================

import torch
import torch.nn as nn

print("=" * 60)
print("WARP: MOVING PIXELS ACCORDING TO FLOW")
print("=" * 60)
#
# Given a frame and a flow field, warp() moves each pixel by its flow vector.
# PyTorch's grid_sample() does this efficiently on the GPU.

import torch.nn.functional as F

def warp(frame, flow):
    """Move pixels in `frame` according to `flow`."""
    B, C, H, W = frame.shape
    # Build a grid of (x, y) coordinates
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=frame.device),
        torch.linspace(-1, 1, W, device=frame.device),
        indexing='ij'
    )
    base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    # Normalise flow to [-1, 1] range that grid_sample expects
    flow_norm = torch.stack([
        flow[:, 0] / (W / 2),   # x-displacement
        flow[:, 1] / (H / 2),   # y-displacement
    ], dim=-1)
    new_grid = base_grid + flow_norm
    return F.grid_sample(frame, new_grid, align_corners=True, mode='bilinear')

# Simulate two frames and a "move right by 5 pixels" flow
H, W = 64, 64
frame0 = torch.randn(1, 3, H, W)
flow   = torch.zeros(1, 2, H, W)
flow[:, 0, :, :] = 5.0   # all pixels shift 5 pixels right

warped = warp(frame0, flow)
print(f"Frame:  {frame0.shape}")
print(f"Flow:   {flow.shape}  (2 channels = x-displacement, y-displacement)")
print(f"Warped: {warped.shape}")
print("Pixel at (0,0,32,20) before:", frame0[0, 0, 32, 20].item():.4f)
print("Should appear at (0,0,32,25) after warp:", warped[0, 0, 32, 25].item():.4f)

print()
print("=" * 60)
print("IFBLOCK: COARSE FLOW ESTIMATION AT ONE SCALE")
print("=" * 60)
#
# Each IFBlock takes:
#   - frame0, frame1 (the two input frames, possibly downsampled)
#   - previous flow estimate (from previous coarser block)
#   - scale factor (1/4, 1/2, or 1)
# And outputs:
#   - refined flow delta (added to the previous estimate)
#   - a soft mask (how much to blend warped frame0 vs warped frame1)

def conv_prelu(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=True),
        nn.PReLU(out_ch),
    )

class IFBlock(nn.Module):
    """
    Simplified IFBlock (matches structure of IFNet.py in the repo).
    Real code has 8 conv layers; this has 4 for clarity.
    """
    def __init__(self, in_channels, scale=1):
        super().__init__()
        self.scale = scale
        # Encode: downsample to work at lower resolution
        self.conv0 = nn.Sequential(
            conv_prelu(in_channels, 64, stride=2),
            conv_prelu(64, 64, stride=2),
        )
        # Process
        self.conv1 = conv_prelu(64, 64)
        self.conv2 = conv_prelu(64, 64)
        # Decode: upsample back and produce flow + mask
        self.conv3 = nn.Conv2d(64, 5, kernel_size=3, padding=1)  # 4=flow, 1=mask

    def forward(self, x, flow, scale):
        # Downsample input if scale < 1
        if scale != 1:
            x = F.interpolate(x, scale_factor=1/scale, mode='bilinear', align_corners=False)
        feat = self.conv0(x)
        feat = self.conv1(feat) + feat   # residual inside block
        feat = self.conv2(feat)
        # Upsample prediction back to full resolution
        out  = self.conv3(feat)
        out  = F.interpolate(out, scale_factor=scale * 4, mode='bilinear', align_corners=False)
        # flow delta = channels 0:4, mask = channel 4
        flow_delta = out[:, :4]
        mask_delta = out[:, 4:5]
        return flow + flow_delta, mask_delta

print("IFBlock created. Channels: 2×RGB + 2×flow = 10 input channels")

print()
print("=" * 60)
print("COARSE-TO-FINE: THREE IFBLOCKS IN SEQUENCE")
print("=" * 60)

# Two video frames
H, W   = 64, 64   # (small for quick demo)
frame0 = torch.randn(1, 3, H, W)
frame1 = torch.randn(1, 3, H, W)

# Start with zero flow
flow = torch.zeros(1, 4, H, W)   # 4 channels: (dx0, dy0, dx1, dy1)
mask = torch.zeros(1, 1, H, W)

# In channels = frame0(3) + frame1(3) + warped0(3) + warped1(3) + mask(1) = 13
# But for this simplified demo we just use frame0+frame1+flow = 10
in_ch = 10

block_coarse = IFBlock(in_ch, scale=4)  # 1/4 resolution
block_medium = IFBlock(in_ch, scale=2)  # 1/2 resolution
block_fine   = IFBlock(in_ch, scale=1)  # full resolution

# Fake x (would be concatenation of frames + current flow in real code)
x = torch.randn(1, in_ch, H, W)

flow, m = block_coarse(x, flow, scale=4)
print(f"After coarse block (1/4 res): flow={flow.shape}, mask={m.shape}")
flow, m = block_medium(x, flow, scale=2)
print(f"After medium block (1/2 res): flow={flow.shape}, mask={m.shape}")
flow, m = block_fine(x, flow, scale=1)
print(f"After fine   block (1/1 res): flow={flow.shape}, mask={m.shape}")

print()
print("Each block ADDS its correction to the previous flow.")
print("That is a residual connection at the architecture level.")
print()
print("RIFE then uses the final flow to warp frame0 toward t,")
print("and frame1 backward toward t, and blends them with the mask.")

# =============================================================================
# EXPECTED OUTPUT:
# After coarse block (1/4 res): flow=torch.Size([1, 4, 64, 64])
# After medium block (1/2 res): flow=torch.Size([1, 4, 64, 64])
# After fine   block (1/1 res): flow=torch.Size([1, 4, 64, 64])
# =============================================================================
