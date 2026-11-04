# =============================================================================
# 11_controlnet_and_ip_adapter / 01 — CONTROLNET
# =============================================================================
#
# LIKE YOU ARE 5:
# Normally you describe what you want with words: "a cat playing piano."
# ControlNet lets you also draw a SKELETON or OUTLINE and say:
# "I want the subject to be in EXACTLY this pose / depth / edge shape."
# The model fills in the details while following the structure you gave it.
#
# C# ANALOGY:
# Like passing a template (interface/contract) alongside your request.
# Without ControlNet: pipe(prompt) → model figures everything out.
# With ControlNet:    pipe(prompt, control_image=skeleton_image)
#   → model is constrained to match the skeleton's structure.
#
# REAL USE IN LOCALVIDEOSTUDIO:
# You have a source video frame (e.g. from inject_scene).
# Extract its depth map or Canny edges.
# Use ControlNet to generate a new frame that matches that structure
# but looks like a different style/subject.
# → Structural video-to-video transfer.
#
# =============================================================================

import torch
import torch.nn as nn
print("=" * 60)
print("WHAT CONTROLNET IS")
print("=" * 60)
print("""
ControlNet (Zhang et al. 2023) adds a parallel "copy" of the UNet encoder.
The copy takes a CONDITIONING IMAGE instead of noise.

Architecture:
  Normal UNet:      noisy_latent → [Encoder] → [Middle] → [Decoder] → clean_latent
  ControlNet adds:  control_image → [ControlNet Encoder] → adds residuals to UNet Decoder

The ControlNet encoder mirrors the UNet encoder exactly (same architecture, different weights).
Its outputs (the "hint") are ADDED to the UNet decoder's skip connections.
Result: generation is steered toward the control image's structure.

ControlNet does NOT change the base UNet weights.
It's a parallel, additive module — like a LoRA but for spatial structure.
""")

print("=" * 60)
print("CONTROLNET TYPES (preprocessors → what they extract)")
print("=" * 60)
print("""
ControlNet type   | Input preprocessor      | What it controls
──────────────────────────────────────────────────────────────────────
Canny             | Canny edge detector     | Edge/outline structure
Depth             | MiDaS / ZoeDepth        | 3D depth layout
HED / Soft-edge   | HED edge detector       | Soft boundaries
OpenPose          | OpenPose skeleton       | Human body pose (17 keypoints)
DWPose            | DWPose detector         | Face + body + hands
Normal map        | Normal estimator        | Surface normals (lighting cues)
Segmentation      | Segformer / ADE20k      | Semantic regions (sky, road, person)
Shuffle           | Random shuffled pixels  | Style transfer / content
Scribble          | Your own drawing        | Rough sketch → detailed image
Tile              | Downscaled then upscaled| High-resolution upscaling
Lineart           | Lineart extractor       | Clean line drawing → colourised
""")

print("=" * 60)
print("USAGE PATTERN")
print("=" * 60)
print("""
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import torch

# 1. Load ControlNet (Canny example)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16,
)

# 2. Build pipeline with ControlNet injected
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()

# 3. Prepare control image
from PIL import Image
import numpy as np
import cv2

source_frame = load_image("my_video_frame.png")

# Extract Canny edges (the "skeleton")
source_array = np.array(source_frame)
edges = cv2.Canny(source_array, threshold1=100, threshold2=200)
edges_rgb = Image.fromarray(np.stack([edges]*3, axis=-1))

# 4. Generate
out = pipe(
    prompt      = "a samurai warrior, cinematic, 8k",
    image       = edges_rgb,          # the control image
    controlnet_conditioning_scale = 0.8,  # 0.0=ignore, 1.0=strict
    num_inference_steps = 20,
    guidance_scale = 7.5,
)
result = out.images[0]
""")

print("=" * 60)
print("CONTROLNET FOR VIDEO (AnimateDiff + ControlNet)")
print("=" * 60)
print("""
# Apply per-frame ControlNet to each frame of a video:

from diffusers import AnimateDiffControlNetPipeline, ControlNetModel, MotionAdapter

motion_adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-2"
)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)
pipe = AnimateDiffControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    motion_adapter=motion_adapter,
    controlnet=controlnet,
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()

# control_frames: list of 16 PIL Images (one per frame), each a Canny edge map
out = pipe(
    prompt             = "a knight walking through forest, cinematic",
    conditioning_frames = control_frames,   # list of control images
    controlnet_conditioning_scale = 0.7,
    num_frames         = 16,
    num_inference_steps = 25,
)
video_frames = out.frames[0]
""")

print("=" * 60)
print("DEMO: CONTROLNET RESIDUAL INJECTION (conceptual)")
print("=" * 60)

class TinyControlNet(nn.Module):
    """
    Conceptual: ControlNet takes a conditioning image and produces
    a residual that is ADDED to the UNet decoder's skip connection.
    """
    def __init__(self, cond_channels, hidden_dim):
        super().__init__()
        self.cond_encoder = nn.Sequential(
            nn.Conv2d(cond_channels, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )

    def forward(self, cond_image):
        return self.cond_encoder(cond_image)   # the residual hint

class TinyUNetDecoder(nn.Module):
    """Conceptual decoder block that accepts a ControlNet residual."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)

    def forward(self, x, controlnet_hint=None):
        if controlnet_hint is not None:
            x = x + controlnet_hint   # <-- this is the key operation
        return self.conv(x)


hidden = 32
controlnet = TinyControlNet(cond_channels=1, hidden_dim=hidden)  # 1-ch edges
decoder    = TinyUNetDecoder(hidden_dim=hidden)

noisy_feat = torch.randn(1, hidden, 16, 16)   # UNet encoder output
canny_map  = torch.randn(1, 1, 16, 16)        # Canny edge image

hint = controlnet(canny_map)
out_with_cn    = decoder(noisy_feat, controlnet_hint=hint)
out_without_cn = decoder(noisy_feat, controlnet_hint=None)

diff = (out_with_cn - out_without_cn).abs().mean().item()
print(f"Output shape:                  {list(out_with_cn.shape)}")
print(f"Mean absolute change from CN:  {diff:.4f}")
print("(In a real model this delta steers generation toward the edge structure)")

print()
print("=" * 60)
print("CONTROLNET SCALE: HOW STRONG TO APPLY IT")
print("=" * 60)
print("""
controlnet_conditioning_scale:
  0.0  → ControlNet ignored, free generation
  0.3  → gentle guidance (useful for loose structure)
  0.7  → standard (most cases)
  1.0  → strict adherence to structure
  1.5+ → may cause artefacts, overpowering the prompt

For pose (OpenPose): 0.7–1.0 (you want exact pose matching)
For depth:          0.5–0.8 (layout guidance, not pixel-perfect)
For canny/lineart:  0.6–0.9 (edge-to-edge correspondence)
For tile (upscale): 0.5–0.7 (upscaling, not rigid structure)

Also useful: guess_mode=True in some implementations
  → ControlNet activates even without a prompt (useful for style transfer)
""")

# =============================================================================
# EXPECTED OUTPUT:
# Output shape:                  [1, 32, 16, 16]
# Mean absolute change from CN:  ~0.3 (nonzero = ControlNet does something)
# =============================================================================
