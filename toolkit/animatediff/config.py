from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GenerationRequest:
    """Parameters for a single AnimateDiff generation run."""
    prompt: str
    negative_prompt: str = (
        "blurry, low quality, worst quality, watermark, text, "
        "deformed, disfigured, ugly, bad anatomy"
    )
    num_frames:          int        = 16    # keep low for 8 GB VRAM
    fps:                 int        = 8
    width:               int        = 512
    height:              int        = 512
    num_inference_steps: int        = 25
    guidance_scale:      float      = 7.5
    seed:                int        = -1    # -1 = random
    conditioning_image:  Path|None  = None  # for IP-Adapter / ControlNet


@dataclass
class GenerationResult:
    """Output from a generation run."""
    frames:      list       # list of PIL Images
    fps:         int
    output_path: Path|None  = None
