from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AnimationRequest:
    """Parameters for a single character animation run."""
    reference_photo:     Path              # photo of the person to animate
    prompt:              str               # what they should be doing
    negative_prompt:     str = (
        "blurry, low quality, worst quality, watermark, text, "
        "deformed, disfigured, bad anatomy, extra limbs, cloned face"
    )
    pose_image:          Path | None = None  # optional OpenPose skeleton image
                                             # if None, pose is inferred from prompt
    num_frames:          int   = 16
    fps:                 int   = 8
    width:               int   = 512
    height:              int   = 512
    num_inference_steps: int   = 25
    guidance_scale:      float = 7.5
    ip_adapter_scale:    float = 0.6     # 0.0 = ignore identity, 1.0 = lock it hard
    controlnet_scale:    float = 0.8     # pose influence strength
    seed:                int   = -1


@dataclass
class AnimationResult:
    """Output from a character animation run."""
    frames:      list        # list of PIL Images
    fps:         int
    output_path: Path | None = None
