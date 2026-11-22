from dataclasses import dataclass
from pathlib import Path


@dataclass
class AnimationRequest:
    """Parameters for a single character animation run."""
    reference_photo:     Path        # photo to animate — becomes frame 0
    prompt:              str = ""    # unused by SVD; kept for CLI compatibility
    negative_prompt:     str = ""    # unused by SVD; kept for CLI compatibility
    num_frames:          int   = 25  # SVD-XT generates 25 frames
    fps:                 int   = 8
    width:               int   = 1024
    height:              int   = 576
    num_inference_steps: int   = 25
    motion_bucket_id:    int   = 127  # 1–255: controls amount of motion
    noise_aug_strength:  float = 0.02
    decode_chunk_size:   int   = 2    # lower = less VRAM during decode
    seed:                int   = -1


@dataclass
class AnimationResult:
    """Output from a character animation run."""
    frames:      list        # list of PIL Images
    fps:         int
    output_path: Path | None = None
