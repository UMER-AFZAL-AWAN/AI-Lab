from dataclasses import dataclass


@dataclass
class EnhancementConfig:
    """Controls ffmpeg-based cinematic enhancement."""
    auto_crop:       bool      = True   # detect and remove native letterbox bars
    denoise:         bool      = True   # hqdn3d noise reduction
    sharpening:      bool      = True   # unsharp mask after denoise
    cinematic_grade: bool      = True   # S-curve + orange-teal color grade
    cinematic_bars:  bool      = True   # 2.39:1 letterbox overlay
    vignette:        bool      = True   # soft edge darkening
    film_grain:      bool      = True   # subtle noise texture
    output_fps:      int|None  = None   # None = keep source fps
    nvenc:           bool      = True   # use h264_nvenc; falls back to libx264
