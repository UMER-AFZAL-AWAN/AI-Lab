from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProcessorConfig:
    pass


@dataclass
class InterpolationConfig(ProcessorConfig):
    """Controls RIFE frame interpolation."""
    exp: int = 1          # 2^exp multiplier: 1 = 2x (30→60 fps), 2 = 4x
    scale: float = 1.0    # optical-flow resolution scale; try 0.5 for 4K
    fp16: bool = True     # use half-precision on Tensor Core GPUs


@dataclass
class EnhancementConfig(ProcessorConfig):
    """Controls ffmpeg-based cinematic enhancement."""
    auto_crop: bool = True          # detect and remove native letterbox bars
    denoise: bool = True
    cinematic_grade: bool = True    # two-stage S-curve + orange-teal color grade
    cinematic_bars: bool = True     # 2.39:1 letterbox overlay (skipped when auto_crop finds bars)
    vignette: bool = True
    film_grain: bool = True
    sharpening: bool = True
    output_fps: int | None = None   # None = keep source fps
    nvenc: bool = True              # use h264_nvenc; falls back to libx264


class IVideoProcessor(ABC):
    """Takes a video file in, returns a processed video file out."""

    @abstractmethod
    def process(
        self,
        input_path: Path,
        output_path: Path,
        config: ProcessorConfig,
    ) -> Path:
        """Returns output_path on success, raises on failure."""
        ...
