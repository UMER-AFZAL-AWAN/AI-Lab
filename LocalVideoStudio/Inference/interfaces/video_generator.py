from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GenerationRequest:
    prompt: str
    negative_prompt: str = (
        "blurry, low quality, worst quality, watermark, text, "
        "deformed, disfigured, ugly, bad anatomy"
    )
    num_frames: int = 16          # keep low for 8 GB VRAM
    fps: int = 8
    width: int = 512
    height: int = 512
    num_inference_steps: int = 25
    guidance_scale: float = 7.5
    seed: int = -1                # -1 = random
    conditioning_image: Path | None = None


@dataclass
class GenerationResult:
    frames: list                  # list of PIL Images
    fps: int
    output_path: Path | None = None


class IVideoGenerator(ABC):
    """Generates video frames from a text prompt (and optionally a seed image)."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate frames. The caller is responsible for saving them."""
        ...

    @abstractmethod
    def generate_seamless_loop(self, request: GenerationRequest) -> GenerationResult:
        """Generate frames that form a seamless ping-pong loop."""
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """True once the model weights are loaded into memory."""
        ...
