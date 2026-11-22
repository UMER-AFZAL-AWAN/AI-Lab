"""
toolkit.character.animator — animate a person from a reference photo using SVD.

Pipeline
────────
1. Load reference photo → resize to 1024×576 (SVD native resolution)
2. Run StableVideoDiffusionPipeline — the photo IS frame 0, so face is preserved exactly
3. Return frames (caller saves to disk)

Memory budget (8 GB VRAM)
──────────────────────────
  - fp16 weights
  - enable_model_cpu_offload() keeps peak VRAM ~6–7 GB
  - decode_chunk_size=2 decodes VAE frames in small batches

Models required on disk
────────────────────────
  C:\\AI-Lab\\Models\\SVD\\    ← stabilityai/stable-video-diffusion-img2vid-xt

See toolkit/character/DOWNLOAD.md for download commands.
"""

import random
import torch
from pathlib import Path
from PIL import Image

from .config import AnimationRequest, AnimationResult
from toolkit.logging import get_logger

_AI_LAB_ROOT = Path(__file__).parent.parent.parent
_SVD_DIR     = _AI_LAB_ROOT / "Models" / "SVD"

log = get_logger(__name__)


class CharacterAnimator:
    """
    Animate a person from a reference photo using Stable Video Diffusion.
    The reference photo becomes frame 0, so face identity is preserved exactly.
    """

    def __init__(self) -> None:
        self._pipe = None

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    def animate(self, request: AnimationRequest) -> AnimationResult:
        _validate_request(request)
        pipe      = self._get_pipeline()
        image     = _prepare_image(request.reference_photo, request.width, request.height)
        generator = _make_generator(request.seed)

        log.step(1, 2, f"Generating {request.num_frames} frames @ {request.width}×{request.height}")
        output = pipe(
            image               = image,
            num_frames          = request.num_frames,
            num_inference_steps = request.num_inference_steps,
            motion_bucket_id    = request.motion_bucket_id,
            noise_aug_strength  = request.noise_aug_strength,
            decode_chunk_size   = request.decode_chunk_size,
            generator           = generator,
        )
        log.step(2, 2, "Done")
        frames = output.frames[0]
        log.done("character.animate", f"{len(frames)} frames")
        return AnimationResult(frames=frames, fps=request.fps)

    def animate_seamless_loop(self, request: AnimationRequest) -> AnimationResult:
        """Generate a seamless ping-pong loop of the animation."""
        result  = self.animate(request)
        forward = result.frames
        reverse = list(reversed(forward[1:-1]))
        log.info("character", f"ping-pong: {len(forward)} + {len(reverse)} frames")
        return AnimationResult(frames=forward + reverse, fps=result.fps)

    def unload(self) -> None:
        self._pipe = None
        torch.cuda.empty_cache()
        log.info("character", "pipeline unloaded, VRAM freed")

    # ── Private ──────────────────────────────────────────────────────────────

    def _get_pipeline(self):
        if self._pipe is not None:
            return self._pipe

        if not _SVD_DIR.exists():
            raise FileNotFoundError(
                f"SVD model not found at {_SVD_DIR}\n"
                "See toolkit/character/DOWNLOAD.md for download commands."
            )

        from diffusers import StableVideoDiffusionPipeline

        log.info("character", "loading SVD pipeline")
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            str(_SVD_DIR),
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        pipe.enable_model_cpu_offload()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()

        log.done("character", "pipeline ready")
        self._pipe = pipe
        return self._pipe


# ── Module-level helpers ─────────────────────────────────────────────────────

def _validate_request(req: AnimationRequest) -> None:
    if not req.reference_photo.exists():
        raise FileNotFoundError(f"[character] Reference photo not found: {req.reference_photo}")


def _prepare_image(photo_path: Path, width: int, height: int) -> Image.Image:
    img = Image.open(photo_path).convert("RGB")
    return img.resize((width, height), Image.LANCZOS)


def _make_generator(seed: int) -> torch.Generator:
    actual_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
    return torch.Generator().manual_seed(actual_seed)
