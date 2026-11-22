"""
toolkit.animatediff.generator — AnimateDiff video generation.

Model layout expected on disk
──────────────────────────────
C:\\AI-Lab\\Models\\
  AnimateDiff-MotionAdapter\\   ← guoyww/animatediff-motion-adapter-v1-5-3
  AnimateDiff-BaseModel\\       ← any SD 1.5 checkpoint (.safetensors or diffusers dir)

Memory budget (RTX 5060 Laptop, 8 GB VRAM)
───────────────────────────────────────────
  - All models loaded in fp16
  - enable_model_cpu_offload() keeps only the active sub-model on GPU
  - enable_vae_slicing() avoids OOM during decode
  - Max safe config: 16 frames @ 512×512, 25 steps
"""

import random
import torch
from pathlib import Path

from .config import GenerationRequest, GenerationResult
from toolkit.logging import get_logger

_AI_LAB_ROOT   = Path(__file__).parent.parent.parent
_MODELS_DIR    = _AI_LAB_ROOT / "Models"
_MOTION_ADAPTER_DIR = _MODELS_DIR / "AnimateDiff-MotionAdapter"
_BASE_MODEL_DIR     = _MODELS_DIR / "AnimateDiff-BaseModel"

log = get_logger(__name__)


class AnimateDiffGenerator:
    """
    Generates video clips from a text prompt using AnimateDiff.

    The pipeline is loaded lazily on first generate() call so importing
    the library does not require diffusers to be installed.
    """

    def __init__(self) -> None:
        self._pipe = None

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate video frames from a text prompt."""
        pipe      = self._get_pipeline()
        generator = _make_generator(request.seed)

        log.info("animatediff", f"generating {request.num_frames} frames @ {request.width}×{request.height}")

        output = pipe(
            prompt              = request.prompt,
            negative_prompt     = request.negative_prompt,
            num_frames          = request.num_frames,
            width               = request.width,
            height              = request.height,
            num_inference_steps = request.num_inference_steps,
            guidance_scale      = request.guidance_scale,
            generator           = generator,
        )

        log.done("animatediff.generate", f"{len(output.frames[0])} frames")
        return GenerationResult(frames=output.frames[0], fps=request.fps)

    def generate_seamless_loop(self, request: GenerationRequest) -> GenerationResult:
        """
        Generate a seamless ping-pong loop.

        Strategy: generate N frames, then append frames N-2 down to 1 in
        reverse (skip endpoints to avoid duplication). The result plays
        forward then backward, creating a perfect visual loop.

        Example: frames [0..15] → loop [0,1..14,15,14..1] (30 frames)
        """
        result  = self.generate(request)
        forward = result.frames
        reverse = list(reversed(forward[1:-1]))
        log.info("animatediff", f"ping-pong loop: {len(forward)} + {len(reverse)} = {len(forward)+len(reverse)} frames")
        return GenerationResult(frames=forward + reverse, fps=result.fps)

    # ── Private ─────────────────────────────────────────────────────────────

    def _get_pipeline(self):
        if self._pipe is not None:
            return self._pipe

        _validate_model_dirs()

        # Deferred import — the rest of the toolkit does not need diffusers
        from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler

        log.info("animatediff", "loading MotionAdapter")
        adapter = MotionAdapter.from_pretrained(
            str(_MOTION_ADAPTER_DIR),
            torch_dtype=torch.float16,
            local_files_only=True,
        )

        single_file = _find_single_file(_BASE_MODEL_DIR)
        if single_file:
            log.info("animatediff", f"loading base model from single file: {single_file.name}")
            pipe = AnimateDiffPipeline.from_single_file(
                str(single_file),
                motion_adapter=adapter,
                torch_dtype=torch.float16,
            )
        else:
            log.info("animatediff", "loading base model from diffusers directory")
            pipe = AnimateDiffPipeline.from_pretrained(
                str(_BASE_MODEL_DIR),
                motion_adapter=adapter,
                torch_dtype=torch.float16,
                local_files_only=True,
            )

        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            beta_schedule="linear",
        )

        # Memory optimisations for 8 GB VRAM.
        # cpu_offload moves each sub-model to GPU only when needed,
        # keeping peak VRAM at ~4-5 GB instead of ~8 GB.
        pipe.enable_model_cpu_offload()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()

        log.done("animatediff", "pipeline ready")
        self._pipe = pipe
        return self._pipe


# ── Module-level helpers ─────────────────────────────────────────────────────

def _validate_model_dirs() -> None:
    missing = [d for d in (_MOTION_ADAPTER_DIR, _BASE_MODEL_DIR) if not d.exists()]
    if missing:
        paths = "\n  ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"AnimateDiff model directories not found:\n  {paths}\n"
            "Run the download commands in LocalVideoStudio/DOWNLOAD.md first."
        )


def _find_single_file(model_dir: Path) -> Path | None:
    """Return a .safetensors path if the directory is a single-file checkpoint."""
    if (model_dir / "model_index.json").exists():
        return None
    candidates = list(model_dir.glob("*.safetensors"))
    return candidates[0] if len(candidates) == 1 else None


def _make_generator(seed: int) -> torch.Generator:
    actual_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
    return torch.Generator().manual_seed(actual_seed)
