"""
AnimateDiff video generator.

Model layout expected on disk
──────────────────────────────
C:\\AI-Lab\\Models\\
  AnimateDiff-MotionAdapter\\   <- guoyww/animatediff-motion-adapter-v1-5-3
  AnimateDiff-BaseModel\\       <- any SD 1.5 checkpoint (e.g. SG161222/Realistic_Vision_V5.1_noVAE)

Both are standard HuggingFace diffusers-format directories.

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
from PIL import Image

from ..interfaces.video_generator import IVideoGenerator, GenerationRequest, GenerationResult
from ..config import ANIMATEDIFF_MOTION_ADAPTER_DIR, ANIMATEDIFF_BASE_MODEL_DIR


class AnimateDiffGenerator(IVideoGenerator):

    def __init__(self) -> None:
        self._pipe = None

    # ------------------------------------------------------------------ #
    # IVideoGenerator                                                      #
    # ------------------------------------------------------------------ #

    def generate(self, request: GenerationRequest) -> GenerationResult:
        pipe = self._get_pipeline()
        generator = self._make_generator(request.seed)

        output = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_frames=request.num_frames,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
        )

        return GenerationResult(frames=output.frames[0], fps=request.fps)

    def generate_seamless_loop(self, request: GenerationRequest) -> GenerationResult:
        """
        Generates a seamless ping-pong loop.

        Strategy: generate N frames, then append frames N-2 down to 1 in
        reverse (skip first and last to avoid duplication).  The result
        plays forward then backward, creating a perfect visual loop.

        Example: frames [0..15]  →  loop [0,1..14,15,14..1]  (30 frames)
        """
        result = self.generate(request)
        forward = result.frames
        reverse = list(reversed(forward[1:-1]))   # strip endpoints to avoid stutter
        loop_frames = forward + reverse

        return GenerationResult(frames=loop_frames, fps=result.fps)

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    # ------------------------------------------------------------------ #
    # Private                                                              #
    # ------------------------------------------------------------------ #

    def _get_pipeline(self):
        if self._pipe is not None:
            return self._pipe

        self._validate_model_dirs()

        # Import here so the rest of the codebase doesn't need diffusers
        # installed just to import this module.
        from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler

        adapter = MotionAdapter.from_pretrained(
            str(ANIMATEDIFF_MOTION_ADAPTER_DIR),
            torch_dtype=torch.float16,
            local_files_only=True,
        )

        # Support two layouts:
        #   - diffusers directory (has model_index.json) → from_pretrained
        #   - single .safetensors file                   → from_single_file
        single_file = self._find_single_file(ANIMATEDIFF_BASE_MODEL_DIR)
        if single_file:
            pipe = AnimateDiffPipeline.from_single_file(
                str(single_file),
                motion_adapter=adapter,
                torch_dtype=torch.float16,
            )
        else:
            pipe = AnimateDiffPipeline.from_pretrained(
                str(ANIMATEDIFF_BASE_MODEL_DIR),
                motion_adapter=adapter,
                torch_dtype=torch.float16,
                local_files_only=True,
            )

        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            beta_schedule="linear",
        )

        # Memory optimisations for 8 GB VRAM.
        # cpu_offload moves each sub-model to GPU only when it is needed,
        # so peak VRAM stays around 4-5 GB instead of ~8 GB.
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()

        self._pipe = pipe
        return self._pipe

    def _validate_model_dirs(self) -> None:
        missing = [
            d for d in (ANIMATEDIFF_MOTION_ADAPTER_DIR, ANIMATEDIFF_BASE_MODEL_DIR)
            if not d.exists()
        ]
        if missing:
            paths = "\n  ".join(str(p) for p in missing)
            raise FileNotFoundError(
                f"AnimateDiff model directories not found:\n  {paths}\n"
                "Run the download commands in LocalVideoStudio/DOWNLOAD.md first."
            )

    @staticmethod
    def _find_single_file(model_dir: Path) -> Path | None:
        """Returns a .safetensors path if the dir is a single-file checkpoint layout."""
        if (model_dir / "model_index.json").exists():
            return None  # diffusers format — use from_pretrained
        candidates = list(model_dir.glob("*.safetensors"))
        return candidates[0] if len(candidates) == 1 else None

    @staticmethod
    def _make_generator(seed: int) -> torch.Generator:
        actual_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        return torch.Generator().manual_seed(actual_seed)
