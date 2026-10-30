"""
StudioPipeline — the high-level orchestrator.

Two main operations
───────────────────
1. enhance_video()
   game_recording.mp4
     → [RIFE: 30fps → 60fps]
     → [ffmpeg: denoise, color grade, letterbox, grain, vignette]
     → enhanced_output.mp4

2. generate_and_inject_loop()
   game_recording.mp4  +  timestamp  +  text prompt
     → [split at timestamp]
     → [AnimateDiff: generate seamless loop clip]
     → [RIFE: interpolate loop to 60fps]
     → [ffmpeg: enhance loop to match main video grade]
     → [ffmpeg concat: part1 + loop + part2]
     → final_output.mp4

Usage example
─────────────
    pipeline = StudioPipeline()

    # Enhance the whole video
    pipeline.enhance_video(
        Path("C:/AI-Lab/InputVideo/input.mp4"),
        Path("C:/AI-Lab/OutputVideo/enhanced.mp4"),
    )

    # Inject a cinematic loop at the 15-second mark
    pipeline.generate_and_inject_loop(
        input_path  = Path("C:/AI-Lab/InputVideo/input.mp4"),
        output_path = Path("C:/AI-Lab/OutputVideo/with_loop.mp4"),
        injection   = InjectionPoint(
            timestamp_seconds = 15.0,
            prompt            = "A knight standing at a castle gate, dramatic lighting, cinematic",
            loop_duration_sec = 4,
        ),
    )
"""

import subprocess
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..interfaces.video_generator import GenerationRequest
from ..interfaces.video_processor import InterpolationConfig, EnhancementConfig
from ..generators.animatediff_generator import AnimateDiffGenerator
from ..processors.rife_interpolator import RifeInterpolator
from ..processors.video_enhancer import VideoEnhancer
from ..config import OUTPUT_VIDEO_DIR


@dataclass
class InjectionPoint:
    timestamp_seconds: float
    prompt: str
    negative_prompt: str = (
        "blurry, low quality, worst quality, watermark, text, "
        "deformed, disfigured, ugly, bad anatomy"
    )
    loop_duration_sec: int = 4    # total loop length in seconds (ping-pong, so generate half)
    width: int = 512
    height: int = 512
    num_inference_steps: int = 25
    guidance_scale: float = 7.5
    seed: int = -1


class StudioPipeline:

    def __init__(self) -> None:
        self._generator   = AnimateDiffGenerator()
        self._interpolator = RifeInterpolator()
        self._enhancer    = VideoEnhancer()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def enhance_video(
        self,
        input_path: Path,
        output_path: Path,
        interpolation: InterpolationConfig | None = None,
        enhancement: EnhancementConfig | None = None,
    ) -> Path:
        """
        Full enhancement pipeline:
          input → frame interpolation → cinematic grading → output
        """
        interp_cfg = interpolation or InterpolationConfig(exp=1)
        enh_cfg    = enhancement   or EnhancementConfig()

        with _TempDir() as tmp:
            interpolated = tmp / "interpolated.mp4"
            print(f"[1/2] Interpolating frames ({2**interp_cfg.exp}×)…")
            self._interpolator.process(input_path, interpolated, interp_cfg)

            print("[2/2] Applying cinematic enhancement…")
            self._enhancer.process(interpolated, output_path, enh_cfg)

        print(f"Enhancement complete -> {output_path}")
        return output_path

    def generate_and_inject_loop(
        self,
        input_path: Path,
        output_path: Path,
        injection: InjectionPoint,
        enhancement: EnhancementConfig | None = None,
    ) -> Path:
        """
        Generates a seamless cinematic loop and splices it into input_path
        at the requested timestamp.

        The loop transitions are abrupt cuts.  For crossfades, call
        add_crossfade_transitions() on the result (planned in next version).
        """
        enh_cfg = enhancement or EnhancementConfig()

        with _TempDir() as tmp:
            # 1. Split the source video at the injection point
            print(f"[1/4] Splitting video at {injection.timestamp_seconds}s…")
            part1 = tmp / "part1.mp4"
            part2 = tmp / "part2.mp4"
            self._split_video(input_path, injection.timestamp_seconds, part1, part2)

            # 2. Generate the loop clip
            print("[2/4] Generating cinematic loop (this takes a while)…")
            raw_loop = tmp / "loop_raw.mp4"
            self._generate_loop_clip(injection, raw_loop)

            # 3. Enhance all three segments consistently
            print("[3/4] Enhancing all segments…")
            part1_enh = tmp / "part1_enh.mp4"
            part2_enh = tmp / "part2_enh.mp4"
            loop_enh  = tmp / "loop_enh.mp4"
            self._enhancer.process(part1, part1_enh, enh_cfg)
            self._enhancer.process(part2, part2_enh, enh_cfg)
            self._enhancer.process(raw_loop, loop_enh, enh_cfg)

            # 4. Concatenate: part1 → loop → part2
            print("[4/4] Assembling final video…")
            concat_list = tmp / "concat.txt"
            concat_list.write_text(
                f"file '{part1_enh}'\nfile '{loop_enh}'\nfile '{part2_enh}'\n",
                encoding="utf-8",
            )
            self._concat_videos(concat_list, output_path)

        print(f"Injection complete -> {output_path}")
        return output_path

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _generate_loop_clip(self, injection: InjectionPoint, output_path: Path) -> None:
        from diffusers.utils import export_to_video

        # AnimateDiff generates at 8 fps; we generate half the desired frames
        # and ping-pong to reach the target duration.
        fps = 8
        half_frames = max(8, (injection.loop_duration_sec * fps) // 2)

        request = GenerationRequest(
            prompt             = injection.prompt,
            negative_prompt    = injection.negative_prompt,
            num_frames         = half_frames,
            fps                = fps,
            width              = injection.width,
            height             = injection.height,
            num_inference_steps= injection.num_inference_steps,
            guidance_scale     = injection.guidance_scale,
            seed               = injection.seed,
        )

        result = self._generator.generate_seamless_loop(request)
        export_to_video(result.frames, str(output_path), fps=result.fps)

    @staticmethod
    def _split_video(source: Path, at_seconds: float, part1: Path, part2: Path) -> None:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(source),
             "-t", str(at_seconds), "-c", "copy", str(part1)],
            check=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(source),
             "-ss", str(at_seconds), "-c", "copy", str(part2)],
            check=True,
        )

    @staticmethod
    def _concat_videos(concat_list: Path, output_path: Path) -> None:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list),
                "-c", "copy",
                str(output_path),
            ],
            check=True,
        )


class _TempDir:
    """Context manager that creates a temporary directory and cleans up after."""

    def __enter__(self) -> Path:
        self._dir = Path(tempfile.mkdtemp(prefix="lvs_"))
        return self._dir

    def __exit__(self, *_) -> None:
        shutil.rmtree(self._dir, ignore_errors=True)
