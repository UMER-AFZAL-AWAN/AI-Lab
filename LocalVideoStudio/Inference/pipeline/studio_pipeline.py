"""
StudioPipeline — high-level orchestrator for LocalVideoStudio.

This module contains NO model logic. It imports everything from the shared
toolkit libraries and coordinates them into end-to-end operations.

Two main operations
───────────────────
1. enhance_video()
     input.mp4
       → [toolkit.rife: 30 fps → 60 fps]
       → [toolkit.color: denoise, grade, letterbox, grain, vignette]
       → enhanced.mp4

2. inject_scene()
     input.mp4 + timestamp + text prompt
       → [split at timestamp]
       → [toolkit.animatediff: generate seamless loop]
       → [toolkit.color: grade loop to match game footage]
       → [toolkit.ffmpeg: freeze-frame anchors + chained xfade transitions]
       → final.mp4
"""

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from toolkit.rife        import RifeInterpolator, InterpolationConfig
from toolkit.color       import VideoEnhancer, EnhancementConfig
from toolkit.animatediff import AnimateDiffGenerator, GenerationRequest
from toolkit.ffmpeg      import ops as ffops, probe as ffprobe
from toolkit.logging     import get_logger

log = get_logger(__name__)


@dataclass
class InjectionPoint:
    timestamp_seconds:   float
    prompt:              str
    negative_prompt:     str   = (
        "blurry, low quality, worst quality, watermark, text, "
        "deformed, disfigured, ugly, bad anatomy"
    )
    loop_duration_sec:   int   = 4      # total loop length (ping-pong, generate half)
    width:               int   = 512
    height:              int   = 512
    num_inference_steps: int   = 25
    guidance_scale:      float = 7.5
    seed:                int   = -1


class StudioPipeline:

    def __init__(self) -> None:
        self._rife      = RifeInterpolator()
        self._enhancer  = VideoEnhancer()
        self._generator = AnimateDiffGenerator()

    # ── Public API ───────────────────────────────────────────────────────────

    def enhance_video(
        self,
        input_path:   Path,
        output_path:  Path,
        interpolation: InterpolationConfig | None = None,
        enhancement:   EnhancementConfig   | None = None,
    ) -> Path:
        """
        Full enhancement pipeline:
          input → frame interpolation → cinematic grading → output
        """
        interp_cfg = interpolation or InterpolationConfig(exp=1)
        enh_cfg    = enhancement   or EnhancementConfig()

        with _TempDir() as tmp:
            interpolated = tmp / "interpolated.mp4"
            log.step(1, 2, f"Interpolating frames ({2**interp_cfg.exp}×)…")
            self._rife.interpolate(input_path, interpolated, interp_cfg)

            log.step(2, 2, "Applying cinematic enhancement…")
            self._enhancer.enhance(interpolated, output_path, enh_cfg)

        log.done("enhance_video", output_path)
        return output_path

    def inject_scene(
        self,
        input_path:          Path,
        output_path:         Path,
        timestamp_seconds:   float,
        prompt:              str,
        negative_prompt:     str   = "blurry, low quality, watermark, text, deformed, ugly",
        loop_duration_sec:   int   = 4,
        transition_duration: float = 0.5,
        width:               int   = 512,
        height:              int   = 512,
        num_inference_steps: int   = 25,
        guidance_scale:      float = 7.5,
        seed:                int   = -1,
    ) -> Path:
        """
        Splits input_path at timestamp_seconds, generates an AnimateDiff
        ping-pong loop, grades it to match the footage, and assembles the
        final video with freeze-frame anchors and chained crossfade transitions.

        Structure:
          game plays → freezes on frame T (1 s still) → dissolves to AI scene →
          dissolves back to same frame T (1 s still) → game resumes from T.

        input_path should already be enhanced — this method does not re-enhance
        the game footage.
        """
        from diffusers.utils import export_to_video

        gen_fps         = 8
        half_frames     = max(8, (loop_duration_sec * gen_fps) // 2)
        video_fps       = 60
        freeze_hold_sec = 1.0

        with _TempDir() as tmp:
            part1 = tmp / "part1.mp4"
            part2 = tmp / "part2.mp4"

            log.step(1, 7, f"Splitting video at {timestamp_seconds}s…")
            ffops.split(input_path, timestamp_seconds, part1, part2)

            log.step(2, 7, "Probing input dimensions…")
            vid_w, vid_h = ffprobe.dimensions(input_path)

            log.step(3, 7, "Freezing last frame of part1…")
            part1_frozen = tmp / "part1_frozen.mp4"
            ffops.freeze_tail(part1, part1_frozen, freeze_hold_sec, video_fps)

            log.step(4, 7, "Freezing first frame of part2…")
            part2_frozen = tmp / "part2_frozen.mp4"
            ffops.freeze_head(part2, part2_frozen, freeze_hold_sec, video_fps)

            log.step(5, 7, "Generating AnimateDiff seamless loop…")
            request = GenerationRequest(
                prompt              = prompt,
                negative_prompt     = negative_prompt,
                num_frames          = half_frames,
                fps                 = gen_fps,
                width               = width,
                height              = height,
                num_inference_steps = num_inference_steps,
                guidance_scale      = guidance_scale,
                seed                = seed,
            )
            loop_raw = tmp / "loop_raw.mp4"
            result   = self._generator.generate_seamless_loop(request)
            export_to_video(result.frames, str(loop_raw), fps=result.fps)

            log.step(6, 7, "Grading loop to match game footage…")
            loop_graded = tmp / "loop_graded.mp4"
            self._enhancer.enhance(
                loop_raw,
                loop_graded,
                EnhancementConfig(auto_crop=False, cinematic_bars=False),
            )

            log.step(7, 7, "Chained xfade: part1_frozen → loop → part2_frozen…")
            ffops.xfade_chain(
                part1_frozen, loop_graded, part2_frozen,
                output_path,
                transition = transition_duration,
                fps        = video_fps,
                width      = vid_w,
                height     = vid_h,
            )

        log.done("inject_scene", output_path)
        return output_path


# ── Helpers ──────────────────────────────────────────────────────────────────

class _TempDir:
    """Context manager: creates a temp directory and cleans up on exit."""

    def __enter__(self) -> Path:
        self._dir = Path(tempfile.mkdtemp(prefix="lvs_"))
        return self._dir

    def __exit__(self, *_) -> None:
        shutil.rmtree(self._dir, ignore_errors=True)
