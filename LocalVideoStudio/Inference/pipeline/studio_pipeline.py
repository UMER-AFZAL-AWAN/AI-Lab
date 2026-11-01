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
from ..interfaces.video_processor import InterpolationConfig, EnhancementConfig, CrossfadeConfig
from ..generators.animatediff_generator import AnimateDiffGenerator
from ..processors.rife_interpolator import RifeInterpolator
from ..processors.video_enhancer import VideoEnhancer
from ..processors.crossfade_transitioner import CrossfadeTransitioner
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
        self._generator    = AnimateDiffGenerator()
        self._interpolator = RifeInterpolator()
        self._enhancer     = VideoEnhancer()
        self._crossfader   = CrossfadeTransitioner()

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

    def inject_scene(
        self,
        input_path: Path,
        output_path: Path,
        timestamp_seconds: float,
        prompt: str,
        negative_prompt: str = "blurry, low quality, watermark, text, deformed, ugly",
        loop_duration_sec: int = 4,
        transition_duration: float = 0.5,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: int = -1,
    ) -> Path:
        """
        Splits input_path at timestamp_seconds, generates an AnimateDiff ping-pong
        loop from prompt, grades it, and assembles the final video with freeze-frame
        anchors and chained crossfade transitions.

        Structure:
          game plays -> freezes on frame T -> dissolves to AI scene ->
          dissolves back to same frame T -> game resumes from T.

        input_path must already be enhanced — this method does not re-enhance
        the game footage.
        """
        from diffusers.utils import export_to_video

        gen_fps = 8
        half_frames = max(8, (loop_duration_sec * gen_fps) // 2)
        video_fps = 60

        with _TempDir() as tmp:
            part1 = tmp / "part1.mp4"
            part2 = tmp / "part2.mp4"

            print(f"[1/8] Splitting video at {timestamp_seconds}s...")
            self._split_video(input_path, timestamp_seconds, part1, part2)

            print("[2/8] Probing input dimensions...")
            vid_w, vid_h = CrossfadeTransitioner._get_dimensions(input_path)

            print("[3/8] Extracting freeze frame from part1...")
            frame_png = tmp / "freeze_frame.png"
            self._extract_frame(part1, frame_png)

            print("[4/8] Creating freeze-hold clip (1s still)...")
            freeze_hold = tmp / "freeze_hold.mp4"
            self._create_freeze_hold(frame_png, freeze_hold,
                                     duration=1.0, fps=video_fps,
                                     width=vid_w, height=vid_h)

            print("[5/8] Building part1_frozen and part2_frozen...")
            part1_frozen = tmp / "part1_frozen.mp4"
            part2_frozen = tmp / "part2_frozen.mp4"

            c1 = tmp / "c1.txt"
            c1.write_text(
                f"file '{part1.as_posix()}'\nfile '{freeze_hold.as_posix()}'\n",
                encoding="utf-8",
            )
            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                 "-i", str(c1), "-c", "copy", str(part1_frozen)],
                check=True, capture_output=True,
            )

            c2 = tmp / "c2.txt"
            c2.write_text(
                f"file '{freeze_hold.as_posix()}'\nfile '{part2.as_posix()}'\n",
                encoding="utf-8",
            )
            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                 "-i", str(c2), "-c", "copy", str(part2_frozen)],
                check=True, capture_output=True,
            )

            print("[6/8] Generating AnimateDiff seamless loop...")
            request = GenerationRequest(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=half_frames,
                fps=gen_fps,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )
            loop_raw = tmp / "loop_raw.mp4"
            result = self._generator.generate_seamless_loop(request)
            export_to_video(result.frames, str(loop_raw), fps=result.fps)

            print("[7/8] Grading loop to match game footage...")
            loop_graded = tmp / "loop_graded.mp4"
            self._enhancer.process(
                loop_raw,
                loop_graded,
                EnhancementConfig(auto_crop=False, cinematic_bars=False),
            )

            print("[8/8] Chained xfade: part1_frozen -> loop -> part2_frozen...")
            self._chained_xfade(
                part1_frozen, loop_graded, part2_frozen,
                output_path,
                transition=transition_duration,
                fps=video_fps,
                width=vid_w,
                height=vid_h,
            )

        print(f"inject_scene complete -> {output_path}")
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

        Deprecated: use inject_scene() instead, which accepts an already-enhanced
        video and adds smooth crossfade transitions at the cut points.
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
    def _create_freeze_hold(
        frame_png: Path,
        output_path: Path,
        duration: float,
        fps: int,
        width: int,
        height: int,
    ) -> None:
        """Turns a single PNG into a short video clip of the given duration."""
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-loop", "1",
                "-framerate", str(fps),
                "-i", str(frame_png),
                "-vf", f"scale={width}:{height}:force_original_aspect_ratio=disable",
                "-t", str(duration),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "fast",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )

    @staticmethod
    def _chained_xfade(
        clip_a: Path,
        clip_b: Path,
        clip_c: Path,
        output_path: Path,
        transition: float,
        fps: int,
        width: int,
        height: int,
    ) -> None:
        """
        Chains two xfades in a single ffmpeg call:
          clip_a -xfade-> clip_b -xfade-> clip_c -> output_path

        Two separate ffmpeg calls would duplicate clip_b; one filter_complex
        call is the only correct approach.

        Offset maths:
          offset1 = dur_a - transition
          offset2 = dur_a + dur_b - 2 * transition
        """
        def _dur(p: Path) -> float:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(p)],
                capture_output=True, text=True, check=True,
            )
            return float(r.stdout.strip())

        dur_a = _dur(clip_a)
        dur_b = _dur(clip_b)
        offset1 = max(0.0, dur_a - transition)
        offset2 = max(0.0, dur_a + dur_b - 2 * transition)

        fc = (
            f"[0:v]fps={fps},scale={width}:{height}:force_original_aspect_ratio=disable[va];"
            f"[1:v]fps={fps},scale={width}:{height}:force_original_aspect_ratio=disable[vb];"
            f"[2:v]fps={fps},scale={width}:{height}:force_original_aspect_ratio=disable[vc];"
            f"[va][vb]xfade=transition=fade:duration={transition:.3f}:offset={offset1:.3f}[x1];"
            f"[x1][vc]xfade=transition=fade:duration={transition:.3f}:offset={offset2:.3f}[out]"
        )

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(clip_a),
                "-i", str(clip_b),
                "-i", str(clip_c),
                "-filter_complex", fc,
                "-map", "[out]",
                "-c:v", "h264_nvenc",
                "-pix_fmt", "yuv420p",
                "-preset", "p4",
                "-cq", "23",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )

    @staticmethod
    def _extract_frame(video_path: Path, out_png: Path) -> None:
        """Extracts the last frame of video_path as a PNG."""
        subprocess.run(
            ["ffmpeg", "-y", "-sseof", "-1",
             "-i", str(video_path),
             "-frames:v", "1", str(out_png)],
            check=True, capture_output=True,
        )

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
