"""
Cinematic video enhancement via ffmpeg filters.

Pipeline (all steps configurable, all optional):
  0. Auto-crop     – cropdetect removes the game's native letterbox bars
  1. Scale/pad     – restores original canvas; natural black bars replace native ones
  2. Denoise       – hqdn3d: removes compression noise
  3. Sharpen       – unsharp mask: recovers edge detail after denoise
  4. S-curve       – deepen shadows, punch midtones, lift highlights (contrast)
  5. Orange-teal   – Hollywood color split: warm highlights/mids, cool shadows
  6. EQ            – saturation +30%, micro-contrast, slight underexposure
  7. Vignette      – soft PI/8 edge darkening
  8. Film grain    – subtle noise texture to break the digital look
  9. Letterbox     – 2.39:1 bars ONLY when auto-crop found no native bars

GPU encode:  h264_nvenc + CUDA hwaccel.
CPU fallback: libx264.
"""

import re
import subprocess
import shutil
from collections import Counter
from pathlib import Path

from ..interfaces.video_processor import IVideoProcessor, ProcessorConfig, EnhancementConfig


class VideoEnhancer(IVideoProcessor):

    def process(
        self,
        input_path: Path,
        output_path: Path,
        config: ProcessorConfig = None,
    ) -> Path:
        cfg = config if isinstance(config, EnhancementConfig) else EnhancementConfig()
        self._validate(input_path)

        orig_w, orig_h = self._get_dimensions(input_path)
        crop_filter = self._detect_crop(input_path) if cfg.auto_crop else None

        vf = self._build_filter_chain(cfg, crop_filter, orig_w, orig_h)
        codec_args = self._build_codec_args(cfg)
        cmd = self._build_command(input_path, output_path, vf, codec_args, cfg)
        self._run(cmd)
        return output_path

    # ------------------------------------------------------------------ #
    # Video probing                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_dimensions(input_path: Path) -> tuple[int, int]:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=p=0", str(input_path)],
            capture_output=True, text=True,
        )
        w, h = result.stdout.strip().split(",")
        return int(w), int(h)

    @staticmethod
    def _detect_crop(input_path: Path) -> str | None:
        """
        Probe a 5-second window for native black bars via cropdetect.
        Returns 'crop=w:h:x:y' when bars >=10px are found, else None.
        Bars this small are almost certainly game UI chrome, not content.
        """
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-ss", "3",
             "-i", str(input_path), "-t", "5",
             "-vf", "cropdetect=limit=24:round=16:skip=0",
             "-f", "null", "-"],
            capture_output=True, text=True,
        )
        matches = re.findall(r"crop=(\d+:\d+:\d+:\d+)", result.stderr)
        if not matches:
            return None
        crop_str = Counter(matches).most_common(1)[0][0]
        w, h, x, y = map(int, crop_str.split(":"))
        if x >= 10 or y >= 10:
            print(f"  [crop] detected native bars -> {crop_str}")
            return f"crop={w}:{h}:{x}:{y}"
        return None

    # ------------------------------------------------------------------ #
    # Filter chain                                                         #
    # ------------------------------------------------------------------ #

    def _build_filter_chain(
        self,
        cfg: EnhancementConfig,
        crop_filter: str | None,
        orig_w: int,
        orig_h: int,
    ) -> str:
        filters: list[str] = []
        has_native_bars = crop_filter is not None

        # Remove native bars then re-letterbox content to original canvas size.
        # The pad step naturally produces clean black bars that replace the game's own.
        if has_native_bars:
            filters.append(crop_filter)
            filters.append(
                f"scale={orig_w}:{orig_h}:force_original_aspect_ratio=decrease"
            )
            filters.append(
                f"pad={orig_w}:{orig_h}:(ow-iw)/2:(oh-ih)/2:black"
            )

        if cfg.denoise:
            filters.append("hqdn3d=1.5:1.5:6:6")

        if cfg.sharpening:
            filters.append("unsharp=5:5:0.8:3:3:0.4")

        if cfg.cinematic_grade:
            # Stage 1 — luminance S-curve: crush shadows, punch midtones, lift highlights
            filters.append(
                "curves=all='0/0 0.08/0.03 0.40/0.35 0.62/0.68 0.92/0.97 1/1'"
            )
            # Stage 2 — orange-teal split:
            #   Red:   lift shadows slightly, push midtone warmth  (+orange feel)
            #   Green: pull midtones down gently                   (reduces green cast)
            #   Blue:  lift shadows (teal darks), pull highlights  (warm tops)
            filters.append(
                "curves="
                "r='0/0.02 0.25/0.30 0.5/0.55 0.75/0.78 1/1.0':"
                "g='0/0.0  0.25/0.22 0.5/0.46 0.75/0.72 1/0.92':"
                "b='0/0.08 0.25/0.28 0.5/0.50 0.75/0.66 1/0.86'"
            )
            # Stage 3 — final eq: +30% saturation, micro-contrast, slight underexposure
            filters.append("eq=contrast=1.08:brightness=-0.02:saturation=1.30:gamma=0.97")

        if cfg.vignette:
            # PI/8 ≈ 22.5 degrees — much softer than the old PI/5 (36 degrees)
            filters.append("vignette=PI/8")

        if cfg.film_grain:
            filters.append("noise=alls=8:allf=t+u")

        # Only add bars when the source had no native bars to begin with.
        # If we already cropped + padded, the pad supplied clean black bars.
        if cfg.cinematic_bars and not has_native_bars:
            filters.append("drawbox=x=0:y=0:w=iw:h=ih*0.114:color=black@1.0:t=fill")
            filters.append(
                "drawbox=x=0:y=ih*0.886:w=iw:h=ih*0.114:color=black@1.0:t=fill"
            )

        return ",".join(filters) if filters else "null"

    # ------------------------------------------------------------------ #
    # Codec + command                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_codec_args(cfg: EnhancementConfig) -> list[str]:
        if cfg.nvenc and shutil.which("ffmpeg"):
            return [
                "-pix_fmt", "yuv420p",
                "-c:v", "h264_nvenc",
                "-preset", "p4",
                "-cq", "23",
                "-b:v", "0",
            ]
        return [
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
        ]

    @staticmethod
    def _build_command(
        input_path: Path,
        output_path: Path,
        vf: str,
        codec_args: list[str],
        cfg: EnhancementConfig,
    ) -> list[str]:
        cmd = [
            "ffmpeg", "-y",
            "-hwaccel", "cuda",
            "-i", str(input_path),
            "-vf", vf,
        ]
        cmd.extend(codec_args)
        if cfg.output_fps is not None:
            cmd.extend(["-r", str(cfg.output_fps)])
        cmd.extend(["-c:a", "copy"])
        cmd.append(str(output_path))
        return cmd

    @staticmethod
    def _run(cmd: list[str]) -> None:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg enhancement failed (exit code {result.returncode})."
            )

    @staticmethod
    def _validate(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Input video not found: {path}")
