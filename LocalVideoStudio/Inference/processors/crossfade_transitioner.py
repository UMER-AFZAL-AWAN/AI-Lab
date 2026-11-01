"""
CrossfadeTransitioner — dissolves two video clips together using ffmpeg xfade.

Single responsibility: take clip A and clip B, produce one output where
B is dissolved into A over a configurable duration.  Knows nothing about
AnimateDiff, enhancement, or any other pipeline step.
"""

import subprocess
from pathlib import Path

from ..interfaces.video_processor import IVideoProcessor, ProcessorConfig, CrossfadeConfig


class CrossfadeTransitioner(IVideoProcessor):

    def process(
        self,
        input_path: Path,       # video A (the "from" clip)
        output_path: Path,
        config: ProcessorConfig = None,
    ) -> Path:
        cfg = config if isinstance(config, CrossfadeConfig) else CrossfadeConfig()
        if cfg.second_clip is None:
            raise ValueError("CrossfadeConfig.second_clip must be set")

        a_duration = self._get_duration(input_path)
        offset = max(0.0, a_duration - cfg.transition_duration)
        a_w, a_h = self._get_dimensions(input_path)

        # Normalise fps and scale so xfade receives identical streams.
        # xfade requires both inputs to have the same resolution and frame rate.
        filter_complex = (
            f"[0:v]fps={cfg.fps}[a];"
            f"[1:v]fps={cfg.fps},scale={a_w}:{a_h}:force_original_aspect_ratio=disable[b];"
            f"[a][b]xfade=transition=fade"
            f":duration={cfg.transition_duration:.3f}:offset={offset:.3f}"
        )

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(input_path),
                "-i", str(cfg.second_clip),
                "-filter_complex", filter_complex,
                "-c:v", "h264_nvenc",
                "-pix_fmt", "yuv420p",
                "-preset", "p4",
                "-cq", "23",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )
        return output_path

    @staticmethod
    def _get_duration(path: Path) -> float:
        """Returns video duration in seconds via ffprobe."""
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())

    @staticmethod
    def _get_dimensions(path: Path) -> tuple[int, int]:
        """Returns (width, height) of the first video stream via ffprobe."""
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        w, h = result.stdout.strip().split(",")
        return int(w), int(h)
