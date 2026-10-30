"""
Wraps the ECCV2022-RIFE inference_video.py script via subprocess.

Why subprocess instead of importing?
  inference_video.py runs argparse at module-import time (top-level code),
  so it cannot be imported safely.  Subprocess is the clean boundary.
"""

import subprocess
import sys
from pathlib import Path

from ..interfaces.video_processor import IVideoProcessor, ProcessorConfig, InterpolationConfig
from ..config import RIFE_SCRIPT_DIR, VENV_PYTHON


class RifeInterpolator(IVideoProcessor):
    """
    Increases frame rate using RIFE optical-flow interpolation.

    Default: exp=1 doubles the frame rate (30 fps → 60 fps).
    Use exp=2 for 4× (30 fps → 120 fps).
    """

    def process(
        self,
        input_path: Path,
        output_path: Path,
        config: ProcessorConfig = None,
    ) -> Path:
        cfg = config if isinstance(config, InterpolationConfig) else InterpolationConfig()
        self._validate(input_path)

        cmd = self._build_command(input_path, output_path, cfg)
        self._run(cmd)

        return output_path

    # ------------------------------------------------------------------ #
    # Private                                                              #
    # ------------------------------------------------------------------ #

    def _build_command(
        self,
        input_path: Path,
        output_path: Path,
        cfg: InterpolationConfig,
    ) -> list[str]:
        script = RIFE_SCRIPT_DIR / "inference_video.py"
        cmd = [
            str(VENV_PYTHON),
            str(script),
            "--video", str(input_path),
            "--output", str(output_path),
            "--exp", str(cfg.exp),
            "--scale", str(cfg.scale),
        ]
        if cfg.fp16:
            cmd.append("--fp16")
        return cmd

    @staticmethod
    def _run(cmd: list[str]) -> None:
        # The output path is the second argument after --output
        output_path = Path(cmd[cmd.index("--output") + 1])

        result = subprocess.run(
            cmd,
            cwd=str(RIFE_SCRIPT_DIR),  # RIFE loads train_log/ relative to its own dir
            capture_output=False,       # let stdout/stderr stream to the terminal
            check=False,
        )

        # RIFE exits non-zero when the source has no audio stream (audio transfer
        # sub-command fails), but the interpolated video frames are written correctly.
        # Treat the run as successful if the output file was produced with content.
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(
                f"RIFE interpolation produced no output (exit code {result.returncode}).\n"
                "Check the output above for details."
            )

    @staticmethod
    def _validate(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Input video not found: {path}")
