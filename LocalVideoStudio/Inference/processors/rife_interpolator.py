"""
Wraps the ECCV2022-RIFE inference_video.py script via subprocess.

Why subprocess instead of importing?
  inference_video.py runs argparse at module-import time (top-level code),
  so it cannot be imported safely.  Subprocess is the clean boundary.

VFR normalisation:
  Game recordings are typically VFR (variable frame rate) — the container
  declares r_frame_rate=59.94 but the actual content runs at ~30 fps with
  duplicate frames padding the gaps.  RIFE's inference_video.py has a
  built-in SSIM > 0.996 "static-frame skip" that fires on those duplicates
  and consumes two input frames per output pair, causing the output to
  advance through the source at 2× speed.  The result is that only the
  first half of the recording appears in the interpolated clip.

  Fix: normalise VFR → CFR at 30 fps with ffmpeg before handing the file
  to RIFE.  This removes all duplicate frames so the SSIM skip never fires.
"""

import shutil
import subprocess
import tempfile
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

        tmp_dir = Path(tempfile.mkdtemp(prefix="lvs_rife_"))
        try:
            cfr_path = tmp_dir / "cfr_input.mp4"
            print("  [rife] normalising VFR -> CFR 30 fps...")
            self._to_cfr(input_path, cfr_path)

            cmd = self._build_command(cfr_path, output_path, cfg)
            self._run(cmd)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return output_path

    # ------------------------------------------------------------------ #
    # Private                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_cfr(input_path: Path, cfr_path: Path) -> None:
        """Convert VFR input to constant 30 fps so RIFE receives no duplicate frames."""
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(input_path),
                "-vf", "fps=30",
                "-vsync", "cfr",
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-c:a", "copy",
                str(cfr_path),
            ],
            check=True,
            capture_output=False,
        )

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
