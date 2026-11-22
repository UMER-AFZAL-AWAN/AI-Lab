"""
RifeInterpolator — increases video frame rate using RIFE optical-flow interpolation.

Wraps ECCV2022-RIFE/inference_video.py via subprocess (the script runs argparse
at import time, so it cannot be safely imported directly).

VFR normalisation
─────────────────
Game recordings are typically VFR (variable frame rate). The container declares
r_frame_rate=59.94 but actual content runs at ~30 fps with duplicate frames
padding the gaps. RIFE's built-in SSIM > 0.996 "static-frame skip" fires on
those duplicates and consumes two input frames per output pair, causing the
output to advance at 2× speed — only the first half of the video appears.

Fix: normalise VFR → CFR at 30 fps with ffmpeg before passing the file to RIFE.
This removes all duplicate frames so the SSIM skip never fires.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

from .config import InterpolationConfig
from toolkit.logging import get_logger

_AI_LAB_ROOT    = Path(__file__).parent.parent.parent          # C:\AI-Lab
_RIFE_DIR       = _AI_LAB_ROOT / "ECCV2022-RIFE-main"
_VENV_PYTHON    = _AI_LAB_ROOT / ".venv" / "Scripts" / "python.exe"

log = get_logger(__name__)


class RifeInterpolator:
    """
    Increases frame rate using RIFE optical-flow interpolation.

    Default: exp=1 doubles the frame rate (30 fps → 60 fps).
    Use exp=2 for 4× (30 fps → 120 fps).
    """

    def interpolate(
        self,
        input_path: Path,
        output_path: Path,
        config: InterpolationConfig | None = None,
    ) -> Path:
        """
        Interpolate input_path and write result to output_path.
        Returns output_path on success, raises on failure.
        """
        cfg = config or InterpolationConfig()
        _validate_input(input_path)

        tmp_dir = Path(tempfile.mkdtemp(prefix="toolkit_rife_"))
        try:
            cfr_path = tmp_dir / "cfr_input.mp4"
            log.info("rife", "normalising VFR → CFR 30 fps")
            _to_cfr(input_path, cfr_path)

            cmd = _build_command(cfr_path, output_path, cfg)
            log.info("rife", f"running interpolation (exp={cfg.exp}, {2**cfg.exp}×)")
            _run(cmd)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        log.done("rife", output_path)
        return output_path


# ── Private helpers ──────────────────────────────────────────────────────────

def _validate_input(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"[rife] Input video not found: {path}")


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


def _build_command(input_path: Path, output_path: Path, cfg: InterpolationConfig) -> list[str]:
    script = _RIFE_DIR / "inference_video.py"
    cmd = [
        str(_VENV_PYTHON),
        str(script),
        "--video",  str(input_path),
        "--output", str(output_path),
        "--exp",    str(cfg.exp),
        "--scale",  str(cfg.scale),
    ]
    if cfg.fp16:
        cmd.append("--fp16")
    return cmd


def _run(cmd: list[str]) -> None:
    output_path = Path(cmd[cmd.index("--output") + 1])

    result = subprocess.run(
        cmd,
        cwd=str(_RIFE_DIR),   # RIFE loads train_log/ relative to its own directory
        capture_output=False,
        check=False,
    )

    # RIFE exits non-zero when the source has no audio stream, but the
    # interpolated video frames are written correctly. Treat the run as
    # successful if the output file was produced with content.
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(
            f"[rife] Interpolation produced no output (exit code {result.returncode}).\n"
            "Check the output above for details."
        )
