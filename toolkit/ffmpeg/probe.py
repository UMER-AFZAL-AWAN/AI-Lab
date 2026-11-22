"""
toolkit.ffmpeg.probe — read video metadata via ffprobe.
"""

import subprocess
from pathlib import Path


def dimensions(path: Path) -> tuple[int, int]:
    """Return (width, height) of the first video stream."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    w, h = result.stdout.strip().split(",")
    return int(w), int(h)


def duration(path: Path) -> float:
    """Return video duration in seconds."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def fps(path: Path) -> float:
    """Return frame rate as a float (e.g. 29.97, 60.0)."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "csv=p=0",
            str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    num, den = result.stdout.strip().split("/")
    return int(num) / int(den)
