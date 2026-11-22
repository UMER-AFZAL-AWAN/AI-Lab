"""
toolkit.ffmpeg.encode — codec argument builders.

Returns ffmpeg argument lists for the available hardware/software encoders.
The caller appends these to their ffmpeg command.

    from toolkit.ffmpeg.encode import best_codec
    cmd = ["ffmpeg", "-y", "-i", src] + best_codec() + [dst]
"""

import shutil


def nvenc() -> list[str]:
    """h264_nvenc — NVIDIA GPU encode (RTX series). Fast, good quality."""
    return [
        "-pix_fmt", "yuv420p",
        "-c:v",     "h264_nvenc",
        "-preset",  "p4",
        "-cq",      "23",
        "-b:v",     "0",
    ]


def libx264(crf: int = 18) -> list[str]:
    """libx264 — CPU software encode. Universal fallback."""
    return [
        "-pix_fmt", "yuv420p",
        "-c:v",     "libx264",
        "-preset",  "slow",
        "-crf",     str(crf),
    ]


def best_codec(prefer_nvenc: bool = True) -> list[str]:
    """Return nvenc if available and preferred, otherwise libx264."""
    if prefer_nvenc and shutil.which("ffmpeg"):
        return nvenc()
    return libx264()
