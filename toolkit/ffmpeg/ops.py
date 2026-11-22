"""
toolkit.ffmpeg.ops — video operations via ffmpeg.

All functions are stateless and work on file paths.
Each raises subprocess.CalledProcessError on ffmpeg failure.
"""

import subprocess
from pathlib import Path

from . import probe
from .encode import nvenc
from toolkit.logging import get_logger

log = get_logger(__name__)


def split(source: Path, at: float, part1: Path, part2: Path) -> None:
    """Split source at `at` seconds into two files."""
    log.info("ffmpeg.split", f"cutting at {at}s → {part1.name} + {part2.name}")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(source),
         "-t", str(at), "-c", "copy", str(part1)],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(source),
         "-ss", str(at), "-c", "copy", str(part2)],
        check=True,
    )


def concat(file_list: Path, output: Path) -> None:
    """Concatenate video files listed in an ffmpeg concat manifest."""
    log.info("ffmpeg.concat", f"→ {output.name}")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(file_list),
            "-c", "copy",
            str(output),
        ],
        check=True,
    )


def freeze_tail(video: Path, output: Path, duration: float, fps: int) -> None:
    """
    Extend video by cloning its last frame for `duration` seconds.
    Uses tpad so the frozen frame is pixel-identical to the final decoded
    frame — no PNG extraction, no codec seam.
    """
    stop_frames = round(duration * fps)
    log.info("ffmpeg.freeze_tail", f"cloning last frame × {stop_frames} frames")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video),
            "-vf", f"tpad=stop_mode=clone:stop={stop_frames}",
        ] + nvenc() + [str(output)],
        check=True, capture_output=True,
    )


def freeze_head(video: Path, output: Path, duration: float, fps: int) -> None:
    """
    Prepend `duration` seconds of the video's first frame.
    Uses tpad so the frozen frame is pixel-identical to the first decoded frame.
    """
    start_frames = round(duration * fps)
    log.info("ffmpeg.freeze_head", f"cloning first frame × {start_frames} frames")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video),
            "-vf", f"tpad=start_mode=clone:start={start_frames}",
        ] + nvenc() + [str(output)],
        check=True, capture_output=True,
    )


def xfade_chain(
    clip_a: Path,
    clip_b: Path,
    clip_c: Path,
    output: Path,
    transition: float,
    fps: int,
    width: int,
    height: int,
) -> None:
    """
    Chain two xfades in one ffmpeg call:
      clip_a → dissolve → clip_b → dissolve → clip_c → output

    Two separate ffmpeg calls would duplicate clip_b frames; a single
    filter_complex call is the only correct approach.

    Offset maths:
      offset1 = dur_a - transition
      offset2 = dur_a + dur_b - 2 * transition
    """
    dur_a   = probe.duration(clip_a)
    dur_b   = probe.duration(clip_b)
    offset1 = max(0.0, dur_a - transition)
    offset2 = max(0.0, dur_a + dur_b - 2 * transition)

    log.info("ffmpeg.xfade_chain", f"3-clip chain (transition={transition}s)")

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
        ] + nvenc() + [str(output)],
        check=True, capture_output=True,
    )


def to_cfr(input_path: Path, output_path: Path, fps: int = 30) -> None:
    """Convert variable frame rate video to constant frame rate."""
    log.info("ffmpeg.to_cfr", f"normalising to {fps} fps CFR")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", f"fps={fps}",
            "-vsync", "cfr",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "copy",
            str(output_path),
        ],
        check=True,
        capture_output=False,
    )
