"""
toolkit.ffmpeg — FFmpeg/ffprobe utility library.

Low-level video I/O operations reusable by any project:
  probe      — read video metadata (duration, fps, dimensions)
  ops        — split, concat, freeze frames, crossfade
  encode     — codec argument builders (nvenc / libx264)

    from toolkit.ffmpeg import probe, ops
    w, h = probe.dimensions(path)
    ops.split(source, at=15.0, part1=p1, part2=p2)
"""

from . import probe
from . import ops
from . import encode

__all__ = ["probe", "ops", "encode"]
