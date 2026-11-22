"""
toolkit.rife — RIFE frame interpolation library.

Wraps ECCV2022-RIFE via subprocess. Reusable by any project.

    from toolkit.rife import RifeInterpolator, InterpolationConfig
    result = RifeInterpolator().interpolate(input_path, output_path)
"""

from .interpolator import RifeInterpolator
from .config import InterpolationConfig

__all__ = ["RifeInterpolator", "InterpolationConfig"]
