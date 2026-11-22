"""
toolkit.color — cinematic color grading and video enhancement library.

    from toolkit.color import VideoEnhancer, EnhancementConfig
    VideoEnhancer().enhance(input_path, output_path)
"""

from .enhancer import VideoEnhancer
from .config import EnhancementConfig

__all__ = ["VideoEnhancer", "EnhancementConfig"]
