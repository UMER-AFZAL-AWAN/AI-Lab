"""
toolkit.animatediff — AnimateDiff text-to-video generation library.

    from toolkit.animatediff import AnimateDiffGenerator, GenerationRequest
    gen = AnimateDiffGenerator()
    result = gen.generate(GenerationRequest(prompt="a dragon flying over a castle"))
"""

from .generator import AnimateDiffGenerator
from .config import GenerationRequest, GenerationResult

__all__ = ["AnimateDiffGenerator", "GenerationRequest", "GenerationResult"]
