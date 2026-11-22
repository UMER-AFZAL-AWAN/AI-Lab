"""
toolkit.character — animate a person from a photo.

Local Higgsfield substitute: takes a reference photo of a person and a pose
(or a motion description) and generates a short video clip where that person
performs the motion while keeping their face and body consistent.

Stack
─────
  IP-Adapter FaceID   — injects identity from a reference photo into the pipeline
  ControlNet OpenPose — drives the body pose frame-by-frame
  AnimateDiff         — base video generation backbone

    from toolkit.character import CharacterAnimator, AnimationRequest
    result = CharacterAnimator().animate(AnimationRequest(
        reference_photo = Path("person.jpg"),
        prompt          = "person walking confidently, cinematic",
        num_frames      = 16,
    ))

Models required (see toolkit/character/DOWNLOAD.md):
  Models/IP-Adapter-FaceID/        ← h94/IP-Adapter-FaceID
  Models/ControlNet-OpenPose/      ← lllyasviel/control_v11p_sd15_openpose
"""

from .animator import CharacterAnimator
from .config   import AnimationRequest, AnimationResult

__all__ = ["CharacterAnimator", "AnimationRequest", "AnimationResult"]
