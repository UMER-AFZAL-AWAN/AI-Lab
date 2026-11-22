"""
toolkit.character.animator — animate a person from a reference photo.

Pipeline
────────
1. Load reference photo → extract face embedding via InsightFace
2. Load IP-Adapter FaceID → injects that identity into the diffusion process
3. Load ControlNet OpenPose (optional) → drives body pose frame-by-frame
4. Run AnimateDiff with both conditionings → video frames
5. Return frames (caller saves to disk)

The result is a short video clip where the person in the reference photo
performs the described motion while their face/body stays consistent.

Memory budget (8 GB VRAM)
──────────────────────────
  - All models in fp16
  - enable_model_cpu_offload() keeps peak VRAM ~5-6 GB
  - Max safe config: 16 frames @ 512×512, 25 steps
  - ControlNet adds ~1 GB VRAM; disable if OOM

Models required on disk
────────────────────────
  C:\\AI-Lab\\Models\\IP-Adapter-FaceID\\        ← h94/IP-Adapter-FaceID
  C:\\AI-Lab\\Models\\ControlNet-OpenPose\\      ← lllyasviel/control_v11p_sd15_openpose
  C:\\AI-Lab\\Models\\AnimateDiff-MotionAdapter\\ (already present)
  C:\\AI-Lab\\Models\\AnimateDiff-BaseModel\\     (already present)

See toolkit/character/DOWNLOAD.md for download commands.
"""

import random
import torch
from pathlib import Path
from PIL import Image

from .config import AnimationRequest, AnimationResult
from toolkit.logging import get_logger

_AI_LAB_ROOT         = Path(__file__).parent.parent.parent
_MODELS_DIR          = _AI_LAB_ROOT / "Models"
_IP_ADAPTER_DIR      = _MODELS_DIR / "IP-Adapter-FaceID"
_CONTROLNET_DIR      = _MODELS_DIR / "ControlNet-OpenPose"
_MOTION_ADAPTER_DIR  = _MODELS_DIR / "AnimateDiff-MotionAdapter"
_BASE_MODEL_DIR      = _MODELS_DIR / "AnimateDiff-BaseModel"

log = get_logger(__name__)


class CharacterAnimator:
    """
    Animate a person from a reference photo.

    The pipeline is loaded lazily on the first animate() call.
    Call unload() to free VRAM when you are done.
    """

    def __init__(self) -> None:
        self._pipe      = None
        self._face_app  = None   # InsightFace app for face embedding extraction

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    def animate(self, request: AnimationRequest) -> AnimationResult:
        """
        Generate an animation of the person in request.reference_photo.
        Returns AnimationResult with a list of PIL Image frames.
        """
        _validate_request(request)
        pipe     = self._get_pipeline(use_controlnet=request.pose_image is not None)
        face_emb = self._extract_face_embedding(request.reference_photo)
        generator = _make_generator(request.seed)

        log.step(1, 2, f"Generating {request.num_frames} frames @ {request.width}×{request.height}")

        kwargs = dict(
            prompt              = request.prompt,
            negative_prompt     = request.negative_prompt,
            num_frames          = request.num_frames,
            width               = request.width,
            height              = request.height,
            num_inference_steps = request.num_inference_steps,
            guidance_scale      = request.guidance_scale,
            ip_adapter_image_embeds = [face_emb],
            generator           = generator,
        )

        # If a pose image is provided, tile it across all frames for ControlNet
        if request.pose_image is not None:
            pose_img    = Image.open(request.pose_image).convert("RGB")
            pose_frames = [pose_img] * request.num_frames
            kwargs["image"] = pose_frames
            kwargs["controlnet_conditioning_scale"] = request.controlnet_scale

        output = pipe(**kwargs)

        log.step(2, 2, "Done")
        log.done("character.animate", f"{len(output.frames[0])} frames")
        return AnimationResult(frames=output.frames[0], fps=request.fps)

    def animate_seamless_loop(self, request: AnimationRequest) -> AnimationResult:
        """Generate a seamless ping-pong loop of the animation."""
        result  = self.animate(request)
        forward = result.frames
        reverse = list(reversed(forward[1:-1]))
        log.info("character", f"ping-pong: {len(forward)} + {len(reverse)} frames")
        return AnimationResult(frames=forward + reverse, fps=result.fps)

    def unload(self) -> None:
        """Free VRAM by unloading the pipeline."""
        self._pipe     = None
        self._face_app = None
        torch.cuda.empty_cache()
        log.info("character", "pipeline unloaded, VRAM freed")

    # ── Private ──────────────────────────────────────────────────────────────

    def _extract_face_embedding(self, photo_path: Path) -> torch.Tensor:
        """Extract face identity embedding from a reference photo via InsightFace."""
        import numpy as np

        if self._face_app is None:
            log.info("character", "loading InsightFace (buffalo_l)")
            from insightface.app import FaceAnalysis
            self._face_app = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._face_app.prepare(ctx_id=0, det_size=(640, 640))

        img   = np.array(Image.open(photo_path).convert("RGB"))
        faces = self._face_app.get(img)
        if not faces:
            raise ValueError(
                f"[character] No face detected in reference photo: {photo_path}\n"
                "Use a clear front-facing photo with good lighting."
            )
        if len(faces) > 1:
            log.warn("character", f"{len(faces)} faces detected — using the largest one")
            faces = sorted(faces, key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)

        # IP-Adapter FaceID expects the embedding as a float16 tensor on CPU
        emb = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0).float()
        log.info("character", f"face embedding extracted (shape {emb.shape})")
        return emb

    def _get_pipeline(self, use_controlnet: bool = False):
        if self._pipe is not None:
            return self._pipe

        _validate_model_dirs(use_controlnet)

        from diffusers import (
            AnimateDiffPipeline,
            MotionAdapter,
            EulerDiscreteScheduler,
        )

        log.info("character", "loading MotionAdapter")
        adapter = MotionAdapter.from_pretrained(
            str(_MOTION_ADAPTER_DIR),
            torch_dtype=torch.float16,
            local_files_only=True,
        )

        # Base model — single .safetensors or diffusers directory
        single_file = _find_single_file(_BASE_MODEL_DIR)
        if single_file:
            log.info("character", f"loading base model: {single_file.name}")
            pipe = AnimateDiffPipeline.from_single_file(
                str(single_file),
                motion_adapter=adapter,
                torch_dtype=torch.float16,
            )
        else:
            log.info("character", "loading base model (diffusers directory)")
            pipe = AnimateDiffPipeline.from_pretrained(
                str(_BASE_MODEL_DIR),
                motion_adapter=adapter,
                torch_dtype=torch.float16,
                local_files_only=True,
            )

        # ControlNet for pose control (optional)
        if use_controlnet:
            from diffusers import ControlNetModel
            log.info("character", "loading ControlNet OpenPose")
            controlnet = ControlNetModel.from_pretrained(
                str(_CONTROLNET_DIR),
                torch_dtype=torch.float16,
                local_files_only=True,
            )
            pipe.controlnet = controlnet

        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            beta_schedule="linear",
        )

        # Load IP-Adapter FaceID weights
        log.info("character", "loading IP-Adapter FaceID")
        pipe.load_ip_adapter(
            str(_IP_ADAPTER_DIR),
            subfolder="",
            weight_name="ip-adapter-faceid_sd15.bin",
            image_encoder_folder=None,   # FaceID uses InsightFace, not CLIP
        )
        pipe.set_ip_adapter_scale(0.6)

        # Memory optimisations for 8 GB VRAM
        pipe.enable_model_cpu_offload()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()

        log.done("character", "pipeline ready")
        self._pipe = pipe
        return self._pipe


# ── Module-level helpers ─────────────────────────────────────────────────────

def _validate_request(req: AnimationRequest) -> None:
    if not req.reference_photo.exists():
        raise FileNotFoundError(f"[character] Reference photo not found: {req.reference_photo}")
    if req.pose_image is not None and not req.pose_image.exists():
        raise FileNotFoundError(f"[character] Pose image not found: {req.pose_image}")


def _validate_model_dirs(use_controlnet: bool) -> None:
    required = [_IP_ADAPTER_DIR, _MOTION_ADAPTER_DIR, _BASE_MODEL_DIR]
    if use_controlnet:
        required.append(_CONTROLNET_DIR)
    missing = [d for d in required if not d.exists()]
    if missing:
        paths = "\n  ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"Required model directories not found:\n  {paths}\n"
            "See toolkit/character/DOWNLOAD.md for download commands."
        )


def _find_single_file(model_dir: Path) -> Path | None:
    if (model_dir / "model_index.json").exists():
        return None
    candidates = list(model_dir.glob("*.safetensors"))
    return candidates[0] if len(candidates) == 1 else None


def _make_generator(seed: int) -> torch.Generator:
    actual_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
    return torch.Generator().manual_seed(actual_seed)
