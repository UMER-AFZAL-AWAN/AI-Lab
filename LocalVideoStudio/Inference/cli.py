"""
LocalVideoStudio — command-line interface.

Examples
────────
# Enhance a game recording (frame interpolation + cinematic grade)
python -m Inference.cli enhance \\
    --input  "C:/AI-Lab/InputVideo/input.mp4" \\
    --output "C:/AI-Lab/OutputVideo/enhanced.mp4"

# Generate an AI scene and inject it at 15 seconds
python -m Inference.cli inject \\
    --input     "C:/AI-Lab/InputVideo/enhanced.mp4" \\
    --output    "C:/AI-Lab/OutputVideo/with_scene.mp4" \\
    --timestamp 15.0 \\
    --prompt    "A knight at a castle gate, dramatic lighting, cinematic, 4k"

# Animate a person from a reference photo (Higgsfield-style)
python -m Inference.cli character \\
    --photo  "C:/AI-Lab/InputVideo/person.jpg" \\
    --output "C:/AI-Lab/OutputVideo/animated.mp4" \\
    --prompt "person walking confidently, cinematic lighting"

Run from C:\\AI-Lab with the project venv active.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # C:\AI-Lab on sys.path

from toolkit.logging import configure_logging
from toolkit.rife.config      import InterpolationConfig
from toolkit.color.config     import EnhancementConfig
from toolkit.character.config import AnimationRequest
from Inference.pipeline.studio_pipeline import StudioPipeline


def cmd_enhance(args: argparse.Namespace) -> None:
    pipeline = StudioPipeline()
    pipeline.enhance_video(
        input_path   = Path(args.input),
        output_path  = Path(args.output),
        interpolation= InterpolationConfig(exp=args.exp, fp16=not args.no_fp16),
        enhancement  = EnhancementConfig(
            auto_crop       = not args.no_autocrop,
            cinematic_grade = not args.no_grade,
            cinematic_bars  = not args.no_bars,
            denoise         = not args.no_denoise,
        ),
    )


def cmd_character(args: argparse.Namespace) -> None:
    from diffusers.utils import export_to_video
    from toolkit.character import CharacterAnimator

    request = AnimationRequest(
        reference_photo     = Path(args.photo),
        prompt              = args.prompt,
        num_frames          = args.frames,
        fps                 = args.fps,
        width               = args.width,
        height              = args.height,
        num_inference_steps = args.steps,
        motion_bucket_id    = args.motion,
        noise_aug_strength  = args.noise_aug,
        seed                = args.seed,
    )
    result = CharacterAnimator().animate_seamless_loop(request) if args.loop \
             else CharacterAnimator().animate(request)
    export_to_video(result.frames, args.output, fps=result.fps)


def cmd_inject(args: argparse.Namespace) -> None:
    pipeline = StudioPipeline()
    pipeline.inject_scene(
        input_path          = Path(args.input),
        output_path         = Path(args.output),
        timestamp_seconds   = args.timestamp,
        prompt              = args.prompt,
        negative_prompt     = args.negative_prompt,
        loop_duration_sec   = args.duration,
        transition_duration = args.transition,
        width               = args.width,
        height              = args.height,
        num_inference_steps = args.steps,
        guidance_scale      = args.cfg,
        seed                = args.seed,
    )


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="lvs",
        description="LocalVideoStudio — local AI video production toolkit",
    )
    root.add_argument("--verbose", action="store_true", help="Debug-level logging")
    sub = root.add_subparsers(dest="command", required=True)

    # ── character ────────────────────────────────────────────────────────── #
    p_chr = sub.add_parser("character", help="Animate a person from a reference photo")
    p_chr.add_argument("--photo",        required=True, help="Reference photo — becomes frame 0")
    p_chr.add_argument("--output",       required=True)
    p_chr.add_argument("--prompt",       default="",    help="Unused by SVD; kept for future use")
    p_chr.add_argument("--frames",       type=int,   default=25)
    p_chr.add_argument("--fps",          type=int,   default=8)
    p_chr.add_argument("--width",        type=int,   default=1024)
    p_chr.add_argument("--height",       type=int,   default=576)
    p_chr.add_argument("--steps",        type=int,   default=25)
    p_chr.add_argument("--motion",       type=int,   default=127,
                       help="Motion amount 1–255 (higher = more motion)")
    p_chr.add_argument("--noise-aug",    type=float, default=0.02,
                       help="Noise augmentation strength (0.02–0.1)")
    p_chr.add_argument("--seed",         type=int,   default=-1)
    p_chr.add_argument("--loop",         action="store_true",
                       help="Generate a seamless ping-pong loop")
    p_chr.set_defaults(func=cmd_character)

    # ── enhance ──────────────────────────────────────────────────────────── #
    p_enh = sub.add_parser("enhance", help="Enhance an existing video cinematically")
    p_enh.add_argument("--input",       required=True)
    p_enh.add_argument("--output",      required=True)
    p_enh.add_argument("--exp",         type=int, default=1,
                       help="Frame interpolation exponent (1=2×, 2=4×)")
    p_enh.add_argument("--no-fp16",     action="store_true")
    p_enh.add_argument("--no-grade",    action="store_true", help="Skip color grading")
    p_enh.add_argument("--no-bars",     action="store_true", help="Skip letterbox bars")
    p_enh.add_argument("--no-denoise",  action="store_true")
    p_enh.add_argument("--no-autocrop", action="store_true",
                       help="Skip native black-bar detection and removal")
    p_enh.set_defaults(func=cmd_enhance)

    # ── inject ───────────────────────────────────────────────────────────── #
    p_inj = sub.add_parser("inject", help="Generate an AI scene and inject it into a video")
    p_inj.add_argument("--input",           required=True)
    p_inj.add_argument("--output",          required=True)
    p_inj.add_argument("--timestamp",       type=float, required=True,
                       help="Injection point in seconds (e.g. 15.0)")
    p_inj.add_argument("--prompt",          required=True)
    p_inj.add_argument("--negative-prompt", default="blurry, low quality, watermark, text, deformed, ugly")
    p_inj.add_argument("--duration",        type=int,   default=4,
                       help="Target loop length in seconds")
    p_inj.add_argument("--transition",      type=float, default=0.5,
                       help="Crossfade duration in seconds")
    p_inj.add_argument("--width",           type=int,   default=512)
    p_inj.add_argument("--height",          type=int,   default=512)
    p_inj.add_argument("--steps",           type=int,   default=25)
    p_inj.add_argument("--cfg",             type=float, default=7.5)
    p_inj.add_argument("--seed",            type=int,   default=-1)
    p_inj.set_defaults(func=cmd_inject)

    return root


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    configure_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    args.func(args)


if __name__ == "__main__":
    main()
