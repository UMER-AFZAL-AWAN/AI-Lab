"""
Command-line interface for LocalVideoStudio.

Examples
────────
# Enhance the game recording
python -m Inference.cli enhance \
    --input  "C:/AI-Lab/InputVideo/input.mp4" \
    --output "C:/AI-Lab/OutputVideo/enhanced.mp4"

# Generate and inject a cinematic loop at 15 seconds
python -m Inference.cli inject \
    --input     "C:/AI-Lab/InputVideo/input.mp4" \
    --output    "C:/AI-Lab/OutputVideo/with_loop.mp4" \
    --timestamp 15.0 \
    --prompt    "A knight at a castle gate, dramatic lighting, cinematic, 4k"

Run from C:\\AI-Lab\\LocalVideoStudio with the project venv active.
"""

import argparse
import sys
from pathlib import Path

# Ensure the package root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from Inference.pipeline.studio_pipeline import StudioPipeline, InjectionPoint
from Inference.interfaces.video_processor import EnhancementConfig, InterpolationConfig


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


def cmd_inject(args: argparse.Namespace) -> None:
    pipeline  = StudioPipeline()
    injection = InjectionPoint(
        timestamp_seconds  = args.timestamp,
        prompt             = args.prompt,
        negative_prompt    = args.negative_prompt,
        loop_duration_sec  = args.loop_duration,
        width              = args.width,
        height             = args.height,
        num_inference_steps= args.steps,
        guidance_scale     = args.cfg,
        seed               = args.seed,
    )
    pipeline.generate_and_inject_loop(
        input_path  = Path(args.input),
        output_path = Path(args.output),
        injection   = injection,
    )


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="lvs",
        description="LocalVideoStudio – local AI video production",
    )
    sub = root.add_subparsers(dest="command", required=True)

    # ── enhance ──────────────────────────────────────────────────────── #
    p_enh = sub.add_parser("enhance", help="Enhance an existing video cinematically")
    p_enh.add_argument("--input",      required=True)
    p_enh.add_argument("--output",     required=True)
    p_enh.add_argument("--exp",        type=int, default=1,
                       help="Frame interpolation multiplier exponent (1=2×, 2=4×)")
    p_enh.add_argument("--no-fp16",      action="store_true")
    p_enh.add_argument("--no-grade",     action="store_true", help="Skip color grading")
    p_enh.add_argument("--no-bars",      action="store_true", help="Skip letterbox bars")
    p_enh.add_argument("--no-denoise",   action="store_true")
    p_enh.add_argument("--no-autocrop",  action="store_true",
                       help="Skip native black-bar detection and removal")
    p_enh.set_defaults(func=cmd_enhance)

    # ── inject ───────────────────────────────────────────────────────── #
    p_inj = sub.add_parser("inject", help="Generate a cinematic loop and inject it")
    p_inj.add_argument("--input",           required=True)
    p_inj.add_argument("--output",          required=True)
    p_inj.add_argument("--timestamp",       type=float, required=True,
                       help="Injection point in seconds (e.g. 15.0)")
    p_inj.add_argument("--prompt",          required=True)
    p_inj.add_argument("--negative-prompt", default="blurry, low quality, watermark, text")
    p_inj.add_argument("--loop-duration",   type=int, default=4,
                       help="Target loop length in seconds")
    p_inj.add_argument("--width",           type=int, default=512)
    p_inj.add_argument("--height",          type=int, default=512)
    p_inj.add_argument("--steps",           type=int, default=25)
    p_inj.add_argument("--cfg",             type=float, default=7.5)
    p_inj.add_argument("--seed",            type=int, default=-1)
    p_inj.set_defaults(func=cmd_inject)

    return root


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
