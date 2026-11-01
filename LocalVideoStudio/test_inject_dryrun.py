"""
Dry-run test for inject_scene pipeline.
Tests every step except AnimateDiff generation (skips model load).
Requires: ffmpeg on PATH, enhanced.mp4 in C:\\AI-Lab\\OutputVideo\\
"""
import subprocess
import tempfile
import shutil
from pathlib import Path


def make_test_clip(output_path: Path, duration: int = 4, fps: int = 8,
                   color: str = "blue") -> None:
    """Creates a solid-colour test video to stand in for AI-generated output."""
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c={color}:size=512x512:rate={fps}:duration={duration}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(output_path),
    ], check=True, capture_output=True)


def run_dryrun():
    input_video = Path("C:/AI-Lab/OutputVideo/enhanced.mp4")
    assert input_video.exists(), f"Run enhance first: {input_video}"

    tmp = Path(tempfile.mkdtemp(prefix="lvs_test_"))
    try:
        print("Step 1: split at 10s...")
        part1 = tmp / "part1.mp4"
        part2 = tmp / "part2.mp4"
        subprocess.run(["ffmpeg", "-y", "-i", str(input_video),
                        "-t", "10", "-c", "copy", str(part1)], check=True, capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-i", str(input_video),
                        "-ss", "10", "-c", "copy", str(part2)], check=True, capture_output=True)
        assert part1.exists() and part1.stat().st_size > 0, "part1 empty"
        assert part2.exists() and part2.stat().st_size > 0, "part2 empty"
        print("  OK")

        print("Step 2: extract last frame of part1...")
        frame_png = tmp / "freeze_frame.png"
        subprocess.run(["ffmpeg", "-y", "-sseof", "-1",
                        "-i", str(part1), "-frames:v", "1", str(frame_png)],
                       check=True, capture_output=True)
        assert frame_png.exists() and frame_png.stat().st_size > 0, "frame PNG empty"
        print("  OK")

        print("Step 3: make synthetic loop clip (stand-in for AnimateDiff)...")
        loop_graded = tmp / "loop_graded.mp4"
        make_test_clip(loop_graded, duration=4, fps=60, color="blue")
        assert loop_graded.exists() and loop_graded.stat().st_size > 0, "loop empty"
        print("  OK")

        print("Step 4: crossfade part1 -> loop...")
        xfade_in = tmp / "xfade_in.mp4"

        result = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(part1)
        ], capture_output=True, text=True, check=True)
        part1_dur = float(result.stdout.strip())
        offset = part1_dur - 0.5

        # probe part1 dimensions so we can scale loop to match
        dim_result = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0", str(part1)
        ], capture_output=True, text=True, check=True)
        p1_w, p1_h = dim_result.stdout.strip().split(",")

        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(part1), "-i", str(loop_graded),
            "-filter_complex",
            (f"[1:v]scale={p1_w}:{p1_h}:force_original_aspect_ratio=disable[b];"
             f"[0:v][b]xfade=transition=fade:duration=0.5:offset={offset:.3f}"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(xfade_in),
        ], check=True, capture_output=True)
        assert xfade_in.exists() and xfade_in.stat().st_size > 0, "xfade_in empty"
        print("  OK")

        print("Step 5: crossfade loop -> part2...")
        xfade_out = tmp / "xfade_out.mp4"

        result = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(loop_graded)
        ], capture_output=True, text=True, check=True)
        loop_dur = float(result.stdout.strip())
        offset2 = loop_dur - 0.5

        # loop_graded is 512x512; scale part2 ref not needed — scale loop up to p1 dims for B->part2
        dim_result2 = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0", str(part2)
        ], capture_output=True, text=True, check=True)
        p2_w, p2_h = dim_result2.stdout.strip().split(",")

        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(loop_graded), "-i", str(part2),
            "-filter_complex",
            (f"[0:v]scale={p2_w}:{p2_h}:force_original_aspect_ratio=disable[a];"
             f"[a][1:v]xfade=transition=fade:duration=0.5:offset={offset2:.3f}"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(xfade_out),
        ], check=True, capture_output=True)
        assert xfade_out.exists() and xfade_out.stat().st_size > 0, "xfade_out empty"
        print("  OK")

        print("Step 6: concat xfade_in + xfade_out...")
        final = tmp / "final_test.mp4"
        concat_txt = tmp / "concat.txt"
        concat_txt.write_text(
            f"file '{xfade_in.as_posix()}'\nfile '{xfade_out.as_posix()}'\n",
            encoding="utf-8"
        )
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_txt),
            "-c", "copy", str(final),
        ], check=True, capture_output=True)
        assert final.exists() and final.stat().st_size > 0, "final empty"
        print("  OK")

        output_copy = Path("C:/AI-Lab/OutputVideo/test_inject_dryrun.mp4")
        shutil.copy(final, output_copy)
        print(f"\nDRY-RUN PASSED. Output: {output_copy}")
        print("Open and verify: game plays to 10s, blue clip for 4s, game resumes.")

    except Exception as e:
        print(f"\nDRY-RUN FAILED: {e}")
        raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run_dryrun()
