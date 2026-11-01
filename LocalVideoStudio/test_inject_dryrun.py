"""
Dry-run test for inject_scene tpad freeze approach.
No AnimateDiff models needed -- a synthetic blue clip stands in for AI output.
Requires: ffmpeg on PATH, enhanced.mp4 in C:\\AI-Lab\\OutputVideo\\
"""
import subprocess
import shutil
import tempfile
from pathlib import Path


def ffprobe_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


def ffprobe_dimensions(path: Path) -> tuple[int, int]:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    w, h = r.stdout.strip().split(",")
    return int(w), int(h)


def run_dryrun():
    input_video = Path("C:/AI-Lab/OutputVideo/enhanced.mp4")
    assert input_video.exists(), f"Run enhance first: {input_video}"

    TIMESTAMP = 10.0
    TRANSITION = 0.5
    FPS = 60
    FREEZE_SEC = 1.0

    tmp = Path(tempfile.mkdtemp(prefix="lvs_dryrun_"))
    try:
        w, h = ffprobe_dimensions(input_video)
        print(f"Input: {w}x{h}")

        # Step 1: split
        print("Step 1: split at 10s...")
        part1 = tmp / "part1.mp4"
        part2 = tmp / "part2.mp4"
        subprocess.run(["ffmpeg", "-y", "-i", str(input_video),
                        "-t", str(TIMESTAMP), "-c", "copy", str(part1)],
                       check=True, capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-i", str(input_video),
                        "-ss", str(TIMESTAMP), "-c", "copy", str(part2)],
                       check=True, capture_output=True)
        assert part1.stat().st_size > 0 and part2.stat().st_size > 0
        print("  OK")

        # Step 2: freeze tail of part1 using tpad (clone last frame)
        print("Step 2: freeze last frame of part1 (tpad clone)...")
        part1_frozen = tmp / "part1_frozen.mp4"
        stop_frames = round(FREEZE_SEC * FPS)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(part1),
             "-vf", f"tpad=stop_mode=clone:stop={stop_frames}",
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             str(part1_frozen)],
            check=True, capture_output=True,
        )
        assert part1_frozen.stat().st_size > 0
        print("  OK")

        # Step 3: freeze head of part2 using tpad (clone first frame)
        print("Step 3: freeze first frame of part2 (tpad clone)...")
        part2_frozen = tmp / "part2_frozen.mp4"
        start_frames = round(FREEZE_SEC * FPS)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(part2),
             "-vf", f"tpad=start_mode=clone:start={start_frames}",
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             str(part2_frozen)],
            check=True, capture_output=True,
        )
        assert part2_frozen.stat().st_size > 0
        print("  OK")

        # Step 4: synthetic AI loop (solid blue, 4s at full resolution and fps)
        print("Step 4: create synthetic AI loop (blue clip)...")
        ai_loop = tmp / "ai_loop.mp4"
        subprocess.run(
            ["ffmpeg", "-y",
             "-f", "lavfi",
             "-i", f"color=c=blue:size={w}x{h}:rate={FPS}:duration=4",
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             str(ai_loop)],
            check=True, capture_output=True,
        )
        assert ai_loop.stat().st_size > 0
        print("  OK")

        # Step 5: chained xfade (3 clips, 2 xfades, 1 ffmpeg call)
        print("Step 5: chained xfade (3 clips, 2 xfades, 1 ffmpeg call)...")
        dur_a = ffprobe_duration(part1_frozen)
        dur_b = ffprobe_duration(ai_loop)
        offset1 = max(0.0, dur_a - TRANSITION)
        offset2 = max(0.0, dur_a + dur_b - 2 * TRANSITION)

        filter_complex = (
            f"[0:v]fps={FPS},scale={w}:{h}:force_original_aspect_ratio=disable[va];"
            f"[1:v]fps={FPS},scale={w}:{h}:force_original_aspect_ratio=disable[vb];"
            f"[2:v]fps={FPS},scale={w}:{h}:force_original_aspect_ratio=disable[vc];"
            f"[va][vb]xfade=transition=fade:duration={TRANSITION:.3f}:offset={offset1:.3f}[x1];"
            f"[x1][vc]xfade=transition=fade:duration={TRANSITION:.3f}:offset={offset2:.3f}[out]"
        )

        final = tmp / "final_dryrun.mp4"
        subprocess.run(
            ["ffmpeg", "-y",
             "-i", str(part1_frozen),
             "-i", str(ai_loop),
             "-i", str(part2_frozen),
             "-filter_complex", filter_complex,
             "-map", "[out]",
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             str(final)],
            check=True, capture_output=True,
        )
        assert final.stat().st_size > 0
        print("  OK")

        dest = Path("C:/AI-Lab/OutputVideo/test_inject_tpad_dryrun.mp4")
        shutil.copy(final, dest)

        dur_final = ffprobe_duration(dest)
        print(f"\nDRY-RUN PASSED.")
        print(f"Output: {dest}")
        print(f"Duration: {dur_final:.2f}s")
        print()
        print("WHAT TO VERIFY when you open the video:")
        print("  0s-10s     : game plays normally (moving footage)")
        print("  ~10s       : game FREEZES on exact last frame (no cut visible)")
        print("  ~10.5s     : frozen frame dissolves into BLUE")
        print("  ~10.5-14.5s: blue clip plays")
        print("  ~14.5s     : blue dissolves back to same frozen frame")
        print("  ~15s       : freeze hold ends, game resumes from 10s")
        print("  15s-end    : game continues")

    except subprocess.CalledProcessError as e:
        print(f"\nFAILED at ffmpeg step.")
        print("stderr:", e.stderr.decode() if e.stderr else "none")
        raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run_dryrun()
