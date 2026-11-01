"""
Dry-run test for inject_scene freeze-frame anchor fix.
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

        # Step 2: extract last frame of part1
        print("Step 2: extract freeze frame...")
        frame_png = tmp / "freeze_frame.png"
        subprocess.run(
            ["ffmpeg", "-y", "-sseof", "-1", "-i", str(part1),
             "-frames:v", "1", str(frame_png)],
            check=True, capture_output=True,
        )
        assert frame_png.stat().st_size > 0
        print("  OK")

        # Step 3: create freeze_hold (1 second of the freeze frame)
        print("Step 3: create freeze_hold clip...")
        freeze_hold = tmp / "freeze_hold.mp4"
        subprocess.run(
            ["ffmpeg", "-y",
             "-loop", "1", "-framerate", str(FPS), "-i", str(frame_png),
             "-vf", f"scale={w}:{h}:force_original_aspect_ratio=disable",
             "-t", "1.0",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
             str(freeze_hold)],
            check=True, capture_output=True,
        )
        assert freeze_hold.stat().st_size > 0
        print("  OK")

        # Step 4: build part1_frozen and part2_frozen
        print("Step 4: concat part1+freeze and freeze+part2...")
        part1_frozen = tmp / "part1_frozen.mp4"
        part2_frozen = tmp / "part2_frozen.mp4"

        c1 = tmp / "c1.txt"
        c1.write_text(f"file '{part1.as_posix()}'\nfile '{freeze_hold.as_posix()}'\n",
                      encoding="utf-8")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(c1), "-c", "copy", str(part1_frozen)],
            check=True, capture_output=True,
        )

        c2 = tmp / "c2.txt"
        c2.write_text(f"file '{freeze_hold.as_posix()}'\nfile '{part2.as_posix()}'\n",
                      encoding="utf-8")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(c2), "-c", "copy", str(part2_frozen)],
            check=True, capture_output=True,
        )
        assert part1_frozen.stat().st_size > 0 and part2_frozen.stat().st_size > 0
        print("  OK")

        # Step 5: synthetic AI loop (solid blue, 4s at video fps and dimensions)
        print("Step 5: create synthetic AI loop (blue clip)...")
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

        # Step 6: chained xfade (3 clips, 2 xfades, 1 ffmpeg call)
        print("Step 6: chained xfade (3 clips, 2 xfades, 1 ffmpeg call)...")
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

        dest = Path("C:/AI-Lab/OutputVideo/test_inject_freeze_dryrun.mp4")
        shutil.copy(final, dest)

        dur_final = ffprobe_duration(dest)
        print(f"\nDRY-RUN PASSED.")
        print(f"Output: {dest}")
        print(f"Duration: {dur_final:.2f}s")
        print()
        print("WHAT TO VERIFY when you open the video:")
        print("  0s-10s   : game plays normally")
        print("  ~10s     : game freezes on exact frame (same frame, no cut)")
        print("  ~10.5s   : freeze frame dissolves into BLUE clip")
        print("  ~10.5-14s: blue clip plays")
        print("  ~14s     : blue clip dissolves back to freeze frame")
        print("  ~14.5s   : freeze frame holds, then game resumes from 10s")
        print("  14.5s-end: game continues")

    except subprocess.CalledProcessError as e:
        print(f"\nFAILED at ffmpeg step.")
        print("stderr:", e.stderr.decode() if e.stderr else "none")
        raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run_dryrun()
