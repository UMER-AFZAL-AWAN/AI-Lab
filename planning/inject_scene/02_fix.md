# Planning Prompt — inject_scene: Freeze Frame Anchor Fix (v2)

## What is wrong right now

Run `inject_scene()` and watch the output. You will see:

1. **No freeze.** The game is moving when it dissolves into the AI scene.
   The viewer sees a busy action shot blending into unrelated generated content.
2. **AI loop appears twice.** You call CrossfadeTransitioner twice
   (part1 to loop, loop to part2), then concat both outputs. Both outputs contain
   the loop. The middle of the video is the loop playing twice back to back.
3. **Wrong structure.** The concat of xfade_in + xfade_out is the wrong
   shape. The loop appears in both halves. That is a duplicate, not a single insertion.

## What the user wants

Game plays -> freezes on the exact game frame at timestamp T -> story scene
evolves -> story dissolves back to the same frozen frame -> game resumes from T.

The frozen game frame is the visual anchor.

## The fix in plain terms

```
BEFORE: game plays to T -> holds frozen on frame T for 1s
IN:     frozen frame dissolves into AI scene (0.5s)
SCENE:  plays (~4s)
OUT:    AI scene dissolves back to frozen frame T (0.5s)
AFTER:  holds on frame T for 1s -> game resumes from T
```

Build these segments:
- freeze_hold.mp4  : 1s still clip of freeze_frame.png at the video's fps/dimensions
- part1_frozen.mp4 : concat(part1, freeze_hold)  [no xfade, -c copy]
- part2_frozen.mp4 : concat(freeze_hold, part2)  [no xfade, -c copy]

Then ONE ffmpeg call with chained filter_complex (3 inputs, 2 xfades):

  ffmpeg -i part1_frozen -i loop_graded -i part2_frozen
    -filter_complex "
      [0:v]fps=60,scale=W:H[va];
      [1:v]fps=60,scale=W:H[vb];
      [2:v]fps=60,scale=W:H[vc];
      [va][vb]xfade=transition=fade:duration=0.5:offset=<dur_A - 0.5>[x1];
      [x1][vc]xfade=transition=fade:duration=0.5:offset=<dur_A + dur_B - 1.0>[out]
    " -map "[out]" -c:v h264_nvenc -pix_fmt yuv420p output.mp4

Offset formula:
  offset1 = dur(part1_frozen) - 0.5
  offset2 = dur(part1_frozen) + dur(loop_graded) - 1.0

DO NOT use two separate CrossfadeTransitioner calls and concat them.
One ffmpeg call, three inputs, two xfades, one output.

## Changes required

Read these files first:
- C:\AI-Lab\LocalVideoStudio\Inference\pipeline\studio_pipeline.py
- C:\AI-Lab\LocalVideoStudio\Inference\processors\crossfade_transitioner.py

**studio_pipeline.py** - rewrite inject_scene():

1. Keep steps 1-4 (split, extract frame, generate loop, grade loop) exactly as-is.
2. Add step 5: create freeze_hold.mp4 from freeze_frame.png.

   @staticmethod
   def _create_freeze_hold(frame_png, output_path, duration, fps, width, height):
       subprocess.run([
           "ffmpeg", "-y", "-loop", "1", "-framerate", str(fps),
           "-i", str(frame_png),
           "-vf", f"scale={width}:{height}:force_original_aspect_ratio=disable",
           "-t", str(duration),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
           str(output_path),
       ], check=True, capture_output=True)

3. Add step 6: concat part1+freeze_hold -> part1_frozen, freeze_hold+part2 -> part2_frozen.
   Use ffmpeg concat demuxer (-f concat -safe 0 -c copy) for each.

4. Replace steps 5-7 with a single _chained_xfade() call:

   @staticmethod
   def _chained_xfade(clip_a, clip_b, clip_c, output_path, transition, fps, width, height):
       def dur(p):
           r = subprocess.run(
               ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "csv=p=0", str(p)],
               capture_output=True, text=True, check=True)
           return float(r.stdout.strip())
       dur_a = dur(clip_a)
       dur_b = dur(clip_b)
       offset1 = max(0.0, dur_a - transition)
       offset2 = max(0.0, dur_a + dur_b - 2 * transition)
       fc = (
           f"[0:v]fps={fps},scale={width}:{height}:force_original_aspect_ratio=disable[va];"
           f"[1:v]fps={fps},scale={width}:{height}:force_original_aspect_ratio=disable[vb];"
           f"[2:v]fps={fps},scale={width}:{height}:force_original_aspect_ratio=disable[vc];"
           f"[va][vb]xfade=transition=fade:duration={transition:.3f}:offset={offset1:.3f}[x1];"
           f"[x1][vc]xfade=transition=fade:duration={transition:.3f}:offset={offset2:.3f}[out]"
       )
       subprocess.run([
           "ffmpeg", "-y",
           "-i", str(clip_a), "-i", str(clip_b), "-i", str(clip_c),
           "-filter_complex", fc,
           "-map", "[out]",
           "-c:v", "h264_nvenc", "-pix_fmt", "yuv420p", "-preset", "p4", "-cq", "23",
           str(output_path),
       ], check=True, capture_output=True)

5. Probe the input video dimensions before _create_freeze_hold.
   Use CrossfadeTransitioner._get_dimensions(input_path).
6. Delete the old steps 5-7 (the two CrossfadeTransitioner calls and concat).

**crossfade_transitioner.py** - no changes needed.

## Dry-run test (run this, no GPU needed)

Create and run a dry-run script at C:\AI-Lab\LocalVideoStudio\test_inject_dryrun.py that:
1. Splits C:\AI-Lab\OutputVideo\enhanced.mp4 at 10s (assert it exists first)
2. Extracts the last frame of part1
3. Creates freeze_hold.mp4 (1s, matching enhanced.mp4 dimensions)
4. Concats part1+freeze_hold and freeze_hold+part2
5. Makes a solid blue clip (4s) as the fake AI loop:
     ffmpeg -f lavfi -i color=c=blue:size=WxH:rate=60:duration=4 ...
6. Runs _chained_xfade logic (inline subprocess, same formula above)
7. Copies output to C:\AI-Lab\OutputVideo\test_freeze_dryrun.mp4
8. Prints expected timestamps for the user to verify:
     0s-10s   : game
     ~10s     : game freezes (same frame, no cut)
     ~10.5s   : dissolves to BLUE
     ~10.5-14.5s: blue
     ~14.5s   : dissolves back to same frozen frame
     ~15s     : game resumes from 10s

**PAUSE here. Ask the user to open test_freeze_dryrun.mp4 and confirm
they see those exact transitions before continuing.**

## Full test (after user confirms dry-run)

C:\AI-Lab\.venv\Scripts\python.exe -m Inference.cli inject ^
  --input "C:\AI-Lab\OutputVideo\enhanced.mp4" ^
  --output "C:\AI-Lab\OutputVideo\with_scene_v2.mp4" ^
  --timestamp 10.0 ^
  --prompt "A footballer stands in golden dramatic light, slow motion, anime cinematic style" ^
  --duration 4 --transition 0.5

**PAUSE. Ask user to open with_scene_v2.mp4 and confirm the freeze and
transitions match the dry-run pattern before committing.**

## Commit (after user confirms)

Stage only changed files. Commit from C:\AI-Lab:

  Fix inject_scene: freeze-frame anchor replaces broken double-loop assembly

  The previous implementation crossfaded moving game footage into the AI
  scene and duplicated the loop by concating two overlapping xfade outputs.

  Fix: extract exact game frame at T, hold as 1-second still clip before
  and after the AI scene, chain both dissolves in one filter_complex call.

  Result: game -> frozen frame -> dissolves to story -> dissolves back to
  same frozen frame -> game resumes.

  Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

## Hard rules

- All print() text: ASCII only. No arrows, no ellipsis, no unicode symbols.
- All temp files inside _TempDir. Nothing written to project dir during runs.
- Do not modify CrossfadeTransitioner. New logic stays in StudioPipeline.
- Probe input dimensions with CrossfadeTransitioner._get_dimensions() -- never hardcode.
- Ask before any assumption not covered above.

---

## ORIGINAL PROMPT (below, superseded by the above fix)

You are fixing an existing implementation in `C:\AI-Lab\LocalVideoStudio`.
Read every file listed below before touching anything.

The previous session implemented `inject_scene()` but the result is wrong.
The user's requirement is:

> Game plays → freezes on exact frame T → AI-generated story scene evolves from
> that frozen frame → story loops back to the same frame → game resumes from T.

**What is currently broken:**

1. The game footage is MOVING when it crossfades into the AI scene. There is no
   freeze. The viewer sees a busy action scene dissolve into unrelated AI content.

2. `inject_scene()` calls `CrossfadeTransitioner` twice (once for part1→loop,
   once for loop→part2), then concats the two outputs. This means the AI loop
   clip appears TWICE in the final video. The assembly is wrong.

3. There is no visual anchor — nothing tells the viewer "the game paused here,
   a scene plays, and we return to this exact moment."

---

## Read these files first

Read all of them. Do not change anything yet.

- `C:\AI-Lab\LocalVideoStudio\Inference\pipeline\studio_pipeline.py`
- `C:\AI-Lab\LocalVideoStudio\Inference\processors\crossfade_transitioner.py`
- `C:\AI-Lab\LocalVideoStudio\Inference\interfaces\video_processor.py`
- `C:\AI-Lab\LocalVideoStudio\Inference\generators\animatediff_generator.py`
- `C:\AI-Lab\LocalVideoStudio\Inference\cli.py`
- `C:\AI-Lab\LocalVideoStudio\Inference\config.py`

**PAUSE. Tell the user: "Files read. Ready to explain the fix before coding."**
Then explain the fix in plain words (the freeze-frame anchor approach below) and
ask the user if they agree before writing any code.

---

## The correct assembly

```
Game video (0 → T)
  last frame = freeze_frame.png

GOAL: game freezes at T, story evolves, game resumes at T

Segments to build:
  part1          = game from 0 → T          (split from input, -c copy)
  part2          = game from T → end         (split from input, -c copy)
  freeze_hold    = 1 second of freeze_frame.png at 60fps 1280x720
                   (a still image turned into a short video clip)
  AI_loop        = AnimateDiff ping-pong loop from text prompt, graded

Assembly:
  part1_frozen   = concat(part1, freeze_hold)    no xfade, -c copy
                   game plays to T, then holds on that frame for 1 second

  part2_frozen   = concat(freeze_hold, part2)    no xfade, -c copy
                   holds on the same frame for 1 second, then game resumes

  Final (ONE ffmpeg call, chained xfade filter_complex):
    [part1_frozen] -xfade(0.5s)-> [AI_loop] -xfade(0.5s)-> [part2_frozen]

What the viewer sees:
  game plays normally
  → last second of game: action slows visually (freeze_hold is the same frame)
  → 0.5s dissolve into AI story scene
  → story plays (~4 seconds)
  → 0.5s dissolve back to the same freeze frame
  → first second: still on that frame
  → game resumes from T
```

The freeze_hold clip before and after the AI scene is the anchor.
Since the last frame of part1 IS the freeze frame, and freeze_hold IS the same
frame, the concat is seamless (no visible cut). Same for freeze_hold + part2.

---

## Section 1 — Create freeze_hold clip (new private method in StudioPipeline)

Add `_create_freeze_hold` to `studio_pipeline.py`:

```python
@staticmethod
def _create_freeze_hold(
    frame_png: Path,
    output_path: Path,
    duration: float,
    fps: int,
    width: int,
    height: int,
) -> None:
    """Turns a single PNG into a short video clip of the given duration."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-loop", "1",
            "-framerate", str(fps),
            "-i", str(frame_png),
            "-vf", f"scale={width}:{height}:force_original_aspect_ratio=disable",
            "-t", str(duration),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )
```

---

## Section 2 — Chained xfade (new private method in StudioPipeline)

Add `_chained_xfade` to `studio_pipeline.py`.
This replaces the two separate `CrossfadeTransitioner.process()` calls.
CrossfadeTransitioner is NOT changed — it keeps its 2-clip responsibility.
This method handles the 3-clip chain that is specific to inject_scene.

```python
@staticmethod
def _chained_xfade(
    clip_a: Path,       # part1_frozen
    clip_b: Path,       # AI_loop (graded)
    clip_c: Path,       # part2_frozen
    output_path: Path,
    transition: float,  # seconds, same for both xfades
    fps: int,
    width: int,
    height: int,
) -> None:
    """
    Chains two xfades in a single ffmpeg call:
      clip_a -xfade-> clip_b -xfade-> clip_c -> output

    Using two separate ffmpeg calls would duplicate clip_b in the output.
    One filter_complex call is the only correct approach.

    Offset maths:
      xfade1 starts when clip_a is almost done:
        offset1 = dur_a - transition

      xfade2 starts when clip_b is almost done in the combined stream.
      In the output of xfade1, clip_b ends at:
        (dur_a - transition) + dur_b
      xfade2 should start transition seconds before that:
        offset2 = (dur_a - transition) + dur_b - transition
                = dur_a + dur_b - 2 * transition
    """
    # Probe durations
    def _duration(p: Path) -> float:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(p)],
            capture_output=True, text=True, check=True,
        )
        return float(r.stdout.strip())

    dur_a = _duration(clip_a)
    dur_b = _duration(clip_b)

    offset1 = max(0.0, dur_a - transition)
    offset2 = max(0.0, dur_a + dur_b - 2 * transition)

    filter_complex = (
        f"[0:v]fps={fps},scale={width}:{height}:force_original_aspect_ratio=disable[va];"
        f"[1:v]fps={fps},scale={width}:{height}:force_original_aspect_ratio=disable[vb];"
        f"[2:v]fps={fps},scale={width}:{height}:force_original_aspect_ratio=disable[vc];"
        f"[va][vb]xfade=transition=fade:duration={transition:.3f}:offset={offset1:.3f}[x1];"
        f"[x1][vc]xfade=transition=fade:duration={transition:.3f}:offset={offset2:.3f}[out]"
    )

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(clip_a),
            "-i", str(clip_b),
            "-i", str(clip_c),
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-c:v", "h264_nvenc",
            "-pix_fmt", "yuv420p",
            "-preset", "p4",
            "-cq", "23",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )
```

---

## Section 3 — Rewrite inject_scene() in StudioPipeline

Replace the entire `inject_scene()` method body with the following logic.
Keep the method signature exactly as-is (same parameters, same return type).

Steps (print [1/8] through [8/8], ASCII only in all print statements):

```
[1/8] Split input_path at timestamp_seconds -> part1.mp4, part2.mp4
      Use _split_video() (already exists, keep as-is)

[2/8] Probe dimensions of input_path (width, height) for freeze hold clip
      Use CrossfadeTransitioner._get_dimensions() OR add a local ffprobe call
      You need the exact pixel dimensions to make freeze_hold match the video

[3/8] Extract last frame of part1 -> freeze_frame.png
      Use _extract_frame() (already exists, keep as-is)

[4/8] Create freeze_hold.mp4 from freeze_frame.png
      Use _create_freeze_hold(frame_png, freeze_hold, duration=1.0, fps=60,
                              width=probed_w, height=probed_h)

[5/8] Build part1_frozen.mp4 = concat(part1, freeze_hold) with -c copy
      Build part2_frozen.mp4 = concat(freeze_hold, part2) with -c copy
      Use a concat list file + ffmpeg concat demuxer for each

[6/8] Generate AnimateDiff seamless loop -> loop_raw.mp4
      Use self._generator.generate_seamless_loop(request)
      export_to_video(result.frames, str(loop_raw), fps=result.fps)

[7/8] Grade loop_raw -> loop_graded.mp4
      Use self._enhancer.process(loop_raw, loop_graded,
                                 EnhancementConfig(auto_crop=False,
                                                   cinematic_bars=False))

[8/8] Chained xfade + final output
      Use _chained_xfade(part1_frozen, loop_graded, part2_frozen,
                         output_path, transition=transition_duration,
                         fps=60, width=probed_w, height=probed_h)
```

Write the full method code (not pseudocode). All temp files go inside `_TempDir`.

**PAUSE. Show the user the full rewritten inject_scene() and _chained_xfade()
and _create_freeze_hold() before continuing. Wait for approval.**

---

## Section 4 — Dry-run test (no GPU needed)

Replace `C:\AI-Lab\LocalVideoStudio\test_inject_dryrun.py` entirely with this:

```python
"""
Dry-run test for inject_scene freeze-frame anchor fix.
No AnimateDiff models needed — a synthetic blue clip stands in for AI output.
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

        # Step 5: synthetic AI loop (solid blue, 4s at 60fps)
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

        # Step 6: chained xfade
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
        print("  ~14.5s   : freeze frame dissolves away, game resumes from 10s")
        print("  14.5s-end: game continues")

    except subprocess.CalledProcessError as e:
        print(f"\nFAILED at ffmpeg step.")
        print("stderr:", e.stderr.decode() if e.stderr else "none")
        raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run_dryrun()
```

Run it:
```
C:\AI-Lab\.venv\Scripts\python.exe C:\AI-Lab\LocalVideoStudio\test_inject_dryrun.py
```

**PAUSE. Copy the printed output here and ask the user to open
`C:\AI-Lab\OutputVideo\test_inject_freeze_dryrun.mp4` and verify the
EXACT sequence printed at the end of the script. Do not continue until
the user confirms each transition point looks correct.**

---

## Section 5 — Full integration test (AnimateDiff models)

Only run after the user confirms the dry-run is correct.

```
C:\AI-Lab\.venv\Scripts\python.exe -m Inference.cli inject ^
    --input "C:\AI-Lab\OutputVideo\enhanced.mp4" ^
    --output "C:\AI-Lab\OutputVideo\with_scene_v2.mp4" ^
    --timestamp 10.0 ^
    --prompt "A footballer stands in dramatic golden light, slow motion, cinematic anime style, high detail" ^
    --duration 4 ^
    --transition 0.5
```

Expected viewer experience:
- Game plays to 10s
- Game freezes on exact frame at 10s (holds ~1s)
- Dissolves into AI scene (0.5s)
- AI scene plays (~4s)
- Dissolves back to the same frozen game frame (0.5s)
- Freeze frame holds ~1s
- Game resumes from 10s and continues to end

**PAUSE. Ask user to open `with_scene_v2.mp4` and confirm this is what they see.**

---

## Section 6 — Commit

Stage only modified/new files:
- `LocalVideoStudio/Inference/pipeline/studio_pipeline.py`
- `LocalVideoStudio/Inference/processors/crossfade_transitioner.py` (if changed)
- `LocalVideoStudio/Inference/interfaces/video_processor.py` (if changed)
- `LocalVideoStudio/Inference/cli.py` (if changed)
- `LocalVideoStudio/test_inject_dryrun.py`

Commit from `C:\AI-Lab`:
```
Fix inject_scene: freeze-frame anchor + correct 3-clip chained xfade

Previous implementation crossfaded moving game footage directly into the
AI scene and duplicated the AI loop in the output by concating two
overlapping xfade outputs.

Fix: extract the exact game frame at timestamp T, hold it as a 1-second
still clip before and after the AI scene, then chain both xfades in a
single ffmpeg filter_complex call so the AI loop appears exactly once.

Result: game plays -> freezes on frame T -> dissolves to story scene ->
dissolves back to same frame T -> game resumes.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

---

## Hard constraints — re-read before every file edit

- ASCII only in all print() statements. No arrows, ellipsis, multiplication signs.
- All temp files inside _TempDir context. Nothing written to the project directory
  during processing.
- SOLID: CrossfadeTransitioner keeps its 2-clip single responsibility unchanged.
  _chained_xfade is a private method of StudioPipeline because it is
  pipeline-assembly logic, not a general processor.
- h264_nvenc + yuv420p on all final re-encoded output.
- freeze_hold must match the enhanced video's exact pixel dimensions.
  Always probe the input first — do not hardcode 1280x720.
- Do not remove or rename any existing public methods or CLI flags.
- Ask the user before making any assumption not covered by this prompt.
