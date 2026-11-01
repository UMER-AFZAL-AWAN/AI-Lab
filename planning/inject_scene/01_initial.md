# Planning Prompt — inject_scene: Initial Implementation

## What you are doing

You are implementing a new `inject` subcommand for the LocalVideoStudio CLI at `C:\AI-Lab\LocalVideoStudio`. This is a Windows 11 machine with RTX 5060 Laptop (8GB VRAM), CUDA 13, Python venv at `C:\AI-Lab\.venv`.

The feature: at a user-specified timestamp in an already-enhanced video, generate a short cinematic scene using AnimateDiff (text prompt), apply a seamless ping-pong loop, crossfade smoothly in and out, and merge it back so the game video resumes from the exact same frame. All print statements must use ASCII only — no Unicode arrows or ellipsis (Windows cp1252 console crashes on them).

**STOP AND ASK THE USER** before proceeding if you have any ambiguity about requirements. Do not guess. Pause for review after completing each numbered section below.

---

## Read these files first before writing anything

Read every file listed here. Understand what already exists.

- `C:\AI-Lab\LocalVideoStudio\Inference\cli.py`
- `C:\AI-Lab\LocalVideoStudio\Inference\config.py`
- `C:\AI-Lab\LocalVideoStudio\Inference\pipeline\studio_pipeline.py`
- `C:\AI-Lab\LocalVideoStudio\Inference\interfaces\video_processor.py`
- `C:\AI-Lab\LocalVideoStudio\Inference\interfaces\video_generator.py`
- `C:\AI-Lab\LocalVideoStudio\Inference\generators\animatediff_generator.py`
- `C:\AI-Lab\LocalVideoStudio\Inference\processors\video_enhancer.py`
- `C:\AI-Lab\LocalVideoStudio\Inference\processors\rife_interpolator.py`

**PAUSE HERE. Tell the user: "I have read all source files. Ready to begin Section 1."**

---

## What already exists (do not rewrite these)

- `AnimateDiffGenerator.generate_seamless_loop()` — already generates a ping-pong loop from a text prompt. Returns `GenerationResult(frames, fps)`.
- `VideoEnhancer.process()` — applies cinematic grade to any video file.
- `StudioPipeline.generate_and_inject_loop()` — an older version of injection that has NO crossfade transitions and runs enhancement on raw input rather than on an already-enhanced video. **This method must be replaced/updated, not kept as-is.**
- `InjectionPoint` dataclass — already defined in `studio_pipeline.py`. You will extend it.
- `_split_video()`, `_concat_videos()` static helpers — keep these, they are fine.
- `_TempDir` context manager — keep this.

---

## Architecture you must follow

SOLID principles throughout. The user is a .NET developer — they care about clean interfaces and single responsibility.

```
cli.py
  └── StudioPipeline.inject_scene()          ← new public method (replaces generate_and_inject_loop)
        ├── _extract_frame()                  ← new private helper
        ├── AnimateDiffGenerator              ← already exists, use as-is
        ├── VideoEnhancer                     ← already exists, use for grading the loop
        ├── CrossfadeTransitioner.apply()     ← NEW processor class (new file)
        └── _concat_videos()                  ← already exists
```

---

## Section 1 — New file: CrossfadeTransitioner

**File:** `C:\AI-Lab\LocalVideoStudio\Inference\processors\crossfade_transitioner.py`

Implements `IVideoProcessor`. Its job: take two video files (A and B) and produce a single output file where B is dissolved into A over a configurable duration.

```python
class CrossfadeTransitioner(IVideoProcessor):
    def process(
        self,
        input_path: Path,       # video A (the "from" clip)
        output_path: Path,
        config: ProcessorConfig = None,  # must be CrossfadeConfig
    ) -> Path:
        ...
```

Add a new config dataclass to `interfaces/video_processor.py`:

```python
@dataclass
class CrossfadeConfig(ProcessorConfig):
    second_clip: Path = None        # video B (the "to" clip)
    transition_duration: float = 0.5  # seconds for the dissolve
    fps: int = 60                   # output fps (must match enhanced video)
```

Implementation uses ffmpeg's `xfade` filter:

```
ffmpeg -i A.mp4 -i B.mp4 \
  -filter_complex "xfade=transition=fade:duration=0.5:offset=<A_duration - 0.5>" \
  -c:v h264_nvenc -pix_fmt yuv420p -preset p4 -cq 23 \
  output.mp4
```

Where `offset = duration_of_A - transition_duration`. You need to probe A's duration with ffprobe to compute offset.

Add a private `_get_duration(path: Path) -> float` static method that runs ffprobe and returns the video duration in seconds.

**PAUSE HERE. Show the user the new file and the addition to video_processor.py for review before continuing.**

---

## Section 2 — New method: StudioPipeline.inject_scene()

**File:** `C:\AI-Lab\LocalVideoStudio\Inference\pipeline\studio_pipeline.py`

Add this new public method. The existing `generate_and_inject_loop` can remain for now but add a deprecation note in its docstring pointing to `inject_scene`.

```python
def inject_scene(
    self,
    input_path: Path,               # the already-enhanced video
    output_path: Path,
    timestamp_seconds: float,       # exact second to inject at
    prompt: str,                    # AnimateDiff text prompt
    negative_prompt: str = "blurry, low quality, watermark, text, deformed, ugly",
    loop_duration_sec: int = 4,     # target total loop duration
    transition_duration: float = 0.5,  # crossfade length in seconds
    width: int = 512,
    height: int = 512,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    seed: int = -1,
) -> Path:
```

Step-by-step inside `inject_scene` (all inside a `_TempDir` context):

```
1. split input_path at timestamp_seconds
       part1.mp4  (0 → timestamp)
       part2.mp4  (timestamp → end)

2. _extract_frame(part1, frame.png)
       extracts the very last frame of part1 as a PNG
       (this is the freeze frame — kept for reference, not used in video yet)

3. AnimateDiffGenerator.generate_seamless_loop(request)
       generates ping-pong loop frames from the text prompt
       exports as loop_raw.mp4 using diffusers export_to_video

4. VideoEnhancer.process(loop_raw.mp4, loop_graded.mp4, EnhancementConfig)
       applies same cinematic grade to the loop so it matches the game footage
       IMPORTANT: pass auto_crop=False and cinematic_bars=False here
       The loop is AI-generated — no native bars to crop, no bars to add

5. CrossfadeTransitioner.apply(part1, loop_graded, xfade_in.mp4, transition_duration)
       dissolves the end of part1 into the start of the loop

6. CrossfadeTransitioner.apply(loop_graded, part2, xfade_out.mp4, transition_duration)
       dissolves the end of the loop into the start of part2

7. concat [xfade_in.mp4, xfade_out.mp4] into output_path
       use the existing _concat_videos helper
```

Print progress at each step using `[1/7]` ... `[7/7]` format. ASCII only in all print statements.

Add a private static method:

```python
@staticmethod
def _extract_frame(video_path: Path, out_png: Path) -> None:
    """Extracts the last frame of video_path as a PNG."""
    # ffmpeg: seek to last frame using sseof=-1/1
    subprocess.run(
        ["ffmpeg", "-y", "-sseof", "-1",
         "-i", str(video_path),
         "-frames:v", "1", str(out_png)],
        check=True, capture_output=True,
    )
```

**PAUSE HERE. Show the user the updated studio_pipeline.py for review before continuing.**

---

## Section 3 — Wire the CLI

**File:** `C:\AI-Lab\LocalVideoStudio\Inference\cli.py`

Add a new `inject` subcommand (rename or replace the existing inject if it exists). The existing inject subcommand in cli.py currently calls `pipeline.generate_and_inject_loop()` — update it to call `pipeline.inject_scene()` instead.

Flags:

| Flag | Type | Default | Required |
|------|------|---------|----------|
| `--input` | str | — | yes |
| `--output` | str | — | yes |
| `--timestamp` | float | — | yes |
| `--prompt` | str | — | yes |
| `--negative-prompt` | str | `"blurry, low quality, watermark, text, deformed, ugly"` | no |
| `--duration` | int | 4 | no |
| `--transition` | float | 0.5 | no |
| `--width` | int | 512 | no |
| `--height` | int | 512 | no |
| `--steps` | int | 25 | no |
| `--cfg` | float | 7.5 | no |
| `--seed` | int | -1 | no |

**PAUSE HERE. Show the user the updated cli.py for review before continuing.**

---

## Section 4 — Dry-run test (no models needed)

Before running the full pipeline (which needs AnimateDiff models loaded), do a dry-run that verifies every non-AI step works correctly.

Create a test script at `C:\AI-Lab\LocalVideoStudio\test_inject_dryrun.py`:

```python
"""
Dry-run test for inject_scene pipeline.
Tests every step except AnimateDiff generation (skips model load).
Requires: ffmpeg on PATH, enhanced.mp4 in C:\AI-Lab\OutputVideo\
"""
import subprocess
import tempfile
import shutil
from pathlib import Path

# Patch AnimateDiffGenerator so it produces a solid-colour test clip
# instead of loading the real model.
# This lets us test split, crossfade, concat, extract_frame without GPU.

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

        # get part1 duration
        result = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(part1)
        ], capture_output=True, text=True, check=True)
        part1_dur = float(result.stdout.strip())
        offset = part1_dur - 0.5

        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(part1), "-i", str(loop_graded),
            "-filter_complex",
            f"xfade=transition=fade:duration=0.5:offset={offset:.3f}",
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

        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(loop_graded), "-i", str(part2),
            "-filter_complex",
            f"xfade=transition=fade:duration=0.5:offset={offset2:.3f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(xfade_out),
        ], check=True, capture_output=True)
        assert xfade_out.exists() and xfade_out.stat().st_size > 0, "xfade_out empty"
        print("  OK")

        print("Step 6: concat xfade_in + xfade_out...")
        final = tmp / "final_test.mp4"
        concat_txt = tmp / "concat.txt"
        concat_txt.write_text(
            f"file '{xfade_in}'\nfile '{xfade_out}'\n", encoding="utf-8"
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
        print("Open and verify the video: game plays to 10s, blue clip for 4s, game resumes.")

    except Exception as e:
        print(f"\nDRY-RUN FAILED: {e}")
        raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    run_dryrun()
```

Run it with:
```
C:\AI-Lab\.venv\Scripts\python.exe C:\AI-Lab\LocalVideoStudio\test_inject_dryrun.py
```

Expected output: all steps print OK, and `test_inject_dryrun.mp4` appears in `C:\AI-Lab\OutputVideo\`.

The user should open `test_inject_dryrun.mp4` and verify:
- Game plays normally from 0 to ~10 seconds
- Dissolves into a blue clip for ~4 seconds
- Dissolves back and game resumes from ~10 seconds

**PAUSE HERE. Run the dry-run, show the output to the user, and wait for them to confirm the video looks correct before continuing.**

---

## Section 5 — Full integration test (requires AnimateDiff models)

Only run this if the user confirms the dry-run passed AND the AnimateDiff models are present at:
- `C:\AI-Lab\Models\AnimateDiff-MotionAdapter\`
- `C:\AI-Lab\Models\AnimateDiff-BaseModel\`

Run the full inject command:

```
C:\AI-Lab\.venv\Scripts\python.exe -m Inference.cli inject ^
    --input "C:\AI-Lab\OutputVideo\enhanced.mp4" ^
    --output "C:\AI-Lab\OutputVideo\with_scene.mp4" ^
    --timestamp 10.0 ^
    --prompt "A footballer stands in dramatic golden light, slow motion, cinematic anime style, high detail" ^
    --duration 4 ^
    --transition 0.5
```

Expected: `with_scene.mp4` in OutputVideo. The game plays to 10s, dissolves into the AI-generated scene for ~4 seconds, then dissolves back to the game continuing from 10s.

**PAUSE HERE. Ask the user to open with_scene.mp4 and describe what they see.**

---

## Section 6 — Commit

Once the user confirms everything works:

1. Stage only the files you changed:
   - `LocalVideoStudio/Inference/processors/crossfade_transitioner.py` (new)
   - `LocalVideoStudio/Inference/interfaces/video_processor.py` (CrossfadeConfig added)
   - `LocalVideoStudio/Inference/pipeline/studio_pipeline.py` (inject_scene added)
   - `LocalVideoStudio/Inference/cli.py` (inject command updated)
   - `LocalVideoStudio/test_inject_dryrun.py` (test script)

2. Commit message:
```
Add inject_scene: AnimateDiff loop with crossfade transitions

New inject subcommand takes an already-enhanced video, a timestamp
and a text prompt. Splits at the timestamp, generates a ping-pong
AnimateDiff loop, grades it to match the game footage, then dissolves
it in and out with ffmpeg xfade so the game resumes from the same frame.

New: CrossfadeTransitioner processor, CrossfadeConfig, inject_scene()
     in StudioPipeline, updated CLI inject subcommand, dry-run test.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

---

## Constraints — read carefully

- **No Unicode in print()** — Windows console is cp1252. Use `->` not `→`, `...` not `…`, `x` not `×`.
- **SOLID principles** — each class has one job. CrossfadeTransitioner does only crossfade. It does not know about AnimateDiff or enhancement.
- **No new dependencies** — everything uses ffmpeg (already on PATH) and the existing diffusers/torch stack. Do not pip install anything.
- **Temp files only in _TempDir** — never write to the project directory during processing.
- **ASCII-safe file paths in concat.txt** — use forward slashes or escape backslashes.
- **EnhancementConfig for loop grading** — pass `auto_crop=False, cinematic_bars=False` when enhancing the AI-generated loop. The loop has no native bars and should not get letterbox bars added.
- **h264_nvenc with yuv420p** — all re-encoded output must be playable in Windows Media Player.
- **Prefer editing existing files over creating new ones** — except CrossfadeTransitioner which must be a new file.
- **Ask before assuming** — if any requirement is unclear, stop and ask the user.
