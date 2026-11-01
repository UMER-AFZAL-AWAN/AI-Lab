# inject_scene -- Freeze Frame Anchor Fix

## What is wrong right now

Read studio_pipeline.py and you will see inject_scene() does this:

  step 5: CrossfadeTransitioner.process(part1, xfade_in, loop_graded)
  step 6: CrossfadeTransitioner.process(loop_graded, xfade_out, part2)
  step 7: concat(xfade_in, xfade_out) -> output

Two bugs:
1. No freeze. The game is moving when it dissolves. The last frame of part1
   is a mid-action frame, not a held pause. The viewer sees busy footage
   blur into unrelated AI content.
2. AI loop appears twice. xfade_in ends with loop_graded playing.
   xfade_out starts with loop_graded playing. Concat puts both together.
   The AI scene plays twice in the output.

## What the user wants

  game plays -> freezes on exact frame at T -> story scene plays ->
  story dissolves back to same frozen frame -> game resumes from T

## The correct approach (tpad, already proven in test_inject_dryrun.py)

test_inject_dryrun.py already shows the right structure. Read it.
It uses the ffmpeg tpad filter -- no PNG extraction needed.

  part1_frozen  = part1 + tpad=stop_mode=clone:stop=<freeze_frames>
                  (clones last frame of part1 for freeze_sec seconds)
  part2_frozen  = tpad=start_mode=clone:start=<freeze_frames> + part2
                  (clones first frame of part2 for freeze_sec seconds)
  ai_loop       = loop_graded (already generated and graded)

Then ONE ffmpeg call, chained filter_complex, 3 inputs, 2 xfades:

  ffmpeg -i part1_frozen -i ai_loop -i part2_frozen
    -filter_complex "
      [0:v]fps=60,scale=W:H[va];
      [1:v]fps=60,scale=W:H[vb];
      [2:v]fps=60,scale=W:H[vc];
      [va][vb]xfade=transition=fade:duration=T:offset=<dur_a - T>[x1];
      [x1][vc]xfade=transition=fade:duration=T:offset=<dur_a + dur_b - 2*T>[out]
    "
    -map [out] -c:v h264_nvenc -pix_fmt yuv420p -preset p4 -cq 23 output.mp4

offset formula:
  offset1 = dur(part1_frozen) - transition_duration
  offset2 = dur(part1_frozen) + dur(ai_loop) - 2 * transition_duration

## Changes required

Read these two files first:
  C:\AI-Lab\LocalVideoStudio\Inference\pipeline\studio_pipeline.py
  C:\AI-Lab\LocalVideoStudio\test_inject_dryrun.py

### studio_pipeline.py -- rewrite inject_scene()

Keep steps 1-4 unchanged (split, extract frame, generate loop, grade loop).
Delete steps 5-7.

Replace with:

  # Step 5: freeze tail/head of splits using tpad
  freeze_fps = 60
  freeze_frames = round(1.0 * freeze_fps)  # 1 second hold

  part1_frozen = tmp / "part1_frozen.mp4"
  subprocess.run([
      "ffmpeg", "-y", "-i", str(part1),
      "-vf", f"tpad=stop_mode=clone:stop={freeze_frames}",
      "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
      str(part1_frozen),
  ], check=True, capture_output=True)

  part2_frozen = tmp / "part2_frozen.mp4"
  subprocess.run([
      "ffmpeg", "-y", "-i", str(part2),
      "-vf", f"tpad=start_mode=clone:start={freeze_frames}",
      "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
      str(part2_frozen),
  ], check=True, capture_output=True)

  # Step 6: chained xfade -- one ffmpeg call, three inputs
  w, h = CrossfadeTransitioner._get_dimensions(input_path)
  dur_a = CrossfadeTransitioner._get_duration(part1_frozen)
  dur_b = CrossfadeTransitioner._get_duration(loop_graded)
  offset1 = max(0.0, dur_a - transition_duration)
  offset2 = max(0.0, dur_a + dur_b - 2 * transition_duration)

  fc = (
      f"[0:v]fps=60,scale={w}:{h}:force_original_aspect_ratio=disable[va];"
      f"[1:v]fps=60,scale={w}:{h}:force_original_aspect_ratio=disable[vb];"
      f"[2:v]fps=60,scale={w}:{h}:force_original_aspect_ratio=disable[vc];"
      f"[va][vb]xfade=transition=fade:duration={transition_duration:.3f}"
      f":offset={offset1:.3f}[x1];"
      f"[x1][vc]xfade=transition=fade:duration={transition_duration:.3f}"
      f":offset={offset2:.3f}[out]"
  )
  subprocess.run([
      "ffmpeg", "-y",
      "-i", str(part1_frozen),
      "-i", str(loop_graded),
      "-i", str(part2_frozen),
      "-filter_complex", fc,
      "-map", "[out]",
      "-c:v", "h264_nvenc", "-pix_fmt", "yuv420p", "-preset", "p4", "-cq", "23",
      str(output_path),
  ], check=True, capture_output=True)

Also remove the _extract_frame call (step 2) -- it is no longer needed.
Remove the frame_png line entirely. Update print labels to [1/6] through [6/6].

crossfade_transitioner.py -- NO changes.

## Verification

### Dry-run (no GPU)

test_inject_dryrun.py already exists and already tests this exact approach.
Run it:

  C:\AI-Lab\.venv\Scripts\python.exe C:\AI-Lab\LocalVideoStudio\test_inject_dryrun.py

PAUSE. Show the user the printed output and ask them to open
  C:\AI-Lab\OutputVideo\test_inject_tpad_dryrun.mp4
and confirm they see:
  0s-10s      : game plays
  ~10s        : game FREEZES (same frame, no cut)
  ~10.5s      : frozen frame dissolves into BLUE
  ~10.5-14.5s : blue
  ~14.5s      : blue dissolves back to same frozen frame
  ~15s        : game resumes from 10s

Do not continue until the user confirms each point.

### Full run (AnimateDiff, after dry-run confirmed)

  C:\AI-Lab\.venv\Scripts\python.exe -m Inference.cli inject ^
    --input "C:\AI-Lab\OutputVideo\enhanced.mp4" ^
    --output "C:\AI-Lab\OutputVideo\with_scene_v2.mp4" ^
    --timestamp 10.0 ^
    --prompt "A footballer stands in golden dramatic light, slow motion, anime cinematic style" ^
    --duration 4 --transition 0.5

PAUSE. Ask the user to open with_scene_v2.mp4 and confirm the freeze and
transitions match the dry-run. Do not commit until they confirm.

## Commit

  Fix inject_scene: tpad freeze anchor + correct 3-clip chained xfade

  The previous implementation crossfaded moving game footage into the AI
  scene and duplicated the loop by concating two separate xfade outputs.

  Fix: use ffmpeg tpad filter to hold the last/first frame of each split
  for 1 second, then chain both dissolves in a single filter_complex call
  so the AI loop appears exactly once.

  Result: game -> freezes at T -> dissolves to story -> dissolves back to
  same frozen frame -> game resumes from T.

  Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

## Hard rules

- ASCII only in all print() text. No unicode arrows or symbols.
- All temp files inside _TempDir.
- Do not modify CrossfadeTransitioner.
- Ask before any assumption not covered above.
