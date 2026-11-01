# BTW: VFR and Why It Broke RIFE

This note explains a subtle bug in phone screen recordings that caused the
RIFE frame-interpolation step to produce only the first half of the video —
and what was done to fix it.

---

## CFR vs VFR

**CFR (Constant Frame Rate)** — every frame is exactly 1/30s apart. Simple, predictable.

```
Frame 0    Frame 1    Frame 2    Frame 3
|----------|----------|----------|----------|
0ms       33ms       66ms       99ms      132ms
```

**VFR (Variable Frame Rate)** — frames arrive at irregular intervals. Some gaps are longer, some shorter.

```
Frame 0   Frame 1        Frame 2   Frame 3
|---------|--------------|---------|--------|
0ms      33ms           99ms     132ms    148ms
```

---

## Why game recordings are VFR

When you screen-record EA FC Mobile on a phone:
- The game itself runs at ~30fps
- The phone's screen recorder declares `r_frame_rate = 59.94` in the container (that is what the screen hardware runs at)
- To fill the gap, it duplicates frames — frame 1 appears twice, frame 2 appears twice, etc.

So the container says "60fps, 1608 frames" but the actual unique content is "30fps, 804 frames."

---

## Why this broke RIFE

RIFE has a duplicate-frame guard: if two consecutive frames are nearly identical
(SSIM > 0.996), it **skips one** — assuming it is a static/frozen moment not
worth interpolating.

Those duplicated frames from VFR recording have SSIM ≈ 1.0 (they are literally
the same image). So RIFE skipped every other frame, effectively seeing only the
odd-numbered frames and consuming the video **twice as fast** as it should.
Result: only the first half of the game appeared in the output.

---

## The fix

Before RIFE sees the file, normalise to true CFR:

```
ffmpeg -vf fps=30 -vsync cfr input.mp4 cfr_input.mp4
```

This collapses all duplicates down to one, giving RIFE 804 truly unique frames
at a steady 30fps. No duplicates → SSIM check never fires → all 804 frames
processed → full game in the output.

This normalisation step runs inside `RifeInterpolator.process()` before the
RIFE model sees the file.

---

# BTW: inject_scene — How the Freeze Frame Anchor Works

The inject feature splices an AI-generated scene into the enhanced game video
at a user-chosen timestamp T. The visual requirement is:

> Game plays → **freezes on exact frame T** → story scene plays → story
> dissolves back to the **same frozen frame** → game resumes from T.

---

## Why two separate xfades are wrong

A naive approach calls CrossfadeTransitioner twice:

```
xfade_in  = crossfade(part1 → loop)
xfade_out = crossfade(loop  → part2)
output    = concat(xfade_in, xfade_out)
```

Two bugs:

1. **No freeze.** The last frame of `part1` is a mid-action game frame (moving
   content). The viewer sees a busy action shot dissolve into unrelated AI
   content — jarring, no visual anchor.

2. **AI loop appears twice.** `xfade_in` ends with the loop playing. `xfade_out`
   starts with the loop playing. Concat puts both back to back. The story
   plays twice.

---

## The fix — tpad + chained xfade

`ffmpeg`'s `tpad` filter clones the last (or first) frame of a clip:

```
# Freeze the last frame of part1 for 1 second
ffmpeg -i part1.mp4 -vf "tpad=stop_mode=clone:stop=60" part1_frozen.mp4

# Freeze the first frame of part2 for 1 second
ffmpeg -i part2.mp4 -vf "tpad=start_mode=clone:start=60" part2_frozen.mp4
```

The last frame of `part1` IS the exact game frame at T. Cloning it means
`part1_frozen` ends with 1 second of that frame held still — no cut, same image.
`part2_frozen` starts with 1 second of the first game frame after T — also
the same image (the cut happens at the same timestamp). The viewer sees the
game pause naturally.

Then **one** `ffmpeg` call chains both dissolves:

```
ffmpeg -i part1_frozen -i ai_loop -i part2_frozen \
  -filter_complex "
    [0:v][1:v] xfade=fade:0.5s at (dur_A - 0.5) [x1];
    [x1][2:v]  xfade=fade:0.5s at (dur_A + dur_B - 1.0) [out]
  " output.mp4
```

Three inputs, two xfades, one output. The AI loop appears exactly once.

---

## What the viewer sees

```
0s ──────── T ─── T+1s ─── T+1.5s ─── T+5.5s ─── T+6s ─── T+7s ───── end
  game plays  freeze  dissolve→  story plays  ←dissolve  freeze  game resumes
              (tpad)  (0.5s)     (4s AI loop)  (0.5s)    (tpad)
```

---

## Key ffmpeg detail — offset arithmetic in a chained xfade

After the first xfade, the output stream is a single timeline. The second
xfade's `offset` is measured from the **start of that combined stream**, not
from the start of the second clip:

```
offset1 = dur(part1_frozen) - transition          # just before part1 ends
offset2 = dur(part1_frozen) + dur(ai_loop) - 2 * transition  # just before loop ends
```

If you use `offset = dur(ai_loop) - transition` for the second xfade (relative
to ai_loop's own start), it will fire too early and corrupt the output.
