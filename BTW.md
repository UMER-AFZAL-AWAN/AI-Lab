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
