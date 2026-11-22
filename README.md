# AI-Lab

Personal local AI toolkit — a growing collection of reusable libraries and
projects for video production, image generation, and machine learning, all
running offline on consumer hardware.

**Hardware target:** RTX 5060 Laptop, 8 GB VRAM, Windows 11

---

## Repository layout

```
AI-Lab/
  toolkit/              ← Shared libraries. Import these from any project.
    rife/               ← Frame interpolation (RIFE optical flow)
    color/              ← Cinematic color grading and video enhancement
    ffmpeg/             ← FFmpeg/ffprobe utilities (probe, ops, encode)
    animatediff/        ← Text-to-video generation (AnimateDiff)
    logging.py          ← Structured logger used by all libraries

  LocalVideoStudio/     ← CLI video production tool (uses toolkit)
  Learning/             ← ML/AI course — 83 runnable Python files

  Models/               ← Downloaded model weights (not in git)
  ECCV2022-RIFE-main/   ← RIFE source (cloned, not modified)
  InputVideo/           ← Drop source videos here
  OutputVideo/          ← Processed outputs land here
```

---

## toolkit — shared libraries

Each library is independently importable. No library imports from a project.

### `toolkit.rife` — Frame interpolation

Increases video frame rate using RIFE optical-flow interpolation.
Doubles (exp=1) or quadruples (exp=2) the frame rate.
Handles VFR game recordings automatically.

```python
from toolkit.rife import RifeInterpolator, InterpolationConfig

RifeInterpolator().interpolate(
    input_path  = Path("input.mp4"),
    output_path = Path("output_60fps.mp4"),
    config      = InterpolationConfig(exp=1, fp16=True),
)
```

### `toolkit.color` — Cinematic color grading

Applies a full Hollywood-style cinematic pipeline via ffmpeg:
auto-crop → denoise → sharpen → S-curve → orange-teal grade → vignette → film grain → letterbox.

```python
from toolkit.color import VideoEnhancer, EnhancementConfig

VideoEnhancer().enhance(
    input_path  = Path("input.mp4"),
    output_path = Path("cinematic.mp4"),
    config      = EnhancementConfig(cinematic_grade=True, cinematic_bars=True),
)
```

### `toolkit.ffmpeg` — Video I/O utilities

Low-level FFmpeg operations: probe metadata, split, concat, freeze frames, xfade.

```python
from toolkit.ffmpeg import probe, ops

w, h = probe.dimensions(Path("video.mp4"))
dur  = probe.duration(Path("video.mp4"))
ops.split(source, at=15.0, part1=p1, part2=p2)
ops.xfade_chain(clip_a, clip_b, clip_c, output, transition=0.5, fps=60, width=w, height=h)
```

### `toolkit.animatediff` — Text-to-video generation

Generates video clips from a text prompt using AnimateDiff locally.
Supports seamless ping-pong loops. Optimised for 8 GB VRAM.

```python
from toolkit.animatediff import AnimateDiffGenerator, GenerationRequest

gen    = AnimateDiffGenerator()
result = gen.generate_seamless_loop(GenerationRequest(
    prompt     = "a dragon flying over a castle, cinematic, 4k",
    num_frames = 16,
    width      = 512,
    height     = 512,
))
```

### `toolkit.logging` — Structured logger

```python
from toolkit.logging import get_logger
log = get_logger(__name__)
log.step(1, 3, "Interpolating frames")
log.info("rife", "normalising VFR → CFR")
log.done("enhance_video", output_path)
```

---

## LocalVideoStudio

CLI tool for cinematic video production. All AI logic lives in `toolkit/`.
This project only orchestrates.

### Setup

```bash
# From C:\AI-Lab with the venv active
cd LocalVideoStudio
pip install -r requirements.txt
```

Model weights are required. See [LocalVideoStudio/DOWNLOAD.md](LocalVideoStudio/DOWNLOAD.md).

### Commands

**Enhance** — frame interpolation + cinematic grade:

```bash
python -m Inference.cli enhance \
    --input  "InputVideo/input.mp4" \
    --output "OutputVideo/enhanced.mp4"
```

Options: `--exp 2` (4× framerate), `--no-grade`, `--no-bars`, `--no-denoise`, `--no-autocrop`, `--verbose`

**Inject** — generate an AI scene and splice it into a video:

```bash
python -m Inference.cli inject \
    --input     "OutputVideo/enhanced.mp4" \
    --output    "OutputVideo/final.mp4" \
    --timestamp 15.0 \
    --prompt    "A knight standing at a castle gate, dramatic lighting, cinematic"
```

Options: `--duration 4`, `--transition 0.5`, `--width 512`, `--height 512`, `--steps 25`, `--seed 42`, `--verbose`

---

## Learning

83 Python files covering ML/AI from math foundations to deployment.
All files run without downloading models. See [Learning/README.md](Learning/README.md).

---

## Design rules

- **toolkit/ is never aware of any project.** Projects import from toolkit; toolkit never imports from a project.
- **Each toolkit library has one job.** `rife` does interpolation. `color` does grading. `ffmpeg` does I/O. No mixing.
- **Projects are thin orchestrators.** Business logic lives in toolkit. Projects wire things together and expose a CLI or UI.
- **New capability = new toolkit library first**, then wire it into whichever projects need it.
