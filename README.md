# LocalVideoStudio

Fully local (offline after setup) AI video production toolkit for game recordings.

Two main capabilities:

1. **Enhance** — takes any screen recording and makes it look like a broadcast sports production: 60fps frame interpolation, Hollywood colour grade, film grain, and vignette.
2. **Inject** — generates a seamless cinematic loop from a text prompt using AnimateDiff and splices it into the recording at a chosen timestamp (for adding story / dialogue scenes to a game video).

---

## Hardware

| | Minimum | This project |
|---|---|---|
| GPU | NVIDIA RTX (8 GB VRAM) | RTX 5060 Laptop 8 GB |
| CUDA | 12+ | 13 |
| Python | 3.10+ | 3.12 |
| OS | Windows 10+ | Windows 11 |

---

## Repository layout

```
C:\AI-Lab\
│
├── LocalVideoStudio\           ← this project
│   ├── Inference\
│   │   ├── cli.py              ← entry point (python -m Inference.cli ...)
│   │   ├── config.py           ← all path constants
│   │   ├── generators\
│   │   │   └── animatediff_generator.py
│   │   ├── interfaces\
│   │   │   ├── video_generator.py
│   │   │   └── video_processor.py
│   │   ├── pipeline\
│   │   │   └── studio_pipeline.py
│   │   └── processors\
│   │       ├── rife_interpolator.py
│   │       └── video_enhancer.py
│   └── requirements.txt
│
├── ECCV2022-RIFE-main\         ← RIFE frame interpolation (separate clone)
│   └── train_log\
│       └── flownet.pkl         ← model weights (download separately, see below)
│
├── Models\                     ← model weights (git-ignored)
│   ├── AnimateDiff-MotionAdapter\
│   └── AnimateDiff-BaseModel\
│
├── InputVideo\                 ← drop source recordings here (git-ignored)
└── OutputVideo\                ← enhanced / injected videos land here (git-ignored)
```

---

## Setup

### 1. Python dependencies

Activate the project venv, then install from inside `LocalVideoStudio\`:

```
C:\AI-Lab\.venv\Scripts\activate
pip install -r C:\AI-Lab\LocalVideoStudio\requirements.txt
```

### 2. ffmpeg

Download the full build from <https://www.gyan.dev/ffmpeg/builds/> (`ffmpeg-release-full.7z`), extract, and add the `bin\` folder to your PATH.
Verify: `ffmpeg -version` and `ffprobe -version` both print output.

---

## Model downloads

Three models are needed. **None are included in this repo** (they are git-ignored under `Models\`).

---

### Model 1 — AnimateDiff Motion Adapter  (~1.7 GB total, fp16 only needed)

**Where it goes:** `C:\AI-Lab\Models\AnimateDiff-MotionAdapter\`

**Files required:**

| File | Size | Purpose |
|------|------|---------|
| `config.json` | 2 KB | adapter architecture |
| `diffusion_pytorch_model.fp16.safetensors` | 190 MB | fp16 weights ✓ |

> The full repo also contains a non-fp16 `.safetensors` (190 MB, redundant) and several JSON files. You can download everything or just the two files above — both work.

**Download command** (run with venv active):

```
C:\AI-Lab\.venv\Scripts\hf.exe download guoyww/animatediff-motion-adapter-v1-5-3 ^
    --local-dir "C:\AI-Lab\Models\AnimateDiff-MotionAdapter"
```

> `hf` is the huggingface_hub 2.x CLI (renamed from `huggingface-cli`).

---

### Model 2 — SD 1.5 Base Model checkpoint  (~2 GB)

**Where it goes:** `C:\AI-Lab\Models\AnimateDiff-BaseModel\`

**Exact file to download:**

| File | Size | Notes |
|------|------|-------|
| `Realistic_Vision_V5.1_fp16-no-ema.safetensors` | ~2.0 GB | single-file checkpoint, fp16, no EMA |

The code auto-detects any single `.safetensors` file in this folder — the filename does not matter.

**Option A — browser download from HuggingFace (recommended):**

1. Go to: <https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE/tree/main>
2. Click **`Realistic_Vision_V5.1_fp16-no-ema.safetensors`** → Download
3. Move the file to `C:\AI-Lab\Models\AnimateDiff-BaseModel\`

**Option B — direct URL (paste in browser or use wget):**

```
https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE/resolve/main/Realistic_Vision_V5.1_fp16-no-ema.safetensors
```

> **Do NOT download the full repo** — it contains fp32 weights + civitai checkpoint + preview images totalling ~25 GB.
> Only the single `fp16-no-ema` file is needed.

**Alternative base model — DreamShaper 8** (better for fantasy/stylised game scenes):

| File | Size | URL |
|------|------|-----|
| `dreamshaper_8.safetensors` | ~2.0 GB | <https://huggingface.co/Lykon/dreamshaper-8/resolve/main/DreamShaper_8_pruned.safetensors> |

Same destination folder. Remove or rename any existing `.safetensors` first so exactly one file is present.

---

### Model 3 — RIFE frame interpolation weights  (~80 MB)

**Where it goes:** `C:\AI-Lab\ECCV2022-RIFE-main\train_log\flownet.pkl`

RIFE source is already cloned. Only the weight file is needed separately.

**Download:**

Google Drive: <https://drive.google.com/file/d/1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_/view>

Place the downloaded `flownet.pkl` at exactly:

```
C:\AI-Lab\ECCV2022-RIFE-main\train_log\flownet.pkl
```

---

## Verify downloads

```
dir "C:\AI-Lab\Models\AnimateDiff-MotionAdapter"
# expect: config.json  diffusion_pytorch_model.fp16.safetensors

dir "C:\AI-Lab\Models\AnimateDiff-BaseModel"
# expect: exactly ONE .safetensors file (~2 GB)

dir "C:\AI-Lab\ECCV2022-RIFE-main\train_log"
# expect: flownet.pkl  IFNet_HDv3.py  RIFE_HDv3.py
```

---

## Usage

Run all commands from `C:\AI-Lab\LocalVideoStudio` with the venv active.

### Enhance a game recording

```
python -m Inference.cli enhance ^
    --input  "C:\AI-Lab\InputVideo\input.mp4" ^
    --output "C:\AI-Lab\OutputVideo\enhanced.mp4"
```

What happens:
1. `cropdetect` — removes the game's native letterbox bars automatically
2. RIFE — doubles frame rate (30 fps → 60 fps)
3. ffmpeg filter chain — denoise → sharpen → contrast S-curve → orange-teal grade → vignette → film grain

### Inject a cinematic loop

```
python -m Inference.cli inject ^
    --input     "C:\AI-Lab\InputVideo\input.mp4" ^
    --output    "C:\AI-Lab\OutputVideo\with_loop.mp4" ^
    --timestamp 15.0 ^
    --prompt    "A warrior stands at a castle gate at sunset, dramatic lighting, cinematic, 4k"
```

AnimateDiff generates a seamless ping-pong loop and splices it in at 15 seconds.

---

## CLI flags

### `enhance`

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | required | source video |
| `--output` | required | output path |
| `--exp` | `1` | frame interpolation exponent (1=2×, 2=4×) |
| `--no-autocrop` | off | skip native black-bar removal |
| `--no-grade` | off | skip colour grade |
| `--no-bars` | off | skip cinematic bars (bars are auto-skipped when autocrop fires) |
| `--no-denoise` | off | skip denoise + sharpen |
| `--no-fp16` | off | run RIFE in fp32 |

### `inject`

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | required | source video |
| `--output` | required | output path |
| `--timestamp` | required | injection point in seconds |
| `--prompt` | required | text description of the loop scene |
| `--loop-duration` | `4` | target loop length in seconds |
| `--width` / `--height` | `512` / `512` | generation resolution |
| `--steps` | `25` | diffusion inference steps |
| `--cfg` | `7.5` | classifier-free guidance scale |
| `--seed` | `-1` | `-1` = random |

---

## Enhancement pipeline detail

```
input.mp4
  └─ [cropdetect]  detect & remove game's native letterbox bars
  └─ [RIFE]        30 fps → 60 fps (optical-flow frame interpolation)
  └─ [hqdn3d]      temporal + spatial denoise
  └─ [unsharp]     5×5 luma sharpen 0.8 strength
  └─ [curves S]    contrast S-curve: crush shadows, punch midtones
  └─ [curves OT]   orange-teal split: warm highlights, teal shadows
  └─ [eq]          saturation +30%, micro-contrast, slight underexposure
  └─ [vignette]    PI/8 soft edge darkening
  └─ [noise]       film grain strength 8
  └─ [h264_nvenc]  GPU encode, yuv420p, CQ 23
     enhanced.mp4
```
