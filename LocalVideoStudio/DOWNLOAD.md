# Model Download Guide

Run all commands with the `.venv` active:
```
C:\AI-Lab\.venv\Scripts\activate
```

---

## Step 1 — Install Python dependencies

```
pip install -r requirements.txt
```

---

## Step 2 — Download AnimateDiff Motion Adapter (~400 MB)

> **Note:** huggingface_hub 2.x renamed the CLI from `huggingface-cli` to `hf`.
> Use the full venv path to be safe.

```
C:\AI-Lab\.venv\Scripts\hf.exe download guoyww/animatediff-motion-adapter-v1-5-3 --local-dir "C:\AI-Lab\Models\AnimateDiff-MotionAdapter"
```

---

## Step 3 — Download Realistic Vision Base Model (fp16 only, ~2.5 GB)

The HuggingFace repo contains fp32 + fp16 + original civitai checkpoint + preview images = 25 GB total.
We only need the fp16 diffusers files. Use ignore_patterns to skip everything else:

```
C:\AI-Lab\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('SG161222/Realistic_Vision_V5.1_noVAE', local_dir='C:/AI-Lab/Models/AnimateDiff-BaseModel', ignore_patterns=['*.ckpt', '*.pt', '*.bin', '*.png', '*.jpg', '*.gif', '*.safetensors'], allow_patterns=['*.fp16.safetensors', '*.json', '*.txt', '*.model'])"
```

> **Alternative — DreamShaper 8 (~2 GB, better for stylised/fantasy game scenes):**
> ```
> C:\AI-Lab\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('Lykon/dreamshaper-8', local_dir='C:/AI-Lab/Models/AnimateDiff-BaseModel', ignore_patterns=['*.ckpt', '*.pt', '*.bin', '*.png', '*.jpg', '*.gif', '*.safetensors'], allow_patterns=['*.fp16.safetensors', '*.json', '*.txt', '*.model'])"
> ```

---

## Step 4 — Verify RIFE weights are present

The RIFE model file should already exist at:
```
C:\AI-Lab\ECCV2022-RIFE-main\train_log\flownet.pkl
```
If it is missing, download from:
https://drive.google.com/file/d/1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_/view

---

## Total disk space needed

| Asset                          | Size   |
|-------------------------------|--------|
| AnimateDiff Motion Adapter    | ~400 MB |
| Realistic_Vision V5.1 Base    | ~2.0 GB |
| RIFE weights (already present)| ~200 MB |
| **Total**                     | **~2.6 GB** |

---

## Test the pipeline (after downloads)

```
cd C:\AI-Lab\LocalVideoStudio

# Test 1: Enhance your game recording
python -m Inference.cli enhance ^
  --input  "C:\AI-Lab\InputVideo\input.mp4" ^
  --output "C:\AI-Lab\OutputVideo\enhanced.mp4"

# Test 2: Generate a cinematic loop at 15 seconds
python -m Inference.cli inject ^
  --input     "C:\AI-Lab\InputVideo\input.mp4" ^
  --output    "C:\AI-Lab\OutputVideo\with_cinematic_loop.mp4" ^
  --timestamp 15.0 ^
  --prompt    "A warrior stands at the edge of a cliff at sunset, epic cinematic, dramatic lighting, 4k"
```
