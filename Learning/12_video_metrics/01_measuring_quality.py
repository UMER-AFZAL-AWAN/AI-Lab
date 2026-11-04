# =============================================================================
# 12_video_metrics / 01 — MEASURING VIDEO QUALITY
# =============================================================================
#
# LIKE YOU ARE 5:
# How do you know if your generated video is good?
# You need a measuring tape, not just your eyes.
# This file shows the measuring tapes researchers use.
# Some compare to a "reference" video (is it close?).
# Some don't need a reference (is it realistic-looking?).
# Some measure if the words matched the video (did the prompt work?).
#
# C# ANALOGY:
# Like Assert.AreEqual() for videos — but the "expected" is fuzzy.
# PSNR/SSIM = strict numerical diff (like Assert.AreEqual on pixel values)
# FVD/FID   = statistical diff (like Assert that distributions match)
# CLIP score = semantic diff (like Assert the concept is present)
#
# WHEN YOU NEED THIS:
# When you compare AnimateDiff vs Wan on the same prompt.
# When you tweak settings (steps, CFG, scheduler) and want a number.
# When you read a paper and want to understand what their numbers mean.
#
# =============================================================================

import torch
import numpy as np
from PIL import Image

print("=" * 60)
print("THE FOUR METRICS AND WHAT THEY MEASURE")
print("=" * 60)
print("""
Metric       | Needs reference? | What it measures          | Good score
─────────────────────────────────────────────────────────────────────────
PSNR         | YES              | Pixel-level accuracy       | > 30 dB
SSIM         | YES              | Structural similarity      | > 0.9
FID          | NO (uses stats)  | Per-frame realism          | < 50
FVD          | NO (uses stats)  | Video temporal realism     | < 300
CLIP score   | NO               | Text-video alignment       | > 0.25

PSNR/SSIM: compare YOUR video to a GROUND-TRUTH video pixel by pixel.
  → Useful for: restoration tasks (denoising, super-res, interpolation)
  → NOT useful for: generation (there's no "correct" output)

FID: compare the distribution of YOUR frames to a large reference set.
  → Uses an Inception-v3 feature extractor; no per-video reference needed.
  → Lower = your frames look more like real videos statistically.

FVD: like FID but uses an I3D (video) feature extractor that sees MOTION.
  → Lower = your video's motion dynamics look realistic.
  → Gold standard for text-to-video papers (AnimateDiff, Wan, Sora papers use FVD).

CLIP score: embed your frames + your prompt into CLIP space; measure cosine similarity.
  → Measures how well the visuals MATCH the text.
  → High CLIP score = video looks like what you asked for.
""")

print("=" * 60)
print("PSNR — PEAK SIGNAL-TO-NOISE RATIO")
print("=" * 60)

def psnr(img_true, img_pred, max_val=255.0):
    """
    img_true, img_pred: numpy arrays (H, W, C) or (H, W), uint8 or float
    Returns: PSNR in dB. Higher = better. Infinity = perfect match.
    """
    img_true = img_true.astype(np.float64)
    img_pred = img_pred.astype(np.float64)
    mse = np.mean((img_true - img_pred) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))

np.random.seed(42)
frame_original = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

# Scenario 1: tiny noise
frame_slight_noise = np.clip(frame_original.astype(int) + np.random.randint(-5, 5, frame_original.shape), 0, 255).astype(np.uint8)
# Scenario 2: heavy noise
frame_heavy_noise  = np.clip(frame_original.astype(int) + np.random.randint(-50, 50, frame_original.shape), 0, 255).astype(np.uint8)
# Scenario 3: totally different
frame_random       = np.random.randint(0, 256, frame_original.shape, dtype=np.uint8)

print(f"Perfect match:    PSNR = {psnr(frame_original, frame_original):.1f} dB  (∞)")
print(f"Slight noise:     PSNR = {psnr(frame_original, frame_slight_noise):.1f} dB  (high — good)")
print(f"Heavy noise:      PSNR = {psnr(frame_original, frame_heavy_noise):.1f} dB  (low — bad)")
print(f"Random image:     PSNR = {psnr(frame_original, frame_random):.1f} dB  (very low)")
print()
print("Interpretation:")
print("  > 40 dB  → visually indistinguishable (lossless quality)")
print("  30–40 dB → good quality (typical for video compression)")
print("  20–30 dB → noticeable degradation")
print("  < 20 dB  → heavily distorted")

print()
print("=" * 60)
print("SSIM — STRUCTURAL SIMILARITY INDEX")
print("=" * 60)

def ssim_simple(img1, img2, C1=6.5025, C2=58.5225):
    """
    Simplified single-channel SSIM.
    Real SSIM uses a Gaussian window; this uses global statistics for clarity.
    img1, img2: float arrays (H, W)
    Returns: float in [-1, 1]. 1.0 = identical.
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mu1, mu2 = img1.mean(), img2.mean()
    s1 = img1.std() ** 2
    s2 = img2.std() ** 2
    s12 = np.mean((img1 - mu1) * (img2 - mu2))
    numerator   = (2*mu1*mu2 + C1) * (2*s12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (s1 + s2 + C2)
    return numerator / denominator

ch = 0  # compare one channel
print(f"Perfect match:    SSIM = {ssim_simple(frame_original[:,:,ch].astype(float), frame_original[:,:,ch].astype(float)):.4f}")
print(f"Slight noise:     SSIM = {ssim_simple(frame_original[:,:,ch].astype(float), frame_slight_noise[:,:,ch].astype(float)):.4f}")
print(f"Heavy noise:      SSIM = {ssim_simple(frame_original[:,:,ch].astype(float), frame_heavy_noise[:,:,ch].astype(float)):.4f}")
print(f"Random image:     SSIM = {ssim_simple(frame_original[:,:,ch].astype(float), frame_random[:,:,ch].astype(float)):.4f}")
print()
print("Interpretation:")
print("  > 0.95  → visually similar")
print("  0.8–0.95 → minor structural changes")
print("  < 0.7   → significant structural difference")
print("  (For VIDEO generation, don't use SSIM — there's no ground truth)")

print()
print("=" * 60)
print("CLIP SCORE — TEXT-VIDEO ALIGNMENT")
print("=" * 60)
print("""
# Install: pip install open-clip-torch  OR  use transformers CLIP

import open_clip
import torch
from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32",
                                                               pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model.eval()

def clip_score(frames, prompt):
    '''
    frames: list of PIL Images
    prompt: str
    Returns: float, cosine similarity (0–1, higher = better alignment)
    '''
    text_tokens = tokenizer([prompt])

    frame_scores = []
    for frame in frames:
        img_tensor = preprocess(frame).unsqueeze(0)
        with torch.no_grad():
            img_feat  = model.encode_image(img_tensor)
            text_feat = model.encode_text(text_tokens)
        img_feat  = img_feat  / img_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        score = (img_feat * text_feat).sum().item()
        frame_scores.append(score)

    return sum(frame_scores) / len(frame_scores)

score = clip_score(generated_frames, "a cat playing piano")
print(f"CLIP score: {score:.3f}")
# > 0.25 is considered good alignment in literature
""")

print("CLIP SCORE WITHOUT DOWNLOADING:")
print("Simulating cosine similarity between two random unit vectors:")
torch.manual_seed(42)
dim = 512
img_feat  = torch.randn(1, dim); img_feat  = img_feat  / img_feat.norm()
text_feat = torch.randn(1, dim); text_feat = text_feat / text_feat.norm()
random_score = (img_feat * text_feat).sum().item()

aligned_feat = img_feat + 0.3 * torch.randn_like(img_feat)
aligned_feat = aligned_feat / aligned_feat.norm()
aligned_score = (aligned_feat * text_feat).sum().item()

print(f"  Random image vs text:  {random_score:.3f}  (baseline — no alignment)")
print(f"  Aligned image vs text: {aligned_score:.3f}  (higher when image matches text)")

print()
print("=" * 60)
print("FID AND FVD: THE HARD ONES")
print("=" * 60)
print("""
FID (Fréchet Inception Distance) — per-frame quality:

  1. Take 10,000+ real video frames (reference distribution)
  2. Take your generated frames (test distribution)
  3. Run both through InceptionV3, get the 2048-dim feature vectors
  4. Fit a Gaussian to each set of vectors: μ_real, Σ_real, μ_gen, Σ_gen
  5. FID = ||μ_real - μ_gen||² + Tr(Σ_real + Σ_gen - 2√(Σ_real·Σ_gen))

  Lower FID = your frames' feature distribution is closer to real video frames.

  Practical use:
    pip install torch-fidelity
    fidelity --input1 real_frames/ --input2 generated_frames/ --fid

FVD (Fréchet Video Distance) — temporal motion quality:

  Same as FID but:
  - Uses an I3D (Inflated 3D ConvNet) feature extractor
  - Extracts CLIP-length (16-frame) features, not single-frame
  - Captures temporal consistency and motion naturalness

  Practical use:
    github.com/google-research/google-research/tree/master/frechet_video_distance
    OR the evaluate library: pip install evaluate
    from evaluate import load
    fvd_metric = load("google/frechet_video_distance")

RULE FOR YOUR EXPERIMENTS:
  Use CLIP score (easy, one number, no reference dataset needed).
  Use PSNR/SSIM only for restoration tasks (RIFE, super-res, denoising).
  Use FVD only when writing a paper or doing formal benchmarks.
""")

print("=" * 60)
print("QUICK BENCHMARK LOGGING PATTERN")
print("=" * 60)
print("""
# After every generation experiment, log these numbers:
import json, time

result = {
    "timestamp"  : time.strftime("%Y-%m-%d %H:%M"),
    "model"      : "animatediff-v1-5-2",
    "prompt"     : "a cat playing piano",
    "num_frames" : 16,
    "steps"      : 25,
    "cfg"        : 7.5,
    "scheduler"  : "DDIM",
    "lora"       : None,
    "clip_score" : 0.287,          # measure with clip_score()
    "peak_vram"  : 5.4,            # from torch.cuda.max_memory_allocated()
    "infer_time" : 42.1,           # seconds
    "notes"      : "baseline run",
}
with open("experiments.jsonl", "a") as f:
    f.write(json.dumps(result) + "\\n")
# → You'll have a table to compare models/settings over time
""")

# =============================================================================
# EXPECTED OUTPUT:
# Perfect match:   PSNR = inf dB
# Slight noise:    PSNR = ~42 dB
# Heavy noise:     PSNR = ~15 dB
# Perfect match:   SSIM = 1.0000
# CLIP score demo: Random ~0.0, Aligned ~0.1-0.2
# =============================================================================
