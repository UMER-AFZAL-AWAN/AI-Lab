# =============================================================================
# 17_data_pipelines / 02 — VIDEO FRAME EXTRACTION
# =============================================================================
#
# LIKE YOU ARE 5:
# A video is just a sequence of still photos played fast (24-30 per second).
# To train a video AI model, we need to:
# 1. Take the video apart into individual frames (photos)
# 2. Select N frames (e.g., 16 frames from a 2-second clip)
# 3. Turn them into tensors the model can read
#
# C# ANALOGY:
# Like reading bytes from a stream — but frame by frame.
# VideoCapture = FileStream for video
# cap.read() = stream.Read(buffer)
# frame_number / fps = timestamp
#
# TOOLS:
# cv2 (OpenCV) — the standard library for video reading in Python
# ffmpeg — command-line tool, faster for large-scale processing
# decord — GPU-accelerated video decoding (best for ML pipelines)
#
# =============================================================================

import torch
from pathlib import Path

print("=" * 60)
print("PART 1: cv2 VIDEO READING (the standard approach)")
print("=" * 60)
print("""
cv2.VideoCapture opens a video file.
cap.read() returns (success, frame) where frame is a numpy BGR array.

Key properties:
  cap.get(cv2.CAP_PROP_FPS)         → frames per second
  cap.get(cv2.CAP_PROP_FRAME_COUNT) → total frames
  cap.get(cv2.CAP_PROP_FRAME_WIDTH) → width in pixels
  cap.get(cv2.CAP_PROP_FRAME_HEIGHT)→ height in pixels
""")

# Full cv2 code (runnable when cv2 is installed):
cv2_extract_code = '''
import cv2
import numpy as np
from PIL import Image

def extract_frames(video_path, num_frames=16, target_fps=None):
    """
    Extract num_frames evenly spaced frames from video_path.
    Returns list of PIL Images.
    """
    cap = cv2.VideoCapture(str(video_path))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)

    # Evenly spaced frame indices
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if not success:
            break
        # cv2 reads BGR → convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames

# Usage:
frames = extract_frames("my_video.mp4", num_frames=16)
# frames: list of 16 PIL Images
'''

print("cv2 frame extraction code:")
print(cv2_extract_code)

print("=" * 60)
print("PART 2: FFMPEG (command-line, fastest for bulk extraction)")
print("=" * 60)
print("""
ffmpeg is a CLI tool for video processing.
Faster than cv2 for extracting many frames.

Extract all frames:
  ffmpeg -i video.mp4 frames/frame_%06d.jpg -q:v 2

Extract at specific FPS (e.g., 8 fps from a 30fps video):
  ffmpeg -i video.mp4 -vf fps=8 frames/frame_%06d.jpg

Extract specific time range (seconds 5 to 10):
  ffmpeg -ss 5 -to 10 -i video.mp4 frames/frame_%06d.jpg

Resize while extracting (512 width, preserve aspect ratio):
  ffmpeg -i video.mp4 -vf "scale=512:-1" frames/frame_%06d.jpg

Frame naming:
  frame_%06d.jpg → frame_000001.jpg, frame_000002.jpg, ...
  The %06d pads to 6 digits → sorted correctly by filename.

Running from Python:
  import subprocess
  subprocess.run(["ffmpeg", "-i", "video.mp4", "-vf", "fps=8",
                  "frames/frame_%06d.jpg"], check=True)
""")

print("=" * 60)
print("PART 3: SAMPLING STRATEGIES")
print("=" * 60)
print("""
Given a video of T total frames, how to select N frames?

1. UNIFORM SAMPLING (most common for training):
   indices = linspace(0, T-1, N)
   → Evenly spaced, guaranteed N frames
   → Best for capturing overall motion

2. STRIDE SAMPLING (AnimateDiff default):
   stride = T // N
   indices = [i * stride for i in range(N)]
   → Regular intervals, easy to reason about
   → Used in AnimateDiff training (stride=1,2,4)

3. RANDOM TEMPORAL CROP (for data augmentation):
   start = random(0, T - N)
   indices = range(start, start + N)
   → N consecutive frames starting at a random position
   → Introduces temporal diversity during training

4. KEY FRAME EXTRACTION (for content understanding):
   Use scene detection to find scene boundaries
   Extract one frame per scene
""")

def uniform_sample(total_frames, num_frames):
    import numpy as np
    return np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

def stride_sample(total_frames, num_frames, stride=None):
    if stride is None:
        stride = max(1, total_frames // num_frames)
    return [min(i * stride, total_frames - 1) for i in range(num_frames)]

def random_crop_sample(total_frames, num_frames):
    if total_frames <= num_frames:
        return list(range(total_frames))
    start = torch.randint(0, total_frames - num_frames, (1,)).item()
    return list(range(start, start + num_frames))

T, N = 120, 16   # 4-second clip at 30fps, want 16 frames
print(f"Video: {T} frames, want {N} frames")
print(f"Uniform:   {uniform_sample(T, N)}")
print(f"Stride:    {stride_sample(T, N)}")
print(f"RandCrop:  {random_crop_sample(T, N)}")

print()
print("=" * 60)
print("PART 4: FRAME → TENSOR PIPELINE")
print("=" * 60)

def pil_frames_to_tensor(frames, size=512):
    """
    Convert a list of PIL Images to a video tensor.

    Input:  list of N PIL Images (H×W×RGB)
    Output: torch.Tensor of shape (C, T, H, W), normalised to [-1, 1]
    """
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),           # 0 to 1
        transforms.Normalize([0.5]*3, [0.5]*3),  # -1 to 1
    ])

    tensors = [transform(frame) for frame in frames]   # list of (C, H, W)
    video   = torch.stack(tensors)                      # (T, C, H, W)
    return video.permute(1, 0, 2, 3)                    # (C, T, H, W)

print("""
Pipeline:
  PIL Image list (N × H × W × 3)
      ↓ Resize, ToTensor, Normalize
  Tensor list (N × 3 × 512 × 512)
      ↓ stack → permute
  Video tensor (3 × N × 512 × 512)  ←  AnimateDiff expects this shape

Shape convention:
  (C, T, H, W) = (channels, time, height, width)
  = (3, 16, 512, 512) for a typical AnimateDiff clip
""")

# Simulate with random "frames" (no actual video needed)
fake_pil_frames = [
    __import__('PIL.Image', fromlist=['Image']).Image.new('RGB', (256, 256))
    for _ in range(16)
]

video_tensor = pil_frames_to_tensor(fake_pil_frames, size=64)
print(f"16 × 256×256 PIL frames → tensor shape: {video_tensor.shape}")
print(f"Value range: [{video_tensor.min().item():.1f}, {video_tensor.max().item():.1f}]  (all same colour → all 1s normalised to same)")

print()
print("=" * 60)
print("PART 5: DECORD (fastest for ML pipelines)")
print("=" * 60)
print("""
decord is a library optimised for ML video loading.
Faster than cv2 for seeking and batch frame access.

  from decord import VideoReader, cpu

  vr     = VideoReader("video.mp4", ctx=cpu(0))
  fps    = vr.get_avg_fps()
  T      = len(vr)

  # Read N frames at once (much faster than a loop with cap.set)
  indices = list(range(0, T, T // 16))[:16]
  frames  = vr.get_batch(indices).asnumpy()   # (N, H, W, 3) numpy, uint8

  # Convert to torch
  tensor  = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

Install: pip install decord
Not available on Windows arm64. Use cv2 on Windows.
""")

print("=" * 60)
print("QUICK REFERENCE: TOOL COMPARISON")
print("=" * 60)
print("""
Library   | Speed    | Windows? | Notes
──────────────────────────────────────────────────────────────
cv2       | Medium   | Yes      | Most common, BGR output (not RGB!)
ffmpeg    | Fastest  | Yes      | CLI, batch extraction, needs install
decord    | Fast     | Partial  | Best for ML, GPU decode, no Windows arm64
PIL       | Slow     | Yes      | Only for single-image tasks
imageio   | Slow     | Yes      | Simple, GIF support

For your RTX 5060 setup: use ffmpeg for bulk extraction,
cv2 for in-code reading, PIL for single-frame operations.
""")

# =============================================================================
# EXPECTED OUTPUT:
# Uniform/Stride/Random frame indices printed
# Video tensor shape: (3, 16, 64, 64)
# No errors (all runs without actual video files)
# =============================================================================
