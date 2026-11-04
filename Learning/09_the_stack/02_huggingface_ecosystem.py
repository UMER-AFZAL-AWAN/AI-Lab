# =============================================================================
# 02 — THE HUGGINGFACE ECOSYSTEM
# =============================================================================
#
# LIKE YOU ARE 5:
# HuggingFace is like a giant supermarket for AI models.
# They host the files (the shop).
# They wrote the Python code that loads and runs the models (the packaging).
# They organised everything into standard interfaces (the labels on the shelves).
#
# FOUR KEY LIBRARIES:
# 1. transformers  — text models (GPT, BERT, Llama, Whisper, CLIP)
# 2. diffusers     — image/video generation (Stable Diffusion, AnimateDiff)
# 3. datasets      — loading and processing training data
# 4. accelerate    — training across multiple GPUs, mixed precision
#
# THE HUB:
# huggingface.co/models — 500,000+ pretrained models, all free to download.
# Each model has a "model card" explaining what it does.
# Download with from_pretrained("owner/repo-name").
# Files go to ~/.cache/huggingface/hub/ on first download.
#
# =============================================================================

import torch
print("=" * 60)
print("TRANSFORMERS LIBRARY: TEXT MODELS")
print("=" * 60)

try:
    from transformers import pipeline as hf_pipeline

    # Sentiment analysis — a 3-line full NLP pipeline
    print("Loading sentiment analysis pipeline...")
    # (comment out if too slow or no internet)
    # sentiment = hf_pipeline("sentiment-analysis")
    # result = sentiment("This AI learning material is great!")
    # print(f"Result: {result}")
    print("Example (don't run to avoid download):")
    print("  from transformers import pipeline")
    print("  sentiment = pipeline('sentiment-analysis')")
    print("  result = sentiment('I love this AI course!')")
    print("  # -> [{'label': 'POSITIVE', 'score': 0.9998}]")
    print()
    print("Other tasks:")
    print("  pipeline('text-generation', model='gpt2')")
    print("  pipeline('automatic-speech-recognition', model='openai/whisper-small')")
    print("  pipeline('image-classification', model='google/vit-base-patch16-224')")
    print("  pipeline('zero-shot-classification')")

except ImportError:
    print("transformers not installed: pip install transformers")

print()
print("=" * 60)
print("DIFFUSERS LIBRARY: IMAGE/VIDEO GENERATION")
print("=" * 60)

try:
    import diffusers
    print(f"diffusers version: {diffusers.__version__}")
    print()
    print("Key classes:")
    from diffusers import (
        StableDiffusionPipeline,
        AnimateDiffPipeline,
        AutoencoderKL,
        DDIMScheduler,
        UNet2DConditionModel,
    )
    print("  StableDiffusionPipeline  — text-to-image")
    print("  AnimateDiffPipeline      — text-to-video (what you use!)")
    print("  AutoencoderKL            — the VAE (encoder/decoder)")
    print("  DDIMScheduler            — the noise scheduler")
    print("  UNet2DConditionModel     — the denoising UNet")
    print()
    print("Usage pattern:")
    print("  pipe = AnimateDiffPipeline.from_pretrained(")
    print("      'guoyww/animatediff-motion-adapter-v1-5-2',")
    print("      torch_dtype=torch.float16)")
    print("  pipe.enable_model_cpu_offload()")
    print("  out = pipe(prompt='a cat playing piano', num_frames=16)")
    print("  frames = out.frames[0]   # list of 16 PIL images")

except ImportError:
    print("diffusers not installed: pip install diffusers")

print()
print("=" * 60)
print("from_pretrained: THE MAGIC DOWNLOAD FUNCTION")
print("=" * 60)
#
# from_pretrained("owner/model-name"):
# 1. Checks ~/.cache/huggingface/hub/ — already downloaded?
# 2. If not: downloads config.json + model files from huggingface.co
# 3. Builds the model from config.json (architecture)
# 4. Loads the weights from model.safetensors or pytorch_model.bin
# 5. Returns a ready-to-use Python object

print("What from_pretrained does:")
print()
print("  AutoencoderKL.from_pretrained('runwayml/stable-diffusion-v1-5',")
print("                                 subfolder='vae')")
print()
print("  Equivalent to:")
print("  1. Download runwayml/stable-diffusion-v1-5/vae/config.json")
print("     -> {'_class_name': 'AutoencoderKL', 'in_channels': 3, ...}")
print("  2. Download vae/diffusion_pytorch_model.safetensors")
print("  3. Build AutoencoderKL(**config)")
print("  4. model.load_state_dict(weights)")
print("  5. Return model (ready to use)")

print()
print("=" * 60)
print("THE HUB: FINDING MODELS")
print("=" * 60)

try:
    from huggingface_hub import list_models
    # (Skipping actual API call to avoid network request)
    pass
except ImportError:
    pass

print("How to search for models:")
print()
print("  from huggingface_hub import HfApi")
print("  api = HfApi()")
print("  models = api.list_models(task='text-generation', sort='downloads', limit=5)")
print()
print("Common model IDs used in AI video:")
print("  'runwayml/stable-diffusion-v1-5'           — base SD 1.5")
print("  'guoyww/animatediff-motion-adapter-v1-5-2'  — AnimateDiff motion module")
print("  'openai/clip-vit-large-patch14'              — CLIP text/image encoder")
print("  'stabilityai/sd-vae-ft-mse'                  — improved VAE for SD")
print("  'ByteDance/RIFE'                             — (GitHub, not HuggingFace)")

print()
print("=" * 60)
print("CACHE LOCATION")
print("=" * 60)

import os
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
hf_home   = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
print(f"Default HuggingFace cache: {hf_home}")
print(f"Model files stored in:     {cache_dir}")
print()
print("AnimateDiff motion module (~1.7 GB) is likely already here from your runs.")
print("SD 1.5 weights (~4 GB) if downloaded via HuggingFace are here too.")
print("Check total size: du -sh ~/.cache/huggingface/hub")
print()
print("To use a custom download location:")
print("  import os")
print("  os.environ['HF_HOME'] = 'D:/my_models'  # before importing diffusers")

# =============================================================================
# EXPECTED OUTPUT:
# diffusers version: 0.42.1  (or current version in your .venv)
# =============================================================================
