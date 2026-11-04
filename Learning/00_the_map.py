# =============================================================================
# 00 — THE MAP: How Every AI Concept Connects
# =============================================================================
#
# LIKE YOU ARE 5:
# Imagine you want to bake a cake.
#   - The RECIPE is the architecture (how the cake is designed).
#   - The OVEN is PyTorch / the execution framework.
#   - The INGREDIENTS are the weights (what was learned from millions of cakes).
#   - The BAKERY PROCESS is the inference pipeline.
#   - The CAKE TIN is the checkpoint file (.safetensors, .onnx, .gguf).
#
# The cake tin (file) does NOT tell you the recipe. You need the recipe separately.
# That is the most important thing to understand about AI models.
#
# =============================================================================
# THE FULL STACK — from your hardware up to your Python code
# =============================================================================
#
#  Your Python / C# code
#       |
#       v
#  Task pipeline          (text generation, image generation, video generation)
#       |
#       v
#  Model architecture     (Transformer / CNN / Diffusion / VAE / GAN)
#       |
#       v
#  Model components       (encoder, decoder, attention, scheduler, tokenizer, VAE)
#       |
#       v
#  Inference engine       (PyTorch, ONNX Runtime, TensorRT)
#       |
#       v
#  Hardware               (CUDA, GPU VRAM, System RAM, CPU)
#
# Each layer is separate. Knowing one does not mean you know the others.
#
# =============================================================================
# THE SIX ARCHITECTURE FAMILIES
# =============================================================================
#
# 1. TRANSFORMERS  (highest priority)
#    - The backbone of GPT, Llama, BERT, Qwen, ViT
#    - Core idea: "attention" — every token looks at every other token
#    - Used in: LLMs, NLP, vision-language, multimodal
#
# 2. DIFFUSION  (highest priority)
#    - The backbone of Stable Diffusion, AnimateDiff, DALL-E
#    - Core idea: start from noise, denoise step by step guided by text
#    - Used in: image generation, video generation, audio generation
#    - YOU ALREADY RAN THIS (AnimateDiff)
#
# 3. CNN (Convolutional Neural Networks)
#    - The backbone of RIFE, ResNet, EfficientNet
#    - Core idea: sliding filters detect patterns in images
#    - Used in: classification, detection, segmentation, super-resolution
#    - YOU ALREADY RAN THIS (RIFE uses IFNet, a CNN)
#
# 4. AUTOENCODER / VAE
#    - Core idea: compress an image to a small vector, reconstruct it
#    - VAE adds randomness so you can GENERATE new things, not just reconstruct
#    - Used in: image compression, latent representations for diffusion
#    - YOU ALREADY RAN THIS (the VAE inside AnimateDiff)
#
# 5. GAN (Generative Adversarial Networks)
#    - Core idea: two networks compete — one generates, one judges
#    - Used in: image synthesis, super-resolution, style transfer
#
# 6. GRAPH / SEQUENTIAL / STATE SPACE
#    - RNN, LSTM, GRU: read sequences one step at a time
#    - Mamba (SSM): a modern efficient alternative to transformers
#    - GNN: learn on graph-structured data
#
# =============================================================================
# THE KEY DISTINCTION  (memorise this)
# =============================================================================
#
#   "Transformer" is an ARCHITECTURE (a design pattern).
#   "PyTorch" is a FRAMEWORK (the oven that executes the design).
#   "HuggingFace Transformers" is a LIBRARY (pre-built recipes using PyTorch).
#   "Diffusers" is a LIBRARY for diffusion pipelines.
#   ".safetensors" is a FILE FORMAT (how weights are stored on disk).
#   "ONNX" is a PORTABLE FORMAT (move weights between frameworks).
#   "ONNX Runtime" is a RUNTIME (execute ONNX models from C# or Python).
#
#   A .safetensors file tells you NOTHING about which architecture to use.
#   You need to know the architecture separately to load the weights correctly.
#
# =============================================================================
# YOUR HARDWARE REALITY
# =============================================================================
#
# RTX 5060 Laptop: 8 GB VRAM
# System RAM: 32 GB
#
# Memory per billion parameters (weights only, before activations):
#   FP32  (32-bit float): 4 GB per billion params
#   FP16  (16-bit float): 2 GB per billion params
#   INT8  (8-bit int):    1 GB per billion params
#   INT4  (4-bit int):    0.5 GB per billion params
#
# A 7B model in FP16 = 14 GB weights alone. Does not fit in 8 GB VRAM.
# A 7B model in INT4 = 3.5 GB weights. Might fit, plus activations.
# "Fits on disk" does not mean "fits in VRAM".
#
# =============================================================================
# THIS FILE IS JUST COMMENTS — nothing to run. Read it, then move on.
# =============================================================================

print("The map is in the comments above. Read them.")
print()
print("Architecture families:")
families = [
    "1. Transformers     — attention, GPT, Llama, BERT, ViT",
    "2. Diffusion        — noise → denoise, AnimateDiff, Stable Diffusion",
    "3. CNN              — sliding filters, RIFE, ResNet",
    "4. Autoencoder/VAE  — compress → reconstruct, latent space",
    "5. GAN              — generator vs discriminator",
    "6. Graph/Sequential — RNN, LSTM, Mamba, GNN",
]
for f in families:
    print(" ", f)
print()
print("Key distinction:")
print("  Architecture != Framework != Library != File format != Runtime")
