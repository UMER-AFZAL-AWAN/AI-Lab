# =============================================================================
# 14_hf_libraries / 02 — TRANSFORMERS LIBRARY
# =============================================================================
#
# LIKE YOU ARE 5:
# "transformers" is a library for text models: CLIP, T5, BERT, GPT, Llama, etc.
# It's the sibling of diffusers. While diffusers handles image/video generation,
# transformers handles text understanding, text encoding, and language generation.
# In diffusion pipelines, the text encoder (CLIP, T5, UMT5) comes from transformers.
#
# C# ANALOGY:
# Like a giant switch statement with every NLP model pre-built.
# AutoModel = Factory.Create(modelType) — auto-selects the right class.
# pipeline("text-generation") = a fluent API builder that wires everything up.
#
# =============================================================================

import torch
print("=" * 60)
print("THE THREE ENTRY POINTS")
print("=" * 60)
print("""
transformers gives you 3 ways to use a model:

1. pipeline()          ← highest level, one line, great for quick tests
2. AutoModel / AutoTokenizer  ← mid level, flexible, for custom code
3. Specific classes    ← lowest level, e.g. CLIPTextModel exactly

── When to use each ───────────────────────────────────
pipeline()         → exploration, quick demo, don't care about internals
AutoModel          → production code, need control over inputs/outputs
Specific classes   → you know exactly what model family you're using
                     (e.g. inside a diffusers pipeline component)
""")

print("=" * 60)
print("AUTO CLASSES: THE FACTORY PATTERN")
print("=" * 60)
print("""
# Auto classes read config.json → pick the right model class → load it.
# You don't need to know the exact class name.

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

# Tokeniser
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
tokens = tokenizer("a cat playing piano", return_tensors="pt")
print(tokens.input_ids.shape)   # (1, 7) — 7 tokens including BOS/EOS

# Text encoder (loads CLIPTextModel automatically)
text_model = AutoModel.from_pretrained(
    "openai/clip-vit-large-patch14",
    torch_dtype=torch.float16,
)
with torch.no_grad():
    out = text_model(**tokens)
print(out.last_hidden_state.shape)   # (1, 7, 768) — per-token embeddings
print(out.pooler_output.shape)       # (1, 768) — single sentence embedding

# Generative LLM
llm = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    device_map="auto",        # fills GPU, spills to CPU
    torch_dtype=torch.float16,
)
llm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
input_ids = llm_tokenizer("Hello, I am", return_tensors="pt").input_ids.cuda()
out = llm.generate(input_ids, max_new_tokens=50)
print(llm_tokenizer.decode(out[0]))
""")

print("=" * 60)
print("THE HIGH-LEVEL pipeline() FUNCTION")
print("=" * 60)
print("""
from transformers import pipeline

# Text generation (like a mini ChatGPT)
gen = pipeline("text-generation", model="gpt2", device=0)
out = gen("Once upon a time", max_new_tokens=30)
print(out[0]["generated_text"])

# Zero-shot classification (no fine-tuning needed)
clf = pipeline("zero-shot-classification",
               model="facebook/bart-large-mnli")
out = clf("A cat playing the piano",
          candidate_labels=["music", "sports", "cooking"])
print(out["labels"][0])   # "music"

# Image classification
ic = pipeline("image-classification",
              model="google/vit-base-patch16-224")
out = ic("my_photo.jpg")
print(out[0]["label"])

# pipeline() tasks:
# "text-generation", "summarization", "translation_en_to_de",
# "question-answering", "fill-mask", "text-classification",
# "zero-shot-classification", "image-classification",
# "object-detection", "image-segmentation", "feature-extraction"
""")

print("=" * 60)
print("CLIP TEXT ENCODER: HOW IT WORKS IN DIFFUSION PIPELINES")
print("=" * 60)

# Demonstrate with tiny tensors — no model download needed
print("""
# Real usage inside diffusers pipeline (you don't call this yourself —
# the pipeline calls it. But knowing it demystifies the internals.)

from transformers import CLIPTokenizer, CLIPTextModel

tokenizer    = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14",
                                             torch_dtype=torch.float16)

prompt = "a cat playing piano, cinematic lighting"
tokens = tokenizer(
    prompt,
    padding="max_length",    # always pad to max_length=77
    max_length=77,
    truncation=True,
    return_tensors="pt"
)
# tokens.input_ids: shape (1, 77) — always exactly 77 token IDs

with torch.no_grad():
    text_embeds = text_encoder(tokens.input_ids)[0]

print(text_embeds.shape)   # torch.Size([1, 77, 768])
# ^ This is the tensor that goes into the UNet's cross-attention layers.
# Each of the 77 positions is a 768-dim vector representing one token.
# The UNet Q/K/V attention uses these as K and V (see 05_transformer/07_cross_attention.py)
""")

# Demonstrate tokenisation mechanics with tiny vocab (no download)
print("=" * 60)
print("TOKENISATION DEMO (synthetic — no download)")
print("=" * 60)

class ToyTokenizer:
    """
    Toy CLIP-style tokenizer.
    Real CLIP uses a 49408-word BPE vocab.
    This shows the same mechanics with 10 words.
    """
    vocab = {"<|startoftext|>": 0, "a": 1, "cat": 2, "plays": 3,
             "piano": 4, "cinematic": 5, "lighting": 6, "<|endoftext|>": 7,
             "<PAD>": 8}
    max_len = 10

    def __call__(self, text):
        words  = ["<|startoftext|>"] + text.lower().split() + ["<|endoftext|>"]
        ids    = [self.vocab.get(w, 8) for w in words]
        # Pad to max_len
        ids   += [8] * (self.max_len - len(ids))
        ids    = ids[:self.max_len]
        return torch.tensor([ids])

tok    = ToyTokenizer()
prompt = "a cat plays piano cinematic lighting"
ids    = tok(prompt)
print(f"Prompt: '{prompt}'")
print(f"Token IDs: {ids.tolist()}")
print(f"Shape: {ids.shape}")  # (1, 10)

# These IDs go into an embedding table: each ID → a 768-dim vector
# In real CLIP: nn.Embedding(49408, 768)
vocab_size, embed_dim = 9, 8   # tiny for demo
embedding = torch.nn.Embedding(vocab_size, embed_dim)
embeds = embedding(ids)
print(f"Embedding shape: {embeds.shape}")  # (1, 10, 8) = (batch, seq, embed_dim)
print("(Real CLIP: (1, 77, 768) — always padded to 77 tokens)")

print()
print("=" * 60)
print("T5 AND UMT5: FOR WAN AND NEWER MODELS")
print("=" * 60)
print("""
# Newer video models (Wan2.1, CogVideoX) use T5 or UMT5 as text encoder.
# T5 produces richer text embeddings (encoder-decoder architecture).
# Key difference from CLIP:
#   CLIP max_length = 77 tokens (short prompts)
#   T5 max_length = 512 tokens (much more detail possible)

from transformers import T5EncoderModel, AutoTokenizer

t5_tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-xxl")
t5_encoder   = T5EncoderModel.from_pretrained("google/t5-v1_1-xxl",
                                               torch_dtype=torch.bfloat16)
# Wan uses UMT5 (multilingual T5):
# from transformers import UMT5EncoderModel
# encoder = UMT5EncoderModel.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B",
#                                            subfolder="text_encoder")
""")

print("=" * 60)
print("QUICK REFERENCE: WHAT GOES WHERE")
print("=" * 60)
print("""
Library        | What it contains
─────────────────────────────────────────────────────────
transformers   | Text encoders: CLIP, T5, UMT5, BERT, Llama, Qwen
               | Vision encoders: ViT, CLIP-ViT
               | AutoModel, AutoTokenizer, pipeline()
               | AutoModelForCausalLM, AutoModelForSeq2SeqLM

diffusers      | Denoising models: UNet, Transformer (DiT), MotionAdapter
               | VAE: AutoencoderKL, AutoencoderKLWan
               | Schedulers: DDIM, DPM++, Euler, FlowMatch
               | Pipelines: StableDiffusion, AnimateDiff, Wan, CogVideoX

accelerate     | device_map, mixed precision training, CPU offload
               | (see 14_hf_libraries/03_accelerate.py)

huggingface_hub | download files, list repos, check sizes before downloading
                | (HfApi, hf_hub_download, snapshot_download)
""")

# =============================================================================
# EXPECTED OUTPUT: (all printed documentation + tokenisation demo)
# Token IDs: [[0, 1, 2, 3, 4, 5, 6, 7, 8, 8]]
# Embedding shape: torch.Size([1, 10, 8])
# =============================================================================
