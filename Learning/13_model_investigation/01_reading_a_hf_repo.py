# =============================================================================
# 01 — READING A HUGGINGFACE REPO BEFORE YOU DOWNLOAD ANYTHING
# =============================================================================
#
# LIKE YOU ARE 5:
# Before buying a LEGO set you read the box:
# how many pieces? what age? which sub-sets are inside?
# A HuggingFace repo is the same. Read the box before downloading 20 GB.
#
# ANATOMY OF A HF MODEL REPO:
#
# SINGLE-FILE MODEL (small, < 10 GB):
#   config.json              ← architecture parameters
#   model.safetensors        ← all weights in one file
#   tokenizer.json           ← (if text model) vocabulary + rules
#   tokenizer_config.json    ← tokenizer class name + settings
#   special_tokens_map.json  ← [PAD], [CLS], [SEP] etc.
#
# SHARDED MODEL (large, > 10 GB split into chunks):
#   model.safetensors.index.json   ← the manifest: which weights → which shard
#   model-00001-of-00008.safetensors  ← shard 1 of 8
#   model-00002-of-00008.safetensors  ← shard 2 of 8
#   ...
#
# DIFFUSION PIPELINE (SD / AnimateDiff / Wan):
#   model_index.json         ← which sub-components exist + their class names
#   vae/                     ← subfolder: AutoencoderKL weights + config
#   unet/                    ← subfolder: UNet weights + config
#   text_encoder/            ← subfolder: CLIP or T5 weights + config
#   tokenizer/               ← subfolder: tokenizer files
#   scheduler/               ← subfolder: scheduler_config.json
#   feature_extractor/       ← (optional) image preprocessor
#
# =============================================================================

import json

print("=" * 60)
print("model_index.json — THE PIPELINE MANIFEST")
print("=" * 60)
#
# This file is what DiffusionPipeline.from_pretrained() reads first.
# It says: "I am an AnimateDiffPipeline made of these components."
# If a component is listed here but the subfolder is missing → crash.

example_model_index = {
    "_class_name": "AnimateDiffPipeline",
    "_diffusers_version": "0.21.0",
    "motion_adapter": ["diffusers", "MotionAdapter"],
    "scheduler": ["diffusers", "DDIMScheduler"],
    "text_encoder": ["transformers", "CLIPTextModel"],
    "tokenizer": ["transformers", "CLIPTokenizer"],
    "unet": ["diffusers", "UNet2DConditionModel"],
    "vae": ["diffusers", "AutoencoderKL"],
}
print(json.dumps(example_model_index, indent=2))
print()
print("Reading this tells you:")
print("  - Pipeline class: AnimateDiffPipeline")
print("  - UNet class: UNet2DConditionModel (from diffusers)")
print("  - Text encoder: CLIPTextModel (from transformers)")
print("  - If the class name is CUSTOM (e.g. 'WanT2VPipeline') → need trust_remote_code")

print()
print("=" * 60)
print("config.json — THE ARCHITECTURE SPEC")
print("=" * 60)
#
# Every sub-component has a config.json.
# It records every __init__ parameter of the class.
# Reading it tells you the EXACT architecture — no need to load the weights.

example_unet_config = {
    "_class_name": "UNet2DConditionModel",
    "in_channels": 4,
    "out_channels": 4,
    "cross_attention_dim": 768,
    "attention_head_dim": [5, 10, 20, 20],
    "block_out_channels": [320, 640, 1280, 1280],
    "down_block_types": [
        "CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D", "DownBlock2D"
    ],
    "layers_per_block": 2,
    "sample_size": 64,
}
print("Example unet/config.json:")
print(json.dumps(example_unet_config, indent=2))
print()
print("What this tells you:")
print(f"  Latent channels:    {example_unet_config['in_channels']}  (input/output for the latent)")
print(f"  Text embed dim:     {example_unet_config['cross_attention_dim']}  (must match the text encoder)")
print(f"  Max latent res:     {example_unet_config['sample_size']}×{example_unet_config['sample_size']}  (512px images at 8x compression)")
print(f"  UNet channel width: {example_unet_config['block_out_channels']}")

print()
print("=" * 60)
print("model.safetensors.index.json — THE SHARD MAP")
print("=" * 60)
#
# For large sharded models, this JSON maps every parameter name to its shard file.
# You can tell how big the model is from the metadata, and
# which shard to download if you only need one part.

example_index = {
    "metadata": {
        "total_size": 14_000_000_000   # 14 GB total
    },
    "weight_map": {
        "model.embed_tokens.weight": "model-00001-of-00004.safetensors",
        "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00004.safetensors",
        "model.layers.0.self_attn.k_proj.weight": "model-00001-of-00004.safetensors",
        "model.layers.16.self_attn.q_proj.weight": "model-00002-of-00004.safetensors",
        "model.layers.31.mlp.down_proj.weight": "model-00004-of-00004.safetensors",
        "lm_head.weight": "model-00004-of-00004.safetensors",
    }
}
total_gb = example_index["metadata"]["total_size"] / 1e9
print(f"Total model size from index: {total_gb:.1f} GB")
print(f"Number of shards: infer from weight_map values")
shards = set(example_index["weight_map"].values())
print(f"  Shards: {sorted(shards)}")
print()
print("from_pretrained downloads all shards and re-assembles automatically.")
print("To check size BEFORE downloading: read the index from HuggingFace raw URL.")
print("  https://huggingface.co/<owner>/<repo>/raw/main/model.safetensors.index.json")

print()
print("=" * 60)
print("WHAT FILES ARE REQUIRED vs OPTIONAL")
print("=" * 60)

required_optional = [
    ("config.json",             True,  "Architecture params — always required"),
    ("model.safetensors",       True,  "Weights — required (or sharded form)"),
    ("tokenizer.json",          False, "Only for text models"),
    ("tokenizer_config.json",   False, "Only for text models"),
    ("model_index.json",        False, "Only for multi-component pipelines"),
    ("generation_config.json",  False, "Default generation args (temp, top_p, etc.)"),
    ("preprocessor_config.json",False, "Only for vision/multimodal models"),
    ("README.md / model card",  False, "Human-readable — not loaded by from_pretrained"),
]

print(f"{'File':<35} {'Required?':>10}  Notes")
print("-" * 75)
for fname, req, note in required_optional:
    r = "✓ YES" if req else "optional"
    print(f"  {fname:<33} {r:>10}  {note}")

print()
print("=" * 60)
print("QUICK INVESTIGATION CHECKLIST (before downloading)")
print("=" * 60)
print("""
1. Open  huggingface.co/<owner>/<model>
2. Click Files and versions tab → see ALL files and their sizes
3. Read README.md → usage example → what pipeline/class is recommended?
4. Read model_index.json (if pipeline) → which sub-components exist
5. Read the config.json of the heaviest component (usually unet or model)
   → architecture params, in_channels, hidden sizes
6. Check model.safetensors.index.json (if sharded) → total_size in metadata
7. Add up: weights + VAE + text encoder + activations → fits in 8 GB?
8. Check if any file is a .py script (custom code) → trust_remote_code needed?
""")

# =============================================================================
# EXPECTED OUTPUT:
# Total model size from index: 14.0 GB
# (all printed documentation about file anatomy)
# =============================================================================
