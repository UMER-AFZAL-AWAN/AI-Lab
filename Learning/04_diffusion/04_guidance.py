# =============================================================================
# 04 — CLASSIFIER-FREE GUIDANCE: STEERING WITH A PROMPT
# =============================================================================
#
# LIKE YOU ARE 5:
# Imagine you are lost in a forest and trying to reach a town.
# Without a compass: you just wander (pure noise -> random image).
# With a compass: every step you check and lean toward the town (guided).
# The stronger the compass, the more directly you go (higher guidance_scale).
# Too strong a compass and you walk so fast you trip over (over-saturated output).
#
# In diffusion:
# - The "town" is a text prompt: "a cat playing piano"
# - The "compass" is the difference between prompted and unprompted noise
# - guidance_scale controls how strongly the compass steers you
#
# HOW IT WORKS (Classifier-Free Guidance = CFG):
# - Run UNet with the text prompt  -> eps_cond
# - Run UNet with empty prompt ""  -> eps_uncond
# - Combine: eps_guided = eps_uncond + scale * (eps_cond - eps_uncond)
# - Use eps_guided for the denoising step
#
# =============================================================================

import torch
import torch.nn as nn

print("=" * 60)
print("TEXT ENCODING: TURNING WORDS INTO VECTORS")
print("=" * 60)
#
# The text prompt is encoded by a text encoder (CLIP) into a tensor.
# "a cat playing piano" -> (1, 77, 768)
#   77 = maximum token sequence length
#   768 = embedding dimension
#
# This tensor is passed to the UNet via CROSS-ATTENTION
# (covered in the Transformer section).
# The UNet "looks at" the text encoding while denoising.

try:
    from transformers import CLIPTokenizer, CLIPTextModel
    print("CLIP text encoder is available. Here is what it produces:")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    tokens = tokenizer(["a cat playing piano"], padding="max_length",
                       max_length=77, return_tensors="pt")
    print(f"Input IDs shape: {tokens.input_ids.shape}   (batch=1, seq_len=77 tokens)")
    print("Note: in AnimateDiff the text encoder is 'openai/clip-vit-large-patch14'")
    print("      which produces embeddings of shape (1, 77, 768)")
except Exception:
    print("(CLIP tokenizer not cached — skipping model load)")
    print("Text prompt tokenisation produces shape (1, 77) token IDs.")
    print("CLIPTextModel encodes those to (1, 77, 768) embeddings.")
    tokens_demo = torch.randint(0, 49408, (1, 77))
    print(f"Demo token IDs shape: {tokens_demo.shape}")
    text_embed  = torch.randn(1, 77, 768)
    print(f"Demo text embedding shape: {text_embed.shape}")

print()
print("=" * 60)
print("CLASSIFIER-FREE GUIDANCE COMPUTATION")
print("=" * 60)

def apply_cfg(eps_cond, eps_uncond, guidance_scale):
    """
    Combine conditional and unconditional noise predictions.
    guidance_scale=1.0 -> no guidance (pure unconditional)
    guidance_scale=7.5 -> standard AnimateDiff guidance
    guidance_scale=15+ -> very strong guidance (over-saturated)
    """
    return eps_uncond + guidance_scale * (eps_cond - eps_uncond)

# Fake UNet outputs
torch.manual_seed(42)
eps_cond   = torch.randn(1, 4, 8, 8)   # from UNet with "a cat playing piano"
eps_uncond = torch.randn(1, 4, 8, 8)   # from UNet with ""

print(f"eps_cond   std: {eps_cond.std().item():.3f}")
print(f"eps_uncond std: {eps_uncond.std().item():.3f}")
print()

for scale in [1.0, 3.0, 7.5, 15.0]:
    guided = apply_cfg(eps_cond, eps_uncond, scale)
    print(f"scale={scale:4.1f}  guided std={guided.std().item():.3f}  "
          + ("(prompt ignored)" if scale == 1.0 else
             "(weak guidance)" if scale == 3.0 else
             "(standard)" if scale == 7.5 else "(strong — may over-saturate)"))

print()
print("=" * 60)
print("THE BATCHED TRICK: ONE UNet FORWARD PASS, NOT TWO")
print("=" * 60)
#
# Running UNet twice per step is expensive.
# In practice, the two inputs are batched together:
# [uncond_latent, cond_latent] -> UNet -> [eps_uncond, eps_cond]
# One forward pass, twice the batch size.

# Simulate batching
latent_uncond = torch.randn(1, 4, 8, 8)
latent_cond   = torch.randn(1, 4, 8, 8)
latent_batch  = torch.cat([latent_uncond, latent_cond], dim=0)   # shape (2, 4, 8, 8)

text_uncond = torch.zeros(1, 77, 768)   # empty prompt embedding
text_cond   = torch.randn(1, 77, 768)   # "a cat playing piano" embedding
text_batch  = torch.cat([text_uncond, text_cond], dim=0)          # shape (2, 77, 768)

print(f"Batched latent:    {latent_batch.shape}  (2 items: uncond + cond)")
print(f"Batched text emb:  {text_batch.shape}  (2 items: empty + prompt)")
print("UNet processes both in one forward pass.")
print("Then split the output and apply CFG formula.")

# After UNet: split
noise_pred_batch = torch.randn(2, 4, 8, 8)   # fake UNet output
eps_uncond_out = noise_pred_batch[:1]
eps_cond_out   = noise_pred_batch[1:]
eps_final      = apply_cfg(eps_cond_out, eps_uncond_out, guidance_scale=7.5)
print(f"\nFinal guided noise shape: {eps_final.shape}")

print()
print("=" * 60)
print("NEGATIVE PROMPTS")
print("=" * 60)
#
# Instead of using an empty uncond prompt,
# you can use a NEGATIVE PROMPT: "blurry, bad quality, distorted"
# eps_uncond is now the noise predicted for THAT text.
# The guidance steers AWAY from the negative prompt and TOWARD the positive.
# This is why AnimateDiff pipelines accept both positive and negative prompts.

print("Positive prompt: 'a cat playing piano, high quality'")
print("Negative prompt: 'blurry, bad quality, deformed hands'")
print()
print("During denoising:")
print("  eps_cond   = UNet('a cat playing piano, high quality')")
print("  eps_uncond = UNet('blurry, bad quality, deformed hands')")
print("  eps_guided = eps_uncond + scale * (eps_cond - eps_uncond)")
print()
print("Effect: move TOWARD quality cat, move AWAY FROM blurry garbage.")

# =============================================================================
# EXPECTED OUTPUT:
# scale= 1.0  guided std=...  (prompt ignored)
# scale= 7.5  guided std=...  (standard)
# Batched latent: torch.Size([2, 4, 8, 8])
# =============================================================================
