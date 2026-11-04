# =============================================================================
# 01 — EMBEDDINGS: TURNING TOKENS INTO VECTORS
# =============================================================================
#
# LIKE YOU ARE 5:
# Computers don't understand words. They understand numbers.
# An EMBEDDING is a dictionary that maps each word (or token) to a vector of numbers.
# The word "cat" might map to [0.3, -1.2, 0.8, 0.5, ...] (512 numbers).
# The word "dog" maps to a nearby vector (cats and dogs are similar).
# The word "car"  maps to a distant vector (car ≠ cat).
#
# After training, similar words end up close in the embedding space.
# "king" - "man" + "woman" ≈ "queen"  (famous example from Word2Vec)
#
# HOW IT WORKS:
# nn.Embedding(vocab_size, embed_dim) = a lookup table of shape (V, D).
# Give it a token ID (integer). Get back a D-dimensional vector.
# The token IDs come from a TOKENIZER which breaks text into sub-word pieces.
#
# =============================================================================

import torch
import torch.nn as nn

print("=" * 60)
print("nn.Embedding: THE LOOKUP TABLE")
print("=" * 60)
#
# C# ANALOGY:
# It is like a Dictionary<int, float[]>.
# token_id -> embedding_vector.
# The vectors are the parameters — they are LEARNED during training.

vocab_size = 100   # small vocabulary for demo (real CLIP: 49,408 tokens)
embed_dim  = 16    # small dimension for demo (real CLIP: 768)

embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
print(f"Embedding table: {embedding.weight.shape}  (100 tokens × 16 dimensions)")
print(f"Num parameters:  {embedding.weight.numel()}")

# Look up a few token IDs
token_ids = torch.tensor([5, 10, 42])
vectors    = embedding(token_ids)
print(f"\nToken IDs: {token_ids.tolist()}")
print(f"Vectors shape: {vectors.shape}  (3 tokens × 16-dim)")
print(f"First token's vector:\n  {vectors[0].data.round(decimals=2)}")

print()
print("=" * 60)
print("TOKENISATION: WORDS -> TOKEN IDS")
print("=" * 60)
#
# Real text is broken into sub-word TOKENS by a tokeniser.
# "unbelievable" might become ["un", "believ", "able"] -> [432, 8821, 712]
# This handles rare words: "unbelievable" might be rare, but "un" + "believ" + "able" aren't.

try:
    from transformers import CLIPTokenizer
    tokeniser = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text = "a cat playing piano in the rain"
    tokens = tokeniser(text)
    ids = tokens["input_ids"]
    words = tokeniser.convert_ids_to_tokens(ids)
    print(f"Text:   '{text}'")
    print(f"Token IDs: {ids}")
    print(f"Tokens:    {words}")
    print(f"Num tokens: {len(ids)}")
except Exception:
    print("(CLIP tokeniser not cached — using demo)")
    # CLIP tokeniser example output
    print("Text:   'a cat playing piano in the rain'")
    print("Tokens: ['<|startoftext|>', 'a', 'cat', 'playing', 'piano', 'in', 'the', 'rain', '<|endoftext|>']")
    print("IDs:    [49406, 320, 2368, 1543, 7373, 530, 518, 1853, 49407]")

print()
print("=" * 60)
print("PADDING TO FIXED LENGTH")
print("=" * 60)
#
# Transformers process a FIXED length sequence (e.g. 77 for CLIP).
# Short sentences are padded with [PAD] tokens.
# The model ignores [PAD] positions via an ATTENTION MASK (True/False).

seq_len   = 10
batch_size = 3

# Different lengths: 4, 7, 10 tokens
input_ids = torch.tensor([
    [1, 2, 3, 4, 0, 0, 0, 0, 0, 0],
    [1, 5, 6, 7, 8, 9, 10, 0, 0, 0],
    [1, 11, 12, 13, 14, 15, 16, 17, 18, 2],
])
attention_mask = (input_ids != 0).long()  # 1 = real token, 0 = padding

print(f"input_ids shape:      {input_ids.shape}")
print(f"attention_mask:\n{attention_mask}")

# Look up embeddings for all tokens
embedding2 = nn.Embedding(20, 8)   # small for demo
embeds = embedding2(input_ids)
print(f"\nEmbeddings shape: {embeds.shape}  (batch=3, seq=10, dim=8)")
print("The attention mask tells the model which positions to ignore.")

print()
print("=" * 60)
print("CLIP vs BERT TOKENISATION DIFFERENCE")
print("=" * 60)
#
# AnimateDiff uses CLIP ViT-L/14 for text encoding.
# CLIP: text encoder from contrastive image-text training.
#       Max 77 tokens. Encodes the whole sentence as one vector OR 77 vectors.
#       Used by: SD 1.5, AnimateDiff, DALL-E 2.
#
# BERT: bidirectional transformer, trained on masked language modelling.
#       Max 512 tokens.
#       Used by: search engines, text classification, NLP tasks.
#
# In AnimateDiff, each of the 77 token embeddings becomes a KEY and VALUE
# in the cross-attention blocks of the UNet. So all 77 tokens "influence"
# the denoising at every spatial position.

print("AnimateDiff text encoding pipeline:")
print("  'a cat playing piano'")
print("  -> CLIP tokeniser -> [49406, 320, 2368, 1543, 7373, 49407, 0...] (77 IDs)")
print("  -> CLIPTextModel  -> (1, 77, 768)  (77 token embeddings, each 768-dim)")
print("  -> UNet cross-attention: each spatial position queries these 77 embeddings")
print()

shape = (1, 77, 768)
params = 1 * 77 * 768
print(f"Text embedding shape: {shape}  = {params:,} values")
print("This is passed to every cross-attention block in the UNet.")

# =============================================================================
# EXPECTED OUTPUT:
# Embedding table: torch.Size([100, 16])
# Vectors shape: torch.Size([3, 16])
# =============================================================================
