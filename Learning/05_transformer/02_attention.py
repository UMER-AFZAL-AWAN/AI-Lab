# =============================================================================
# 02 — ATTENTION: THE HEART OF THE TRANSFORMER
# =============================================================================
#
# LIKE YOU ARE 5:
# In the sentence "The animal didn't cross the street because IT was too tired",
# what does "IT" refer to? The animal. Not the street.
# To understand "it", you need to look back at "animal".
#
# ATTENTION is the mechanism that lets each word "look at" other words
# and decide which ones are relevant.
# "it" attends strongly to "animal" and weakly to "street".
#
# HOW IT WORKS:
# Every token becomes three vectors via linear layers:
#   Q (Query):  "what am I looking for?"
#   K (Key):    "what do I contain?"
#   V (Value):  "what will I contribute if selected?"
#
# Attention score = Q @ K.T / sqrt(d_k)
# Then softmax -> weights that sum to 1
# Output = weights @ V   (weighted sum of values)
#
# C# ANALOGY:
# It's like a soft dictionary lookup:
# instead of exact key match, every key is matched partially
# and you get a weighted average of all values.
#
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("=" * 60)
print("MANUAL SELF-ATTENTION")
print("=" * 60)

# Configuration
seq_len   = 5     # number of tokens
embed_dim = 8     # size of each token embedding
d_k       = 8     # size of Q/K/V vectors (can equal embed_dim)

# Input: 5 token embeddings, each 8-dimensional
x = torch.randn(1, seq_len, embed_dim)   # (batch=1, seq=5, dim=8)
print(f"Input (token embeddings): {x.shape}")

# Three projection matrices: Q, K, V
W_Q = nn.Linear(embed_dim, d_k, bias=False)
W_K = nn.Linear(embed_dim, d_k, bias=False)
W_V = nn.Linear(embed_dim, d_k, bias=False)

Q = W_Q(x)   # (1, 5, 8) — Queries
K = W_K(x)   # (1, 5, 8) — Keys
V = W_V(x)   # (1, 5, 8) — Values
print(f"Q shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"V shape: {V.shape}")

# Step 1: Compute raw attention scores
# Each query attends to ALL keys: how relevant is key j to query i?
scores = Q @ K.transpose(-2, -1)   # (1, 5, 5)
print(f"\nRaw attention scores: {scores.shape}  (5 queries × 5 keys)")
print(f"scores[0]:\n{scores[0].data.round(decimals=2)}")

# Step 2: Scale by sqrt(d_k) to prevent vanishing gradients
scores = scores / math.sqrt(d_k)

# Step 3: Softmax — make each row sum to 1 (attention weights)
weights = F.softmax(scores, dim=-1)   # (1, 5, 5)
print(f"\nAttention weights (after softmax):\n{weights[0].data.round(decimals=3)}")
print("Each row sums to 1:", weights[0].sum(dim=-1).data.round(decimals=4).tolist())

# Step 4: Weighted sum of values
output = weights @ V   # (1, 5, 8)
print(f"\nAttention output: {output.shape}  (same shape as input)")

print()
print("=" * 60)
print("CAUSAL MASKING (decoder / language model)")
print("=" * 60)
#
# In a language model (GPT, Llama), tokens can only attend to PAST tokens.
# "cat" cannot see the word that comes after it in the sentence.
# This is enforced by setting future positions to -infinity BEFORE softmax.
# -infinity -> 0 after softmax (zero weight for future tokens).

mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
print(f"Causal mask (True = cannot attend):\n{mask.int()}")

scores_causal = scores.clone()
scores_causal = scores_causal.masked_fill(mask, float('-inf'))
weights_causal = F.softmax(scores_causal, dim=-1)
print(f"\nCausal attention weights (future positions = 0):")
print(weights_causal[0].data.round(decimals=3))

print()
print("=" * 60)
print("USING PyTorch's nn.MultiheadAttention")
print("=" * 60)
#
# PyTorch provides this as a built-in module.
# It handles multiple heads (see 03_multi_head_attention.py for why).
# scaled_dot_product_attention (torch.nn.functional) is the fused CUDA kernel.

mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=2, batch_first=True)
out, attn_weights = mha(x, x, x)   # self-attention: Q=K=V=x
print(f"MultiheadAttention output: {out.shape}")
print(f"Attention weights:         {attn_weights.shape}  (1, seq, seq)")

# Flash Attention (PyTorch 2.0+) — much faster, same result
# torch.nn.functional.scaled_dot_product_attention uses it automatically
out_flash = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
print(f"\nFlash attention output:    {out_flash.shape}  (same shape)")
print("Flash attention is 2-4x faster and uses less VRAM (no materialised attention matrix)")

# =============================================================================
# EXPECTED OUTPUT:
# Input (token embeddings): torch.Size([1, 5, 8])
# Attention output: torch.Size([1, 5, 8])
# Each row sums to 1: [1.0, 1.0, 1.0, 1.0, 1.0]
# =============================================================================
