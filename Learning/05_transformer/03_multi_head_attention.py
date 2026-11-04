# =============================================================================
# 03 — MULTI-HEAD ATTENTION: LOOKING IN MULTIPLE WAYS AT ONCE
# =============================================================================
#
# LIKE YOU ARE 5:
# When you read a sentence, different parts of your brain look for different things:
# Part 1: "who is doing the action?" (subject)
# Part 2: "what are they doing?" (verb)
# Part 3: "where is it happening?" (place)
#
# Multi-Head Attention = run attention H times in parallel,
# each with its own Q/K/V projection (its own "head").
# Then concatenate all H results and project back to the original size.
#
# Why H=8 or H=16?
# One attention head can only see one kind of relationship at a time.
# Multiple heads can see many different relationships simultaneously.
# Head 1 might learn subject-verb relationships.
# Head 2 might learn pronoun-reference relationships.
# Head 3 might learn positional proximity.
#
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("=" * 60)
print("MANUAL MULTI-HEAD ATTENTION")
print("=" * 60)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads   # each head sees embed_dim/H dims

        # Three projections for all heads combined (more efficient than separate)
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)   # output projection

    def forward(self, q, k, v, mask=None):
        B, Sq, D = q.shape    # batch, seq_len, embed_dim
        _, Sk, _ = k.shape

        # Project and split into H heads
        Q = self.W_Q(q).view(B, Sq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(k).view(B, Sk, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(v).view(B, Sk, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: (B, num_heads, seq, head_dim)

        # Attention for each head
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)   # (B, H, Sq, Sk)
        out = weights @ V                     # (B, H, Sq, head_dim)

        # Concatenate heads and project back
        out = out.transpose(1, 2).contiguous().view(B, Sq, D)  # (B, Sq, embed_dim)
        return self.W_O(out), weights

# Demo
embed_dim = 32
num_heads = 4
seq_len   = 6
x = torch.randn(2, seq_len, embed_dim)   # batch=2

mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
out, attn_weights = mha(x, x, x)

print(f"Input:            {x.shape}")
print(f"Output:           {out.shape}   (same shape)")
print(f"Attention weights:{attn_weights.shape}  (batch, heads, seq, seq)")
print(f"head_dim = {mha.head_dim}  (each head sees {mha.head_dim} of the {embed_dim} dims)")

print()
print("What each head's attention pattern looks like:")
for h in range(num_heads):
    # Average weight across the sequence
    mean_weight = attn_weights[0, h].mean(dim=0)
    print(f"  Head {h}: top attended tokens = {mean_weight.argsort(descending=True).tolist()}")

print()
print("=" * 60)
print("HOW MULTI-HEAD IS USED IN SD UNET (AnimateDiff)")
print("=" * 60)
#
# In Stable Diffusion's UNet, there are TWO types of attention:
#
# 1. SELF-ATTENTION:  q=k=v = spatial features
#    Each spatial position attends to all other spatial positions.
#    Helps maintain global consistency (the cat's eye matches its fur).
#
# 2. CROSS-ATTENTION: q = spatial features, k=v = text embeddings
#    Each spatial position attends to all text tokens.
#    "What part of the text is relevant to this spatial region?"
#    This is HOW the text prompt steers the image content.

# Simulate cross-attention in UNet block
spatial_features = torch.randn(2, 16, 32)   # (batch, H*W, channels) flattened spatial
text_embeddings  = torch.randn(2, 77, 32)   # (batch, 77 tokens, 768 -> 32 projected)

cross_mha = MultiHeadAttention(embed_dim=32, num_heads=4)
# Q from spatial features, K and V from text
cross_out, cross_weights = cross_mha(
    q=spatial_features,
    k=text_embeddings,
    v=text_embeddings,
)
print(f"Cross-attention:")
print(f"  Q (spatial): {spatial_features.shape}")
print(f"  K,V (text):  {text_embeddings.shape}")
print(f"  Output:      {cross_out.shape}  (spatial features updated by text)")
print(f"  Weights:     {cross_weights.shape}  (batch, heads, spatial_pos, text_tokens)")
print()
print("Each spatial position (of 16) attends to all 77 text tokens.")
print("The position near 'piano' in the image will attend strongly to 'piano' token.")

print()
print("=" * 60)
print("PARAMETER COUNT SCALING")
print("=" * 60)

for embed_dim_val in [64, 256, 512, 768]:
    for num_heads_val in [4, 8]:
        mha_tmp = MultiHeadAttention(embed_dim_val, num_heads_val)
        params = sum(p.numel() for p in mha_tmp.parameters())
        head_dim = embed_dim_val // num_heads_val
        print(f"  embed={embed_dim_val}, heads={num_heads_val}, head_dim={head_dim:3d}: {params:,} params")

# =============================================================================
# EXPECTED OUTPUT:
# Input:            torch.Size([2, 6, 32])
# Output:           torch.Size([2, 6, 32])
# Attention weights:torch.Size([2, 4, 6, 6])
# =============================================================================
