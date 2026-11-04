# =============================================================================
# 07 — CROSS-ATTENTION: HOW PROMPTS STEER IMAGE GENERATION
# =============================================================================
#
# LIKE YOU ARE 5:
# SELF-ATTENTION:  "I look at other words in the SAME sentence."
# CROSS-ATTENTION: "I look at words in a DIFFERENT sequence."
#
# In AnimateDiff's UNet:
# - Q (query) comes from the spatial image features (what the UNet currently "sees")
# - K and V (key, value) come from the TEXT EMBEDDINGS (what the prompt describes)
#
# Every spatial position in the image asks: "which part of the text is relevant to me?"
# The region where the sky will appear attends to "sky" and "blue" text tokens.
# The region where a cat will appear attends to "cat" text tokens.
#
# This is how the text controls the image content.
# Without cross-attention, the UNet would generate images ignoring the prompt.
#
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("=" * 60)
print("CROSS-ATTENTION: Q from one sequence, K/V from another")
print("=" * 60)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads

        # Q comes from the spatial features
        self.W_Q = nn.Linear(query_dim,   embed_dim, bias=False)
        # K, V come from the text context
        self.W_K = nn.Linear(context_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(context_dim, embed_dim, bias=False)
        self.W_O = nn.Linear(embed_dim,   query_dim, bias=False)

    def forward(self, x, context):
        """
        x:       (B, num_spatial_positions, query_dim)   — spatial features
        context: (B, num_text_tokens,       context_dim) — text embeddings
        """
        B, Nx, _ = x.shape
        _, Nc, _ = context.shape
        H, D = self.num_heads, self.head_dim

        # Project to Q, K, V and reshape into heads
        Q = self.W_Q(x).view(B, Nx, H, D).transpose(1, 2)        # (B, H, Nx, D)
        K = self.W_K(context).view(B, Nc, H, D).transpose(1, 2)  # (B, H, Nc, D)
        V = self.W_V(context).view(B, Nc, H, D).transpose(1, 2)  # (B, H, Nc, D)

        # Scaled dot-product attention
        scores  = Q @ K.transpose(-2, -1) / math.sqrt(D)   # (B, H, Nx, Nc)
        weights = F.softmax(scores, dim=-1)                  # (B, H, Nx, Nc)
        out     = weights @ V                                # (B, H, Nx, D)

        # Merge heads and project
        out = out.transpose(1, 2).contiguous().view(B, Nx, H * D)
        return self.W_O(out), weights

# Simulate a UNet cross-attention block at 64x64 resolution (flattened to 4096 spatial positions)
B          = 2
spatial_res= 16   # 16×16 = 256 spatial positions (smaller for demo)
query_dim  = 320  # UNet spatial feature channels
context_dim= 768  # CLIP text embedding dim
embed_dim  = 320  # attention embed dim
num_heads  = 8
num_tokens = 77   # CLIP max text length

spatial_features = torch.randn(B, spatial_res * spatial_res, query_dim)
text_embeddings  = torch.randn(B, num_tokens, context_dim)

cross_attn = CrossAttention(query_dim, context_dim, embed_dim, num_heads)
out, attn_weights = cross_attn(spatial_features, text_embeddings)

print(f"Spatial features (Q): {spatial_features.shape}  ({spatial_res}×{spatial_res} flattened)")
print(f"Text embeddings (K,V): {text_embeddings.shape}  (77 tokens)")
print(f"Output:                {out.shape}  (spatial features updated by text)")
print(f"Attention weights:     {attn_weights.shape}  (batch, heads, spatial, text_tokens)")

# Which text token does each spatial position attend to most?
# (average over batch and heads)
mean_weights = attn_weights.mean(dim=[0, 1])   # (spatial, text_tokens)
top_tokens   = mean_weights.argmax(dim=-1)     # (spatial,) — most attended text token per position
print(f"\nTop attended text token per spatial position (first 10):")
print(f"  {top_tokens[:10].tolist()}")

print()
print("=" * 60)
print("THE SPATIAL TRANSFORMER BLOCK IN SD/ANIMATEDIFF")
print("=" * 60)
#
# In Stable Diffusion, each "attention block" in the UNet is actually
# a "Spatial Transformer" block containing:
#   1. Self-attention (spatial features attend to each other)
#   2. Cross-attention (spatial features attend to text embeddings)
#   3. FFN
#   Each with residual connections + LayerNorm

class SpatialTransformerBlock(nn.Module):
    def __init__(self, spatial_dim, text_dim, embed_dim, num_heads):
        super().__init__()
        # Self-attention
        self.norm1    = nn.LayerNorm(spatial_dim)
        self.self_attn= nn.MultiheadAttention(spatial_dim, num_heads, batch_first=True)
        # Cross-attention
        self.norm2      = nn.LayerNorm(spatial_dim)
        self.cross_attn = CrossAttention(spatial_dim, text_dim, embed_dim, num_heads)
        # FFN
        self.norm3 = nn.LayerNorm(spatial_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(spatial_dim, spatial_dim * 4),
            nn.GELU(),
            nn.Linear(spatial_dim * 4, spatial_dim),
        )

    def forward(self, x, text):
        # 1. Self-attention
        n = self.norm1(x)
        sa_out, _ = self.self_attn(n, n, n, need_weights=False)
        x = x + sa_out
        # 2. Cross-attention
        ca_out, _ = self.cross_attn(self.norm2(x), text)
        x = x + ca_out
        # 3. FFN
        x = x + self.ffn(self.norm3(x))
        return x

block   = SpatialTransformerBlock(spatial_dim=320, text_dim=768, embed_dim=320, num_heads=8)
out     = block(spatial_features, text_embeddings)
print(f"SpatialTransformerBlock:")
print(f"  Input spatial: {spatial_features.shape}")
print(f"  Input text:    {text_embeddings.shape}")
print(f"  Output:        {out.shape}  (spatial features updated by both self+cross attention)")
params = sum(p.numel() for p in block.parameters())
print(f"  Params:        {params:,}")
print()
print("AnimateDiff's UNet contains ~16 of these blocks (one per attention block level).")
print("Each one runs 50 times during one video generation.")
print("Total cross-attention ops: 16 blocks × 50 steps × 256 positions × 77 text tokens.")

# =============================================================================
# EXPECTED OUTPUT:
# Spatial features (Q): torch.Size([2, 256, 320])
# Text embeddings (K,V): torch.Size([2, 77, 768])
# Output: torch.Size([2, 256, 320])
# =============================================================================
