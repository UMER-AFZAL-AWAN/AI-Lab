# =============================================================================
# 04 — POSITIONAL ENCODING: TEACHING ORDER TO A ORDERLESS SYSTEM
# =============================================================================
#
# LIKE YOU ARE 5:
# "Dog bites man" ≠ "Man bites dog".
# The same words in a different order mean very different things.
#
# PROBLEM: attention treats the input as a SET (bag of tokens).
# "cat sat mat" and "mat sat cat" look the same to attention
# because every token attends to every other token regardless of order.
#
# FIX: add a positional signal to each token's embedding.
# Token at position 0 gets a unique "tag" added to its vector.
# Token at position 1 gets a different "tag".
# Now "cat" at position 0 ≠ "cat" at position 2.
#
# Original Transformers (Attention is All You Need 2017):
#   Sinusoidal encoding: fixed math formula based on sine and cosine waves.
#   Each position → a unique pattern of sines at different frequencies.
#
# Modern LLMs (Llama, Mistral):
#   RoPE (Rotary Position Embedding): rotates Q and K vectors by an angle
#   that depends on position. This makes relative positions matter naturally.
#
# =============================================================================

import torch
import torch.nn as nn
import math

print("=" * 60)
print("SINUSOIDAL POSITIONAL ENCODING")
print("=" * 60)

def get_sinusoidal_encoding(seq_len, embed_dim):
    """
    Returns a (seq_len, embed_dim) tensor of positional encodings.
    Even dimensions: sin(pos / 10000^(2i/d))
    Odd dimensions:  cos(pos / 10000^(2i/d))
    """
    pe = torch.zeros(seq_len, embed_dim)
    pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq, 1)
    # Frequencies: 10000^(2i/d) for i in 0..d/2
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
    pe[:, 0::2] = torch.sin(pos * div_term)   # even indices: sin
    pe[:, 1::2] = torch.cos(pos * div_term)   # odd indices: cos
    return pe   # (seq_len, embed_dim)

seq_len   = 8
embed_dim = 16
pe = get_sinusoidal_encoding(seq_len, embed_dim)
print(f"Positional encoding shape: {pe.shape}  (8 positions × 16-dim)")
print(f"\nPosition 0: {pe[0].round(decimals=2).tolist()}")
print(f"Position 1: {pe[1].round(decimals=2).tolist()}")
print(f"Position 7: {pe[7].round(decimals=2).tolist()}")
print()
print("Note: adjacent positions are similar, distant positions differ.")
dot_01 = (pe[0] @ pe[1]).item()
dot_07 = (pe[0] @ pe[7]).item()
print(f"Dot product pos 0 vs pos 1: {dot_01:.3f}  (similar, close positions)")
print(f"Dot product pos 0 vs pos 7: {dot_07:.3f}  (less similar, distant)")

print()
print("=" * 60)
print("USING POSITIONAL ENCODING")
print("=" * 60)

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding    = get_sinusoidal_encoding(max_seq_len, embed_dim)
        self.dropout         = nn.Dropout(dropout)
        self.scale           = math.sqrt(embed_dim)

    def forward(self, token_ids):
        seq_len = token_ids.shape[1]
        # Token embedding (scaled by sqrt(d) to match PE magnitude)
        tok_emb = self.token_embedding(token_ids) * self.scale
        # Add positional encoding for the first seq_len positions
        pos_emb = self.pos_encoding[:seq_len].to(tok_emb.device)
        return self.dropout(tok_emb + pos_emb)

emb_layer = TransformerEmbedding(vocab_size=100, embed_dim=16, max_seq_len=50)
ids = torch.randint(0, 100, (2, 8))  # batch=2, seq=8
out = emb_layer(ids)
print(f"Token IDs shape:  {ids.shape}")
print(f"Embedding output: {out.shape}  (batch, seq, embed_dim)")

print()
print("=" * 60)
print("LEARNED POSITIONAL EMBEDDING (alternative)")
print("=" * 60)
#
# Many models (BERT, some ViT) use a LEARNED positional embedding instead of sinusoidal.
# nn.Embedding(max_seq_len, embed_dim) — the positions are treated like tokens.
# The model learns the best positional signals during training.
# Limitation: can't extrapolate beyond max_seq_len (sinusoidal can, in theory).

learned_pos = nn.Embedding(100, 16)   # max 100 positions
positions   = torch.arange(8).unsqueeze(0)   # (1, 8) positions [0,1,...,7]
pos_vecs    = learned_pos(positions)
print(f"Learned position embeddings: {pos_vecs.shape}  (1 batch, 8 positions, 16-dim)")

print()
print("=" * 60)
print("ROPE: ROTARY POSITION EMBEDDING (Llama, Mistral, Qwen)")
print("=" * 60)
#
# RoPE rotates each Q and K vector by an angle proportional to its position.
# A rotation of angle θ × pos.
# Key insight: when you compute Q @ K.T, only the RELATIVE angle (pos_q - pos_k) matters.
# So the model naturally learns to pay attention based on RELATIVE distance.
# Also: extrapolates to longer sequences than training (with tricks like YaRN).

def apply_rope(q, k, dim):
    """Simplified RoPE: rotate each pair of dimensions by position-dependent angle."""
    seq_len = q.shape[1]
    # Frequencies: theta_i = 1 / (10000 ^ (2i / dim))
    freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    t     = torch.arange(seq_len, dtype=torch.float)
    theta = torch.outer(t, freqs)   # (seq, dim/2) — angle for each position and freq
    cos   = theta.cos()   # (seq, dim/2)
    sin   = theta.sin()   # (seq, dim/2)

    # Rotate: (x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)
    q_even, q_odd = q[..., 0::2], q[..., 1::2]
    k_even, k_odd = k[..., 0::2], k[..., 1::2]

    q_rot = torch.stack([q_even * cos - q_odd * sin,
                          q_even * sin + q_odd * cos], dim=-1).flatten(-2)
    k_rot = torch.stack([k_even * cos - k_odd * sin,
                          k_even * sin + k_odd * cos], dim=-1).flatten(-2)
    return q_rot, k_rot

q = torch.randn(2, 8, 16)   # (batch, seq, head_dim)
k = torch.randn(2, 8, 16)
q_rot, k_rot = apply_rope(q, k, dim=16)
print(f"Original Q shape: {q.shape}")
print(f"RoPE-rotated Q:   {q_rot.shape}  (same shape, but each position rotated)")
print("After RoPE, attention score Q@K captures relative position, not absolute.")

# =============================================================================
# EXPECTED OUTPUT:
# Positional encoding shape: torch.Size([8, 16])
# Embedding output: torch.Size([2, 8, 16])
# =============================================================================
