# =============================================================================
# 05 — ENCODER, DECODER, AND THE FULL TRANSFORMER BLOCK
# =============================================================================
#
# LIKE YOU ARE 5:
# ENCODER: reads the whole source sentence and understands it.
#   "Je mange une pomme" (French: I eat an apple)
#   After encoding: a rich representation of meaning in that sentence.
#
# DECODER: generates the target sentence word by word, attending to the encoder.
#   Step 1: "I"    — looks at the French sentence encoding
#   Step 2: "eat"  — looks at "I" and the French sentence encoding
#   Step 3: "an"   — looks at "I eat" and the French sentence encoding
#   ...
#
# ENCODER-ONLY (BERT, CLIP text encoder):
#   Only the encoder stack. Good for UNDERSTANDING tasks.
#   Every token attends to every other token (bidirectional).
#
# DECODER-ONLY (GPT, Llama, Mistral):
#   Only the decoder stack. Good for GENERATING text.
#   Each token only attends to previous tokens (causal).
#   AnimateDiff's prompt uses CLIP encoder. NOT a decoder.
#
# ENCODER-DECODER (T5, Whisper, original Transformer):
#   Full architecture. Translation, summarisation, speech recognition.
#
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("=" * 60)
print("ONE TRANSFORMER BLOCK (the standard building unit)")
print("=" * 60)
#
# Every Transformer (GPT, BERT, Llama, the SD UNet's attention blocks) is
# made of STACKED copies of the same block:
#
# TransformerBlock(x):
#   x = x + MultiHeadAttention(LayerNorm(x))   <- residual + norm
#   x = x + FFN(LayerNorm(x))                  <- residual + norm
#   return x
#
# FFN is a two-layer MLP (feed-forward network):
#   FFN(x) = Linear(GELU(Linear(x)))
#   The inner dimension is typically 4× the embedding dimension.
#   It adds per-token non-linear processing (attention can't do non-linear).

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_mult=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True,
                                            dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_mult),
            nn.GELU(),   # Gaussian Error Linear Unit — smoother than ReLU
            nn.Linear(embed_dim * ffn_mult, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-norm: normalise BEFORE each sub-layer (more stable than post-norm)
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed,
                                 attn_mask=mask, need_weights=False)
        x = x + self.dropout(attn_out)       # residual connection 1
        x = x + self.dropout(self.ffn(self.norm2(x)))  # residual connection 2
        return x

block = TransformerBlock(embed_dim=64, num_heads=4)
x = torch.randn(2, 10, 64)   # batch=2, seq=10, embed=64
out = block(x)
print(f"Input:  {x.shape}")
print(f"Output: {out.shape}  (Transformer block preserves shape)")
params = sum(p.numel() for p in block.parameters())
print(f"Params: {params:,}")

print()
print("=" * 60)
print("ENCODER (BERT / CLIP style): BIDIRECTIONAL")
print("=" * 60)

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.blocks    = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, token_ids):
        x = self.embedding(token_ids)   # (B, seq, embed)
        for block in self.blocks:
            x = block(x)               # no causal mask: bidirectional
        return self.norm(x)             # (B, seq, embed)

encoder = TransformerEncoder(vocab_size=1000, embed_dim=64, num_heads=4, num_layers=3)
ids = torch.randint(0, 1000, (2, 15))
enc_out = encoder(ids)
print(f"Encoder input:  {ids.shape}  (token IDs)")
print(f"Encoder output: {enc_out.shape}  (contextual embeddings for each token)")
print("Every token's output attends to ALL other tokens (bidirectional).")

print()
print("=" * 60)
print("DECODER (GPT style): CAUSAL (LEFT-TO-RIGHT)")
print("=" * 60)

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.blocks    = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.norm   = nn.LayerNorm(embed_dim)
        self.lm_head= nn.Linear(embed_dim, vocab_size)   # predict next token

    def forward(self, token_ids):
        seq_len = token_ids.shape[1]
        # Causal mask: upper triangle = -inf, so future tokens cannot be attended to
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        mask = mask.to(token_ids.device)

        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x, mask=mask)        # causal: each token only sees past
        x = self.norm(x)
        logits = self.lm_head(x)           # (B, seq, vocab_size) — scores for next token
        return logits

decoder = TransformerDecoder(vocab_size=1000, embed_dim=64, num_heads=4, num_layers=3)
ids = torch.randint(0, 1000, (2, 10))
logits = decoder(ids)
print(f"Decoder input:  {ids.shape}  (token IDs)")
print(f"Decoder output: {logits.shape}  (next-token logits for each position)")
print(f"To get next token: argmax(logits[:, -1, :]) = {logits[:, -1].argmax(dim=-1).tolist()}")

print()
print("=" * 60)
print("LAYER COUNT IN REAL MODELS")
print("=" * 60)

models = [
    ("GPT-2 small",         768,  12, 12),
    ("GPT-2 XL",           1600,  48, 25),
    ("Llama 3 8B",         4096,  32, 32),
    ("Llama 3 70B",        8192,  80, 64),
    ("CLIP ViT-L/14",       768,  12, 24),
]
print(f"{'Model':<22} {'embed_dim':>10} {'num_heads':>10} {'num_layers':>11}")
print("-" * 55)
for name, d, h, L in models:
    params_estimate = 4 * d * d * L   # rough: 4 × d² × L (attn + ffn)
    print(f"{name:<22} {d:>10} {h:>10} {L:>10}  (~{params_estimate//1e6:.0f}M params from attn+ffn alone)")

# =============================================================================
# EXPECTED OUTPUT:
# Input:  torch.Size([2, 10, 64])
# Output: torch.Size([2, 10, 64])
# Encoder output: torch.Size([2, 15, 64])
# Decoder output: torch.Size([2, 10, 1000])
# =============================================================================
