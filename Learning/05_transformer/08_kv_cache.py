# =============================================================================
# 05_transformer / 08 — KV CACHE
# =============================================================================
#
# LIKE YOU ARE 5:
# When you write a sentence word by word, you don't re-read the whole
# sentence from scratch every time you add a new word.
# You remember what you've already written (the K and V of past tokens).
# KV cache = that memory.
# Without it: generating 100 tokens costs 100 * 100 = 10,000 attention ops.
# With it: 100 * 1 + a bit of bookkeeping = 100 ops. 100x faster.
#
# C# ANALOGY:
# Like a Dictionary<int, (K, V)> that stores computed key-value pairs.
# Instead of calling an expensive Compute(token) every time, you check the
# cache first and only compute for the NEW token.
#
# WHERE IT MATTERS FOR YOU:
# KV cache is used in Transformer LANGUAGE MODELS (Llama, Qwen, GPT).
# In diffusion models (SD, AnimateDiff, Wan) the denoiser runs all tokens at
# once (not autoregressively), so KV cache is NOT used there.
# But the TEXT ENCODER (CLIP, T5) inside a diffusion pipeline DOES have
# self-attention — though since it runs just once per prompt, caching there
# is less critical than for LLMs.
#
# =============================================================================

import torch
import torch.nn as nn
import math
import time

print("=" * 60)
print("BACKGROUND: AUTOREGRESSIVE GENERATION")
print("=" * 60)
print("""
Language models generate one token at a time:
  Input: [<BOS>, "a", "cat"]
  → predicts next token → "plays"
  Input: [<BOS>, "a", "cat", "plays"]
  → predicts next token → "piano"
  ...

Each step runs the FULL transformer on the GROWING sequence.
Without KV cache: at step t, we compute K and V for ALL t tokens.
  Total work ∝ T² (quadratic in sequence length)

With KV cache: at step t, we keep K and V from steps 0..t-1 in memory.
  At step t, only compute K and V for the NEW token.
  Total work ∝ T (linear in sequence length) — massive speedup!
""")

print("=" * 60)
print("THE KV CACHE MECHANICS")
print("=" * 60)

# ---------------------------------------------------------------------------
# Toy single-head attention that supports KV caching
# ---------------------------------------------------------------------------

class SingleHeadAttentionWithCache(nn.Module):
    """
    Minimal attention with KV cache.
    Real implementations (transformers library) use optimized versions
    of exactly this pattern.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale = embed_dim ** -0.5

    def forward(self, x, kv_cache=None):
        """
        x:         (B, 1, D) for cached (new token only) OR (B, T, D) for prefill
        kv_cache:  {'k': (B, T_past, D), 'v': (B, T_past, D)} or None

        Returns:
          output:    (B, Tx, D)
          new_cache: {'k': (B, T_past+Tx, D), 'v': (B, T_past+Tx, D)}
        """
        B, Tx, D = x.shape

        Q = self.W_Q(x)        # (B, Tx, D) — Q only for the new tokens
        K = self.W_K(x)        # (B, Tx, D) — K for the new tokens
        V = self.W_V(x)        # (B, Tx, D) — V for the new tokens

        if kv_cache is not None:
            # Concatenate past keys and values with new ones
            K = torch.cat([kv_cache['k'], K], dim=1)   # (B, T_past+Tx, D)
            V = torch.cat([kv_cache['v'], V], dim=1)   # (B, T_past+Tx, D)

        new_cache = {'k': K, 'v': V}   # save for next step

        # Attention: Q queries against ALL past K, weighted by all past V
        scores  = Q @ K.transpose(-2, -1) * self.scale  # (B, Tx, T_past+Tx)
        weights = torch.softmax(scores, dim=-1)
        out     = weights @ V                             # (B, Tx, D)
        return self.W_O(out), new_cache


# ---------------------------------------------------------------------------
# DEMO: prefill + decode with cache
# ---------------------------------------------------------------------------

D = 32   # small embed_dim for demo
attn = SingleHeadAttentionWithCache(D)
attn.eval()

# --- Prefill: process the prompt "a cat plays" (3 tokens) all at once ---
prompt_len = 3
prompt     = torch.randn(1, prompt_len, D)   # (1, 3, 32)

with torch.no_grad():
    prefill_out, kv_cache = attn(prompt, kv_cache=None)

print(f"Prefill input:  {list(prompt.shape)}")
print(f"Prefill output: {list(prefill_out.shape)}")
print(f"Cache K shape:  {list(kv_cache['k'].shape)}")   # (1, 3, 32)
print(f"Cache V shape:  {list(kv_cache['v'].shape)}")

# --- Decode: generate 5 new tokens, one at a time ---
print(f"\nGenerating 5 tokens with KV cache:")
for step in range(5):
    new_token = torch.randn(1, 1, D)   # one new token (1, 1, 32)
    with torch.no_grad():
        out, kv_cache = attn(new_token, kv_cache=kv_cache)
    # K and V grow by 1 each step
    print(f"  Step {step+1}: new_token (1,1,{D}) → output {list(out.shape)}, "
          f"cache K {list(kv_cache['k'].shape)}")

print()
print("=" * 60)
print("KV CACHE VRAM COST")
print("=" * 60)
print("""
For one Transformer layer:
  K cache: (batch, seq_len, num_heads, head_dim)
  V cache: same
  Total per layer: 2 × batch × seq_len × num_heads × head_dim × bytes_per_elem

For Llama 3 8B:
  - 32 layers, 32 heads, head_dim=128, FP16 (2 bytes)
  - At seq_len=4096: 2 × 1 × 4096 × 32 × 128 × 2 × 32 layers
    = 2 × 4096 × 32 × 128 × 2 × 32 / 1e9
""")

layers, heads, head_dim, seq_len = 32, 32, 128, 4096
kv_bytes = 2 * seq_len * heads * head_dim * 2 * layers   # FP16
print(f"  KV cache for Llama 3 8B at seq=4096: {kv_bytes/1e9:.2f} GB")
print(f"  At seq=1024: {kv_bytes/4/1e9:.2f} GB")
print(f"  At seq=256:  {kv_bytes/16/1e9:.2f} GB")

print()
print("THAT'S WHY long context (100k tokens) needs so much VRAM —")
print("the KV cache alone can be larger than the weights!")

print()
print("=" * 60)
print("SPEEDUP MEASUREMENT")
print("=" * 60)

class SlowAttn(nn.Module):
    """No cache: recompute everything from scratch each step."""
    def __init__(self, D):
        super().__init__()
        self.W_Q = nn.Linear(D, D, bias=False)
        self.W_K = nn.Linear(D, D, bias=False)
        self.W_V = nn.Linear(D, D, bias=False)
        self.scale = D ** -0.5

    def forward(self, full_seq):
        Q = self.W_Q(full_seq)
        K = self.W_K(full_seq)
        V = self.W_V(full_seq)
        return torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1) @ V

slow_attn  = SlowAttn(D).eval()
fast_attn  = SingleHeadAttentionWithCache(D).eval()
n_steps    = 20
slow_times = []
fast_times = []

# Slow: recompute full sequence every step
for step in range(1, n_steps + 1):
    full_seq = torch.randn(1, step, D)
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = slow_attn(full_seq)
    slow_times.append(time.perf_counter() - t0)

# Fast: cache
kv = None
for step in range(1, n_steps + 1):
    token = torch.randn(1, 1, D)
    t0 = time.perf_counter()
    with torch.no_grad():
        _, kv = fast_attn(token, kv)
    fast_times.append(time.perf_counter() - t0)

print(f"{'Step':>5} | {'No cache (ms)':>14} | {'With cache (ms)':>16}")
print("-" * 42)
for i in range(0, n_steps, 4):
    s = slow_times[i] * 1000
    f = fast_times[i] * 1000
    print(f"  {i+1:>3}  |     {s:>8.3f}      |      {f:>8.3f}")

total_slow = sum(slow_times) * 1000
total_fast = sum(fast_times) * 1000
print(f"\nTotal {n_steps} steps:")
print(f"  No cache:   {total_slow:.2f} ms")
print(f"  With cache: {total_fast:.2f} ms")
if total_fast > 0:
    print(f"  Speedup:    {total_slow/total_fast:.1f}x")

print()
print("=" * 60)
print("WHEN YOU ENCOUNTER KV CACHE IN THE WILD")
print("=" * 60)
print("""
In HuggingFace transformers:
  model.generate(input_ids, use_cache=True)   ← default; KV cache enabled
  model.generate(input_ids, use_cache=False)  ← disables cache (slower)

  output = model(input_ids, past_key_values=None)
  # output.past_key_values = the KV cache for next call

In diffusers (AnimateDiff, Wan):
  → NOT used in the denoiser (all patches processed together, not autoregressively)
  → The TEXT ENCODER (CLIP, T5) processes all tokens at once; no generation loop
  → So KV cache is irrelevant for image/video generation inference

In LocalVideoStudio:
  → No KV cache needed. AnimateDiff is not autoregressive.
  → If you add an LLM for prompt enhancement (Qwen, Llama) → use_cache=True.
""")

# =============================================================================
# EXPECTED OUTPUT:
# Prefill input:  [1, 3, 32]
# Prefill output: [1, 3, 32]
# Cache K shape:  [1, 3, 32]
# Step 1: new_token (1,1,32) → output [1, 1, 32], cache K [1, 4, 32]
# Step 2: ... cache K [1, 5, 32]
# ...
# KV cache for Llama 3 8B at seq=4096: 2.15 GB
# =============================================================================
