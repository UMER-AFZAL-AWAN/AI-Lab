# =============================================================================
# 05_transformer / 09 — MIXTURE OF EXPERTS (MoE)
# =============================================================================
#
# LIKE YOU ARE 5:
# Imagine a school with 8 specialist teachers: one for math, one for art,
# one for history, etc. For every question, a router (the head teacher)
# picks the 2 best teachers. The student only talks to those 2.
# Total teachers: 8. Work per question: 2. Efficient!
#
# MoE works the same:
#   - Replace the FFN layer in a Transformer with N expert FFNs.
#   - A router picks the top K experts for each token.
#   - Only K out of N experts run (the rest do nothing).
#   - Total parameters: huge. Active parameters per token: much smaller.
#
# C# ANALOGY:
# Like a strategy pattern where only 2 of 8 IStrategy implementations
# are instantiated per request, chosen by a Router factory.
# The others exist in memory (parameters counted) but don't execute.
#
# REAL EXAMPLES:
#   Qwen2-MoE:      14.3B total params, 2.7B active per token
#   Mixtral 8×7B:   47B total params,   13B active per token (top 2 of 8)
#   GPT-4 (rumored): dense experts, ~1.8T total
#
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("PART 1: THE STANDARD TRANSFORMER FFN (baseline)")
print("=" * 60)

class StandardFFN(nn.Module):
    """
    Standard Transformer feedforward block.
    Every token uses this same FFN every time.
    hidden_dim usually 4x embed_dim.
    """
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.up   = nn.Linear(embed_dim, hidden_dim)
        self.down = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        return self.down(F.silu(self.up(x)))   # SiLU common in modern LLMs

embed_dim  = 16
hidden_dim = 64

ffn    = StandardFFN(embed_dim, hidden_dim)
params = sum(p.numel() for p in ffn.parameters())
print(f"Standard FFN parameters: {params:,}")
print(f"  Parameters per token (active = total): {params:,}")

x = torch.randn(2, 5, embed_dim)   # batch=2, seq=5, dim=16
print(f"Input:  {list(x.shape)}")
print(f"Output: {list(ffn(x).shape)}")

print()
print("=" * 60)
print("PART 2: MIXTURE OF EXPERTS FFN")
print("=" * 60)

class MoEFFN(nn.Module):
    """
    Replace a single FFN with N expert FFNs + a router.
    Only top_k experts activate per token.

    The key property: total_params = N × ffn_params
                      active_params_per_token = top_k × ffn_params
    """
    def __init__(self, embed_dim, hidden_dim, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k

        # N expert FFNs
        self.experts = nn.ModuleList([
            StandardFFN(embed_dim, hidden_dim) for _ in range(num_experts)
        ])

        # Router: maps each token's embedding → scores over N experts
        self.router = nn.Linear(embed_dim, num_experts, bias=False)

    def forward(self, x):
        """
        x: (B, T, D)
        Returns: (B, T, D) — same shape as StandardFFN
        """
        B, T, D = x.shape
        x_flat = x.view(B * T, D)         # (B*T, D) — treat each token independently

        # Router: score each token for each expert
        router_logits = self.router(x_flat)                   # (B*T, num_experts)
        router_probs  = F.softmax(router_logits, dim=-1)       # (B*T, num_experts)

        # Pick top-k experts per token
        top_k_probs, top_k_idx = torch.topk(router_probs, self.top_k, dim=-1)
        # top_k_probs: (B*T, top_k)  — the weights for each selected expert
        # top_k_idx:   (B*T, top_k)  — which expert (0..num_experts-1)

        # Re-normalise weights so they sum to 1
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Run each selected expert and combine with weighted sum
        output = torch.zeros_like(x_flat)   # (B*T, D)

        for k in range(self.top_k):
            expert_ids   = top_k_idx[:, k]            # which expert for each token
            expert_probs = top_k_probs[:, k]           # weight for that expert

            for e_idx in range(self.num_experts):
                # Find tokens routed to expert e_idx at position k
                mask = (expert_ids == e_idx)           # bool tensor (B*T,)
                if not mask.any():
                    continue
                tokens_for_e = x_flat[mask]            # only the tokens this expert handles
                expert_out   = self.experts[e_idx](tokens_for_e)  # run the expert
                # Weight the output and add back
                output[mask] += expert_probs[mask].unsqueeze(-1) * expert_out

        return output.view(B, T, D)


num_experts = 8
top_k       = 2

moe = MoEFFN(embed_dim, hidden_dim, num_experts=num_experts, top_k=top_k)
total_params  = sum(p.numel() for p in moe.parameters())
router_params = sum(p.numel() for p in moe.router.parameters())
expert_params = sum(p.numel() for p in moe.experts[0].parameters())
active_params = top_k * expert_params + router_params

print(f"MoE ({num_experts} experts, top-{top_k}):")
print(f"  Total parameters: {total_params:,}")
print(f"    - Router:   {router_params:,}")
print(f"    - Experts:  {num_experts} × {expert_params:,} = {num_experts*expert_params:,}")
print(f"  Active per token: {active_params:,}  ({100*active_params/total_params:.0f}% of total)")
print(f"  Standard FFN was: {params:,}")
print(f"  MoE total / Standard ratio: {total_params/params:.1f}x more params")
print(f"  But active / Standard ratio: {active_params/params:.1f}x (same FLOPs!)")

print()
print("INPUT → OUTPUT:")
x = torch.randn(2, 5, embed_dim)
print(f"  Input:  {list(x.shape)}")
out = moe(x)
print(f"  Output: {list(out.shape)}")

print()
print("=" * 60)
print("PART 3: ROUTER VISUALISATION — WHICH EXPERTS GET TOKENS?")
print("=" * 60)

# Freeze the MoE and inspect routing decisions
with torch.no_grad():
    flat_x = x.view(-1, embed_dim)
    logits = moe.router(flat_x)
    probs  = F.softmax(logits, dim=-1)
    top2   = torch.topk(probs, 2, dim=-1)

print(f"{'Token':>6} | {'Top-1 expert':>12} | {'Top-2 expert':>12} | {'Top-1 prob':>12}")
print("-" * 50)
for i in range(flat_x.shape[0]):
    e1, e2 = top2.indices[i, 0].item(), top2.indices[i, 1].item()
    p1     = top2.values[i, 0].item()
    print(f"  {i:>4}   |      {e1:>4}        |      {e2:>4}        |    {p1:.3f}")

print()
print("=" * 60)
print("PART 4: REAL NUMBERS (Qwen2-MoE, Mixtral)")
print("=" * 60)
print("""
Qwen2-MoE (Qwen2-57B-A14B):
  Total:  57B parameters
  Active: 14B per token  (top-4 of 64 experts)
  Layers: some are dense (always active), some are MoE
  VRAM:   ~28 GB at FP16 (weights only), ~8-9 GB at INT4

Mixtral 8×7B:
  Total:  47B parameters
  Active: ~13B per token  (top-2 of 8 experts per layer)
  VRAM:   ~24 GB at FP16, ~12 GB at INT4

Qwen2.5-3B (small dense, NOT MoE, fits on 8 GB):
  Total:  3B parameters, all active
  VRAM:   ~6 GB at FP16, ~2 GB at INT4

RULE FOR 8 GB VRAM:
  MoE model fits if: ACTIVE params × 2 bytes + KV cache ≤ 7 GB
  (you pay for ALL weights in memory but only run top-k)
  So Qwen2-57B-A14B still needs ~28 GB VRAM just for weights — doesn't fit.
  Use Qwen2.5-3B or Llama 3.2-3B for text tasks on 8 GB.
""")

print("=" * 60)
print("PART 5: LOAD BALANCING — THE EXPERT COLLAPSE PROBLEM")
print("=" * 60)
print("""
Problem: a poorly trained router sends MOST tokens to 1-2 experts.
Result:  the other experts are never trained → useless.
         The model is effectively a small dense model, not a large MoE.

Fix: auxiliary load balancing loss during training:
  L_aux = coefficient × (fraction of tokens per expert)² summed
  Penalises lopsided expert usage.

You don't implement this yourself — it's part of training.
When running inference, you just use the already-trained model.
But it explains WHY real MoE models (Mixtral, Qwen-MoE) outperform
equivalently-parameterised dense models:
  - More diverse learned representations per layer
  - Each expert specialises in different token patterns
""")

print("=" * 60)
print("WHAT THIS MEANS FOR LOCALVIDEOSTUDIO")
print("=" * 60)
print("""
Current stack: AnimateDiff — no MoE involved.
Text encoder (CLIP): dense, 123M params, 0.25 GB.

Possible upgrade paths that use MoE:
  Qwen2.5-3B for prompt enhancement → dense, fine.
  Qwen-MoE for caption generation → needs 24+ GB, cloud only.

If a future video model uses MoE for the denoiser:
  - Check "active" params, not total params, for VRAM estimate.
  - Load with device_map="auto" and INT4 quantisation.
  - Performance = active params × 2B per step (not total params).
""")

# =============================================================================
# EXPECTED OUTPUT:
# MoE (8 experts, top-2):
#   Total parameters: ~8,448
#   Active per token: ~2,208 (26% of total)
# Input → Output: same shape
# Router table: which 2 experts each token goes to
# =============================================================================
