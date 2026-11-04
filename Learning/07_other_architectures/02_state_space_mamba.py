# =============================================================================
# 02 — STATE SPACE MODELS AND MAMBA
# =============================================================================
#
# LIKE YOU ARE 5:
# Transformers are like a classroom where every student can talk to every other student
# at the same time. Very powerful. But if the class has 100,000 students,
# that's 100,000 × 100,000 = 10 BILLION conversations at once. Too expensive.
#
# RNNs are like a telephone chain: student 1 tells student 2, who tells student 3.
# Fast, but long chains forget what student 1 said.
#
# State Space Models (SSM) are a third way:
# Like RNNs (can process one step at a time, O(1) memory per step),
# but trainable in PARALLEL like a Transformer (using convolutions).
# They are essentially a learnable linear filter over the sequence.
#
# MAMBA (2023) added "selectivity" to SSMs:
# The model can CHOOSE which parts of the input to remember
# based on the current input (like LSTM gates, but more mathematically principled).
# Result: matches Transformer quality at a fraction of the compute for long sequences.
#
# PRACTICAL STATUS:
# Mamba is used in long-context language models and long video understanding.
# Not yet dominant but growing fast (2024-2025).
#
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("STATE SPACE MODELS: CONTINUOUS-TIME INTUITION")
print("=" * 60)
#
# A linear SSM is defined by four matrices: A, B, C, D
#
# Continuous time:
#   h'(t) = A * h(t) + B * x(t)   <- state update
#   y(t)  = C * h(t) + D * x(t)   <- output
#
# Discrete time (what we implement):
#   h[t] = A_bar * h[t-1] + B_bar * x[t]
#   y[t] = C * h[t]
#
# This is linear RNN — it CAN be unrolled as a convolution (parallel training).
# The key: matrix powers A_bar^k decay, so the state "forgets" old inputs.

print("Simple 1D state space model:")

class SimpleSSM(nn.Module):
    """Minimal discrete SSM demonstration."""
    def __init__(self, state_dim=4, input_dim=1):
        super().__init__()
        # Learnable matrices
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.1)
        self.B = nn.Parameter(torch.randn(state_dim, input_dim) * 0.1)
        self.C = nn.Parameter(torch.randn(1, state_dim) * 0.1)

    def forward_sequential(self, x):
        """Process sequence one step at a time (like RNN)."""
        B_batch, T, _ = x.shape
        h = torch.zeros(B_batch, self.A.shape[0])
        outputs = []
        for t in range(T):
            h = h @ self.A.T + x[:, t] @ self.B.T   # state update
            y = h @ self.C.T                           # output
            outputs.append(y)
        return torch.stack(outputs, dim=1)   # (B, T, 1)

ssm = SimpleSSM(state_dim=4, input_dim=1)
x   = torch.randn(2, 10, 1)   # batch=2, seq=10, features=1
out = ssm.forward_sequential(x)
print(f"Input:  {x.shape}")
print(f"Output: {out.shape}  (sequential processing)")

print()
print("=" * 60)
print("MAMBA'S KEY INNOVATION: SELECTIVE STATE SPACE")
print("=" * 60)
#
# In vanilla SSM: A, B, C are FIXED for all input tokens.
# In Mamba: B and C are FUNCTIONS of the input x.
# This is called "selectivity" or "input-dependent" SSM.
#
# B(x) = Linear(x)   <- how much to let new input IN depends on x
# C(x) = Linear(x)   <- how much to expose the state depends on x
#
# This is similar to LSTM gates, but applied to the SSM formulation.
# Mamba can "zoom in" on important tokens and "ignore" unimportant ones.

class SelectiveSSMBlock(nn.Module):
    """Simplified Mamba-like selective SSM block."""
    def __init__(self, d_model=16, d_state=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Input projections
        self.in_proj = nn.Linear(d_model, d_model * 2)   # expand

        # SSM parameters
        self.A_log   = nn.Parameter(torch.randn(d_model, d_state))
        self.D       = nn.Parameter(torch.ones(d_model))

        # Selective projections (input-dependent B, C, dt)
        self.dt_proj = nn.Linear(d_model, d_model)
        self.B_proj  = nn.Linear(d_model, d_state)
        self.C_proj  = nn.Linear(d_model, d_state)

        self.out_proj= nn.Linear(d_model, d_model)
        self.norm    = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, D = x.shape
        residual = x

        # Split into two paths: one for SSM, one for gating (SiLU)
        xz = self.in_proj(x)                              # (B, T, 2*D)
        x_, z = xz.chunk(2, dim=-1)                       # each (B, T, D)

        # Compute input-dependent SSM parameters
        dt = F.softplus(self.dt_proj(x_))                 # (B, T, D)
        B_in = self.B_proj(x_)                            # (B, T, d_state)
        C_in = self.C_proj(x_)                            # (B, T, d_state)

        # (Simplified: just use a linear transform as stand-in for the actual SSM scan)
        # Real Mamba uses a parallel associative scan for the recurrence
        y = x_ * dt                                       # simplified placeholder

        # Gate with z (SiLU gating like in LLaMA FFN)
        y = y * F.silu(z)
        y = self.out_proj(y)
        return self.norm(y + residual)

block = SelectiveSSMBlock(d_model=16)
x     = torch.randn(2, 20, 16)
out   = block(x)
print(f"Mamba block input:  {x.shape}")
print(f"Mamba block output: {out.shape}  (same shape)")
params = sum(p.numel() for p in block.parameters())
print(f"Parameters: {params:,}")

print()
print("=" * 60)
print("COMPLEXITY COMPARISON")
print("=" * 60)

print(f"{'Model':<18} {'Training':>12} {'Inference':>12} {'Long-seq?':>10}")
print("-" * 55)
print(f"{'Transformer':<18} {'O(T²·D)':>12} {'O(T²·D)':>12} {'Poor':>10}")
print(f"{'RNN/LSTM':<18} {'O(T·D²)':>12} {'O(D²)':>12} {'OK':>10}")
print(f"{'Mamba/SSM':<18} {'O(T·D·N)':>12} {'O(D·N)':>12} {'Good':>10}")
print()
print("T = sequence length, D = model dim, N = state dim")
print()
print("At T=1000, D=512: Transformer attention = 512M ops/step")
print("                  Mamba with N=16     =   8M ops/step (64x cheaper)")
print()
print("Real-world: Mamba-2 (2024) matches GPT-4 quality on long documents")
print("while being 2-5x faster on sequences > 2048 tokens.")

# =============================================================================
# EXPECTED OUTPUT:
# Input:  torch.Size([2, 10, 1])
# Output: torch.Size([2, 10, 1])
# Mamba block output: torch.Size([2, 20, 16])
# =============================================================================
