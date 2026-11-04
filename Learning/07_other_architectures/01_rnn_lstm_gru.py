# =============================================================================
# 01 — RNN, LSTM, GRU: SEQUENTIAL MEMORY
# =============================================================================
#
# LIKE YOU ARE 5:
# When you read a sentence word by word, you remember what you read before.
# "The cat sat on the ___" — you expect "mat" because you remember "cat sat".
#
# A RNN (Recurrent Neural Network) processes sequences one item at a time.
# It has a HIDDEN STATE — a summary of everything seen so far.
# At each step: new_hidden = f(previous_hidden, current_input)
# This is the "memory" passed from one step to the next.
#
# PROBLEMS with vanilla RNN:
# - Long-term memory fails: by step 100, it has forgotten step 1.
#   The hidden state gets overwritten completely at every step.
#   Gradients also vanish over long sequences.
#
# LSTM FIX (Long Short-Term Memory, 1997):
# - Add a CELL STATE (long-term memory) separate from the hidden state.
# - Three GATES control what to remember and forget:
#   FORGET GATE: how much of the old cell state to keep.
#   INPUT GATE:  how much of the new info to write to cell state.
#   OUTPUT GATE: how much of the cell state to expose as hidden state.
#
# GRU (Gated Recurrent Unit, 2014):
# - Simplified version of LSTM. Two gates instead of three.
# - Often similar performance to LSTM, faster to train.
#
# STATUS: mostly replaced by Transformers for NLP (attention handles long-range
#         better). Still used in real-time/streaming applications (video games,
#         robotics) where you must process one frame at a time.
#
# =============================================================================

import torch
import torch.nn as nn

print("=" * 60)
print("VANILLA RNN")
print("=" * 60)

rnn = nn.RNN(
    input_size=16,    # size of each input step (e.g. word embedding dim)
    hidden_size=32,   # size of the hidden state (the memory)
    num_layers=1,
    batch_first=True, # input shape: (batch, seq, features) — easier to reason about
)

# Process a sequence of 10 tokens, each 16-dimensional
x = torch.randn(2, 10, 16)   # batch=2, seq=10, features=16
h0 = torch.zeros(1, 2, 32)   # initial hidden state (num_layers, batch, hidden)

out, hn = rnn(x, h0)
print(f"Input:   {x.shape}")
print(f"Output:  {out.shape}   (hidden state at every step)")
print(f"hn:      {hn.shape}    (hidden state at the LAST step)")
print("out[-1] == hn[0]: all close?", torch.allclose(out[:, -1], hn[0]))
print("(The last output step IS the final hidden state)")

print()
print("=" * 60)
print("LSTM: LONG-TERM MEMORY")
print("=" * 60)

lstm = nn.LSTM(
    input_size=16,
    hidden_size=32,
    num_layers=2,     # stack 2 LSTM layers
    batch_first=True,
    bidirectional=False,
)

h0 = torch.zeros(2, 2, 32)   # (num_layers, batch, hidden)
c0 = torch.zeros(2, 2, 32)   # cell state (the long-term memory)

out, (hn, cn) = lstm(x, (h0, c0))
print(f"LSTM output:        {out.shape}")
print(f"Final hidden state: {hn.shape}  (num_layers, batch, hidden)")
print(f"Final cell state:   {cn.shape}  (long-term memory)")
print()
print("The cell state (cn) is the long-term memory.")
print("The hidden state (hn) is what gets exposed at each step.")
print("The forget gate lets the network keep important info across many steps.")

print()
print("=" * 60)
print("GRU: SIMPLER THAN LSTM")
print("=" * 60)

gru = nn.GRU(input_size=16, hidden_size=32, num_layers=1, batch_first=True)
out, hn = gru(x)
print(f"GRU output: {out.shape}  (similar to LSTM but no separate cell state)")
params_lstm = sum(p.numel() for p in lstm.parameters())
params_gru  = sum(p.numel() for p in gru.parameters())
print(f"LSTM params (2 layers): {params_lstm:,}")
print(f"GRU  params (1 layer):  {params_gru:,}")

print()
print("=" * 60)
print("USING LSTM FOR SEQUENCE CLASSIFICATION")
print("=" * 60)
#
# Common pattern: encode a sequence, take the final hidden state,
# classify with a Linear layer.

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.emb  = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, num_classes)

    def forward(self, token_ids):
        emb = self.emb(token_ids)                # (B, seq, embed_dim)
        out, (hn, _) = self.lstm(emb)            # hn: (1, B, hidden_dim)
        return self.fc(hn.squeeze(0))            # (B, num_classes)

clf = LSTMClassifier(vocab_size=1000, embed_dim=32, hidden_dim=64, num_classes=5)
ids = torch.randint(0, 1000, (4, 20))   # batch=4, seq=20
out = clf(ids)
print(f"Input:  {ids.shape}  (4 sentences of 20 tokens)")
print(f"Output: {out.shape}  (4 sentences, 5 class scores each)")

print()
print("WHEN TO STILL USE RNN/LSTM:")
print("- Streaming audio/sensor data (process one sample at a time)")
print("- When sequence position memory matters more than global context")
print("- Edge devices with limited compute (Transformers need quadratic memory)")

# =============================================================================
# EXPECTED OUTPUT:
# Output:  torch.Size([2, 10, 32])
# LSTM output: torch.Size([2, 10, 32])
# GRU output: torch.Size([2, 10, 32])
# =============================================================================
