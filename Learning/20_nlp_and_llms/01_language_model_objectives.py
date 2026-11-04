# =============================================================================
# 20_nlp_and_llms / 01 — LANGUAGE MODEL OBJECTIVES
# =============================================================================
#
# LIKE YOU ARE 5:
# BERT: "I ___ to school" — fill in the blank (masked word prediction).
#   Like a crossword puzzle: given surrounding context, guess the hidden word.
# GPT: "The cat sat on the ___" — predict the NEXT word.
#   Like auto-complete on your phone: what word comes next?
#
# C# ANALOGY:
# BERT (bidirectional): look at the whole string, fill in a gap.
#   Like: string.Contains() — can see left AND right context.
# GPT (causal): only see characters to the LEFT of cursor.
#   Like: string.Substring(0, cursorPos) — only past context.
#
# =============================================================================

import torch
import torch.nn as nn
import math

torch.manual_seed(42)

print("=" * 60)
print("PART 1: BERT — MASKED LANGUAGE MODELLING (MLM)")
print("=" * 60)
print("""
BERT training objective:
  1. Take a sentence: "The cat sat on the mat"
  2. Randomly mask 15% of tokens: "The [MASK] sat on the [MASK]"
  3. Train the model to predict the MASKED tokens from CONTEXT
  4. BERT sees both left and right context (bidirectional)

Why bidirectional?
  "The bank is by the [MASK]" — river or money? Needs "bank" context.
  Bidirectional lets BERT use both sides to disambiguate.

What BERT is good for:
  - Text classification (sentiment, spam detection)
  - Named entity recognition (find "New York" = location)
  - Question answering (find the answer span in a document)
  - Semantic similarity (embedding similarity)

BERT is NOT a text generator — it can't auto-complete sentences.
""")

# Demonstrate attention mask (what BERT sees)
def show_attention_mask(tokens, masked_positions, vocab_size=1000):
    """Show what BERT's input looks like."""
    # In BERT, [MASK] token has a special ID (103 in BERT-base)
    MASK_ID = 103
    ids = list(range(1, len(tokens)+1))   # fake token IDs
    for pos in masked_positions:
        ids[pos] = MASK_ID
    return ids

sentence = ["The", "cat", "sat", "on", "the", "mat"]
masked   = show_attention_mask(sentence, masked_positions=[1, 5])
print("BERT input (masked):")
for tok, id_ in zip(sentence, masked):
    marker = " ← [MASK]" if id_ == 103 else ""
    print(f"  {tok:8s} → token_id={id_}{marker}")

print()
print("=" * 60)
print("PART 2: GPT — CAUSAL LANGUAGE MODELLING (CLM)")
print("=" * 60)
print("""
GPT training objective:
  1. Take a sentence: "The cat sat on the mat"
  2. At each position, predict the NEXT token
  3. At position i, only see tokens 0..i-1 (causal, left-to-right)

Training: given the token sequence, the target is the same sequence
shifted by 1 position:
  Input:  [The, cat, sat, on, the]
  Target: [cat, sat, on,  the, mat]  ← predict next token at every position

Causal mask: a triangular mask prevents the model from "cheating"
by looking at future tokens during training.

What GPT is good for:
  - Text generation (auto-complete, story writing)
  - In-context learning ("few-shot" examples in the prompt)
  - Instruction following (GPT-4, Claude, etc.)
  - Code generation (Codex, GitHub Copilot)
""")

# The causal attention mask (lower triangular)
seq_len = 6
causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
print(f"Causal attention mask ({seq_len}×{seq_len}):")
print("  Position: 0  1  2  3  4  5")
for i in range(seq_len):
    row = "  ".join("1" if causal_mask[i, j] else "0" for j in range(seq_len))
    print(f"  Token {i}:  {row}")
print("  1 = can attend, 0 = future (blocked)")
print("  Token 0 sees only itself; Token 5 sees all 6 tokens")

print()
print("=" * 60)
print("PART 3: TOKENISATION")
print("=" * 60)
print("""
Before text enters any model, it must be converted to integers (token IDs).

Tokenisation levels:
  Character:    "Hello" → [H, e, l, l, o] → [72, 101, 108, 108, 111]
                Too fine-grained; "Hello" = 5 tokens (expensive)

  Word:         "Hello" → [Hello] → [19875]
                Unknown words (OOV) problem; huge vocabulary

  Subword (BPE): "Hello" → ["Hello"] → [15496]
                 "unforgettable" → ["un", "forget", "table"] → [470, 20074, 2096]
                 Balances vocabulary size with sequence length
                 Used by all modern LLMs (GPT-2, BERT, LLaMA, Claude)

BPE (Byte Pair Encoding):
  Start with individual bytes.
  Repeatedly merge the most common adjacent pair.
  After 50,000 merges → 50,257-token vocabulary (GPT-2).
  "Common" words are one token; rare words are split into subwords.
""")

# Using HuggingFace tokenizer (requires: pip install transformers)
try:
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    sentences = [
        "Hello, world!",
        "unforgettable",
        "PyTorch is a deep learning framework",
    ]
    for sent in sentences:
        ids    = tok.encode(sent)
        tokens = tok.convert_ids_to_tokens(ids)
        print(f'  "{sent}"')
        print(f'    → {tokens}')
        print(f'    → {ids}\n')
except Exception:
    print("""
(transformers not installed — documentation mode)
  'Hello, world!'  → ['Hello', ',', ' world', '!'] → [15496, 11, 995, 0]
  'unforgettable'  → ['un', 'forget', 'table'] → [470, 20074, 2096]

  Install: pip install transformers
""")

print("=" * 60)
print("PART 4: NEXT-TOKEN PREDICTION IN DETAIL")
print("=" * 60)
print("""
Language model training is simply:
  Input:   token sequence [t_0, t_1, ..., t_{n-1}]
  Target:  same sequence shifted by 1: [t_1, t_2, ..., t_n]

  At every position, predict the NEXT token.
  Loss = CrossEntropyLoss averaged over all positions.

This is called "teacher forcing" — always give the correct previous tokens,
even if the model's prediction was wrong.

At inference (generation), the model is "autoregressive":
  Give it "The cat" → predicts "sat"
  Give it "The cat sat" → predicts "on"
  Give it "The cat sat on" → predicts "the"
  ... and so on until [EOS] token or max_length

Sampling strategies:
  Greedy:        always pick the token with highest probability
                 Deterministic but often repetitive
  Temperature:   logits / T → lower T = more confident, higher T = more random
  Top-K:         only sample from top K tokens (K=50 is common)
  Top-P (nucleus): sample from tokens whose cumulative prob >= P (P=0.9)
""")

# Demonstrate temperature's effect on sampling distribution
logits    = torch.tensor([2.0, 1.5, 0.5, -0.5, -2.0])
vocab     = ["cat", "dog", "bird", "fish", "rock"]
print("Temperature effect on token probabilities:")
for T in [0.1, 0.5, 1.0, 2.0]:
    probs = torch.softmax(logits / T, dim=0)
    top2  = [(vocab[i], f"{probs[i].item():.3f}") for i in probs.argsort(descending=True)[:2]]
    print(f"  T={T:.1f}: {top2[0]} | {top2[1]} ... (sum={probs.sum().item():.3f})")

print()
print("=" * 60)
print("PART 5: BERT vs GPT QUICK COMPARISON")
print("=" * 60)
print("""
                BERT                    GPT
──────────────────────────────────────────────────────────────
Architecture    Encoder only            Decoder only
Attention       Bidirectional           Causal (left-to-right)
Objective       Masked LM + NSP         Next token prediction
Strength        Understanding text      Generating text
Use cases       Classification, NER,    Generation, chat,
                Q&A, embeddings         code, reasoning
Training data   Large text corpus       Large text corpus
Examples        BERT, RoBERTa, ALBERT   GPT-2/3/4, LLaMA,
                                        Claude, Mistral

Encoder-Decoder (T5, BART):
  Encoder: bidirectional (like BERT) — understands input
  Decoder: causal (like GPT) — generates output
  Best for: translation, summarisation, question answering
""")

# =============================================================================
# EXPECTED OUTPUT:
# Causal mask: lower triangular of 1s
# Temperature: T=0.1 → one token dominates; T=2.0 → more uniform distribution
# =============================================================================
