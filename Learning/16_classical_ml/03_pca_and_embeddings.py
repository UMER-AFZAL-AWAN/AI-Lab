# =============================================================================
# 16_classical_ml / 03 — PCA AND EMBEDDINGS
# =============================================================================
#
# LIKE YOU ARE 5:
# PCA: imagine a cloud of points in 3D space.
#   Most points lie roughly in a flat pancake (2D plane in 3D).
#   PCA finds that pancake and projects every point onto it.
#   You went from 3D → 2D without losing much information.
#
# Embeddings: convert anything (words, images, videos) into a point in
#   high-dimensional space, where SIMILAR things are NEAR each other.
#   "King - Man + Woman ≈ Queen" is an embedding relationship.
#
# C# ANALOGY:
# PCA = data compression. Like JPEG that discards low-energy frequencies.
# Embeddings = hashing, but preserving semantic meaning.
#   HashSet<int> loses relationships. EmbeddingSpace preserves them.
#
# =============================================================================

import torch
import torch.nn as nn
import math

torch.manual_seed(42)

print("=" * 60)
print("PART 1: PCA FROM FIRST PRINCIPLES")
print("=" * 60)
print("""
Principal Component Analysis (PCA):
  1. Centre the data (subtract mean)
  2. Compute covariance matrix: how much do features co-vary?
  3. Find eigenvectors (principal components) = directions of max variance
  4. Project data onto the top K eigenvectors

The first principal component is the direction of GREATEST variance.
The second is perpendicular to the first, capturing the second-most variance.
""")

# ── Create correlated 2D data (a tilted ellipse) ─────────────────────────────
n = 200
t = torch.linspace(0, 2*math.pi, n)
# Data aligned at 45 degrees with different spreads
x1 = 3 * torch.cos(t) + 0.3 * torch.randn(n)
x2 = 1 * torch.sin(t) + 0.3 * torch.randn(n)
angle = math.pi / 4  # rotate 45°
X = torch.stack([
    x1 * math.cos(angle) - x2 * math.sin(angle),
    x1 * math.sin(angle) + x2 * math.cos(angle),
], dim=1)   # shape (200, 2)

# ── Step 1: Centre ────────────────────────────────────────────────────────────
mean = X.mean(dim=0)
Xc   = X - mean

# ── Step 2: Covariance matrix ─────────────────────────────────────────────────
# Cov[i,j] = (1/N) × X[:,i]·X[:,j]
cov = (Xc.T @ Xc) / (n - 1)
print(f"Data shape: {X.shape}")
print(f"Covariance matrix:\n{cov}")
print(f"Off-diagonal is non-zero → features are correlated (expected for rotated ellipse)")

# ── Step 3: Eigenvectors (principal components) ───────────────────────────────
eigenvalues, eigenvectors = torch.linalg.eigh(cov)
# torch.linalg.eigh returns sorted ascending; we want descending
idx          = eigenvalues.argsort(descending=True)
eigenvalues  = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]  # columns are eigenvectors

print(f"\nEigenvalues (variance explained): {eigenvalues.tolist()}")
print(f"PC1 direction: {eigenvectors[:, 0].tolist()}")
print(f"PC2 direction: {eigenvectors[:, 1].tolist()}")

# ── Step 4: Project onto PCs ──────────────────────────────────────────────────
X_pca_2d = Xc @ eigenvectors               # still 2D (keeping both PCs)
X_pca_1d = Xc @ eigenvectors[:, :1]        # reduce to 1D (top PC only)

variance_explained = eigenvalues / eigenvalues.sum()
print(f"\nVariance explained by PC1: {variance_explained[0].item()*100:.1f}%")
print(f"Variance explained by PC2: {variance_explained[1].item()*100:.1f}%")
print(f"→ PC1 alone captures {variance_explained[0].item()*100:.1f}% of information")

# ── 3D → 2D example ──────────────────────────────────────────────────────────
print()
print("3D → 2D reduction example:")
X3d = torch.randn(100, 3)
X3d[:, 2] = 0.1 * X3d[:, 0] + 0.05 * torch.randn(100)  # dim 3 is nearly redundant

Xc3 = X3d - X3d.mean(0)
cov3 = (Xc3.T @ Xc3) / 99
ev3, evec3 = torch.linalg.eigh(cov3)
ev3   = ev3.flip(0)
evec3 = evec3.flip(1)

ve3 = ev3 / ev3.sum()
print(f"3D Variance explained: PC1={ve3[0].item()*100:.1f}%, PC2={ve3[1].item()*100:.1f}%, PC3={ve3[2].item()*100:.1f}%")
print(f"PC1+PC2 captures {(ve3[0]+ve3[1]).item()*100:.1f}% — safe to drop PC3")

X_2d = Xc3 @ evec3[:, :2]  # project onto top 2 PCs
print(f"Original shape: {X3d.shape}  →  Reduced shape: {X_2d.shape}")

print()
print("=" * 60)
print("PART 2: USING PYTORCH'S torch.pca_lowrank")
print("=" * 60)
print("""
In practice, don't implement PCA yourself.
Use torch.pca_lowrank (faster via SVD) or sklearn.decomposition.PCA.
""")

U, S, V = torch.pca_lowrank(X, q=2)   # q = number of components to keep
# V columns are the principal components
# U @ diag(S) = projections
X_proj = X @ V  # project data
print(f"torch.pca_lowrank: projected shape = {X_proj.shape}")
print(f"Singular values (∝ eigenvalues): {S.tolist()}")

print()
print("=" * 60)
print("PART 3: EMBEDDINGS — SEMANTIC VECTORS")
print("=" * 60)
print("""
Embeddings convert discrete tokens (words, pixels, frames) into
continuous vectors in a high-dimensional space.

Key property: semantic similarity = geometric proximity
  "cat" and "kitten" → vectors that are close together
  "cat" and "skyscraper" → vectors far apart

How to measure similarity: cosine similarity
  sim(a, b) = (a · b) / (|a| × |b|)
  = 1.0: identical direction (most similar)
  = 0.0: perpendicular (unrelated)
  = -1.0: opposite (most different)
""")

def cosine_sim(a, b):
    return (a @ b) / (a.norm() * b.norm())

# Fake word embeddings (normally these come from a trained model)
# But we can verify the similarity property:
dim = 8
torch.manual_seed(1)
cat     = torch.randn(dim)
kitten  = cat + 0.3 * torch.randn(dim)   # close to cat
dog     = cat + 0.8 * torch.randn(dim)   # somewhat close
car     = torch.randn(dim)               # completely different direction

print(f"cos_sim(cat, kitten)  = {cosine_sim(cat, kitten).item():.3f}  (should be high)")
print(f"cos_sim(cat, dog)     = {cosine_sim(cat, dog).item():.3f}  (medium)")
print(f"cos_sim(cat, car)     = {cosine_sim(cat, car).item():.3f}  (should be low/random)")

print()
print("=" * 60)
print("PART 4: nn.Embedding — the learnable lookup table")
print("=" * 60)
print("""
In a neural network, nn.Embedding is a learnable table:
  input: integer token id (0 to vocab_size-1)
  output: a dense vector of size embed_dim

It's just a matrix W of shape (vocab_size, embed_dim).
embedding(token_id) = W[token_id]  ← just a row lookup (like a dict)

The vectors are learned via backpropagation.
After training, similar tokens have similar vectors.
""")

vocab_size = 10   # 10 possible tokens
embed_dim  = 4

embedding_layer = nn.Embedding(vocab_size, embed_dim)
print(f"Embedding table shape: {embedding_layer.weight.shape}  ({vocab_size} tokens × {embed_dim} dims)")

# Look up embeddings for tokens 0, 3, 7
token_ids = torch.tensor([0, 3, 7])
embeds    = embedding_layer(token_ids)
print(f"Token IDs: {token_ids.tolist()}")
print(f"Embeddings shape: {embeds.shape}")
print(f"Token 0 embedding: {embeds[0].tolist()}")

print()
print("Sentence embedding (average pooling):")
sentence_tokens = torch.tensor([1, 5, 3, 7, 2])    # 5 tokens
sentence_embeds = embedding_layer(sentence_tokens)   # (5, 4)
sentence_vector = sentence_embeds.mean(dim=0)        # (4,) — average all tokens
print(f"5 tokens → mean pooled → shape {sentence_vector.shape}")

print()
print("=" * 60)
print("PART 5: WHERE EMBEDDINGS APPEAR IN AI MODELS")
print("=" * 60)
print("""
Text models:
  Token embedding: each word/subword → 768-dim vector (BERT, GPT)
  Positional embedding: position 0,1,2,... → added to token embeddings
  → Combined: the input to every Transformer

Image models:
  Patch embedding (ViT): each 16×16 patch → flattened → linear → 768-dim
  CLIP: image encoder maps image → 512-dim vector
        text encoder maps caption → 512-dim vector
        Training: push matching pairs TOGETHER, non-matching APART

Video models (AnimateDiff):
  Frame embeddings: each frame is a patch-embedded ViT feature
  Temporal embedding: frame index → added to frame features

Retrieval systems (RAG):
  Encode documents as vectors → store in vector database
  Encode query as vector → find nearest documents by cosine similarity
  → retrieve-augmented generation (RAG, see 20_nlp_and_llms/)
""")

print("=" * 60)
print("QUICK REFERENCE: PCA vs EMBEDDINGS")
print("=" * 60)
print("""
            PCA                         Embeddings
Purpose     Reduce dimensionality       Represent anything as a vector
How         Find axes of max variance   Train network to encode semantics
Input       Numerical features          Anything (text, image, audio, video)
Output      Compressed features         Dense vector (e.g., 768-dim)
Training    Closed form (SVD)           Gradient descent
Interpretable Yes (variance explained)  Hard (high-dimensional)
Use case    Visualise data, speed up ML All modern deep learning
""")

# =============================================================================
# EXPECTED OUTPUT:
# PC1 captures ~75-85% of variance (most info in one direction)
# cosine_sim(cat, kitten) > cosine_sim(cat, car)
# Embedding table: (10, 4) learnable matrix
# =============================================================================
