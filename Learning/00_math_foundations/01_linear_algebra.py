# =============================================================================
# 00_math_foundations / 01 — LINEAR ALGEBRA FOR AI
# =============================================================================
#
# LIKE YOU ARE 5:
# Every number in AI lives in a grid. A single number is a dot (scalar).
# A row of numbers is a ruler (vector). A grid of numbers is a table (matrix).
# A stack of tables is a cube (tensor). All of AI is just maths on these grids.
#
# C# ANALOGY:
# scalar = int/float
# vector = float[]
# matrix = float[,]
# tensor = float[,,,]  (n-dimensional)
# matrix multiply = not element-wise — it's a "dot product of rows × columns"
#
# WHY THIS MATTERS:
# Every layer weight is a matrix. Every forward pass is matrix multiplication.
# "Linear" in "Linear layer" literally means: y = W @ x + b (a matrix multiply).
# Knowing this demystifies every shape error you will ever see.
#
# =============================================================================

import torch
import numpy as np

print("=" * 60)
print("SCALARS, VECTORS, MATRICES, TENSORS")
print("=" * 60)

scalar = torch.tensor(3.14)
vector = torch.tensor([1.0, 2.0, 3.0])          # shape (3,)
matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]]) # shape (2, 2)
tensor3d = torch.randn(2, 3, 4)                  # shape (2, 3, 4)

print(f"Scalar:  {scalar}          shape={scalar.shape}")
print(f"Vector:  {vector}  shape={vector.shape}")
print(f"Matrix:\n{matrix}  shape={matrix.shape}")
print(f"3D tensor: shape={tensor3d.shape}  ({tensor3d.numel()} numbers total)")

print()
print("=" * 60)
print("THE DOT PRODUCT — foundation of attention")
print("=" * 60)
#
# Dot product of two vectors: multiply element-wise, then sum.
# a · b = a[0]*b[0] + a[1]*b[1] + ... = ||a|| * ||b|| * cos(θ)
#
# GEOMETRIC MEANING:
# Large positive → vectors point the same direction (similar)
# Near zero      → vectors are perpendicular (unrelated)
# Large negative → vectors point opposite directions (dissimilar)
#
# This is EXACTLY what attention's Q·K computes: how similar is query to key?

a = torch.tensor([1.0, 0.0, 0.0])   # points "right"
b = torch.tensor([1.0, 0.0, 0.0])   # same direction
c = torch.tensor([0.0, 1.0, 0.0])   # perpendicular
d = torch.tensor([-1.0, 0.0, 0.0])  # opposite

print(f"a·a (same direction):   {torch.dot(a, a):.2f}  ← maximum similarity")
print(f"a·c (perpendicular):    {torch.dot(a, c):.2f}  ← no similarity")
print(f"a·d (opposite):         {torch.dot(a, d):.2f}  ← maximum dissimilarity")
print()
print("In attention:  scores = Q @ K.T")
print("  → each score is the dot product between one query and one key")
print("  → high score = this token should attend to that token")

print()
print("=" * 60)
print("MATRIX MULTIPLICATION — the core operation of every neural network")
print("=" * 60)
#
# C = A @ B:
#   A is (m, k), B is (k, n) → C is (m, n)
#   C[i, j] = dot product of row i of A with column j of B
#
# RULE: inner dimensions must match. (m, K) @ (K, n) → (m, n)
# This is the most important shape rule in all of deep learning.

A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])   # (2, 2)
B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])   # (2, 2)
C = A @ B
print(f"A:\n{A}")
print(f"B:\n{B}")
print(f"A @ B:\n{C}")
print()
# Manual verification of C[0,0]:
c00 = A[0, 0]*B[0, 0] + A[0, 1]*B[1, 0]
print(f"C[0,0] manually: {A[0,0]}×{B[0,0]} + {A[0,1]}×{B[1,0]} = {c00}")

print()
print("SHAPE RULE EXAMPLES:")
shapes = [
    ((1, 768), (768, 3072), "Linear layer: 1 token × (768→3072 FFN)"),
    ((2, 77, 768), (768, 768), "Batched: 2 × 77 tokens, project 768→768"),
    ((1, 8, 77, 96), (1, 8, 96, 77), "Attention: Q(1,H,T,d) @ K.T(1,H,d,T)"),
]
for a_shape, b_shape, desc in shapes:
    # Check inner dim matches
    ok = a_shape[-1] == b_shape[-2] if len(b_shape) >= 2 else False
    out = (*a_shape[:-1], b_shape[-1])
    print(f"  {str(a_shape):20} @ {str(b_shape):25} → {str(out):20}  {desc}")

print()
print("=" * 60)
print("TRANSPOSE — flipping rows and columns")
print("=" * 60)
W = torch.randn(4, 3)   # 4 output features, 3 input features
print(f"W shape: {W.shape}  (out_features, in_features)")
print(f"W.T shape: {W.T.shape}  (transposed)")
print()
print("nn.Linear stores W as (out, in).")
print("In forward(): output = input @ W.T + b")
print("That's why nn.Linear(3, 4) takes (batch, 3) → (batch, 4)")

print()
print("=" * 60)
print("NORMS — measuring the size of a vector")
print("=" * 60)
#
# L2 norm (Euclidean length): √(x₀² + x₁² + ... + xₙ²)
# Used in: attention scaling (÷√d), cosine similarity, weight decay
#
v = torch.tensor([3.0, 4.0])
l2 = torch.norm(v)                      # = 5.0
cosine = torch.dot(a, b) / (torch.norm(a) * torch.norm(b))

print(f"v = [3, 4]")
print(f"||v||₂ = √(3²+4²) = {l2:.1f}  (Pythagorean theorem)")
print(f"cosine(a, b) = {cosine:.4f}  (1.0 = same direction)")

print()
print("=" * 60)
print("EIGENVALUES (concept only — rarely computed manually in DL)")
print("=" * 60)
print("""
Eigenvalue equation:  A·v = λ·v
  v = eigenvector: a direction the matrix doesn't rotate, only scales
  λ = eigenvalue:  how much it scales in that direction

WHY IT MATTERS:
  PCA (Principal Component Analysis) finds eigenvectors of the covariance matrix.
  Eigenvectors = the directions of maximum variance in your data.
  → Used in dimensionality reduction (see 16_classical_ml/03_pca.py)

  Gradient explosion/vanishing relates to eigenvalues of weight matrices:
  λ > 1 → gradients grow each layer → explosion
  λ < 1 → gradients shrink each layer → vanishing
  → Fixed by normalisation layers and careful initialisation
""")
M = torch.tensor([[2.0, 0.0], [0.0, 3.0]])   # diagonal → eigenvalues obvious
eigenvalues, eigenvectors = torch.linalg.eig(M)
print(f"Diagonal matrix eigenvalues: {eigenvalues.real}")
print("(For a diagonal matrix the eigenvalues are just the diagonal entries)")

print()
print("=" * 60)
print("BROADCASTING — operating on different-shaped tensors")
print("=" * 60)
#
# Broadcasting: PyTorch automatically expands smaller tensors to match larger ones.
# RULE: align shapes from the right; a size-1 dimension stretches to match.
#
# This is everywhere: adding a bias, applying layer norm, CFG in diffusion.

x     = torch.randn(2, 3, 4)       # (batch=2, seq=3, dim=4)
bias  = torch.randn(4)              # (4,) — will broadcast to (2, 3, 4)
out   = x + bias                   # adds the same bias to every (batch, seq) position
print(f"x shape:    {x.shape}")
print(f"bias shape: {bias.shape}  → broadcasts to {x.shape}")
print(f"x + bias:   {out.shape}")

mean  = x.mean(dim=-1, keepdim=True)   # (2, 3, 1)
out2  = x - mean                       # (2, 3, 4) - (2, 3, 1) → (2, 3, 4)
print(f"mean shape (keepdim=True): {mean.shape}  → broadcasts correctly for subtraction")

# =============================================================================
# EXPECTED OUTPUT:
# Scalar: 3.14   shape=torch.Size([])
# A @ B:  [[19, 22], [43, 50]]
# ||v||₂ = 5.0
# Diagonal matrix eigenvalues: [2., 3.]
# =============================================================================
