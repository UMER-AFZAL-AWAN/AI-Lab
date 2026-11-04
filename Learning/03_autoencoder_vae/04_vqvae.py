# =============================================================================
# 04 — VQ-VAE: DISCRETE LATENT SPACE (codebook)
# =============================================================================
#
# LIKE YOU ARE 5:
# Normal VAE: the latent code is a continuous number, like 3.47 or -1.22.
# VQ-VAE: the latent code is a whole number INDEX into a dictionary.
# Like emojis: instead of describing a face in words, you pick emoji #42.
#
# The dictionary is called the CODEBOOK. It has N entries (e.g. 512).
# The encoder outputs a continuous vector.
# Then it looks up the NEAREST entry in the codebook.
# That nearest entry's INDEX is the compressed code.
#
# WHY IT MATTERS:
# - Discrete codes can be used with language models (predict the next token!).
# - Very stable training compared to VAE.
# - Used in DALL-E (original), AudioCodec, VideoCodec.
# - The "token" in image generation = a VQ-VAE codebook index.
#
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("VECTOR QUANTISATION: FIND THE NEAREST CODEBOOK ENTRY")
print("=" * 60)

class VectorQuantiser(nn.Module):
    """
    Quantises a continuous latent vector to the nearest codebook entry.
    Returns the quantised vector and the codebook indices.
    """
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        # The codebook: num_embeddings entries, each embedding_dim long
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim

    def forward(self, z):
        # z: (B, C, H, W) where C = embedding_dim
        B, C, H, W = z.shape
        # Flatten spatial dims: (B*H*W, C)
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, C)

        # Compute distances from each z to each codebook entry
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 * z @ e.T
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)       # (N, 1)
            + self.codebook.weight.pow(2).sum(dim=1)     # (K,)
            - 2 * z_flat @ self.codebook.weight.T        # (N, K)
        )  # shape: (N, K)  — distance from each z to each codebook entry

        # Pick the nearest codebook entry for each z
        indices = distances.argmin(dim=1)  # (N,) — nearest codebook index
        # Look up the actual codebook vectors
        z_q = self.codebook(indices)       # (N, C)
        # Reshape back
        z_q = z_q.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # Straight-through estimator: gradient passes through quantisation
        # During forward: use quantised z_q
        # During backward: treat as if z_q = z (gradient flows to encoder)
        z_q_st = z + (z_q - z).detach()

        indices = indices.view(B, H, W)
        return z_q_st, indices

# Demo
vq = VectorQuantiser(num_embeddings=512, embedding_dim=4)
z  = torch.randn(2, 4, 8, 8)  # (batch=2, channels=4, H=8, W=8)
z_q, indices = vq(z)

print(f"Continuous latent z:    {z.shape}")
print(f"Quantised latent z_q:   {z_q.shape}  (same shape, but snapped to codebook)")
print(f"Codebook indices:       {indices.shape}  (each spatial cell has an index)")
print(f"Index range:            0 to {indices.max().item()} (out of 512 entries)")
print()
print("Each spatial cell is now described by ONE integer (the codebook index).")
print("The 8x8 grid of indices IS the compressed representation.")
print(f"Compression: {4*8*8} floats -> {8*8} integers")

print()
print("=" * 60)
print("A SIMPLE VQ-VAE")
print("=" * 60)

class VQVAE(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=4):
        super().__init__()
        # Encoder: 1x28x28 -> 4x7x7
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),  # 1x28x28 -> 16x14x14
            nn.ReLU(),
            nn.Conv2d(16, embedding_dim, kernel_size=4, stride=2, padding=1),  # -> 4x7x7
        )
        self.vq  = VectorQuantiser(num_embeddings, embedding_dim)
        # Decoder: 4x7x7 -> 1x28x28
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z     = self.enc(x)
        z_q, indices = self.vq(z)
        recon = self.dec(z_q)
        return recon, z, z_q, indices

vqvae  = VQVAE()
x      = torch.rand(4, 1, 28, 28)
recon, z, z_q, indices = vqvae(x)

print(f"Input:    {x.shape}")
print(f"Latent z: {z.shape}   (continuous)")
print(f"Codes:    {indices.shape}  (discrete integers)")
print(f"Recon:    {recon.shape}")

# VQ-VAE loss has 3 components
recon_loss     = F.mse_loss(recon, x)
codebook_loss  = F.mse_loss(z_q.detach(), z)      # push codebook toward encoder output
commitment_loss= F.mse_loss(z.detach(),  z_q)     # push encoder toward nearest codebook entry

total_loss = recon_loss + codebook_loss + 0.25 * commitment_loss
print(f"\nReconstruction loss: {recon_loss.item():.4f}")
print(f"Codebook loss:       {codebook_loss.item():.4f}")
print(f"Commitment loss:     {commitment_loss.item():.4f}")
print(f"Total loss:          {total_loss.item():.4f}")

print()
print("After training, the indices ARE the compressed image.")
print("You could train a GPT-like model to predict these indices autoregressively.")
print("That is exactly what DALL-E (original 2021) did.")

# =============================================================================
# EXPECTED OUTPUT:
# Continuous latent z:    torch.Size([2, 4, 8, 8])
# Quantised latent z_q:   torch.Size([2, 4, 8, 8])
# Codebook indices:       torch.Size([2, 8, 8])
# =============================================================================
