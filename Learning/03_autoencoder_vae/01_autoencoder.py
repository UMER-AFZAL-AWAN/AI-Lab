# =============================================================================
# 01 — AUTOENCODER: COMPRESS AND RECONSTRUCT
# =============================================================================
#
# LIKE YOU ARE 5:
# Imagine playing "telephone". You tell a friend a long story.
# They write it in 10 bullet points on a card (ENCODING = compression).
# You send the card to another person.
# That person reads the 10 bullets and tries to retell the whole story (DECODING).
#
# An autoencoder is trained to:
#   1. COMPRESS an input into a small "code" (the latent vector)
#   2. RECONSTRUCT the original from the code
#
# After training, the model learns the best 10 bullet points for typical stories.
# The code (latent vector) is a compressed representation of the input.
#
# WHY IT MATTERS:
# Stable Diffusion and AnimateDiff do NOT work in pixel space.
# They work in LATENT space — the compressed representation.
# The autoencoder turns pixels into a 4-channel 64x64 latent (instead of
# 3-channel 512x512 pixels). That is 48x fewer numbers. Much faster.
#
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("A SIMPLE AUTOENCODER")
print("=" * 60)

class Encoder(nn.Module):
    """Compress a 28x28 image to a 32-dimensional code."""
    def __init__(self):
        super().__init__()
        # Flatten 28*28=784 pixels -> 128 -> 32
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 32)   # 32-dim latent code

    def forward(self, x):
        x = x.flatten(1)                # (B, 1, 28, 28) -> (B, 784)
        x = F.relu(self.fc1(x))
        return self.fc2(x)              # (B, 32) — the compressed code

class Decoder(nn.Module):
    """Reconstruct a 28x28 image from a 32-dimensional code."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 784)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = torch.sigmoid(self.fc2(z))  # sigmoid -> output in [0, 1]
        return z.view(-1, 1, 28, 28)   # reshape to image

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        code = self.encoder(x)
        recon = self.decoder(code)
        return recon, code

ae = Autoencoder()
print(f"Encoder parameters: {sum(p.numel() for p in ae.encoder.parameters()):,}")
print(f"Decoder parameters: {sum(p.numel() for p in ae.decoder.parameters()):,}")

# Simulate a batch of 8 greyscale images (28x28)
images = torch.rand(8, 1, 28, 28)
recon, code = ae(images)

print(f"\nInput:         {images.shape}  ({images.numel() // 8} values per image)")
print(f"Latent code:   {code.shape}    (32 values per image)")
print(f"Reconstruction:{recon.shape}")
print(f"Compression ratio: {784 / 32:.1f}x  (784 -> 32)")

print()
print("=" * 60)
print("RECONSTRUCTION LOSS")
print("=" * 60)
#
# The model is trained to minimise the difference between
# the original image and its reconstruction.
# Common loss: Mean Squared Error (MSE) — how far off each pixel is.

loss_fn = nn.MSELoss()
loss = loss_fn(recon, images)
print(f"Reconstruction loss (before training): {loss.item():.4f}")
print("(High at first — model is random. Goes lower as it trains.)")
print()

# After training, the latent code should capture structure.
# Example: images of same digit cluster near each other in latent space.

print("=" * 60)
print("CONVOLUTIONAL AUTOENCODER (better for images)")
print("=" * 60)
#
# Fully-connected autoencoders ignore spatial structure.
# A convolutional autoencoder uses Conv2d (encoder) and ConvTranspose2d (decoder).
# ConvTranspose2d is the "reverse convolution" — it UPSAMPLES.

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: 1x28x28 -> 16x7x7
        self.enc = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),   # 1x28x28 -> 8x14x14
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # 8x14x14 -> 16x7x7
            nn.ReLU(),
        )
        # Decoder: 16x7x7 -> 1x28x28
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> 8x14x14
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # -> 1x28x28
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.enc(x)        # encode
        return self.dec(z), z  # decode and return latent

cae = ConvAutoencoder()
x   = torch.rand(4, 1, 28, 28)
recon, z = cae(x)
print(f"Input:   {x.shape}")
print(f"Latent:  {z.shape}  ({z.numel() // 4} values per image)")
print(f"Recon:   {recon.shape}")

# =============================================================================
# EXPECTED OUTPUT:
# Latent code:   torch.Size([8, 32])
# Compression ratio: 24.5x  (784 -> 32)
# Latent:  torch.Size([4, 16, 7, 7])  (784 values per image)
# =============================================================================
