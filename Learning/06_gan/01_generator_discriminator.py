# =============================================================================
# 01 — GAN: GENERATOR VS DISCRIMINATOR
# =============================================================================
#
# LIKE YOU ARE 5:
# Imagine a FORGER who makes fake paintings and a DETECTIVE who catches fakes.
# The forger wants to fool the detective. The detective wants to catch the forger.
# They compete. The forger keeps getting better to fool the detective.
# The detective keeps getting better to catch the forger.
# Eventually the forger makes paintings that are indistinguishable from real.
#
# That is a GAN (Generative Adversarial Network):
# GENERATOR: makes fake data. Takes random noise -> makes an image.
# DISCRIMINATOR: decides if an image is real (from dataset) or fake (from generator).
#
# TRAINING:
# 1. Train discriminator: show it real images (label=1) and fake images (label=0).
#    It learns to tell them apart.
# 2. Train generator: generate fake images, show them to the discriminator.
#    The generator's loss = how often the discriminator says "fake".
#    The generator wants the discriminator to say "real" (be fooled).
#    So it updates to produce more convincing fakes.
# Repeat until the discriminator can't tell the difference (50/50 accuracy).
#
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim

print("=" * 60)
print("THE GENERATOR")
print("=" * 60)
#
# Input: random noise vector z ~ N(0,1) of size latent_dim
# Output: a fake image of size (C, H, W)
# Architecture: fully connected or convolutional, depending on complexity

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_size=28):
        super().__init__()
        self.img_size = img_size
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, img_size * img_size),
            nn.Tanh(),   # output in [-1, 1] (real images also normalised to this range)
        )

    def forward(self, z):
        out = self.net(z)
        return out.view(-1, 1, self.img_size, self.img_size)

latent_dim = 100
G = Generator(latent_dim=latent_dim, img_size=28)
z = torch.randn(8, latent_dim)   # 8 random noise vectors
fake_images = G(z)
print(f"Input noise:    {z.shape}  ({latent_dim}-dim per sample)")
print(f"Generated images: {fake_images.shape}  (8 fake 28x28 images)")
params_G = sum(p.numel() for p in G.parameters())
print(f"Generator parameters: {params_G:,}")

print()
print("=" * 60)
print("THE DISCRIMINATOR")
print("=" * 60)
#
# Input: an image (real or fake)
# Output: a single number (probability that the image is real)
# 1.0 = "definitely real", 0.0 = "definitely fake"

class Discriminator(nn.Module):
    def __init__(self, img_size=28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_size * img_size, 512),
            nn.LeakyReLU(0.2),   # LeakyReLU (not ReLU) — avoids dead neurons
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),        # output: probability of being real
        )

    def forward(self, img):
        flat = img.flatten(1)   # (B, 1, 28, 28) -> (B, 784)
        return self.net(flat)

D = Discriminator(img_size=28)
real_images = torch.randn(8, 1, 28, 28)   # batch of "real" images (simulated)
pred_real = D(real_images)
pred_fake = D(fake_images.detach())   # .detach() so gradients don't flow to G here
print(f"Real images:         {real_images.shape}")
print(f"D(real) scores:      {pred_real.T.data.round(decimals=3)}")  # should be near 1.0
print(f"D(fake) scores:      {pred_fake.T.data.round(decimals=3)}")  # should be near 0.0 after training

print()
print("=" * 60)
print("THE TRAINING LOOP (one step)")
print("=" * 60)

criterion = nn.BCELoss()   # binary cross-entropy: measures how wrong D and G are
opt_G     = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D     = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

real_labels = torch.ones(8, 1)    # label: "this is real"
fake_labels = torch.zeros(8, 1)   # label: "this is fake"

# --- Step 1: Train Discriminator ---
opt_D.zero_grad()
# Loss on real images: D should output 1
loss_D_real = criterion(D(real_images), real_labels)
# Loss on fake images: D should output 0
z_new = torch.randn(8, latent_dim)
fake  = G(z_new).detach()   # detach so G doesn't get D's gradients
loss_D_fake = criterion(D(fake), fake_labels)
loss_D = (loss_D_real + loss_D_fake) / 2
loss_D.backward()
opt_D.step()
print(f"Discriminator loss: {loss_D.item():.4f}  (smaller = D is more confident)")

# --- Step 2: Train Generator ---
opt_G.zero_grad()
# G wants D to say its fakes are real -> use real_labels as target
z_new = torch.randn(8, latent_dim)
fake  = G(z_new)
loss_G = criterion(D(fake), real_labels)   # G wants D to say "real"
loss_G.backward()
opt_G.step()
print(f"Generator loss:     {loss_G.item():.4f}  (smaller = G is fooling D better)")

print()
print("Mode collapse risk: G might find one fake that always fools D and produce ONLY that.")
print("Solution: diverse noise inputs, spectral normalisation, Wasserstein loss (WGAN).")

# =============================================================================
# EXPECTED OUTPUT:
# Input noise:    torch.Size([8, 100])
# Generated images: torch.Size([8, 1, 28, 28])
# Discriminator loss: ~0.69  (at start, D is roughly at chance)
# Generator loss:     ~0.69  (at start, G can't fool D yet)
# =============================================================================
