# =============================================================================
# 02 — ADVERSARIAL TRAINING: THE GAN GAME
# =============================================================================
#
# LIKE YOU ARE 5:
# The forger and the detective play a game.
# Every round the detective gets better. Every round the forger gets better.
# After thousands of rounds, neither can win easily.
# That is Nash Equilibrium — neither player can improve by changing their strategy.
#
# PROBLEMS with GAN training:
#
# 1. MODE COLLAPSE: the generator finds a few fakes that always fool the
#    discriminator and only produces those. No diversity.
#    (The forger finds ONE style that works and never tries anything else.)
#
# 2. TRAINING INSTABILITY: loss oscillates, doesn't converge.
#    The discriminator becomes too strong -> gradient vanishes -> G can't learn.
#
# FIXES:
# - Wasserstein GAN (WGAN): better mathematical loss function.
# - Label smoothing: real labels = 0.9 (not 1.0) — prevents D from being too confident.
# - Gradient penalty: prevents D from having extreme gradients.
# - Spectral normalisation: normalises D's weights to stabilise training.
#
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim

print("=" * 60)
print("LABEL SMOOTHING: DON'T LET D BE TOO CONFIDENT")
print("=" * 60)
#
# If D correctly classifies everything with 99.9% confidence,
# the gradient flowing to G is near zero (G can't learn where to improve).
# Label smoothing caps D's confidence: real labels = 0.9 instead of 1.0.

smooth_real = 0.9    # instead of 1.0
smooth_fake = 0.0    # (can also use 0.1 for two-sided smoothing)
print(f"Real image labels:  {smooth_real}  (instead of 1.0 — prevents overconfidence)")
print(f"Fake image labels:  {smooth_fake}")

print()
print("=" * 60)
print("DCGAN: CONVOLUTIONAL GAN FOR IMAGES")
print("=" * 60)
#
# DCGAN (Deep Convolutional GAN, 2016) replaced FC layers with:
# Generator: ConvTranspose2d (upsampling)
# Discriminator: Conv2d (downsampling)
# This improved image quality significantly.

class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        # z (100,) -> (512, 4, 4) -> (256, 8, 8) -> (128, 16, 16) -> (3, 32, 32)
        self.net = nn.Sequential(
            # Input: (B, 100, 1, 1)
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),   # -> 512×4×4
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),          # -> 256×8×8
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),          # -> 128×16×16
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128,   3, 4, 2, 1, bias=False),          # -> 3×32×32
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z.view(-1, 100, 1, 1))

class DCGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Input: (B, 3, 32, 32)
            nn.Conv2d(3, 128, 4, 2, 1, bias=False),  # -> 128×16×16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), # -> 256×8×8
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), # -> 512×4×4
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),   # -> 1×1×1
            nn.Sigmoid(),
        )

    def forward(self, x): return self.net(x).view(-1, 1)

G = DCGANGenerator(latent_dim=100)
D = DCGANDiscriminator()

z    = torch.randn(4, 100)
fake = G(z)
pred = D(fake)
print(f"Generator output: {fake.shape}  (4 RGB 32×32 images)")
print(f"Discriminator:    {pred.shape}  (probability each is real)")

print()
print("=" * 60)
print("TRAINING BALANCE")
print("=" * 60)
#
# The key numbers to watch during GAN training:
# - D loss ≈ 0.69  (ln(2) ≈ 0.693): discriminator at chance (ideal)
# - G loss decreasing: generator getting better at fooling D
# - D accuracy ≈ 50%: D can't tell real from fake (ideal)
#
# If D loss goes to 0: D is perfect, G gets zero gradients -> mode collapse risk
# If G loss goes to 0: D always fooled -> D collapsed (not useful)

criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

# Simulate 5 training steps
real_images = torch.randn(4, 3, 32, 32)
print(f"{'Step':>4}  {'D_loss':>8}  {'G_loss':>8}  {'D_acc_real':>12}  {'D_acc_fake':>12}")
print("-" * 50)
for step in range(5):
    # Train D
    opt_D.zero_grad()
    real_preds = D(real_images)
    fake_imgs  = G(torch.randn(4, 100)).detach()
    fake_preds = D(fake_imgs)
    loss_D = (criterion(real_preds, torch.full((4,1), smooth_real)) +
              criterion(fake_preds, torch.zeros(4,1))) / 2
    loss_D.backward(); opt_D.step()

    # Train G
    opt_G.zero_grad()
    fake_imgs  = G(torch.randn(4, 100))
    fake_preds = D(fake_imgs)
    loss_G = criterion(fake_preds, torch.ones(4,1))
    loss_G.backward(); opt_G.step()

    acc_real = (D(real_images) > 0.5).float().mean().item()
    acc_fake = (D(G(torch.randn(4,100)).detach()) < 0.5).float().mean().item()
    print(f"{step+1:>4}  {loss_D.item():>8.4f}  {loss_G.item():>8.4f}  {acc_real:>12.2f}  {acc_fake:>12.2f}")

# =============================================================================
# EXPECTED OUTPUT (untrained so values are volatile):
# Generator output: torch.Size([4, 3, 32, 32])
# D_loss ≈ 0.69, G_loss ≈ 0.69  (at the start — chance performance)
# =============================================================================
