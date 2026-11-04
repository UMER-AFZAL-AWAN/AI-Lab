# =============================================================================
# 03 — VARIATIONAL AUTOENCODER (VAE)
# =============================================================================
#
# LIKE YOU ARE 5:
# Normal autoencoder problem: it memorises specific examples.
# Ask it to "generate a new cat" by picking a random point in latent space
# and you get garbage — because random points between memorised cats
# don't decode to anything sensible.
#
# VAE fix: instead of encoding to ONE point, encode to a BLOB (a Gaussian).
# The blob has a centre (mean) and a size (standard deviation).
# During training you sample a random point FROM the blob.
# Because the blob is continuous, nearby points decode to similar things.
# Now you CAN sample random points and get sensible outputs.
#
# TECHNICAL: the encoder outputs mu (mean) and log_var (log of variance).
# A sample z = mu + eps * std  where eps is random noise.
# This "reparameterisation trick" lets gradients flow through the random step.
#
# C# ANALOGY:
# Normal autoencoder: an encoder that returns a Dictionary<string, float>.
# VAE encoder: an encoder that returns a GaussianDistribution with mean and std.
#              You sample from it, not just take the mean.
#
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("THE REPARAMETERISATION TRICK")
print("=" * 60)
#
# Problem: sampling is NOT differentiable. You can't backprop through random.
# Fix: instead of z = sample(Normal(mu, std))
#      write z = mu + eps * std  where eps ~ Normal(0, 1)
# Now z depends on mu and std in a differentiable way.
# The randomness comes from eps, which is external — not backprop'd.

def reparameterise(mu, log_var):
    """
    mu:      (B, latent_dim) — the mean of the distribution
    log_var: (B, latent_dim) — log of the variance (using log for numerical stability)
    """
    std = torch.exp(0.5 * log_var)   # std = sqrt(var) = exp(0.5 * log_var)
    eps = torch.randn_like(std)       # random noise, same shape as std
    return mu + eps * std             # the reparameterised sample

# Example
mu      = torch.tensor([[1.0, 2.0, 3.0]])
log_var = torch.tensor([[0.0, 0.0, 0.0]])   # log(1) = 0 -> std = 1

z1 = reparameterise(mu, log_var)
z2 = reparameterise(mu, log_var)
print(f"mu:  {mu}")
print(f"z1:  {z1}  (random sample near mu)")
print(f"z2:  {z2}  (different sample, same mu)")
print("Different every call, but centred on mu.")

print()
print("=" * 60)
print("A SIMPLE VAE")
print("=" * 60)

class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=16):
        super().__init__()
        # Encoder: input -> (mu, log_var)
        self.enc_shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
        )
        self.enc_mu      = nn.Linear(256, latent_dim)
        self.enc_log_var = nn.Linear(256, latent_dim)

        # Decoder: z -> reconstructed input
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h      = self.enc_shared(x.flatten(1))
        mu     = self.enc_mu(h)
        log_var= self.enc_log_var(h)
        return mu, log_var

    def decode(self, z):
        return self.dec(z).view(-1, 1, 28, 28)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z    = reparameterise(mu, log_var)
        recon= self.decode(z)
        return recon, mu, log_var

vae = VAE(input_dim=784, latent_dim=16)
x   = torch.rand(8, 1, 28, 28)
recon, mu, log_var = vae(x)

print(f"Input:   {x.shape}")
print(f"mu:      {mu.shape}   (mean of latent distribution)")
print(f"log_var: {log_var.shape}   (log variance of latent distribution)")
print(f"Recon:   {recon.shape}")

print()
print("=" * 60)
print("THE VAE LOSS = RECONSTRUCTION LOSS + KL DIVERGENCE")
print("=" * 60)
#
# The loss has TWO parts:
#
# 1. RECONSTRUCTION LOSS: how well does the decoder reconstruct the input?
#    (same as normal autoencoder — MSE or binary cross-entropy)
#
# 2. KL DIVERGENCE LOSS: how close is the latent distribution to Normal(0,1)?
#    "Don't let the blobs drift too far from the origin or become too small."
#    If KL = 0 the latent codes are all nice Normal blobs around the origin.
#    That means you can sample from Normal(0,1) and always get valid outputs.
#    Formula: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
#
# Total loss = recon_loss + beta * kl_loss
# beta controls how "smooth" the latent space is vs reconstruction quality.

def vae_loss(recon, x, mu, log_var, beta=1.0):
    # Reconstruction: how close is recon to x?
    recon_loss = F.binary_cross_entropy(recon, x, reduction='sum')
    # KL: how close is q(z|x) to N(0,1)?
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

total, recon_l, kl_l = vae_loss(recon, x, mu, log_var)
print(f"Reconstruction loss: {recon_l.item():.1f}")
print(f"KL divergence loss:  {kl_l.item():.1f}")
print(f"Total loss:          {total.item():.1f}")

print()
print("=" * 60)
print("GENERATING NEW SAMPLES (the VAE superpower)")
print("=" * 60)
#
# Now that the latent space is smooth (thanks to KL loss),
# we can sample from Normal(0,1) and decode to get new images.

z_new = torch.randn(4, 16)   # 4 random latent vectors from Normal(0,1)
with torch.no_grad():
    generated = vae.decode(z_new)
print(f"Generated {generated.shape[0]} new images from random noise")
print(f"Each: {generated.shape[1:]}  — no input needed, pure generation!")
print()
print("This is the GENERATION capability that normal autoencoders lack.")
print("The VAE's smooth latent space makes random sampling produce valid outputs.")

# =============================================================================
# EXPECTED OUTPUT:
# z1:  tensor([[~1, ~2, ~3]])  (close to mu, with small random noise)
# mu:      torch.Size([8, 16])
# Generated 4 new images from random noise
# =============================================================================
