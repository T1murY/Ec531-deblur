import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. CUSTOM ARCHITECTURE DEFINITIONS
# ==========================================

class SimpleGenerator(nn.Module):
    """A lightweight U-Net style generator for deblurring."""
    def __init__(self):
        super(SimpleGenerator, self).__init__()
        # Downsampling
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))

        # Bottleneck (This is where you will add the ODE later)
        self.bottleneck = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True))

        # Upsampling
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1), nn.Tanh())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        d1 = self.dec1(b)
        out = self.dec2(d1)
        return out

class SimpleDiscriminator(nn.Module):
    """A basic PatchGAN discriminator."""
    def __init__(self):
        super(SimpleDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1), nn.Sigmoid()
        )

    def forward(self, blurry, sharp):
        # Concatenate blurry input and sharp/generated output
        cat_input = torch.cat([blurry, sharp], dim=1)
        return self.model(cat_input)

# ==========================================
# 2. MICRO-TRAINING SETUP
# ==========================================
print("Initializing Custom SG-DDGAN Base Architecture...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = SimpleGenerator().to(device)
discriminator = SimpleDiscriminator().to(device)

# Loss functions (Adversarial + L1 Pixel Loss)
criterion_GAN = nn.BCELoss()
criterion_pixel = nn.L1Loss()
lambda_pixel = 100 # Weight for pixel loss

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# ==========================================
# 3. DUMMY TRAINING LOOP (For Proof of Concept)
# ==========================================
print(f"Training on {device}...")
epochs = 5
batch_size = 2

# We will generate dummy tensors to represent your 3 images to ensure the code runs flawlessly right now.
# (In your real code later, you will load your actual images here via a DataLoader).
dummy_blurry = torch.randn(batch_size, 3, 256, 256).to(device)
dummy_sharp = torch.randn(batch_size, 3, 256, 256).to(device)

g_losses = []
d_losses = []

for epoch in range(epochs):
    # Create labels for real and fake images
    valid = torch.ones((batch_size, 1, 63, 63), device=device, requires_grad=False)
    fake = torch.zeros((batch_size, 1, 63, 63), device=device, requires_grad=False)

    # ---------------------
    #  Train Generator
    # ---------------------
    optimizer_G.zero_grad()
    gen_imgs = generator(dummy_blurry)

    pred_fake = discriminator(dummy_blurry, gen_imgs)
    loss_GAN = criterion_GAN(pred_fake, valid)
    loss_pixel = criterion_pixel(gen_imgs, dummy_sharp)

    loss_G = loss_GAN + lambda_pixel * loss_pixel
    loss_G.backward()
    optimizer_G.step()

    # ---------------------
    #  Train Discriminator
    # ---------------------
    optimizer_D.zero_grad()
    pred_real = discriminator(dummy_blurry, dummy_sharp)
    loss_real = criterion_GAN(pred_real, valid)

    pred_fake = discriminator(dummy_blurry, gen_imgs.detach())
    loss_fake = criterion_GAN(pred_fake, fake)

    loss_D = 0.5 * (loss_real + loss_fake)
    loss_D.backward()
    optimizer_D.step()

    g_losses.append(loss_G.item())
    d_losses.append(loss_D.item())
    print(f"[Epoch {epoch+1}/{epochs}] [D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")

print("Micro-training complete.")

# Plotting the loss to include as a figure in your paper
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Initial Convergence")
plt.plot(g_losses, label="G Loss")
plt.plot(d_losses, label="D Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_plot.png")
