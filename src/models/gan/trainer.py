import os
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import matplotlib.pyplot as plt
from .model import Generator, Discriminator
from pytorch_fid import fid_score

def train_gan(dataset_name="mnist", data_dir="./.data", device="cpu",
              latent_dim=100, epochs=20, batch_size=64, save_path="./weights/gan_mnist"):

    os.makedirs(save_path, exist_ok=True)
    fid_real_dir = os.path.join(save_path, "fid_real")
    fid_fake_dir = os.path.join(save_path, "fid_fake")
    os.makedirs(fid_real_dir, exist_ok=True)
    os.makedirs(fid_fake_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for idx, (img, _) in enumerate(DataLoader(dataset, batch_size=1)):
        save_image((img + 1) / 2, os.path.join(fid_real_dir, f"{idx:05d}.png"))
        if idx >= 999: break

    G = Generator(latent_dim=latent_dim, img_channels=1).to(device)
    D = Discriminator(img_channels=1).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(16, latent_dim, 1, 1, device=device)

    g_losses, d_losses = [], []

    for epoch in range(epochs):
        G.train()
        D.train()
        pbar = tqdm(loader, desc=f"Epoch [{epoch+1}/{epochs}]")

        total_g_loss, total_d_loss = 0.0, 0.0

        for real_imgs, _ in pbar:
            real_imgs = real_imgs.to(device)
            b_size = real_imgs.size(0)

            real_labels = torch.ones(b_size, device=device)
            fake_labels = torch.zeros(b_size, device=device)

            z = torch.randn(b_size, latent_dim, 1, 1, device=device)
            fake_imgs = G(z).detach()

            d_real = D(real_imgs)
            d_fake = D(fake_imgs)

            d_loss = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)
            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            z = torch.randn(b_size, latent_dim, 1, 1, device=device)
            fake_imgs = G(z)
            g_loss = criterion(D(fake_imgs), real_labels)
            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
            pbar.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item())

        g_losses.append(total_g_loss / len(loader))
        d_losses.append(total_d_loss / len(loader))

        G.eval()
        with torch.no_grad():
            gen_imgs = G(fixed_noise).detach().cpu()
            gen_imgs = (gen_imgs + 1) / 2
            grid = make_grid(gen_imgs, nrow=4)
            save_image(grid, os.path.join(save_path, f"samples_epoch_{epoch+1:03d}.png"))

    plt.figure(figsize=(10, 4))
    plt.plot(g_losses, label="Generator Loss")
    plt.plot(d_losses, label="Discriminator Loss")
    plt.title("GAN Training Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "loss_curve.png"))
    plt.close()

    G.eval()
    with torch.no_grad():
        for i in range(1000):
            z = torch.randn(1, latent_dim, 1, 1, device=device)
            fake_img = G(z).detach().cpu()
            fake_img = (fake_img + 1) / 2
            save_image(fake_img, os.path.join(fid_fake_dir, f"{i:05d}.png"))

    fid = fid_score.calculate_fid_given_paths([fid_real_dir, fid_fake_dir], batch_size, device, dims=2048)
    print(f"✅ FID Score: {fid:.2f}")

    torch.save(G.state_dict(), os.path.join(save_path, "generator.pt"))
    print(f"✅ Trained Generator saved to {save_path}/generator.pt")
