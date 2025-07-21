import os
import torch
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torchvision.utils import save_image
from pytorch_fid import fid_score

from .model import VQVAE

import logging

logger = logging.getLogger("AugmentationPipeline")

def get_dataset(name, data_dir, transform):
    if name == "cifar10":
        return datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    elif name == "mnist":
        return datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    elif name == "imagenet":
        return datasets.ImageFolder(os.path.join(data_dir, "tiny-imagenet-200", "train"), transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

def train_vqvae(dataset_name="cifar10", data_dir="./data", device="cpu", epochs=5, train_size=0.9, test_size=0.1):
    transform = [transforms.Resize((64, 64))]
    if dataset_name == "mnist":
        transform.append(transforms.Grayscale(num_output_channels=3))
    transform.append(transforms.ToTensor())
    transform = transforms.Compose(transform)

    dataset = get_dataset(dataset_name, data_dir, transform)

    n = len(dataset)
    train_len = int(n * train_size)
    test_len = n - train_len
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

    loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = VQVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    losses_per_epoch = []

    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0
        for x, _ in tqdm(loader, desc=f"VQ-VAE Epoch {epoch+1}/{epochs}"):
            x = x.to(device)
            recon, loss = model(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / n_batches
        losses_per_epoch.append(avg_loss)
        logger.info(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

    # Save model and loss curve
    model_dir = f"./weights/{dataset_name}_vqvae"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "generator.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

    # Save loss curve plot
    plt.figure()
    plt.plot(losses_per_epoch, marker="o", label="Loss")
    plt.title(f"VQ-VAE Loss Curve: {dataset_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    loss_plot_path = os.path.join(model_dir, "loss_curve.png")
    plt.savefig(loss_plot_path)
    plt.close()
    logger.info(f"Loss curve saved to {loss_plot_path}")

    # ========== FID PREP ========== #
    fid_real_dir = os.path.join(model_dir, "fid_real")
    fid_fake_dir = os.path.join(model_dir, "fid_fake")
    os.makedirs(fid_real_dir, exist_ok=True)
    os.makedirs(fid_fake_dir, exist_ok=True)

    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    # Collect 1000 real images
    for idx, (img, _) in enumerate(DataLoader(train_dataset, batch_size=1)):
        save_image(img.squeeze(0), os.path.join(fid_real_dir, f"{idx:05d}.png"))
        if idx >= 999: break

    # Collect 1000 reconstructed (fake) images
    model.eval()
    img_count = 0
    with torch.no_grad():
        for x, _ in DataLoader(train_dataset, batch_size=1):
            x = x.to(device)
            z, _ = model.encode(x)
            recon = model.decode(z)
            recon = recon.clamp(0, 1).cpu()
            save_image(recon.squeeze(0), os.path.join(fid_fake_dir, f"{img_count:05d}.png"))
            img_count += 1
            if img_count >= 1000: break

    # Compute FID
    fid_value = fid_score.calculate_fid_given_paths(
        [fid_real_dir, fid_fake_dir],
        batch_size=64,
        device=device,
        dims=2048
    )

    fid_path = os.path.join(model_dir, "fid_score.json")
    with open(fid_path, "w") as f:
        json.dump({"fid": fid_value}, f, indent=2)

    logger.info(f"✅ FID Score for {dataset_name} VQ-VAE: {fid_value:.2f}")
    logger.info(f"FID saved to {fid_path}")

    return model
