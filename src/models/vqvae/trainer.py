from .model import VQVAE
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch, os

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

    return model
