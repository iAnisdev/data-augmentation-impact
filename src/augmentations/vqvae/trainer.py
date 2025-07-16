from .model import VQVAE
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch, os

def get_dataset(name, data_dir, transform):
    if name == "cifar10":
        return datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    elif name == "mnist":
        return datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    elif name == "imagenet":
        return datasets.ImageFolder(os.path.join(data_dir, "tiny-imagenet-200", "train"), transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

def train_vqvae(dataset_name="cifar10", data_dir="./data", device="cpu"):
    transform = [transforms.Resize((64, 64))]
    if dataset_name == "mnist":
        transform.append(transforms.Grayscale(num_output_channels=3))
    transform.append(transforms.ToTensor())
    transform = transforms.Compose(transform)

    dataset = get_dataset(dataset_name, data_dir, transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = VQVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    for epoch in range(30):
        for x, _ in loader:
            x = x.to(device)
            recon, loss = model(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

    save_path = f"./weights/vqvae_{dataset_name}.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return model
