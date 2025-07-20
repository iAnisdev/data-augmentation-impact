import os
import logging
import torch
from models.vqvae.model import VQVAE
from models.vqvae.trainer import train_vqvae
from torchvision import transforms, datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger("AugmentationPipeline")

def get_vqvae_model(dataset_name, weights_dir="./weights", device="cpu", retrain=False):
    datasets = ["cifar10", "mnist", "imagenet"] if dataset_name == "all" else [dataset_name]
    
    models = {}
    for ds in datasets:
        os.makedirs(weights_dir, exist_ok=True)
        model_path = os.path.join(weights_dir, f"vqvae_{ds}.pt")
        if not os.path.exists(model_path) or retrain:
            logger.info(f"Training VQ-VAE model for {ds}...")
            train_vqvae(dataset_name=ds, data_dir="./.data", device=device, epochs=5)
        else:
            logger.info(f"Using existing VQ-VAE model: {model_path}")
        
        model = VQVAE().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models[ds] = model
    
    return models if dataset_name == "all" else models[dataset_name]

def apply_vqvae(dataset_name, model, device="cpu", out_root="./processed", batch_size=64):
    datasets_list = ["cifar10", "mnist", "imagenet"] if dataset_name == "all" else [dataset_name]
    
    for ds in datasets_list:
        current_model = model[ds] if isinstance(model, dict) else model
        
        transform = [transforms.Resize((64, 64))]
        if ds == "mnist":
            transform.append(transforms.Grayscale(num_output_channels=3))
        transform.append(transforms.ToTensor())
        transform = transforms.Compose(transform)

        dataset_cls = {
            "cifar10": datasets.CIFAR10,
            "mnist": datasets.MNIST,
            "imagenet": lambda root, transform: datasets.ImageFolder(os.path.join(root, "tiny-imagenet-200", "train"), transform=transform)
        }[ds]

        dataset = dataset_cls(root="./.data", train=True, download=True, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        out_dir = os.path.join(out_root, ds, "train", "vqvae")
        os.makedirs(out_dir, exist_ok=True)

        to_pil = transforms.ToPILImage()
        img_count = 0

        for batch in tqdm(loader, desc=f"Augmenting {ds} with VQ-VAE"):
            imgs, _ = batch
            imgs = imgs.to(device)
            with torch.no_grad():
                z, _ = current_model.encode(imgs)
                recons = current_model.decode(z)

            for img in recons:
                img = to_pil(img.clamp(0, 1).cpu())
                save_image(transforms.ToTensor()(img), os.path.join(out_dir, f"{img_count:05d}.png"))
                img_count += 1

        logger.info(f"Saved {img_count} VQ-VAE images to {out_dir}")
