from .base import Augmentation
from models.vqvae.model import VQVAE 
import torch
import torchvision.transforms as T
from PIL import Image
import os

class VQVAEAugmentation(Augmentation):
    def __init__(self, model_path: str, device: str = "cpu"):
        super().__init__("vqvae")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"VQ-VAE model not found at: {model_path}")

        self.device = device
        self.model = VQVAE().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor()
        ])
        self.to_pil = T.ToPILImage()

    def __call__(self, img: Image.Image) -> Image.Image:
        x = self.transform(img).unsqueeze(0).to(self.device)  # [1, C, H, W]
        with torch.no_grad():
            z, _ = self.model.encode(x)
            x_recon = self.model.decode(z)
        return self.to_pil(x_recon.squeeze(0).clamp(0, 1).cpu())
