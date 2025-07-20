from .base import Augmentation
from models.vqvae import VQVAE 
import torch
import torchvision.transforms as T
from PIL import Image

class VQVAEAugmentation(Augmentation):
    def __init__(self, model_path, device="cpu"):
        super().__init__("vqvae")
        self.device = device
        self.model = VQVAE().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.to_tensor = T.ToTensor()
        self.to_pil = T.ToPILImage()
        self.resize = T.Resize((64, 64))

    def __call__(self, img: Image.Image) -> Image.Image:
        x = self.to_tensor(self.resize(img)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            z, _ = self.model.encode(x)
            x_recon = self.model.decode(z)
        return self.to_pil(x_recon.squeeze(0).clamp(0, 1).cpu())
