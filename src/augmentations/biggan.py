import random
import torch
import torch.nn.functional as F
from pytorch_pretrained_biggan import BigGAN, one_hot_from_int, truncated_noise_sample
from utils.mappings import DATASET_TO_IMAGENET
from models.gan.model import Generator
import os
from torchvision.utils import make_grid

class BigGANAugmentation:
    def __init__(self, dataset_name="cifar10", truncation=0.4, device="cpu"):
        self.device = device
        self.truncation = truncation
        self.dataset_name = dataset_name.lower()

        if self.dataset_name == "mnist":
            self.model = self._load_mnist_gan()
        else:
            self.model = BigGAN.from_pretrained("biggan-deep-256").to(device)
            self.model.eval()
            self.class_ids = self._get_class_ids(dataset_name)

    def _get_class_ids(self, dataset_name):
        mapping = DATASET_TO_IMAGENET.get(dataset_name.lower())
        if mapping == "native":
            return list(range(200))  # Tiny ImageNet range
        return list(mapping.values())

    def _load_mnist_gan(self):
        gen = Generator(latent_dim=100, img_channels=1).to(self.device)
        path = "./weights/gan_mnist/generator.pt"
        if not os.path.exists(path):
            raise FileNotFoundError("Trained MNIST GAN generator not found.")
        gen.load_state_dict(torch.load(path, map_location=self.device))
        gen.eval()
        return gen

    def __call__(self, batch_size=1):
        if self.dataset_name == "mnist":
            noise = torch.randn(batch_size, 100, 1, 1, device=self.device)
            with torch.no_grad():
                fake_imgs = self.model(noise)
            # Upscale to 64x64 and convert to 3 channels for consistency
            fake_imgs = F.interpolate(fake_imgs, size=(64, 64), mode="bilinear", align_corners=False)
            fake_imgs = fake_imgs.expand(-1, 3, -1, -1)  # [B, 1, H, W] -> [B, 3, H, W]
            normalized = (fake_imgs - 0.5) / 0.5
            return normalized.cpu()

        selected_ids = [random.choice(self.class_ids) for _ in range(batch_size)]
        noise_vector = truncated_noise_sample(truncation=self.truncation, batch_size=batch_size)
        class_vector = one_hot_from_int(selected_ids, batch_size=batch_size)

        noise_vector = torch.from_numpy(noise_vector).to(self.device)
        class_vector = torch.from_numpy(class_vector).to(self.device)

        with torch.no_grad():
            output = self.model(noise_vector, class_vector, truncation=self.truncation)

        output = (output + 1) / 2
        resized = F.interpolate(output, size=(64, 64), mode="bilinear", align_corners=False)
        normalized = (resized - 0.5) / 0.5
        return normalized.cpu()