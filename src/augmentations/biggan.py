import os
import random
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from pytorch_pretrained_biggan import BigGAN, one_hot_from_int, truncated_noise_sample

from models.gan.model import Generator
from models.gan.trainer import train_gan
from utils.mappings import DATASET_TO_IMAGENET


class BigGANAugmentation:
    def __init__(self, dataset_name="cifar10", truncation=0.4, device="cpu"):
        self.dataset_name = dataset_name.lower()
        self.device = device
        self.truncation = truncation

        if self.dataset_name == "mnist":
            self.model = self._load_or_train_mnist_gan()
        else:
            self.model = BigGAN.from_pretrained("biggan-deep-256").to(self.device)
            self.model.eval()
            self.class_ids = self._get_class_ids(self.dataset_name)

    def _get_class_ids(self, dataset_name):
        mapping = DATASET_TO_IMAGENET.get(dataset_name.lower())
        if mapping == "native":
            return list(range(200))  # Tiny ImageNet
        return list(mapping.values())

    def _load_or_train_mnist_gan(self):
        weight_path = "./weights/gan_mnist/generator.pt"
        if not os.path.exists(weight_path):
            print("⚠️ Trained MNIST GAN generator not found. Training now...")
            train_gan(dataset_name="mnist", device=self.device)

        G = Generator(latent_dim=100, img_channels=1).to(self.device)
        G.load_state_dict(torch.load(weight_path, map_location=self.device))
        G.eval()
        return G

    def __call__(self, batch_size=1):
        if self.dataset_name == "mnist":
            z = torch.randn(batch_size, 100, 1, 1, device=self.device)
            with torch.no_grad():
                out = self.model(z)
                out = (out + 1) / 2  # scale to [0, 1]
                out = out.repeat(1, 3, 1, 1)  # 1 channel → 3 channels
                resized = F.interpolate(out, size=(64, 64), mode="bilinear", align_corners=False)
                normalized = (resized - 0.5) / 0.5
                return normalized.cpu()

        else:
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
