# augmentations/biggan.py

import random
import torch
import torch.nn.functional as F
from pytorch_pretrained_biggan import BigGAN, one_hot_from_int, truncated_noise_sample
from utils.mappings import DATASET_TO_IMAGENET

class BigGANAugmentation:
    def __init__(self, dataset_name="cifar10", truncation=0.4, device="cpu"):
        if dataset_name.lower() == "mnist":
            raise ValueError("BigGAN augmentation is not supported for MNIST.")

        self.device = device
        self.truncation = truncation
        self.model = BigGAN.from_pretrained("biggan-deep-256").to(device)
        self.model.eval()
        self.class_ids = self._get_class_ids(dataset_name)

    def _get_class_ids(self, dataset_name):
        mapping = DATASET_TO_IMAGENET.get(dataset_name.lower())
        if mapping == "native":
            return list(range(200))  # Tiny ImageNet range
        return list(mapping.values())

    def __call__(self, batch_size=1):
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
