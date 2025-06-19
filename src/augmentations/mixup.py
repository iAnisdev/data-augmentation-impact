from .base import Augmentation
from torchvision import transforms
from PIL import Image
import numpy as np

class MixupAugmentation(Augmentation):
    def __init__(self, alpha=0.4):
        super().__init__('mixup')
        self.alpha = alpha
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __call__(self, img1, img2):
        arr1 = np.array(img1).astype(np.float32)
        arr2 = np.array(img2).astype(np.float32)

        lam = np.random.beta(self.alpha, self.alpha)
        mixed = lam * arr1 + (1 - lam) * arr2
        mixed = np.clip(mixed, 0, 255).astype(np.uint8)
        return Image.fromarray(mixed)
