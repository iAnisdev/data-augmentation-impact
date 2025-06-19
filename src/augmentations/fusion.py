from .base import Augmentation
from torchvision import transforms
from PIL import Image

class FusionAugmentation(Augmentation):
    def __init__(self):
        super().__init__('fusion')
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ColorJitter(brightness=0.3, contrast=0.3)
        ])

    def __call__(self, img1, img2):
        img1_aug = self.transform(img1)
        img2_aug = self.transform(img2)
        alpha = 0.6
        return Image.blend(img1_aug, img2_aug, alpha)
