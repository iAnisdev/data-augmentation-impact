from .base import Augmentation
from torchvision import transforms
from PIL import Image

class MiAMixAugmentation(Augmentation):
    def __init__(self):
        super().__init__('miamix')
        self.transform = transforms.Compose([
            transforms.RandomAffine(20),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)
        ])

    def __call__(self, img1, img2):
        img1_aug = self.transform(img1)
        img2_aug = self.transform(img2)
        alpha = 0.5
        return Image.blend(img1_aug, img2_aug, alpha)
