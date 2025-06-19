from .base import Augmentation
from torchvision import transforms

class TraditionalAugmentation(Augmentation):
    def __init__(self):
        super().__init__('traditional')
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
        ])

    def __call__(self, img):
        return self.transform(img)
