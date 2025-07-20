from .base import Augmentation
import torchvision.transforms as T
from PIL import Image

class AutoAugmentAugmentation(Augmentation):
    def __init__(self, policy="imagenet"):
        super().__init__("autoaugment")
        self.resize = T.Resize((64, 64))
        self.to_tensor = T.ToTensor()
        self.to_pil = T.ToPILImage()

        policy_map = {
            "imagenet": T.AutoAugmentPolicy.IMAGENET,
            "cifar10": T.AutoAugmentPolicy.CIFAR10,
            "svhn": T.AutoAugmentPolicy.SVHN,
        }
        self.augment = T.AutoAugment(policy=policy_map.get(policy.lower(), T.AutoAugmentPolicy.IMAGENET))

    def __call__(self, img: Image.Image) -> Image.Image:
        img = self.resize(img)
        img = self.augment(img)
        return img
