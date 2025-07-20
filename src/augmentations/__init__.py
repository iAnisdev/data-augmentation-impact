from .auto import AutoAugmentAugmentation
from .base import Augmentation
from .traditional import TraditionalAugmentation
from .miamix import MiAMixAugmentation
from .mixup import MixupAugmentation
from .lsb import LSBAugmentation
from .vqvae_augment import VQVAEAugmentation
from .fusion import FusionAugmentation
from .biggan import BigGANAugmentation

__all__ = [
    "Augmentation",
    "AutoAugmentAugmentation",
    "BigGANAugmentation",
    "TraditionalAugmentation",
    "MiAMixAugmentation",
    "MixupAugmentation",
    "LSBAugmentation",
    "VQVAEAugmentation",
    "FusionAugmentation",
]
