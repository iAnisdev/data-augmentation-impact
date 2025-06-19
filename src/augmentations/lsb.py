from .base import Augmentation
import numpy as np
from PIL import Image

class LSBAugmentation(Augmentation):
    def __init__(self, secret_byte=0b10101010):
        super().__init__('lsb')
        self.secret_byte = secret_byte

    def __call__(self, img):
        arr = np.array(img)
        arr[:, :, 0] = (arr[:, :, 0] & ~1) | (self.secret_byte & 1)
        return Image.fromarray(arr)
