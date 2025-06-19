from .base import Augmentation
import numpy as np
from PIL import Image

class VQVAEAugmentation(Augmentation):
    def __init__(self):
        super().__init__('vqvae')

    def __call__(self, img):
        arr = np.array(img)
        arr = arr + np.random.randint(-10, 10, arr.shape, dtype='int')
        arr = np.clip(arr, 0, 255)
        return Image.fromarray(arr.astype('uint8'))
