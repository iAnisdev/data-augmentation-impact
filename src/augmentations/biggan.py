import torch
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Normalize, Compose, ToPILImage
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_int, truncated_noise_sample)

class BigGANAugmentation:
    def __init__(self, class_id=207, truncation=0.4, device="cpu"):
        self.device = device
        self.class_id = class_id
        self.truncation = truncation
        self.model = BigGAN.from_pretrained("biggan-deep-256").to(device)
        self.model.eval()

        self.to_pil = ToPILImage()
        self.to_tensor = Compose([
            Resize((64, 64)),
            ToTensor(),
            Normalize([0.5]*3, [0.5]*3),
        ])

    def __call__(self, _=None):
        noise_vector = truncated_noise_sample(truncation=self.truncation, batch_size=1)
        class_vector = one_hot_from_int([self.class_id], batch_size=1)

        noise_vector = torch.from_numpy(noise_vector).to(self.device)
        class_vector = torch.from_numpy(class_vector).to(self.device)

        with torch.no_grad():
            output = self.model(noise_vector, class_vector, truncation=self.truncation)

        output = (output + 1) / 2
        img = output.squeeze(0).cpu()
        return self.to_tensor(self.to_pil(img))
