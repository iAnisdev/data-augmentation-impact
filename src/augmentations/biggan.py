import torch
from pytorch_pretrained_biggan import BigGAN, one_hot_from_int, truncated_noise_sample
import torch.nn.functional as F

class BigGANAugmentation:
    def __init__(self, class_id=207, truncation=0.4, device="cpu"):
        self.device = device
        self.class_id = class_id
        self.truncation = truncation
        self.model = BigGAN.from_pretrained("biggan-deep-256").to(device)
        self.model.eval()

    def __call__(self, batch_size=1):
        noise_vector = truncated_noise_sample(truncation=self.truncation, batch_size=batch_size)
        class_vector = one_hot_from_int([self.class_id] * batch_size, batch_size=batch_size)

        noise_vector = torch.from_numpy(noise_vector).to(self.device)
        class_vector = torch.from_numpy(class_vector).to(self.device)

        with torch.no_grad():
            output = self.model(noise_vector, class_vector, truncation=self.truncation)

        # Output is in [-1, 1], rescale to [0, 1]
        output = (output + 1) / 2

        # Resize to 64x64 in batch using interpolation
        resized = F.interpolate(output, size=(64, 64), mode="bilinear", align_corners=False)

        # Normalize to [-1, 1]
        normalized = (resized - 0.5) / 0.5

        return normalized.cpu()  # Return as tensor batch
