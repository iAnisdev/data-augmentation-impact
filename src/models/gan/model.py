import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=1, feature_maps=64):
        super().__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_maps * 4, 7, 1, 0),   # 7x7
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1),  # 14x14
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 2, img_channels, 4, 2, 1),     # 28x28
            nn.Tanh()
        )

    def forward(self, z):
        return self.gen(z)

class Discriminator(nn.Module):
    def __init__(self, img_channels=1, feature_maps=64):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels, feature_maps * 2, 4, 2, 1),   # 14x14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1),  # 7x7
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 4, 1, 7, 1, 0),  # 1x1
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.disc(img).view(-1)
