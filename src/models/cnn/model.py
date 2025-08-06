import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(
        self, in_channels: int = 1, num_classes: int = 10, image_size: int = 28
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        input = torch.zeros(1, in_channels, image_size, image_size)
        with torch.no_grad():
            conv_out = self.features(input)
            self.flatten_dim = conv_out.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
