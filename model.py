import torch
import torch.nn as nn


class ImageRestorationCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Upscale 128x128 -> 256x256
        self.upsample = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False
            ),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, 3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.upsample(x)
        return x


def create_model():
    return ImageRestorationCNN()