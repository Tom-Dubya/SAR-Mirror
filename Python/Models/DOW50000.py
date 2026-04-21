import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from Models.Components.ConvBlock import ConvBlock
from Models.Components.ResidualBlock import ResidualBlock
from Models.Components.ResidualBlockBottleneck import ResidualBlockBottleneck
from Models.Components.ResidualBlockDownSample import ResidualBlockDownSample


class DOW50000(nn.Module):
    def __init__(self, num_classes : int):
        super(DOW50000, self).__init__()

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.features = nn.Sequential(
            ConvBlock(1, 16, depth=2, kernel_size=7, stride=1, padding=1, batch_norm=True),
            ConvBlock(16, 32, depth=2, kernel_size=5, stride=1, padding=1, batch_norm=True),

            ResidualBlockDownSample(32, 64, depth=1, kernel_size=5, batch_norm=True),
            ResidualBlock(64, 64, depth=2, kernel_size=3, batch_norm=True),

            ResidualBlockDownSample(64, 128, depth=2, kernel_size=3, batch_norm=True),
            ResidualBlock(128, 128, depth=2, kernel_size=3, batch_norm=True),

            ResidualBlockBottleneck(128, 64, 256, stride=2),
            ConvBlock(256, 256, depth=2, kernel_size=3, stride=1, padding=1, batch_norm=True),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

        def transforms_to_PIL(image):
            if isinstance(image, Image.Image):
                return image
            return transforms.ToPILImage()(image)

        self.train_transform = transforms.Compose([
            transforms_to_PIL,
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomResizedCrop(128, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.1, 0.1),
                scale=(0.95, 1.05),
                shear=None
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.test_transform = transforms.Compose([
            transforms_to_PIL,
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x