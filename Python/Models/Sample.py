import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from Models.Components.ConvBlock import ConvBlock
from Models.Components.ResidualBlock import ResidualBlock
from Models.Components.ResidualBlockBottleneck import ResidualBlockBottleneck
from Models.Components.ResidualBlockDownSample import ResidualBlockDownSample


class Sample(nn.Module):
    def __init__(self, num_classes : int):
        super(Sample, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            self.pool,

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            self.pool,

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            self.pool,

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            self.pool,
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, num_classes),
        )

        def transforms_to_PIL(image):
            if isinstance(image, Image.Image):
                return image
            return transforms.ToPILImage()(image)

        self.train_transform = transforms.Compose([
            transforms_to_PIL,
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.test_transform = transforms.Compose([
            transforms_to_PIL,
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x