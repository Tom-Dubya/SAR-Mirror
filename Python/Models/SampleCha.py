import torch.nn as nn
from PIL import Image
from torchvision import transforms


class SampleCha(nn.Module):
    def __init__(self, num_classes : int):
        super(SampleCha, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.features = nn.Sequential(
            nn.Conv2d(1, 9, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            self.pool,

            nn.Conv2d(9, 18, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            self.pool,

            nn.Conv2d(18, 36, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            self.pool,

            nn.Conv2d(36, 60, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2160, 60),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(60, num_classes),
            nn.Softmax(dim=1),
        )

        def transforms_to_PIL(image):
            if isinstance(image, Image.Image):
                return image
            return transforms.ToPILImage()(image)

        self.train_transform = transforms.Compose([
            transforms_to_PIL,
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.test_transform = transforms.Compose([
            transforms_to_PIL,
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x