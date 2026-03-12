import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class VGG19(nn.Module):
    def __init__(self, num_classes : int):
        super(VGG19, self).__init__()

        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(4 * 4 * 512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        self.features = nn.Sequential(
            self.conv1_1,
            nn.ReLU(),
            self.conv1_2,
            nn.ReLU(),
            self.pool,

            self.conv2_1,
            nn.ReLU(),
            self.conv2_2,
            nn.ReLU(),
            self.pool,

            self.conv3_1,
            nn.ReLU(),
            self.conv3_2,
            nn.ReLU(),
            self.conv3_3,
            nn.ReLU(),
            self.conv3_4,
            nn.ReLU(),
            self.pool,

            self.conv4_1,
            nn.ReLU(),
            self.conv4_2,
            nn.ReLU(),
            self.conv4_3,
            nn.ReLU(),
            self.conv4_4,
            nn.ReLU(),
            self.pool,

            self.conv5_1,
            nn.ReLU(),
            self.conv5_2,
            nn.ReLU(),
            self.conv5_3,
            nn.ReLU(),
            self.conv5_4,
            nn.ReLU(),
            self.pool,
        )

        self.classifier = nn.Sequential(
            self.fc1,
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            self.fc2,
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            self.fc3
        )

        def transforms_to_PIL(image):
            if isinstance(image, Image.Image):
                return image
            return transforms.ToPILImage()(image)

        self.train_transform = transforms.Compose([
            transforms_to_PIL,
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((128, 128)),
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
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x