import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class VGG19BNLite(nn.Module):
    def __init__(self, num_classes : int):
        super(VGG19BNLite, self).__init__()

        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.batchNorm1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.batchNorm1_2 = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.batchNorm2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.batchNorm2_2 = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.batchNorm3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.batchNorm3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.batchNorm3_3 = nn.BatchNorm2d(256)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.batchNorm3_4 = nn.BatchNorm2d(256)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.batchNorm4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.batchNorm4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.batchNorm4_3 = nn.BatchNorm2d(512)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.batchNorm4_4 = nn.BatchNorm2d(512)


        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.batchNorm5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.batchNorm5_2 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.batchNorm5_3 = nn.BatchNorm2d(512)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.batchNorm5_4 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(4 * 4 * 512, 4096)
        self.fc2 = nn.Linear(4096, num_classes)

        self.features = nn.Sequential(
            self.conv1_1,
            self.batchNorm1_1,
            nn.ReLU(),
            self.conv1_2,
            self.batchNorm1_2,
            nn.ReLU(),
            self.pool,

            self.conv2_1,
            self.batchNorm2_1,
            nn.ReLU(),
            self.conv2_2,
            self.batchNorm2_2,
            nn.ReLU(),
            self.pool,

            self.conv3_1,
            self.batchNorm3_1,
            nn.ReLU(),
            self.conv3_2,
            self.batchNorm3_2,
            nn.ReLU(),
            self.conv3_3,
            self.batchNorm3_3,
            nn.ReLU(),
            self.conv3_4,
            self.batchNorm3_4,
            nn.ReLU(),
            self.pool,

            self.conv4_1,
            self.batchNorm4_1,
            nn.ReLU(),
            self.conv4_2,
            self.batchNorm4_2,
            nn.ReLU(),
            self.conv4_3,
            self.batchNorm4_3,
            nn.ReLU(),
            self.conv4_4,
            self.batchNorm4_4,
            nn.ReLU(),
            self.pool,

            self.conv5_1,
            self.batchNorm5_1,
            nn.ReLU(),
            self.conv5_2,
            self.batchNorm5_2,
            nn.ReLU(),
            self.conv5_3,
            self.batchNorm5_3,
            nn.ReLU(),
            self.conv5_4,
            self.batchNorm5_4,
            nn.ReLU(),
            self.pool,
        )

        self.classifier = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(p=0.5),
            self.fc2
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