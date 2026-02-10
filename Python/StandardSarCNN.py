import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class StandardSarCNN(nn.Module):
    def __init__(self, num_classes : int):
        super(StandardSarCNN, self).__init__()
        # Assume padding = 1 for MatLab's 'same'
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(256, 512, kernel_size=6, stride=1, padding=0)

        self.batchNorm1 = nn.BatchNorm2d(32)
        self.batchNorm2 = nn.BatchNorm2d(64)
        self.batchNorm3 = nn.BatchNorm2d(128)
        self.batchNorm4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)

        # Input formula is (H * K - 1) * (W * K - 1) * C
        # By the time our forward pass reaches FC1, the H and W are reduced from 128x128 to 8x8 through the 4 pools.
        # K (kernel size) is 6 and C (channel) is 512, both from convolution layer #9.
        self.fc1 = nn.Linear(3 * 3 * 512, 512)

        # 10 reflects the number of MSTAR classes
        self.fc2 = nn.Linear(512, num_classes)

        self.features = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            self.batchNorm1,
            nn.ReLU(),
            self.pool,

            self.conv3,
            nn.ReLU(),
            self.conv4,
            self.batchNorm2,
            nn.ReLU(),
            self.pool,

            self.conv5,
            nn.ReLU(),
            self.conv6,
            self.batchNorm3,
            nn.ReLU(),
            self.pool,

            self.conv7,
            nn.ReLU(),
            self.conv8,
            self.batchNorm4,
            nn.ReLU(),
            self.pool,

            self.conv9,
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            self.fc1,
            nn.ReLU(),
            self.fc2
        )

    def forward(self, x):
        x = self.features(x)
        # Must flatten input layers before FC-ing. MATLAB does this step implicitly.
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    @staticmethod
    def transform():
        def transforms_to_PIL(image):
            if isinstance(image, Image.Image):
                return image
            return transforms.ToPILImage()(image)

        return transforms.Compose([
            transforms_to_PIL,
            transforms.Grayscale(num_output_channels=1),   # force grayscale if model expects 1 channel
            transforms.Resize((128, 128)),                 # resize to match input size
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))           # normalize to [-1,1]
        ])