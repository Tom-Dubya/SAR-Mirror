from torch import nn

import torch.nn.functional as F

class ResidualBlockBottleneck(nn.Module):
    def __init__(self, in_channels: int, reduced_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, stride=stride, padding=0),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, reduced_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, out_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )

        self.skip = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0),
                        nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = self.skip(x)
        x = self.layers(x)
        x = x + identity
        return F.relu(x)
