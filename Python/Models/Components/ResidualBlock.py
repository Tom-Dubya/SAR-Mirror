from torch import nn

import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 depth: int = 3,
                 padding: int = -1,
                 kernel_size: int = 3,
                 batch_norm: bool = False):
        super().__init__()
        if padding == -1:
            padding = (kernel_size - 1) // 2
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding))
        if batch_norm:
            self.layers.append(nn.BatchNorm2d(out_channels))
        self.layers.append(nn.ReLU(inplace=True))

        for i in range(depth):
            conv = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
            self.layers.append(conv)
            if batch_norm:
                self.layers.append(nn.BatchNorm2d(out_channels))

            if i < depth - 1:
                self.layers.append(nn.ReLU(inplace=True))

        self.skip = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                        nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity())

    def forward(self, x):
        identity = self.skip(x)
        for layer in self.layers:
            x = layer(x)
        x = x + identity
        return F.relu(x)
