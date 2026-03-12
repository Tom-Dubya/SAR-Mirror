from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 depth: int = 3,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = -1,
                 batch_norm: bool = False):
        super().__init__()
        if padding == -1:
            padding = (kernel_size - 1) // 2
        self.depth = depth
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding))
        if batch_norm:
            self.layers.append(nn.BatchNorm2d(out_channels))
        self.layers.append(nn.ReLU(inplace=True))

        for _ in range(depth):
            conv = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
            self.layers.append(conv)
            if batch_norm:
                self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(nn.ReLU(inplace=True))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
