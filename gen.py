import torch
import torch.nn as nn


class C7S1Block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=7,
                      padding=3,
                      stride=1,
                      bias=True,
                      padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DKBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1,
                      stride=2,
                      bias=True,
                      padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      padding_mode='reflect'),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels,
                      out_channels=channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      padding_mode='reflect'),
            nn.InstanceNorm2d(channels),
            nn.Identity()
        )

    def forward(self, x):
        y = self.conv(x)
        return x + self.conv(x)


class UKBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UKBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Generator(nn.Module):

    def __init__(self, in_channels=3, img_size=128):
        super(Generator, self).__init__()
        layers = []
        layers.extend([C7S1Block(in_channels, 64),
                       DKBlock(64, 128),
                       DKBlock(128, 256)])
        if img_size >= 256:
            for _ in range(9):
                layers.append(ResidualBlock(256))
        else:
            for _ in range(6):
                layers.append(ResidualBlock(256))
        layers.append(UKBlock(256, 128))
        layers.append(UKBlock(128, 64))
        self.last = C7S1Block(64, in_channels)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = self.layer(x)
        return torch.tanh(self.last(x))


def test():
    img = torch.randn((5, 3, 256, 256))
    model = Generator(3, 256)
    print(model(img).shape)
    layers = []
    layers.extend([3, 4])
    print(layers)

if __name__ == '__main__':
    test()
