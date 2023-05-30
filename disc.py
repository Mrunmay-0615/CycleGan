import torch
import torch.nn as nn

class Block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      stride,
                      padding,
                      padding_mode='reflect',
                      bias=True),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2))

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):

    def __init__(self, in_channels=3, features=(64, 128, 256, 512)):
        super().__init__()
        self.initial_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=features[0],
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      padding_mode='reflect'),
            nn.LeakyReLU(0.2),
        )
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block(in_channels=in_channels,
                                out_channels=feature,
                                kernel_size=4,
                                stride= 1 if feature == features[-1] else 2,
                                padding=1,
            ))
            in_channels = feature

        layers.append(nn.Conv2d(in_channels,
                                1, 4, 1, 1, padding_mode='reflect'))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial_layer(x)
        return torch.sigmoid(self.model(x))



def test():
    disc = Discriminator(3)
    x = torch.randn((5, 3, 256, 256))
    out = disc(x)
    print(out.shape)


if __name__ == '__main__':
    test()