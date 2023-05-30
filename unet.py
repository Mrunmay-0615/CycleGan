import torch
from torch import nn


class DownConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DownConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)




class UNET(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.convts = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        self.features = features
        self.bottom_conv1 = nn.Conv2d(512, 1024, 3, 1, 1, bias=False)
        self.bottom_batch_norm1 = nn.BatchNorm2d(1024)
        self.bottom_conv2 = nn.Conv2d(1024, 1024, 3, 1, 1, bias=False)
        self.bottom_batch_norm2 = nn.BatchNorm2d(1024)
        self.out_conv = nn.Conv2d(features[0], out_channels, 3, 1, 1)
        self.relu = nn.ReLU()

        # Downs
        for channels in features:
            self.downs.append(DownConvBlock(in_channels=in_channels, out_channels=channels))
            in_channels = channels

        in_channels = in_channels * 2

        # Ups
        for channels in reversed(features):
            self.ups.append(DownConvBlock(in_channels=in_channels, out_channels=channels))
            in_channels = channels

        in_channels = 1024

        # Transposed Convs
        for i in range(4):
            self.convts.append(nn.ConvTranspose2d(in_channels, in_channels//2, 2, 2, 0))
            in_channels = in_channels//2


    def forward(self, x):

        downs = []
        ups = []
        for module in self.downs:
            x = module(x)
            downs.append(x)
            x = self.pool(x)

        x = self.relu(self.bottom_batch_norm1(self.bottom_conv1(x)))
        x = self.relu(self.bottom_batch_norm2(self.bottom_conv2(x)))

        for i, module in enumerate(self.ups):
            print(i)
            x = self.convts[i](x)
            x = torch.concat([x, downs[3-i]], dim=1)
            # print(f"Before passing shape: {x.shape}")
            x = module(x)
            # print(f"After passing shape: {x.shape}")

        mask = torch.sigmoid(self.out_conv(x))
        return mask
