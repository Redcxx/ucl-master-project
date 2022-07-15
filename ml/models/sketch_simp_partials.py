import torch
from torch import nn


def conv1(fin, fout):
    return nn.Conv2d(fin, fout, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))


def conv3(fin, fout, stride=1, no_relu=False):
    return nn.Sequential(
        nn.Conv2d(fin, fout, kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1)),
        nn.Identity() if no_relu else nn.ReLU(True)
    )


def conv5(fin, fout, stride=1, no_relu=False):
    return nn.Sequential(
        nn.Conv2d(fin, fout, kernel_size=(5, 5), stride=(stride, stride), padding=(2, 2)),
        nn.Identity() if no_relu else nn.ReLU(True)
    )


def deconv2(fin, fout):
    return nn.Sequential(
        nn.ConvTranspose2d(fin, fout, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
        nn.BatchNorm2d(fout),
        nn.ReLU(True)
    )


def bn(fin):
    return nn.BatchNorm2d(fin)


class SketchSimpModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            conv5(1, 48, 2),
            conv3(48, 128),
            conv3(128, 128),
            conv3(128, 256, 2),
            conv3(256, 256),
            conv3(256, 256),
            conv3(256, 256, 2),
            conv3(256, 512),
            conv3(512, 1024),
            conv3(1024, 1024),
            conv3(1024, 1024),
            conv3(1024, 1024),
            conv3(1024, 512),
            conv3(512, 256),
            deconv2(256, 256),
            conv3(256, 256),
            conv3(256, 128),
            deconv2(128, 128),
            conv3(128, 128),
            conv3(128, 48),
            deconv2(48, 48),
            conv3(48, 24),
            conv3(24, 1, 1, True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class SketchSimpNetD(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            conv5(2, 64),
            conv3(64, 128, 4),
            conv3(128, 256, 4),
            conv3(256, 256, 4),
            conv3(256, 256, 4),
            conv3(256, 256, 2),
        )

        self.linear = nn.Linear(256, 32)

    def forward(self, inp, gen):
        x = torch.cat((inp, gen), dim=1)
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)

        return x
