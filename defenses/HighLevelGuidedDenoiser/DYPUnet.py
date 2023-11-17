import torch
from torch import nn
from typing import Tuple


class BasicDownBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, stride: int = 1, padding: int = 1):
        super(BasicDownBlock, self).__init__()
        self.m = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.m(x)


class C2(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(C2, self).__init__()
        middle: int = (in_dim + out_dim) // 2
        self.m1 = BasicDownBlock(in_dim, middle)
        self.m2 = BasicDownBlock(middle, out_dim)

    def forward(self, x):
        x = self.m1(x)
        x = self.m2(x)
        return x


class C3(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, down=True):
        super(C3, self).__init__()
        if down:
            self.down = BasicDownBlock(in_dim, out_dim, stride=2)
        else:
            self.down = BasicDownBlock(in_dim, out_dim, stride=1)
        self.fuse = C2(out_dim, out_dim)

    def forward(self, x):
        x = self.down(x)
        x = self.fuse(x)
        return x


class Fuse(nn.Module):
    def __init__(self, size: Tuple[int, int], in_dim, out_dim):
        super(Fuse, self).__init__()
        self.up = nn.Upsample(size=size, mode='bilinear')
        self.c3 = C3(in_dim, out_dim, down=False)

    def forward(self, x, res):
        x = self.up(x)
        x = torch.cat([x, res], dim=1)
        x = self.c3(x)
        return x


class DYPUnet(nn.Module):
    def __init__(self):
        super(DYPUnet, self).__init__()
        self.first = C2(3, 64)
        self.second = C3(64, 128)
        self.third = C3(128, 256)
        self.fourth = C3(256, 256)
        self.fifth = C3(256, 256)
        self.f1 = Fuse((38, 38), 512, 256)
        self.f2 = Fuse((75, 75), 512, 256)
        self.f3 = Fuse((150, 150), 384, 128)
        self.f4 = Fuse((299, 299), 192, 64)
        self.final = nn.Conv2d(64, 3, stride=1, kernel_size=1)

    def forward(self, x):
        x1 = self.first(x)
        x2 = self.second(x1)
        x3 = self.third(x2)
        x4 = self.fourth(x3)
        x5 = self.fifth(x4)
        # print(x5.shape)
        x4 = self.f1(x5, x4)
        x3 = self.f2(x4, x3)
        x2 = self.f3(x3, x2)
        x1 = self.f4(x2, x1)
        x1 = self.final(x1)  # - adv perturb
        return x + x1


if __name__ == '__main__':
    idol = DYPUnet()
    x = torch.randn(1, 3, 299, 299)
    print(x.shape)
    print(idol(x).shape)