"""ShuffleNetV2 in PyTorch.
See the paper "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design" for more details.

adding hyperparameter norm_layer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ShuffleV2_aux", "ShuffleV2"]


class ShuffleBlock(nn.Module):
    def __init__(self, groups=2):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C // g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)


class SplitBlock(nn.Module):
    def __init__(self, ratio):
        super(SplitBlock, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        c = int(x.size(1) * self.ratio)
        return x[:, :c, :, :], x[:, c:, :, :]


class BasicBlock(nn.Module):
    def __init__(self, in_channels, split_ratio=0.5, is_last=False, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.split = SplitBlock(split_ratio)
        in_channels = int(in_channels * split_ratio)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn1 = norm_layer(in_channels)
        self.conv2 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.bn2 = norm_layer(in_channels)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn3 = norm_layer(in_channels)
        self.shuffle = ShuffleBlock()

    def forward(self, x):
        x1, x2 = self.split(x)
        out = F.relu(self.bn1(self.conv1(x2)))
        out = self.bn2(self.conv2(out))
        preact = self.bn3(self.conv3(out))
        out = F.relu(preact)
        # out = F.relu(self.bn3(self.conv3(out)))
        preact = torch.cat([x1, preact], 1)
        out = torch.cat([x1, out], 1)
        out = self.shuffle(out)
        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(DownBlock, self).__init__()
        mid_channels = out_channels // 2
        # left
        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        # right
        self.conv3 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.conv4 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=mid_channels,
            bias=False,
        )
        self.bn4 = nn.BatchNorm2d(mid_channels)
        self.conv5 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(mid_channels)

        self.shuffle = ShuffleBlock()

    def forward(self, x):
        # left
        out1 = self.bn1(self.conv1(x))
        out1 = F.relu(self.bn2(self.conv2(out1)))
        # right
        out2 = F.relu(self.bn3(self.conv3(x)))
        out2 = self.bn4(self.conv4(out2))
        out2 = F.relu(self.bn5(self.conv5(out2)))
        # concat
        out = torch.cat([out1, out2], 1)
        out = self.shuffle(out)
        return out


class ShuffleNetV2(nn.Module):
    def __init__(self, net_size, num_classes=100, norm_layer=nn.BatchNorm2d):
        super(ShuffleNetV2, self).__init__()
        out_channels = configs[net_size]["out_channels"]
        num_blocks = configs[net_size]["num_blocks"]

        # self.conv1 = nn.Conv2d(3, 24, kernel_size=3,
        #                        stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(3, 24, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.in_channels = 24
        self.layer1 = self._make_layer(out_channels[0], num_blocks[0], norm_layer)
        self.layer2 = self._make_layer(out_channels[1], num_blocks[1], norm_layer)
        self.layer3 = self._make_layer(out_channels[2], num_blocks[2], norm_layer)
        self.conv2 = nn.Conv2d(
            out_channels[2], out_channels[3], kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels[3])
        self.linear = nn.Linear(out_channels[3], num_classes)
        self.last_channel = out_channels[3]

    def _make_layer(self, out_channels, num_blocks, norm_layer=nn.BatchNorm2d):
        layers = [DownBlock(self.in_channels, out_channels)]
        for i in range(num_blocks):
            layers.append(BasicBlock(out_channels, is_last=(i == num_blocks - 1), norm_layer=norm_layer))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        return feat_m

    def get_bn_before_relu(self):
        raise NotImplementedError('ShuffleNetV2 currently is not supported for "Overhaul" teacher2')

    def forward(self, x, is_feat=False, preact=False):
        out = F.relu(self.bn1(self.conv1(x)))
        # out = F.max_pool2d(out, 3, stride=2, padding=1)
        out = self.layer1(out)
        f1 = out
        out = self.layer2(out)
        f2 = out
        out = self.layer3(out)
        f3 = out
        out = F.relu(self.bn2(self.conv2(out)))
        f4 = out
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if is_feat:
            return [f1, f2, f3, f4], out
        else:
            return out


class Auxiliary_Classifier(nn.Module):
    def __init__(self, net_size, num_classes=100, norm_layer=nn.BatchNorm2d):
        super(Auxiliary_Classifier, self).__init__()
        out_channels = configs[net_size]["out_channels"]
        num_blocks = configs[net_size]["num_blocks"]

        self.in_channels = out_channels[0]
        self.block_extractor1 = nn.Sequential(
            *[
                self._make_layer(out_channels[1], num_blocks[1]),
                self._make_layer(out_channels[2], num_blocks[2]),
                nn.Conv2d(
                    out_channels[2], out_channels[3], kernel_size=1, stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(out_channels[3]),
                nn.ReLU(inplace=True),
            ]
        )

        self.in_channels = out_channels[1]
        self.block_extractor2 = nn.Sequential(
            *[
                self._make_layer(out_channels[2], num_blocks[2]),
                nn.Conv2d(
                    out_channels[2], out_channels[3], kernel_size=1, stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(out_channels[3]),
                nn.ReLU(inplace=True),
            ]
        )

        self.in_channels = out_channels[2]
        self.block_extractor3 = nn.Sequential(
            *[
                self._make_layer(out_channels[2], num_blocks[2], stride=1),
                nn.Conv2d(
                    out_channels[2], out_channels[3], kernel_size=1, stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(out_channels[3]),
                nn.ReLU(inplace=True),
            ]
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(out_channels[3], num_classes)
        self.fc2 = nn.Linear(out_channels[3], num_classes)
        self.fc3 = nn.Linear(out_channels[3], num_classes)

    def _make_layer(self, out_channels, num_blocks, stride=2):
        layers = [DownBlock(self.in_channels, out_channels, stride=stride)]
        for i in range(num_blocks):
            layers.append(BasicBlock(out_channels, is_last=(i == num_blocks - 1)))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        ss_logits = []
        for i in range(len(x)):
            idx = i + 1
            out = getattr(self, "block_extractor" + str(idx))(x[i])
            out = self.avg_pool(out)
            out = out.view(out.size(0), -1)
            out = getattr(self, "fc" + str(idx))(out)
            ss_logits.append(out)
        return ss_logits


class ShuffleNetV2_Auxiliary(nn.Module):
    def __init__(self, net_size, num_classes=100):
        super(ShuffleNetV2_Auxiliary, self).__init__()
        self.backbone = ShuffleNetV2(net_size, num_classes=num_classes)
        self.auxiliary_classifier = Auxiliary_Classifier(net_size, num_classes=num_classes * 4)

    def forward(self, x, grad=False):
        feats, logit = self.backbone(x, is_feat=True)
        if grad is False:
            for i in range(len(feats)):
                feats[i] = feats[i].detach()
        ss_logits = self.auxiliary_classifier(feats)
        return logit, ss_logits


configs = {
    0.2: {"out_channels": (40, 80, 160, 512), "num_blocks": (3, 3, 3)},
    0.3: {"out_channels": (40, 80, 160, 512), "num_blocks": (3, 7, 3)},
    0.5: {"out_channels": (48, 96, 192, 1024), "num_blocks": (3, 7, 3)},
    1: {"out_channels": (116, 232, 464, 1024), "num_blocks": (3, 7, 3)},
    1.5: {"out_channels": (176, 352, 704, 1024), "num_blocks": (3, 7, 3)},
    2: {"out_channels": (224, 488, 976, 2048), "num_blocks": (3, 7, 3)},
}


def ShuffleV2(**kwargs):
    model = ShuffleNetV2(net_size=1, **kwargs)
    return model


def ShuffleV2_aux(**kwargs):
    model = ShuffleNetV2_Auxiliary(net_size=1, **kwargs)
    return model
