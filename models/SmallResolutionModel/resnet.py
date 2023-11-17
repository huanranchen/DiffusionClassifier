from __future__ import absolute_import

"""Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei


adding hyperparameter norm_layers

"""
import math

import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "resnet56_aux",
    "resnet20_aux",
    "resnet32x4_aux",
    "resnet8x4_aux",
    "resnet8",
    "resnet8x4",
    "resnet20",
    "resnet32",
    "resnet56",
    "resnet110",
    "resnet8_spkd",
    "resnet20_spkd",
    "resnet56_spkd",
    "resnet8x4_spkd",
    "resnet32x4_spkd",
    "resnet32x4",
    "resnet8_crd",
    "resnet20_crd",
    "resnet56_crd",
    "resnet8x4_crd",
    "resnet32x4_crd",
]


class Normalizer4CRD(nn.Module):
    def __init__(self, linear, power=2):
        super().__init__()
        self.linear = linear
        self.power = power

    def forward(self, x):
        x = x.flatten(1)
        z = self.linear(x)
        norm = z.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        out = z.div(norm)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False,
                 norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False,
                 norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, depth, num_filters, block_name="BasicBlock", num_classes=10,
                 norm_layer=nn.BatchNorm2d):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == "basicblock":
            assert (
                           depth - 2
                   ) % 6 == 0, "When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202"
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == "bottleneck":
            assert (
                           depth - 2
                   ) % 9 == 0, "When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199"
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError("block_name shoule be Basicblock or Bottleneck")

        self.inplanes = num_filters[0]
        self.conv1 = nn.Conv2d(3, num_filters[0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, num_filters[1], n, norm_layer)
        self.layer2 = self._make_layer(block, num_filters[2], n, norm_layer, stride=2)
        self.layer3 = self._make_layer(block, num_filters[3], n, norm_layer, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_filters[3] * block.expansion, num_classes)
        self.last_channel = num_filters[3] * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, norm_layer, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list([])
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.relu)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        return feat_m

    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck):
            bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
        else:
            raise NotImplementedError("ResNet unknown block error !!!")

        return [bn1, bn2, bn3]

    def forward(self, x, is_feat=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32

        x = self.layer1(x)  # 32x32
        f1 = x
        x = self.layer2(x)  # 16x16
        f2 = x
        x = self.layer3(x)  # 8x8
        f3 = x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if is_feat:
            return [f1, f2, f3, f3], x
        else:
            return x


class Auxiliary_Classifier(nn.Module):
    def __init__(self, depth, num_filters, block_name="BasicBlock", num_classes=100):
        super(Auxiliary_Classifier, self).__init__()
        if block_name.lower() == "basicblock":
            assert (
                           depth - 2
                   ) % 6 == 0, "When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202"
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == "bottleneck":
            assert (
                           depth - 2
                   ) % 9 == 0, "When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199"
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError("block_name shoule be Basicblock or Bottleneck")

        self.inplanes = num_filters[1] * block.expansion
        self.block_extractor1 = nn.Sequential(
            *[
                self._make_layer(block, num_filters[2], n, stride=2),
                self._make_layer(block, num_filters[3], n, stride=2),
            ]
        )
        self.inplanes = num_filters[2] * block.expansion
        self.block_extractor2 = nn.Sequential(
            *[self._make_layer(block, num_filters[3], n, stride=2)]
        )
        self.inplanes = num_filters[3] * block.expansion
        self.block_extractor3 = nn.Sequential(
            *[self._make_layer(block, num_filters[3], n, stride=1)]
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(num_filters[3] * block.expansion, num_classes)
        self.fc2 = nn.Linear(num_filters[3] * block.expansion, num_classes)
        self.fc3 = nn.Linear(num_filters[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list([])
        layers.append(block(self.inplanes, planes, stride, downsample, is_last=(blocks == 1)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, is_last=(i == blocks - 1)))

        return nn.Sequential(*layers)

    def forward(self, x):
        ss_logits = []
        ss_feats = []
        for i in range(len(x)):
            idx = i + 1
            out = getattr(self, "block_extractor" + str(idx))(x[i])
            out = self.avg_pool(out)
            out = out.view(out.size(0), -1)
            ss_feats.append(out)
            out = getattr(self, "fc" + str(idx))(out)
            ss_logits.append(out)

        return ss_feats, ss_logits


class ResNet_Auxiliary(nn.Module):
    def __init__(self, depth, num_filters, block_name="BasicBlock", num_classes=10):
        super(ResNet_Auxiliary, self).__init__()
        self.backbone = ResNet(depth, num_filters, block_name, num_classes)
        self.auxiliary_classifier = Auxiliary_Classifier(
            depth, num_filters, block_name, num_classes=num_classes * 4
        )

    def forward(self, x, grad=False, att=False):
        feats, logit = self.backbone(x, is_feat=True)
        if grad is False:
            for i in range(len(feats)):
                feats[i] = feats[i].detach()
        ss_feats, ss_logits = self.auxiliary_classifier(feats)
        if att is False:
            return logit, ss_logits
        else:
            return logit, ss_logits, feats


class ResNet_SPKD(ResNet):
    def __init__(self, depth, num_filters, block_name="BasicBlock", num_classes=10):
        super(ResNet_SPKD, self).__init__(depth, num_filters, block_name, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32
        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        f3 = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return f3, x


class ResNet_CRD(nn.Module):
    def __init__(self, depth, num_filters, block_name="BasicBlock", num_classes=10):
        super(ResNet_CRD, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == "basicblock":
            assert (
                           depth - 2
                   ) % 6 == 0, "When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202"
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == "bottleneck":
            assert (
                           depth - 2
                   ) % 9 == 0, "When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199"
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError("block_name shoule be Basicblock or Bottleneck")

        self.inplanes = num_filters[0]
        self.conv1 = nn.Conv2d(3, num_filters[0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, num_filters[1], n)
        self.layer2 = self._make_layer(block, num_filters[2], n, stride=2)
        self.layer3 = self._make_layer(block, num_filters[3], n, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_filters[3] * block.expansion, num_classes)
        linear = nn.Linear(num_filters[3] * block.expansion, 128, bias=True)
        self.normalizer = Normalizer4CRD(linear, power=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list([])
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.relu)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        return feat_m

    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck):
            bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
        else:
            raise NotImplementedError("ResNet unknown block error !!!")

        return [bn1, bn2, bn3]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        crdout = x
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        crdout = self.normalizer(crdout)
        return crdout, x


def resnet8(**kwargs):
    return ResNet(8, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet14(**kwargs):
    return ResNet(14, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet20(**kwargs):
    return ResNet(20, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet8_spkd(**kwargs):
    return ResNet_SPKD(8, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet14_spkd(**kwargs):
    return ResNet_SPKD(14, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet20_spkd(**kwargs):
    return ResNet_SPKD(20, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet8_crd(**kwargs):
    return ResNet_CRD(8, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet14_crd(**kwargs):
    return ResNet_CRD(14, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet20_crd(**kwargs):
    return ResNet_CRD(20, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet20_aux(**kwargs):
    return ResNet_Auxiliary(20, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet14x05(**kwargs):
    return ResNet(14, [8, 8, 16, 32], "basicblock", **kwargs)


def resnet20x05(**kwargs):
    return ResNet(20, [8, 8, 16, 32], "basicblock", **kwargs)


def resnet20x0375(**kwargs):
    return ResNet(20, [6, 6, 12, 24], "basicblock", **kwargs)


def resnet32(**kwargs):
    return ResNet(32, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet44(**kwargs):
    return ResNet(44, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet56(**kwargs):
    return ResNet(56, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet56_aux(**kwargs):
    return ResNet_Auxiliary(56, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet56_spkd(**kwargs):
    return ResNet_SPKD(56, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet56_crd(**kwargs):
    return ResNet_CRD(56, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet110(**kwargs):
    return ResNet(110, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet8x4(**kwargs):
    return ResNet(8, [32, 64, 128, 256], "basicblock", **kwargs)


def resnet8x4_aux(**kwargs):
    return ResNet_Auxiliary(8, [32, 64, 128, 256], "basicblock", **kwargs)


def resnet8x4_spkd(**kwargs):
    return ResNet_SPKD(8, [32, 64, 128, 256], "basicblock", **kwargs)


def resnet8x4_crd(**kwargs):
    return ResNet_CRD(8, [32, 64, 128, 256], "basicblock", **kwargs)


def resnet32x4(**kwargs):
    return ResNet(32, [32, 64, 128, 256], "basicblock", **kwargs)


def resnet32x4_aux(**kwargs):
    return ResNet_Auxiliary(32, [32, 64, 128, 256], "basicblock", **kwargs)


def resnet32x4_spkd(**kwargs):
    return ResNet_SPKD(32, [32, 64, 128, 256], "basicblock", **kwargs)


def resnet32x4_crd(**kwargs):
    return ResNet_CRD(32, [32, 64, 128, 256], "basicblock", **kwargs)
