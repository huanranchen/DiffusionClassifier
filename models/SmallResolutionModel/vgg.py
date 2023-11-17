"""VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei


adding hyperparameter norm_layer
"""

import math

import torch.nn as nn
import torch.nn.functional as F

__all__ = ["vgg13_bn_aux", "vgg13_bn", "vgg13_bn_spkd", "vgg13_bn_crd", "vgg8_bn"]

model_urls = {
    "vgg11": "https://download.pytorch.org/models/vgg11-bbd30ac9.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-c768596a.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
}


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


class VGG(nn.Module):
    def __init__(self, cfg, batch_norm=False, num_classes=1000,
                 norm_layer=nn.BatchNorm2d):
        super(VGG, self).__init__()
        self.block0 = self._make_layers(cfg[0], batch_norm, 3, norm_layer)
        self.block1 = self._make_layers(cfg[1], batch_norm, cfg[0][-1], norm_layer)
        self.block2 = self._make_layers(cfg[2], batch_norm, cfg[1][-1], norm_layer)
        self.block3 = self._make_layers(cfg[3], batch_norm, cfg[2][-1], norm_layer)
        self.block4 = self._make_layers(cfg[4], batch_norm, cfg[3][-1], norm_layer)

        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.last_channel = 512
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.block0)
        feat_m.append(self.pool0)
        feat_m.append(self.block1)
        feat_m.append(self.pool1)
        feat_m.append(self.block2)
        feat_m.append(self.pool2)
        feat_m.append(self.block3)
        feat_m.append(self.pool3)
        feat_m.append(self.block4)
        feat_m.append(self.pool4)
        return feat_m

    def get_bn_before_relu(self):
        bn1 = self.block1[-1]
        bn2 = self.block2[-1]
        bn3 = self.block3[-1]
        bn4 = self.block4[-1]
        return [bn1, bn2, bn3, bn4]

    def forward(self, x, is_feat=False, preact=False):
        h = x.shape[2]
        x = F.relu(self.block0(x))
        f0 = x

        x = self.pool0(x)
        x = self.block1(x)
        x = F.relu(x)
        f1 = x

        x = self.pool1(x)
        x = self.block2(x)
        x = F.relu(x)
        f2 = x

        x = self.pool2(x)
        x = self.block3(x)
        x = F.relu(x)
        if h == 64:
            x = self.pool3(x)
        x = self.block4(x)
        x = F.relu(x)
        f3 = x

        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        if is_feat:
            return [f0, f1, f2, f3], x
        else:
            return x

    @staticmethod
    def _make_layers(cfg, batch_norm=False, in_channels=3, norm_layer=nn.BatchNorm2d):
        layers = []
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, norm_layer(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        layers = layers[:-1]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class Auxiliary_Classifier(nn.Module):
    def __init__(self, cfg, batch_norm=False, num_classes=100):
        super(Auxiliary_Classifier, self).__init__()

        self.block_extractor1 = nn.Sequential(
            *[
                nn.MaxPool2d(kernel_size=2, stride=2),
                self._make_layers(cfg[1], batch_norm, cfg[0][-1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                self._make_layers(cfg[2], batch_norm, cfg[1][-1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                self._make_layers(cfg[3], batch_norm, cfg[2][-1]),
                nn.ReLU(inplace=True),
                self._make_layers(cfg[4], batch_norm, cfg[3][-1]),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            ]
        )

        self.block_extractor2 = nn.Sequential(
            *[
                nn.MaxPool2d(kernel_size=2, stride=2),
                self._make_layers(cfg[2], batch_norm, cfg[1][-1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                self._make_layers(cfg[3], batch_norm, cfg[2][-1]),
                nn.ReLU(inplace=True),
                self._make_layers(cfg[4], batch_norm, cfg[3][-1]),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            ]
        )

        self.block_extractor3 = nn.Sequential(
            *[
                nn.MaxPool2d(kernel_size=2, stride=2),
                self._make_layers(cfg[3], batch_norm, cfg[2][-1]),
                nn.ReLU(inplace=True),
                self._make_layers(cfg[4], batch_norm, cfg[3][-1]),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            ]
        )

        self.block_extractor4 = nn.Sequential(
            *[
                self._make_layers(cfg[3], batch_norm, cfg[4][-1]),
                nn.ReLU(inplace=True),
                self._make_layers(cfg[4], batch_norm, cfg[3][-1]),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            ]
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(512, num_classes)
        self.fc4 = nn.Linear(512, num_classes)

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    n = m.weight.size(1)
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()

    @staticmethod
    def _make_layers(cfg, batch_norm=False, in_channels=3):
        layers = []
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        layers = layers[:-1]
        return nn.Sequential(*layers)

    def forward(self, x):
        ss_logits = []
        for i in range(len(x)):
            idx = i + 1
            out = getattr(self, "block_extractor" + str(idx))(x[i])
            out = out.view(-1, 512)
            out = getattr(self, "fc" + str(idx))(out)
            ss_logits.append(out)
        return ss_logits


class VGG_Auxiliary(nn.Module):
    def __init__(self, cfg, batch_norm=False, num_classes=100):
        super(VGG_Auxiliary, self).__init__()
        self.backbone = VGG(cfg, batch_norm=batch_norm, num_classes=num_classes)
        self.auxiliary_classifier = Auxiliary_Classifier(
            cfg, batch_norm=batch_norm, num_classes=num_classes * 4
        )

    def forward(self, x, grad=False):
        feats, logit = self.backbone(x, is_feat=True)
        if grad is False:
            for i in range(len(feats)):
                feats[i] = feats[i].detach()
        ss_logits = self.auxiliary_classifier(feats)
        return logit, ss_logits


class VGG_SPKD(VGG):
    def __init__(self, cfg, batch_norm=False, num_classes=1000):
        super(VGG_SPKD, self).__init__(cfg, batch_norm, num_classes)

    def forward(self, x):
        h = x.shape[2]
        x = F.relu(self.block0(x))

        x = self.pool0(x)
        x = self.block1(x)
        x = F.relu(x)

        x = self.pool1(x)
        x = self.block2(x)
        x = F.relu(x)

        x = self.pool2(x)
        x = self.block3(x)
        x = F.relu(x)
        if h == 64:
            x = self.pool3(x)
        x = self.block4(x)
        x = F.relu(x)
        f3 = x

        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return f3, x


cfg = {
    "A": [[64], [128], [256, 256], [512, 512], [512, 512]],
    "B": [[64, 64], [128, 128], [256, 256], [512, 512], [512, 512]],
    "D": [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]],
    "E": [[64, 64], [128, 128], [256, 256, 256, 256], [512, 512, 512, 512], [512, 512, 512, 512]],
    "S": [[64], [128], [256], [512], [512]],
}


class VGG_CRD(nn.Module):
    def __init__(self, cfg, batch_norm=False, num_classes=1000):
        super(VGG_CRD, self).__init__()
        self.block0 = self._make_layers(cfg[0], batch_norm, 3)
        self.block1 = self._make_layers(cfg[1], batch_norm, cfg[0][-1])
        self.block2 = self._make_layers(cfg[2], batch_norm, cfg[1][-1])
        self.block3 = self._make_layers(cfg[3], batch_norm, cfg[2][-1])
        self.block4 = self._make_layers(cfg[4], batch_norm, cfg[3][-1])

        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Linear(512, num_classes)
        linear = nn.Linear(512, 128, bias=True)
        self.normalizer = Normalizer4CRD(linear, power=2)
        self._initialize_weights()

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.block0)
        feat_m.append(self.pool0)
        feat_m.append(self.block1)
        feat_m.append(self.pool1)
        feat_m.append(self.block2)
        feat_m.append(self.pool2)
        feat_m.append(self.block3)
        feat_m.append(self.pool3)
        feat_m.append(self.block4)
        feat_m.append(self.pool4)
        return feat_m

    def get_bn_before_relu(self):
        bn1 = self.block1[-1]
        bn2 = self.block2[-1]
        bn3 = self.block3[-1]
        bn4 = self.block4[-1]
        return [bn1, bn2, bn3, bn4]

    def forward(self, x):
        h = x.shape[2]
        x = F.relu(self.block0(x))

        x = self.pool0(x)
        x = self.block1(x)
        x = F.relu(x)

        x = self.pool1(x)
        x = self.block2(x)
        x = F.relu(x)

        x = self.pool2(x)
        x = self.block3(x)
        x = F.relu(x)
        if h == 64:
            x = self.pool3(x)
        x = self.block4(x)
        x = F.relu(x)

        x = self.pool4(x)
        crdout = x
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        crdout = self.normalizer(crdout)
        return crdout, x

    @staticmethod
    def _make_layers(cfg, batch_norm=False, in_channels=3):
        layers = []
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        layers = layers[:-1]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def vgg8(**kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg["S"], **kwargs)
    return model


def vgg8_bn(**kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg["S"], batch_norm=True, **kwargs)
    return model


def vgg8_bn_aux(**kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_Auxiliary(cfg["S"], batch_norm=True, **kwargs)
    return model


def vgg8_bn_spkd(**kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_SPKD(cfg["S"], batch_norm=True, **kwargs)
    return model


def vgg8_bn_crd(**kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_CRD(cfg["S"], batch_norm=True, **kwargs)
    return model


def vgg11(**kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg["A"], **kwargs)
    return model


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(cfg["A"], batch_norm=True, **kwargs)
    return model


def vgg13(**kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg["B"], **kwargs)
    return model


def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(cfg["B"], batch_norm=True, **kwargs)
    return model


def vgg13_bn_aux(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG_Auxiliary(cfg["B"], batch_norm=True, **kwargs)
    return model


def vgg13_bn_spkd(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG_SPKD(cfg["B"], batch_norm=True, **kwargs)
    return model


def vgg13_bn_crd(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG_CRD(cfg["B"], batch_norm=True, **kwargs)
    return model


def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg["D"], **kwargs)
    return model


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(cfg["D"], batch_norm=True, **kwargs)
    return model


def vgg19(**kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg["E"], **kwargs)
    return model


def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG(cfg["E"], batch_norm=True, **kwargs)
    return model
