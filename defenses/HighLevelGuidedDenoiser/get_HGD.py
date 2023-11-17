from .inception import get_net, Conv
import torch
from torch import nn
from torchvision import transforms
from .res152_wide import get_model as get_model1
from .inres import get_model as get_model2
from .v3 import get_model as get_model3
from .resnext101 import get_model as get_model4
from models.BaseNormModel import BaseNormModel, LogitEnsemble

mean_torch = [0.485, 0.456, 0.406]
std_torch = [0.229, 0.224, 0.225]
mean_tf = [0.5, 0.5, 0.5]
std_tf = [0.5, 0.5, 0.5]
__base_hgd_ckpt_path__ = "./resources/checkpoints/HGD/"


def get_HGD_inception():
    config = dict()
    config["flip"] = True
    config["loss_idcs"] = [1]
    config["net_type"] = "inceptionresnetv2"
    input_size = [299, 299]
    block = Conv
    fwd_out = [64, 128, 256, 256, 256]
    num_fwd = [2, 3, 3, 3, 3]
    back_out = [64, 128, 256, 256]
    num_back = [2, 3, 3, 3]
    n = 1
    hard_mining = 0
    loss_norm = False
    net = get_net(input_size, block, fwd_out, num_fwd, back_out, num_back, n, hard_mining, loss_norm)

    class HGD(nn.Module):
        def __init__(self):
            super(HGD, self).__init__()
            self.resizer = transforms.Resize((299, 299))
            self.model = net

        def forward(self, x):
            x = self.resizer(x)
            x = self.model(x)
            return x

    # pretrain_dict = torch.load('adv_inceptionv3.pth')
    return HGD()


def get_HGD_model():
    config, resmodel = get_model1()
    config, inresmodel = get_model2()
    config, incepv3model = get_model3()
    config, rexmodel = get_model4()
    net1 = resmodel.net
    net2 = inresmodel.net
    net3 = incepv3model.net
    net4 = rexmodel.net
    checkpoint = torch.load(__base_hgd_ckpt_path__ + "denoise_res_015.ckpt")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        resmodel.load_state_dict(checkpoint["state_dict"])
    else:
        resmodel.load_state_dict(checkpoint)

    checkpoint = torch.load(__base_hgd_ckpt_path__ + "denoise_inres_014.ckpt")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        inresmodel.load_state_dict(checkpoint["state_dict"])
    else:
        inresmodel.load_state_dict(checkpoint)

    checkpoint = torch.load(__base_hgd_ckpt_path__ + "denoise_incepv3_012.ckpt")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        incepv3model.load_state_dict(checkpoint["state_dict"])
    else:
        incepv3model.load_state_dict(checkpoint)

    checkpoint = torch.load(__base_hgd_ckpt_path__ + "denoise_rex_001.ckpt")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        rexmodel.load_state_dict(checkpoint["state_dict"])
    else:
        rexmodel.load_state_dict(checkpoint)

    class GetLastOut(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.resizer = transforms.Resize((299, 299))
            self.model = model

        def forward(self, x):
            x = self.resizer(x)
            return self.model(x)[-1]

    net1 = BaseNormModel(GetLastOut(net1), transform=transforms.Normalize(mean=mean_torch, std=std_torch))
    net2 = BaseNormModel(GetLastOut(net2), transform=transforms.Normalize(mean=mean_tf, std=std_tf))
    net3 = BaseNormModel(GetLastOut(net3), transform=transforms.Normalize(mean=mean_tf, std=std_tf))
    net4 = BaseNormModel(GetLastOut(net4), transform=transforms.Normalize(mean=mean_torch, std=std_torch))
    model = LogitEnsemble([net1, net2, net3, net4])
    return model
