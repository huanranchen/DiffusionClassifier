import torch
from data import get_someset_loader
from tester import test_acc, test_apgd_dlr_acc
from models.unets import get_guided_diffusion_unet, get_edm_imagenet_64x64_cond
from defenses.PurificationDefenses.DiffPure import EDMEulerIntegralDC, EDMEulerIntegralWraped
from utils.seed import set_seed
from torchvision import transforms
from defenses.PurificationDefenses.DiffPure import EDMEulerIntegralLM, VP2EDM
from defenses.PurificationDefenses.DiffPure.DiffPure.DiffusionLikelihoodMaximizer import (
    diffusion_likelihood_maximizer_defense,
)
import argparse
from models import BaseNormModel, resnet50, RegionClassSelectionModel

parser = argparse.ArgumentParser()
parser.add_argument("--begin", type=int)
parser.add_argument("--end", type=int)
args = parser.parse_args()
begin, end = args.begin, args.end

set_seed(1)
loader = get_someset_loader(
    "./resources/RestrictedImageNet256",
    "./resources/RestrictedImageNet256/gt.npy",
    batch_size=1,
    shuffle=False,
    transform=transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    ),
)
loader = [item for i, item in enumerate(loader) if begin <= i < end]
unet = get_guided_diffusion_unet(resolution=256)

dc = EDMEulerIntegralDC(VP2EDM(unet))
lm = EDMEulerIntegralLM(dc)

edm_unet = get_edm_imagenet_64x64_cond()
edm_unet.load_state_dict(torch.load("../../../resources/checkpoints/EDM/edm-imagenet-64x64-cond.pt"))


def edm_64x64_transform(x):
    x = transforms.Resize((64, 64))(x)
    x = (x - 0.5) * 2
    return x


dc_wraped = EDMEulerIntegralWraped(
    edm_unet,
    target_class=(151, 281, 30, 33, 80, 365, 389, 118, 300),
    num_classes=1000,
    transform=edm_64x64_transform
)
defensed = diffusion_likelihood_maximizer_defense(
    dc_wraped, lambda x: lm.likelihood_maximization_T1(x, iter_step=5, t_min=0.4, t_max=0.6)
)
# test_acc(dc, loader)
test_apgd_dlr_acc(defensed, loader=loader, eps=4 / 255, bs=1)

# 目前的策略是，虽然edm没有uncond，但我似然仍然可以用uncond，因为他似乎能泛化到uncond
# 还有个策略就是看下embedding，我能不能取这些class embedding的平均值。
