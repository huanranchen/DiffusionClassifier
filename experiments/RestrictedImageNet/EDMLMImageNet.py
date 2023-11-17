import torch
from data import get_someset_loader
from tester import test_acc, test_apgd_dlr_acc
from models.unets import get_guided_diffusion_unet
from defenses.PurificationDefenses.DiffPure import EDMEulerIntegralDC, EDMEulerIntegralWraped
from utils.seed import set_seed
from torchvision import transforms
from defenses.PurificationDefenses.DiffPure import EDMEulerIntegralLM, VP2EDM
from defenses.PurificationDefenses.DiffPure.DiffPure.DiffusionLikelihoodMaximizer import (
    diffusion_likelihood_maximizer_defense,
)
import argparse
from models import Salman2020Do_R50, RegionClassSelectionModel

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
classifier = RegionClassSelectionModel(
    Salman2020Do_R50(pretrained=True),
    target_class=RegionClassSelectionModel.restricted_imagenet_target_class(),
)

defensed = diffusion_likelihood_maximizer_defense(
    classifier, lambda x: lm.likelihood_maximization_T1(x, iter_step=5, t_min=0.4, t_max=0.6)
)
# test_acc(dc, loader)
test_apgd_dlr_acc(defensed, loader=loader, eps=4 / 255, bs=2)
