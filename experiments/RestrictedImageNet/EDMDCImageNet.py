import torch
from data import get_someset_loader
from tester import test_acc, test_apgd_dlr_acc
from models.unets.EDM import get_edm_imagenet_64x64_cond
from defenses.PurificationDefenses.DiffPure import EDMEulerIntegralDC, EDMEulerIntegralWraped
from utils.seed import set_seed
from torchvision import transforms
from defenses.PurificationDefenses.DiffPure.LikelihoodMaximization import EDMEulerIntegralLM
from defenses.PurificationDefenses.DiffPure.DiffPure.DiffusionLikelihoodMaximizer import (
    diffusion_likelihood_maximizer_defense,
)
import argparse

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
)
# loader = get_restricted_imagenet_test_loader(batch_size=256, shuffle=True)
# save_dataset(x, y, path='./resources/RestrictedImageNet256', gt_saving_name='gt.npy')
loader = [item for i, item in enumerate(loader) if begin <= i < end]
unet = get_edm_imagenet_64x64_cond()
unet.load_state_dict(torch.load("../../../resources/checkpoints/EDM/edm-imagenet-64x64-cond.pt"))
resizer = transforms.Resize((64, 64), antialias=True)


def edm_64x64_transform(x):
    x = resizer(x)
    x = (x - 0.5) * 2
    return x

dc = EDMEulerIntegralWraped(
    unet, target_class=(151, 281, 30, 33, 80, 365, 389, 118, 300), num_classes=1000, transform=edm_64x64_transform
)
test_apgd_dlr_acc(dc, loader=loader, eps=4 / 255)
# dc = EDMEulerIntegralWraped(
#     unet=unet,
#     target_class=(151, 281, 30, 33, 80, 365, 389, 118, 300),
#     num_classes=1000,
#     transform=edm_64x64_transform,
#     timesteps=torch.tensor([1.0, 2.0]),
# )
