import torch
from models.unets import get_edm_cifar_cond
from data import get_CIFAR10_test
from tester import test_acc, test_apgd_dlr_acc
import argparse
from defenses.PurificationDefenses.DiffPure import EDMEulerIntegralDC, DiffusionClassifierSingleHeadBaseWraped

"""
Example code of Diffusion Classifier!
"""

parser = argparse.ArgumentParser()
parser.add_argument("--begin", type=int, default=0)
parser.add_argument("--end", type=int, default=128)
parser.add_argument("--steps", type=int, default=126)
args = parser.parse_args()
begin, end, steps = args.begin, args.end, args.steps

model = get_edm_cifar_cond().cuda()
test_loader = get_CIFAR10_test(batch_size=1)
test_loader = [item for i, item in enumerate(test_loader) if begin <= i < end]

dc = EDMEulerIntegralDC(unet=model, timesteps=torch.linspace(1e-4, 3, steps))
dc.share_noise = True
dc_wraped = DiffusionClassifierSingleHeadBaseWraped(dc)
test_acc(dc_wraped, test_loader)
test_apgd_dlr_acc(dc_wraped, loader=test_loader)
