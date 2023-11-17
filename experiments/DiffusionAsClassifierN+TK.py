import torch
from models.unets import get_edm_cifar_uncond, get_edm_cifar_cond
from data import get_CIFAR10_test
from tester import test_acc, test_apgd_dlr_acc
import argparse
from defenses.PurificationDefenses.DiffPure.DiffPure.DiffusionLikelihoodMaximizer import (
    diffusion_likelihood_maximizer_defense,
)
from defenses.PurificationDefenses.DiffPure import EDMEulerIntegralDC, EDMEulerIntegralWraped
from defenses.PurificationDefenses.DiffPure.LikelihoodMaximization import EDMEulerIntegralLM

parser = argparse.ArgumentParser()
parser.add_argument("--begin", type=int)
parser.add_argument("--end", type=int)
args = parser.parse_args()
begin, end = args.begin, args.end

edm_unet = get_edm_cifar_uncond()
edm_unet.load_state_dict(torch.load("../../resources/checkpoints/EDM/edm_cifar_uncond_vp.pt"))
purify_dc = EDMEulerIntegralDC(unet=edm_unet)
lm = EDMEulerIntegralLM(purify_dc)

model = get_edm_cifar_cond().cuda()
test_loader = get_CIFAR10_test(batch_size=1)
test_loader = [item for i, item in enumerate(test_loader) if begin <= i < end]
model.load_state_dict(torch.load("../../resources/checkpoints/EDM/edm_cifar_cond.pt"))

dc = EDMEulerIntegralWraped(unet=model)
defensed = diffusion_likelihood_maximizer_defense(dc, lm.likelihood_maximization_T1)

test_apgd_dlr_acc(defensed, loader=test_loader, norm="L2", eps=0.5)
