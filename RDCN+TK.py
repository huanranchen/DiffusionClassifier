import torch
from models.unets import get_edm_cifar_uncond, get_edm_cifar_cond
from data import get_CIFAR10_test
from tester import test_acc
import argparse
from defenses.PurificationDefenses.DiffPure import EDMEulerIntegralDC, EDMEulerIntegralLM, \
    diffusion_likelihood_maximizer_defense


"""
Example code of Robust Diffusion Classifier!
"""

parser = argparse.ArgumentParser()
parser.add_argument("--begin", type=int, default=0)
parser.add_argument("--end", type=int, default=512)
parser.add_argument("--steps", type=int, default=126)
args = parser.parse_args()
begin, end, steps = args.begin, args.end, args.steps

model = get_edm_cifar_cond().cuda()
test_loader = get_CIFAR10_test(batch_size=1)
test_loader = [item for i, item in enumerate(test_loader) if begin <= i < end]

dc = EDMEulerIntegralDC(unet=model, timesteps=torch.linspace(1e-4, 3, steps))
dc.share_noise = True

uncond_edm_net = get_edm_cifar_uncond()
uncond_edm_dc = EDMEulerIntegralDC(unet=uncond_edm_net)
lm = EDMEulerIntegralLM(uncond_edm_dc)

rdc = diffusion_likelihood_maximizer_defense(
    dc,
    lm.likelihood_maximization_T1,
)

test_acc(rdc, test_loader)
