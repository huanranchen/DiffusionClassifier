import torch
from models.unets import get_edm_cifar_uncond, get_edm_cifar_cond
from data import get_CIFAR10_test, get_someset_loader
from tester import test_acc, test_apgd_dlr_acc, test_robustness
import argparse
from defenses.PurificationDefenses.DiffPure.DiffPure.DiffusionLikelihoodMaximizer import (
    diffusion_likelihood_maximizer_defense,
)
from defenses.PurificationDefenses.DiffPure import EDMEulerIntegralDC
from defenses.PurificationDefenses.DiffPure.LikelihoodMaximization import EDMEulerIntegralLM
from models import WideResNet_70_16_dropout, LogitEnsemble
from attacks import StAdvAttack, BIM

loader = get_CIFAR10_test(batch_size=16)

parser = argparse.ArgumentParser()
parser.add_argument("--begin", type=int)
parser.add_argument("--end", type=int)
args = parser.parse_args()
begin, end = args.begin, args.end
loader = [item for i, item in enumerate(loader) if begin <= i < end]
# loader = get_someset_loader(img_path="./debug/LMnofixnoise", gt_path="./debug/LMfixnoise/labels.npy", batch_size=16)

uncond_edm_net = get_edm_cifar_uncond()
uncond_edm_net.load_state_dict(torch.load("../../resources/checkpoints/EDM/edm_cifar_uncond_vp.pt"))
uncond_edm_dc = EDMEulerIntegralDC(unet=uncond_edm_net)
lm = EDMEulerIntegralLM(uncond_edm_dc)


defensed = diffusion_likelihood_maximizer_defense(
    WideResNet_70_16_dropout(),
    lm.likelihood_maximization_T1,
)


defensed.eval().requires_grad_(False)
# result = []
# for i in tqdm(range(10)):
#     result.extend(lm.SDS_T1(loader[i][0].cuda(), return_intermediate_result=True, eps=300, iter_step=20)[1])
# concatenate_image(result, col=20, row=10)
test_acc(defensed, loader)
test_apgd_dlr_acc(defensed, loader=loader, norm="Linf", eps=8 / 255, bs=64)
test_apgd_dlr_acc(defensed, loader=loader, norm="L2", eps=0.5, bs=64)
attacker = StAdvAttack(defensed, num_iterations=100, bound=0.05)
test_robustness(attacker, loader, [defensed])
