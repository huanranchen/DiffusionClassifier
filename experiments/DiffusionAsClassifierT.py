import torch
from models.unets import get_NCSNPP_cached, DiffusionClassifierCached, get_edm_cifar_uncond
from data import get_CIFAR10_test
from tester import test_acc, test_apgd_dlr_acc
import argparse
from defenses.PurificationDefenses.DiffPure import EDMEulerIntegralDC
from attacks import StAdvAttack
from tester import test_robustness

parser = argparse.ArgumentParser()
parser.add_argument("--begin", type=int)
parser.add_argument("--end", type=int)
args = parser.parse_args()
begin, end = args.begin, args.end


model = get_NCSNPP_cached(grad_checkpoint=True).cuda()
test_loader = get_CIFAR10_test(batch_size=1)
test_loader = [item for i, item in enumerate(test_loader) if begin <= i < end]
# actually is ema_3050
model.load_state_dict(
    torch.load("/workspace/home/chenhuanran2022/work/Diffusion/" "checkpoints/cached_kd_edm_all/ema_new.pt")
)

edm_unet = get_edm_cifar_uncond()
edm_unet.load_state_dict(torch.load("./resources/checkpoints/EDM/edm_cifar_uncond_vp.pt"))
purify_dc = EDMEulerIntegralDC(unet=edm_unet)

classifier = DiffusionClassifierCached(model)

test_apgd_dlr_acc(classifier, loader=test_loader, norm="Linf", eps=8/255)
test_apgd_dlr_acc(classifier, loader=test_loader, norm="L2", eps=0.5)
# attacker = StAdvAttack(classifier, num_iterations=100, bound=0.05)
# test_robustness(attacker, test_loader, [classifier])
