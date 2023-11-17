import torch
from models.unets import get_edm_cifar_uncond, get_edm_cifar_cond
from data import get_CIFAR10_test
from tester import test_acc, test_apgd_dlr_acc
import argparse
from defenses.PurificationDefenses.DiffPure.DiffPure.DiffusionLikelihoodMaximizer import (
    diffusion_likelihood_maximizer_defense,
)
from defenses.PurificationDefenses.DiffPure import EDMEulerIntegralDC, EDMEulerIntegralWraped

parser = argparse.ArgumentParser()
parser.add_argument("--begin", type=int)
parser.add_argument("--end", type=int)
args = parser.parse_args()
begin, end = args.begin, args.end

edm_unet = get_edm_cifar_uncond()
edm_unet.load_state_dict(torch.load("./resources/checkpoints/EDM/edm_cifar_uncond_vp.pt"))
purify_dc = EDMEulerIntegralDC(unet=edm_unet)

model = get_edm_cifar_cond().cuda()
test_loader = get_CIFAR10_test(batch_size=1)
test_loader = [item for i, item in enumerate(test_loader) if begin <= i < end]
model.load_state_dict(torch.load("./resources/checkpoints/EDM/edm_cifar_cond.pt"))

from tqdm import tqdm
from tester.utils import cosine_similarity
from models import WideResNet_70_16_dropout
from attacks import BIM

dc = EDMEulerIntegralWraped(unet=model)
defensed = diffusion_likelihood_maximizer_defense(WideResNet_70_16_dropout(), purify_dc)
num_grad = 10

x, y = test_loader[0]
x, y = x.cuda(), y.cuda()
attacker = BIM([defensed], step_size=1 / 255, total_step=1000)
x = attacker(x, y)
grad = []
for i in range(10):
    grad.clear()
    for _ in tqdm(range(num_grad)):
        x.requires_grad_(True)
        loss = defensed(x).squeeze()[i]
        loss.backward()
        grad.append(x.grad.clone())
        x.requires_grad_(False)
        x.grad = None

    print("class ", i, " cosine similarity: ", cosine_similarity(grad))
# test_apgd_dlr_acc(dc, loader=test_loader, norm='L2', eps=0.5)
