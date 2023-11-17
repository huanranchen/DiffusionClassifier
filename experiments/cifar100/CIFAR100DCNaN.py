from defenses.PurificationDefenses.DiffPure import (
    EDMEulerIntegralDC,
    EDMEulerIntegralLM,
    diffusion_likelihood_maximizer_defense,
    DiffusionClassifierSingleHeadBaseWraped,
)
from models.unets import get_edm_cifar_unet
import torch
from data import get_CIFAR100_test
from tester import test_acc, test_apgd_dlr_acc
import argparse
from utils.seed import set_seed
from torch.amp import autocast

set_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument("--begin", type=int)
parser.add_argument("--end", type=int)
parser.add_argument("--t_steps", type=int, default=126)
args = parser.parse_args()
begin, end, t_steps = args.begin, args.end, args.t_steps

loader = get_CIFAR100_test(batch_size=1)
loader = [item for i, item in enumerate(loader) if begin <= i < end]
device = torch.device("cuda")

torch.autograd.set_detect_anomaly(True)
cond_edm = get_edm_cifar_unet(dataset="cifar100", cond=True, use_fp16=True)
classifier = EDMEulerIntegralDC(cond_edm, timesteps=torch.linspace(1e-4, 3, steps=t_steps), num_classes=100)
classifier.share_noise = True
classifier = DiffusionClassifierSingleHeadBaseWraped(classifier)
classifier.eval().requires_grad_(False).to(device)


def nan_hook(module, f_in, f_out):
    # print(module)
    # print(torch.max(f_in[0]), torch.max(f_out[0]))
    func = torch.isnan
    if all([(torch.sum(func(i)) <= 0).item() for i in f_in]) and any([(torch.sum(func(i)) > 0).item() for i in f_out]):
        print("nan!!!!!!")
        print(module, [(torch.sum(func(i)) <= 0).item() for i in f_in], [(torch.sum(func(i)) > 0).item() for i in f_out])
        assert False
    func = torch.isinf
    if all([(torch.sum(func(i)) <= 0).item() for i in f_in]) and any([(torch.sum(func(i)) > 0).item() for i in f_out]):
        print("inf!!!!")
        print(module, [(torch.sum(func(i)) <= 0).item() for i in f_in], [(torch.sum(func(i)) > 0).item() for i in f_out])
        assert False


for submodule in classifier.modules():
    submodule.register_forward_hook(nan_hook)

test_acc(classifier, loader, verbose=True)
# g = Smooth(classifier, batch_size=32, verbose=False)
# result, nAs, ns, radii = certify_robustness(g, loader, n0=10, n=1000)
