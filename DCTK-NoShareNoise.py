import torch
from models.unets import get_edm_cifar_cond
from data import get_CIFAR10_test
from tester import test_acc, test_apgd_dlr_acc
import argparse
from defenses.PurificationDefenses.DiffPure import EDMEulerIntegralDC, EDMEulerIntegralWraped


parser = argparse.ArgumentParser()
parser.add_argument("--begin", type=int, default=0)
parser.add_argument("--end", type=int, default=128)
args = parser.parse_args()
begin, end = args.begin, args.end

model = get_edm_cifar_cond().cuda()
test_loader = get_CIFAR10_test(batch_size=1)
test_loader = [item for i, item in enumerate(test_loader) if begin <= i < end]

dc = EDMEulerIntegralWraped(unet=model, timesteps=torch.linspace(1e-4, 3, 1001))
test_acc(dc, test_loader)
test_apgd_dlr_acc(dc, loader=test_loader)
