import torch
from data import get_CIFAR10_test
from tester import test_apgd_dlr_acc
from torchvision import transforms
import numpy as np
import random
from defenses.PurificationDefenses.DiffPure import RobustDiffusionClassifier
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--begin', type=int)
parser.add_argument('--end', type=int)
args = parser.parse_args()
begin, end = args.begin, args.end
print(args.begin, args.end)

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

to_img = transforms.ToPILImage()
loader = get_CIFAR10_test(batch_size=1)
device = torch.device('cuda')

diffpure = RobustDiffusionClassifier(
    bpda=True,
    likelihood_maximization=True,
    diffpure=False,
    second_order=False
)

diffpure.eval().requires_grad_(False).to(device)
xs, ys = [], []
for x, y in loader:
    xs.append(x)
    ys.append(y)
x = torch.concat(xs, dim=0).cuda()
y = torch.concat(ys, dim=0).cuda()
x, y = x[begin:end], y[begin:end]

test_apgd_dlr_acc(diffpure, x=x, y=y, bs=1, log_path=f'./Linfbpda-{begin}-{end}.txt',
                  eps=8 / 255, norm='Linf')
