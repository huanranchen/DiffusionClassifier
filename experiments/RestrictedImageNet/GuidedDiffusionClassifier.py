import torch
from data import get_CIFAR10_test
from models.unets import get_NCSNPP
from tester import test_apgd_dlr_acc, test_transfer_attack_acc
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

unet = get_NCSNPP()
diffpure = RobustDiffusionClassifier(
    bpda=False,
    likelihood_maximization=False,
    diffpure=False,
    second_order=False,
    unet=unet,
)

diffpure.eval().requires_grad_(False).to(device)
xs, ys = [], []
for x, y in loader:
    xs.append(x)
    ys.append(y)
x = torch.concat(xs, dim=0).cuda()
y = torch.concat(ys, dim=0).cuda()
x, y = x[begin:end], y[begin:end]
loader = [item for i, item in enumerate(loader) if begin <= i < end]

target = torch.tensor([[0.1200, 0.1195, 0.1197, 0.1195, 0.1199, 0.1199, 0.1198, 0.1211, 0.1198, 0.1198]], device=device)
ce = torch.nn.CrossEntropyLoss()


# def loss(x, y):
#     return -torch.mean((x - target.repeat(x.shape[0], 1)) ** 2)
def loss(x, y):
    return -torch.mean((x - target.repeat(x.shape[0], 1)) ** 2) + ce(x, y)


#
# test_apgd_dlr_acc(diffpure, x=x, y=y, bs=1, log_path=f'./Linfbpda-{begin}-{end}.txt',
#                   eps=8 / 255, norm='Linf')
from attacks import PGD

attacker = PGD([diffpure], epsilon=8 / 255, total_step=20, step_size=1 / 255, criterion=loss,
               eot_step=1, eot_batch_size=1)
test_transfer_attack_acc(attacker, loader, [diffpure], )
