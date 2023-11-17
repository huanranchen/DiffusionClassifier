import torch
from data import get_CIFAR10_test
from tester import test_apgd_dlr_acc
from torchvision import transforms
import numpy as np
import random
from defenses.PurificationDefenses.DiffPure import diffusion_likelihood_maximizer_defense
from models import WideResNet_70_16_dropout

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

to_img = transforms.ToPILImage()
loader = get_CIFAR10_test(batch_size=1)
device = torch.device('cuda')

model = WideResNet_70_16_dropout().eval().requires_grad_(False)
diffpure = diffusion_likelihood_maximizer_defense(model)

xs, ys = [], []
for x, y in loader:
    xs.append(x)
    ys.append(y)
x = torch.concat(xs, dim=0).cuda()
y = torch.concat(ys, dim=0).cuda()
x, y = x[0:100], y[0:100]
test_apgd_dlr_acc(diffpure, x=x, y=y, bs=1, log_path='./maximizer.txt')
