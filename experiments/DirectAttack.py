import torch
from data import get_CIFAR10_test
from tester import test_apgd_dlr_acc
from torchvision import transforms
import numpy as np
import random
from defenses.PurificationDefenses.DiffPure import RobustDiffusionClassifier

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

to_img = transforms.ToPILImage()
loader = get_CIFAR10_test(batch_size=1)
device = torch.device('cuda')


def save_img(x, name='test'):
    if x.ndim == 4:
        x = x[0]
    img = to_img(x)
    img.save(f'{name}.png')


diffpure = RobustDiffusionClassifier(second_order=True, likelihood_maximization=True)
unet = diffpure.unet
diffpure.eval().requires_grad_(False).to(device)
xs, ys = [], []
for x, y in loader:
    xs.append(x)
    ys.append(y)
x = torch.concat(xs, dim=0).cuda()
y = torch.concat(ys, dim=0).cuda()
x, y = x[:512], y[:512]
test_apgd_dlr_acc(diffpure, x=x, y=y, bs=1, log_path='./direct.txt')
