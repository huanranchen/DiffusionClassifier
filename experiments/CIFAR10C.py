import torch
from data import get_CIFAR10_test, get_cifar_10_c_loader
from tester import test_apgd_dlr_acc, test_transfer_attack_acc, test_acc
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
corruptions = [
    'glass_blur',
    'gaussian_noise',
    'shot_noise',
    'speckle_noise',
    'impulse_noise',
    'defocus_blur',
    'gaussian_blur',
    'motion_blur',
    'zoom_blur',
    'snow',
    'fog',
    'brightness',
    'contrast',
    'elastic_transform',
    'pixelate',
    'jpeg_compression',
    'spatter',
    'saturate',
    'frost',
]

device = torch.device('cuda')


def save_img(x, name='test'):
    if x.ndim == 4:
        x = x[0]
    img = to_img(x)
    img.save(f'{name}.png')


diffpure = RobustDiffusionClassifier(
    False, False, False
).cuda().requires_grad_(False).eval()
accs = []
for name in corruptions:
    loader = get_cifar_10_c_loader(name=name, batch_size=1)
    print(f'now testing **{name}**, total images {len(loader)}')
    _, now_acc = test_acc(diffpure, loader)
    accs.append(now_acc)
    print('-' * 100)
acc = sum(accs) / len(accs)
