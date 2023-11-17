import torch
from data import get_CIFAR10_test
from tester import test_apgd_dlr_acc
from torchvision import transforms
import numpy as np
import random
from defenses import OptimalDiffusionClassifier

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

to_img = transforms.ToPILImage()
loader = get_CIFAR10_test(batch_size=1)
device = torch.device('cuda')

loader = [(x, y) for i, ((x, y)) in enumerate(loader) if i < 100]
diffpure = OptimalDiffusionClassifier(loader)
test_apgd_dlr_acc(diffpure, loader=loader)
