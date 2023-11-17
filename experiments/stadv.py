from robustbench import load_model
import torch
from data import get_CIFAR10_test
from tester import test_transfer_attack_acc
import numpy as np
import random
from attacks.stadv_eot.attacks import StAdvAttack
from defenses import RobustDiffusionClassifier

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

loader = get_CIFAR10_test(batch_size=1)

models = [load_model(model_name="Rebuffi2021Fixing_70_16_cutmix_ddpm", dataset="cifar10", threat_model="L2"),
          load_model(model_name="Wang2023Better_WRN-70-16", dataset="cifar10", threat_model="L2"),
          load_model(model_name="Rebuffi2021Fixing_70_16_cutmix_ddpm", dataset="cifar10", threat_model="Linf"),
          load_model(model_name="Wang2023Better_WRN-70-16", dataset="cifar10", threat_model="Linf")]
#
loader = [(x, y) for i, ((x, y)) in enumerate(loader) if 96 <= i < 128]

result = []
for model in models:
    model.eval().requires_grad_(False).cuda()
    attacker = StAdvAttack(model, num_iterations=100, bound=0.05)
    result.append(test_transfer_attack_acc(attacker, loader, [model]))
print(result)

model = RobustDiffusionClassifier(False, False, False).eval().requires_grad_(False).cuda()
attacker = StAdvAttack(model, num_iterations=100, bound=0.05)
test_transfer_attack_acc(attacker, loader, [model])
