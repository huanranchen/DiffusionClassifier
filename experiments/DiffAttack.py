import torch
from attacks import BIM
from data import get_CIFAR10_test
from tester import test_robustness, test_apgd_dlr_acc
from defenses import DiffusionPure
from utils import set_seed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--begin", type=int)
parser.add_argument("--end", type=int)
args = parser.parse_args()
begin, end = args.begin, args.end

set_seed(1)
loader = get_CIFAR10_test(batch_size=1)
device = torch.device("cuda")
diffpure = DiffusionPure(mode="sde", grad_checkpoint=True).eval().requires_grad_(False)

loader = [(x, y) for i, ((x, y)) in enumerate(loader) if begin <= i < end]
attacker = BIM([diffpure], step_size=0.1, total_step=200, eot_step=32, epsilon=0.5, norm="L2", eot_batch_size=32)
test_robustness(
    attacker,
    loader,
    [diffpure],
)
# test_apgd_dlr_acc(diffpure, loader=loader, bs=16, eot_iter=20, norm="L2", eps=0.5)
