import torch
from attacks import BIM
from models import BaseNormModel
from data import get_CIFAR100_test
from tester import test_robustness, test_apgd_dlr_acc, test_acc
from models.unets import get_edm_cifar_unet
from models.SmallResolutionModel import wrn_40_2
from defenses import DiffusionPure
from utils import set_seed
import argparse
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument("--begin", type=int)
parser.add_argument("--end", type=int)
args = parser.parse_args()
begin, end = args.begin, args.end

set_seed(1)
loader = get_CIFAR100_test(batch_size=1)
device = torch.device("cuda")
unet = get_edm_cifar_unet(pretrained=True, dataset="cifar100", cond=False)
model = wrn_40_2(num_classes=100)
model.load_state_dict(torch.load("../../../resources/checkpoints/cifar100/wrn_40_2.pth")["model"])
model = BaseNormModel(model, transform=transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]))
diffpure = DiffusionPure(mode="edm", grad_checkpoint=True, unet=unet, model=model).eval().requires_grad_(False)

loader = [(x, y) for i, ((x, y)) in enumerate(loader) if begin <= i < end]
test_acc(diffpure, loader)
# attacker = BIM([diffpure], step_size=0.1, total_step=200, eot_step=32, epsilon=0.5, norm="L2", eot_batch_size=32)
# test_robustness(
#     attacker,
#     loader,
#     [diffpure],
# )
# test_apgd_dlr_acc(diffpure, loader=loader, bs=16, eot_iter=20, norm="L2", eps=0.5)
