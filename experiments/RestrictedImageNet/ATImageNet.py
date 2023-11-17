import torch
from data import get_someset_loader
from tester import test_acc, test_apgd_dlr_acc
from utils.seed import set_seed
from models import Wong2020Fast, ClassSelectionModel, RegionClassSelectionModel
from models.RobustBench.imagenet import *
from torch.nn import functional as F

set_seed(1)
loader = get_someset_loader(
    "./resources/RestrictedImageNet256",
    "./resources/RestrictedImageNet256/gt.npy",
    batch_size=32,
)
# loader = get_restricted_imagenet_test_loader(batch_size=256, shuffle=True)
x, y = next(iter(loader))
x, y = x.cuda(), y.cuda()
# save_dataset(x, y, path='./resources/RestrictedImageNet256', gt_saving_name='gt.npy')
models = [Engstrom2019Robustness(), Wong2020Fast(), Salman2020Do_R50(), Debenedetti2022Light_XCiT_S12()]
for model in models:
    dc = RegionClassSelectionModel(model,
                                   target_class=RegionClassSelectionModel.restricted_imagenet_target_class()
                                   ).cuda()
    # test_acc(dc, loader)
    test_apgd_dlr_acc(dc, loader=loader, eps=4 / 255, bs=32)
# for model in models:
#     dc = ClassSelectionModel(model, target_class=(151, 281, 30, 33, 80, 365, 389, 118, 300)).cuda()
#     x.requires_grad = True
#     loss = F.cross_entropy(dc(x), y)
#     loss.backward()
#     print(torch.mean(x.grad.abs()))
#     x.requires_grad = False
#     x.grad = None
