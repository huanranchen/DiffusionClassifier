# from autoattack import AutoAttack
from attacks.autoattack import AutoAttack
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader


def test_autoattack_acc(model: nn.Module, loader: DataLoader, bs=16, log_path=None):
    adversary = AutoAttack(model, eps=8 / 255, log_path=log_path)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    x = torch.concat(xs, dim=0).cuda()
    y = torch.concat(ys, dim=0).cuda()
    adversary.run_standard_evaluation(x, y, bs=bs)


def test_apgd_dlr_acc(model: nn.Module,
                      x: Tensor = None,
                      y: Tensor = None,
                      loader: DataLoader = None,
                      bs=1,
                      log_path=None, device=torch.device('cuda'),
                      eps=8 / 255,
                      norm='Linf',
                      eot_iter=1,
                      n_iter=100,
                      ):
    model.eval().requires_grad_(False).cuda()
    if loader is not None:
        xs, ys = [], []
        for x, y in loader:
            xs.append(x)
            ys.append(y)
        x = torch.concat(xs, dim=0).cuda()
        y = torch.concat(ys, dim=0).cuda()
    adversary = AutoAttack(model, norm=norm, eps=eps, version='custom',
                           log_path=log_path, device=device)
    adversary.attacks_to_run = ['apgd-dlr']
    adversary.apgd.eot_iter = eot_iter
    adversary.apgd.n_iter = n_iter
    adversary.run_standard_evaluation(x, y, bs=bs)

##
#
#
#
#
#
# def test_apgd_dlr_multi_gpu(model: nn.Module, loader: DataLoader, bs=1,
#                             ngpus=4):
#     xs, ys = [], []
#     for x, y in loader:
#         xs.append(x)
#         ys.append(y)
#     x = torch.concat(xs, dim=0)
#     y = torch.concat(ys, dim=0)
#     x = x.split(math.floor(x.shape[0] / ngpus))
#     y = y.split(math.floor(y.shape[0] / ngpus))
#     processes = []
#     for i in range(ngpus):
#         now_device = torch.device(f'cuda:{i}')
#         now = multiprocessing.Process(target=test_apgd_dlr_acc,
#                                       args=(
#                                           model.to(now_device),
#                                           x[i].to(now_device),
#                                           y[i].to(now_device),
#                                           bs,
#                                           f'apgd_device_{i}.txt',
#                                           now_device,
#                                       )
#                                       )
#         processes.append(now)
#     for now in processes:
#         now.start()
#     for now in processes:
#         now.join()
