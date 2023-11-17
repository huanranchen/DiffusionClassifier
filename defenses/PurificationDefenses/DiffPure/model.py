from typing import Tuple
import torch
from torch import nn, Tensor
from models.unets import get_NCSNPP
from models.unets import get_guided_diffusion_unet


def get_unet(mode='cifar') -> Tuple[nn.Module, Tensor, Tuple[int, int, int]]:
    if mode == 'imagenet':
        img_shape = (3, 256, 256)
        model = get_guided_diffusion_unet()
        model.load_state_dict(torch.load(f'./resources/checkpoints/DiffPure/256x256_diffusion_uncond.pt'))
    elif mode == 'cifar':
        img_shape = (3, 32, 32)
        model = get_NCSNPP()
        model.load_state_dict(torch.load('./resources/checkpoints/DiffPure/32x32_diffusion.pth'), strict=False)
    else:
        assert False, 'We only provide CIFAR10 and ImageNet diffusion unet model'
    betas = torch.linspace(0.1 / 1000, 20 / 1000, 1000, device=torch.device('cuda'))
    return model, betas, img_shape
