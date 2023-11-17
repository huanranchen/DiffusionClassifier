import torch
from torch import nn, Tensor
from abc import abstractmethod
import random


class BaseLoss():
    def __init__(self,
                 unet: nn.Module,
                 device: torch.device = torch.device('cuda'),
                 ):
        self.unet = unet
        self.device = device

    @abstractmethod
    def __call__(self, x: Tensor, y: Tensor = None) -> Tensor:
        pass


class VPDiscrete(BaseLoss):
    def __init__(self,
                 unet: nn.Module,
                 device=torch.device('cuda'),
                 beta: Tensor = torch.linspace(0.1 / 1000, 20 / 1000, 1000, device=torch.device('cuda')),
                 T: int = 1000,
                 transform=lambda x: (x - 0.5) * 2,
                 weight=None,
                 p_unconditional=0.1,
                 ):
        super(VPDiscrete, self).__init__(unet, device)
        self.beta = beta
        alpha = (1 - beta)
        self.alpha_bar = alpha.cumprod(dim=0).to(self.device)
        self.T = T
        self.weight = weight
        self.transform = transform
        self.p_unconditional = p_unconditional

    def __call__(self, x: Tensor, y: Tensor = None) -> Tensor:
        x, y = x.to(self.device), y.to(self.device)
        # some preprocess
        x = self.transform(x)
        # train
        t = torch.randint(1000, (x.shape[0],), device=self.device)
        tensor_t = t
        noise = torch.randn_like(x)
        noised_x = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1) * x + \
                   torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1, 1) * noise

        if random.random() < self.p_unconditional:
            pre = self.unet(noised_x, tensor_t)
        else:
            pre = self.unet(noised_x, tensor_t, y)
        target = noise
        loss = torch.mean((pre - target) ** 2, dim=[1, 2, 3])
        loss = torch.mean(loss) if self.weight is None else torch.mean(self.weight[t] * loss)
        return loss


class VPDiscreteMLE(VPDiscrete):
    def __init__(self, *args, **kwargs):
        super(VPDiscreteMLE, self).__init__(*args, **kwargs)
        self.weight = self.beta / (1 - self.alpha_bar)


class VPContinuous(BaseLoss):
    def __init__(self,
                 unet: nn.Module,
                 device=torch.device('cuda'),
                 beta_min=0.1,
                 beta_max=20,
                 transform=lambda x: (x - 0.5) * 2,
                 p_unconditional=0.1,
                 ):
        super(VPContinuous, self).__init__(unet, device)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.transform = transform
        self.p_unconditional = p_unconditional
        self.criterion = nn.MSELoss()

    def __call__(self, x: Tensor, y: Tensor = None) -> Tensor:
        x, y = x.to(self.device), y.to(self.device)
        # some preprocess
        x = self.transform(x)
        # train
        t = torch.rand((x.shape[0],), device=self.device)
        log_mean = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean).view(-1, 1, 1, 1)
        std = torch.sqrt(1 - torch.exp(2 * log_mean)).view(-1, 1, 1, 1)
        tensor_t = t * 999
        noise = torch.randn_like(x)
        noised_x = mean * x + std * noise
        if random.random() < self.p_unconditional:
            pre = self.unet(noised_x, tensor_t)[:, :3, :, :]
        else:
            pre = self.unet(noised_x, tensor_t, y)[:, :3, :, :]
        target = noise
        loss = self.criterion(pre, target)
        return loss
