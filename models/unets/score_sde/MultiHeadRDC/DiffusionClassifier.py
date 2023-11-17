import torch
from torch import nn
from .utils import *


class DiffusionClassifierCached(nn.Module):
    def __init__(self,
                 unet: nn.Module,
                 beta: Tensor = torch.linspace(0.1 / 1000, 20 / 1000, 1000),
                 num_classes=10,
                 ts: Tensor = torch.arange(1000),
                 cfg=0,
                 ):
        super(DiffusionClassifierCached, self).__init__()
        self.device = torch.device('cuda')
        self.unet = unet
        alpha = (1 - beta)
        self.alpha_bar = alpha.cumprod(dim=0).to(self.device)
        self.T = 1000
        self.ts = ts.to(self.device)
        self._init()

        # storing
        self.num_classes = num_classes
        self.unet_criterion = nn.MSELoss()
        self.cfg = cfg

    def _init(self):
        self.eval().requires_grad_(False)
        self.to(self.device)
        self.transform = lambda x: (x - 0.5) * 2

    def get_one_instance_prediction(self, x: Tensor) -> Tensor:
        """
        :param x: 1, C, H, D
        """
        loss = self.unet_loss_without_grad(x)
        loss = loss * -1  # convert into logit where greatest is the target
        return loss

    def forward(self, x: Tensor) -> Tensor:
        xs = x.split(1)  # 1, C, H, D
        y = []
        for now_x in xs:
            y.append(self.get_one_instance_prediction(now_x))
        y = torch.stack(y)  # N, num_classes
        return y

    def unet_loss_without_grad(self,
                               x: Tensor,
                               ) -> Tensor:
        """
        Calculate the diffusion loss
        :param x: in range [0, 1]
        """
        t = self.ts
        x = x.repeat(t.numel(), 1, 1, 1)
        x = self.transform(x)  # T, C, H, D
        T, C, H, D = x.shape
        noise = torch.randn_like(x)
        noised_x = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1) * x + \
                   torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1, 1) * noise  # T, C, H, D
        pre = self.unet(noised_x, t)  # T, num_classes*C, H, D
        pre = pre.reshape(T, -1, C, H, D).permute(1, 0, 2, 3, 4)  # num_classes, T, C, H, D
        if self.cfg != 0:
            unconditional_prediction = pre[-1:]
            conditional_prediction = pre[:-1]
            pre = (1 + self.cfg) * conditional_prediction - \
                  self.cfg * unconditional_prediction.repeat(self.num_classes, 1, 1, 1, 1)
        else:
            pre = pre[:self.num_classes]
        loss = torch.mean((pre - noise) ** 2, dim=[1, 2, 3, 4])  # num_classes
        # pre = pre.reshape(self.num_classes, -1)
        # print(cosine_similarity(list(pre.split(1, dim=0))))
        return loss
