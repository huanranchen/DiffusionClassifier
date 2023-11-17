import torch
from torch import nn, Tensor
from .BaseSampler import BaseSampler


class DDIM(BaseSampler):
    def __init__(self,
                 unet: nn.Module = None,
                 beta=torch.linspace(0.1 / 1000, 20 / 1000, 1000, ),
                 img_shape=(3, 32, 32), T=1000, stride=1,
                 ddpm=False,
                 *args, **kwargs):
        super(DDIM, self).__init__(unet, *args, **kwargs)
        # calculate VP configuration
        self.alpha = (1 - beta)
        self.alpha_bar = self.alpha.cumprod(dim=0).to(self.device)
        self.beta = beta
        self.T = T
        self.stride = stride
        self.img_shape = img_shape
        self.state_size = img_shape[0] * img_shape[1] * img_shape[2]
        self.ddpm = ddpm

    @staticmethod
    def convert(x):
        x = (x + 1) * 0.5
        x = torch.clamp(x, min=0, max=1)
        return x

    def sample(self, batch_size=64, stride=1):
        x = torch.randn((batch_size, *self.img_shape), device=self.device)
        alpha_bar = self.alpha_bar.flip(0)[::self.stride].flip(0)
        embedding_t = list(range(self.T))[::-1][::self.stride][::-1]
        for t in range(alpha_bar.shape[0] - 1, 0, -1):
            sigma = torch.sqrt(1 - self.alpha[t]) * torch.sqrt(1 - alpha_bar[t - 1]) \
                    / torch.sqrt(1 - alpha_bar[t]) if self.ddpm else 0
            tensor_t = torch.zeros((x.shape[0]), device=self.device) + embedding_t[t]
            predict = self.unet(x, tensor_t)[:, :3, :, :]
            x0 = (x - torch.sqrt(1 - alpha_bar[t]) * predict) / torch.sqrt(alpha_bar[t])
            if t > 1:
                noise = torch.randn_like(x) * sigma
                x = torch.sqrt(alpha_bar[t - 1]) * x0 + \
                    torch.sqrt(1 - alpha_bar[t - 1] - sigma ** 2) * predict + \
                    noise
            else:
                x = x0
        return self.convert(x)

    def purify(self, x: Tensor, noise_level=100, add_noise=True, normalize=True, scale=False):
        if normalize:
            x = (x - 0.5) * 2
        if add_noise:
            x = torch.sqrt(self.alpha_bar[noise_level - 1]) * x + \
                torch.randn_like(x, requires_grad=False) * torch.sqrt(1 - self.alpha_bar[noise_level - 1])
        elif scale:
            x = torch.sqrt(self.alpha_bar[noise_level - 1]) * x
        alpha_bar = self.alpha_bar[:noise_level].flip(0)[::self.stride].flip(0)
        embedding_t = list(range(self.T))[:noise_level][::-1][::self.stride][::-1]
        for t in range(alpha_bar.shape[0] - 1, -1, -1):
            sigma = torch.sqrt(1 - self.alpha[t]) * torch.sqrt(1 - alpha_bar[t - 1]) \
                    / torch.sqrt(1 - alpha_bar[t]) if self.ddpm else 0
            tensor_t = torch.zeros((x.shape[0]), device=self.device) + embedding_t[t]
            predict = self.unet(x, tensor_t)[:, :3, :, :]
            x0 = (x - torch.sqrt(1 - alpha_bar[t]) * predict) / torch.sqrt(alpha_bar[t])
            if t > 0:
                noise = torch.randn_like(x) * sigma
                x = torch.sqrt(alpha_bar[t - 1]) * x0 + \
                    torch.sqrt(1 - alpha_bar[t - 1] - sigma ** 2) * predict + \
                    noise
            else:
                x = x0
        x = self.convert(x)
        return x

    def inverse(self, x: Tensor, noise_level=100):
        x = (x - 0.5) * 2
        alpha_bar = self.alpha_bar[:noise_level][::self.stride]
        embedding_t = list(range(self.T))[:noise_level][::self.stride]
        for t in range(alpha_bar.shape[0] - 1):
            tensor_t = torch.zeros((x.shape[0]), device=self.device) + embedding_t[t]
            t_plus_one = embedding_t[t + 1]
            predict = self.unet(x, tensor_t)[:, :3, :, :]
            x = torch.sqrt(self.alpha_bar[t_plus_one] / self.alpha_bar[t]) * x + (
                    torch.sqrt(1 / self.alpha_bar[t_plus_one] - 1) - torch.sqrt(1 / self.alpha_bar[t] - 1)
            ) * predict * torch.sqrt(self.alpha_bar[t_plus_one])
        x = (x + 1) * 0.5
        return x
