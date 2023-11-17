import torch
from torch import nn, Tensor


class EDM2VP(nn.Module):
    def __init__(
        self,
        edm_unet: nn.Module,
        beta=torch.linspace(0.1 / 1000, 20 / 1000, 1000, device=torch.device("cuda")),
        device=torch.device("cuda"),
    ):
        self.device = device
        super(EDM2VP, self).__init__()
        self.unet = edm_unet
        alpha = 1 - beta
        self.alpha_bar = alpha.cumprod(dim=0).to(self.device)

    def forward(self, x, t, *args, **kwargs):
        """
        VP Inputs
        :param x:
        :param t:
        :param args:
        :param kwargs:
        :return: VP predictions
        """
        t = t.long()
        x0 = self.unet(
            x / torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1),
            torch.sqrt(1 - self.alpha_bar[t]) / torch.sqrt(self.alpha_bar[t]),
            *args,
            **kwargs
        )
        epsilon = (x - torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1) * x0) / torch.sqrt(1 - self.alpha_bar[t]).view(
            -1, 1, 1, 1
        )
        return epsilon


class VP2EDM(nn.Module):
    def __init__(
        self,
        vp_unet: nn.Module,
        beta=torch.linspace(0.1 / 1000, 20 / 1000, 1000, device=torch.device("cuda")),
        device=torch.device("cuda"),
    ):
        self.device = device
        super(VP2EDM, self).__init__()
        self.unet = vp_unet
        alpha = 1 - beta
        self.alpha_bar = alpha.cumprod(dim=0).to(self.device)
        self.edm_sigma = torch.sqrt((1 - self.alpha_bar) / self.alpha_bar)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def forward(self, x, sigma, *args, **kwargs):
        """
        EDM Inputs
        :param x: N, C, H, D
        :param sigma: N,
        :param args:
        :param kwargs:
        :return: EDM predictions
        """
        # chose the first t greater than sigma
        t = torch.max((sigma.view(-1, 1) <= self.edm_sigma.repeat(sigma.numel(), 1)), dim=1)[1]
        vp_x_t = x * torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1)
        epsilon = self.unet(vp_x_t, t, *args, **kwargs)
        # x_0 = x - sigma * epsilon. Will this be better?
        x_0 = (vp_x_t - torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1, 1) * epsilon) / torch.sqrt(
            self.alpha_bar[t]
        ).view(-1, 1, 1, 1)
        return x_0
