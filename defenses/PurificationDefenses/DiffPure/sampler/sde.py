import torch
import torchsde
from .BaseSampler import BaseSampler
from torch import Tensor, nn
from copy import deepcopy


class DiffusionSde(BaseSampler):
    def __init__(self, unet: nn.Module = None,
                 beta=torch.linspace(0.1 / 1000, 20 / 1000, 1000, device=torch.device('cuda')),
                 img_shape=(3, 32, 32),
                 T=1000, dt=1e-3, cfg=3,
                 *args, **kwargs):
        super(DiffusionSde, self).__init__(unet, *args, **kwargs)
        # Calculate VP config
        alpha = (1 - beta)
        self.alpha_bar = alpha.cumprod(dim=0).to(self.device)
        self.beta = beta * T
        self.T = T
        self.dt = dt
        self.img_shape = img_shape
        self.state_size = img_shape[0] * img_shape[1] * img_shape[2]
        self.cfg = cfg
        self.init()

    def init(self):
        self.eval().to(self.device).requires_grad_(False)
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        self.diffusion_kwargs = dict()
        print(f'dt is {self.dt}')

    def convert(self, x):
        x = (x + 1) * 0.5
        return torch.clamp(x, min=0, max=1)

    def diffusion_forward(self, x: torch.Tensor, t: int):
        assert len(x.shape) == 2, 'x should be N, D'
        N = x.shape[0]
        diffusion = torch.sqrt(self.beta[t]).view(1, 1).repeat(N, x.shape[1])
        drift = -0.5 * self.beta[t] * x
        return drift, diffusion

    def reverse_diffusion_forward(self, x: torch.int, t: int, return_type='diffusion'):
        N = x.shape[0]
        tensor_t = torch.zeros((N,), device=self.device, dtype=torch.int) + (self.T - t - 1)
        forward_drift, forward_diffusion = self.diffusion_forward(x, self.T - t - 1)
        diffusion = forward_diffusion
        if return_type == 'diffusion':
            return diffusion
        if 'y' in self.diffusion_kwargs:
            pre_with_y = self.unet(x.view(N, *self.img_shape), tensor_t,
                                   **self.diffusion_kwargs).view(N, -1)
            kwargs_without_y = deepcopy(self.diffusion_kwargs)
            kwargs_without_y.pop('y')
            pre_without_y = self.unet(x.view(N, *self.img_shape), tensor_t,
                                      **kwargs_without_y).view(N, -1)
            pre = pre_with_y * (self.cfg + 1) - pre_without_y * self.cfg
        else:
            pre = self.unet(x.view(N, *self.img_shape), tensor_t,
                            **self.diffusion_kwargs).view(N, -1)
        score = - pre / torch.sqrt(1 - self.alpha_bar[self.T - t - 1])
        drift = forward_drift - diffusion ** 2 * score
        return -drift

    def f(self, t: float, x):
        f = self.reverse_diffusion_forward(x, round(float(t) * self.T), return_type='drift')
        return f

    def g(self, t: float, x):
        g = self.reverse_diffusion_forward(x, round(float(t) * self.T), return_type='diffusion')
        return g

    @torch.no_grad()
    def sample(self, batch_size=64, **kwargs):
        self.diffusion_kwargs.update(kwargs)
        x = torch.randn((batch_size, *self.img_shape), device=self.device).view((batch_size, self.state_size))
        ts = torch.tensor([0., 1. - 1e-5], device=self.device)
        x = torchsde.sdeint(self, x, ts, method='euler')
        x = x[-1]  # N, 3, 256, 256
        x = x.reshape(batch_size, *self.img_shape)
        self.diffusion_kwargs.clear()
        return self.convert(x)

    def purify(self, x, noise_level=100, end_step=0,
               store_midian_result=False,
               add_noise=True,
               scale=True,
               **kwargs):
        self.diffusion_kwargs.update(kwargs)
        x = (x - 0.5) * 2
        if add_noise:
            x = torch.sqrt(self.alpha_bar[noise_level - 1]) * x + \
                torch.randn_like(x, requires_grad=False) * torch.sqrt(1 - self.alpha_bar[noise_level - 1])
        elif scale:
            x = torch.sqrt(self.alpha_bar[noise_level - 1]) * x
        N = x.shape[0]
        x = x.view(N, -1)
        if store_midian_result:
            ts = torch.linspace(1 - noise_level * 1e-3, 1 - 1e-4 - end_step * 1e-3, noise_level)
        else:
            ts = torch.linspace(1 - noise_level * 1e-3, 1 - 1e-4 - end_step * 1e-3, 2)
        x = torchsde.sdeint(self, x, ts, method='euler', dt=self.dt)
        x = x.view(ts.shape[0], N, *self.img_shape)
        if store_midian_result:
            return self.convert(x)
        x = x[-1]
        x = x.view(N, *self.img_shape)
        self.diffusion_kwargs.clear()
        return self.convert(x)


class DiffusionOde(DiffusionSde):
    def __init__(self, *args, **kwargs):
        super(DiffusionOde, self).__init__(*args, **kwargs)

    def reverse_diffusion_forward(self, x: torch.int, t: int, return_type='diffusion'):
        N = x.shape[0]
        tensor_t = torch.zeros((N,), device=self.device, dtype=torch.int) + (self.T - t - 1)
        forward_drift, forward_diffusion = self.diffusion_forward(x, self.T - t - 1)
        diffusion = forward_diffusion
        if return_type == 'diffusion':
            return diffusion
        pre = self.unet(x.view(N, *self.img_shape), tensor_t)[:, :3, :, :].view(N, -1)
        score = - pre / torch.sqrt(1 - self.alpha_bar[self.T - t - 1])
        drift = forward_drift - 0.5 * diffusion ** 2 * score
        return -drift

    def f(self, t: torch.tensor, x):
        f = self.reverse_diffusion_forward(x, round(float(t) * self.T), return_type='drift')
        return f

    def g(self, t: float, x):
        return torch.zeros_like(x)
