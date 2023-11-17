import torch
from torch import nn, Tensor
import torchdiffeq
import math
from ..model import get_unet
from typing import Callable, Tuple


class SBGC(nn.Module):
    def __init__(
        self,
        unet: nn.Module = None,
        beta: Tensor = torch.linspace(0.1 / 1000, 20 / 1000, 1000),
        T=1000,
        device=torch.device("cuda"),
        img_shape=(3, 32, 32),
        num_classes=10,
        transform: Callable = lambda x: (x - 0.5) * 2,
        cfg: float = 0,
        mode: str = "Skilling-Hutchinson trace estimator",
    ):
        assert mode in ["Skilling-Hutchinson trace estimator", "huanran approximator"]
        self.mode = mode
        self.cfg = cfg
        super(SBGC, self).__init__()
        if unet is None:
            unet, beta, img_shape = get_unet()
        self.unet = unet
        alpha = 1 - beta
        self.alpha_bar = alpha.cumprod(dim=0).to(device)
        self.beta = beta.to(device) * T
        self.T = T
        self.device = device
        self.img_shape = img_shape
        self._init()
        self.num_classes = num_classes
        self.transform = transform

    def _init(self):
        self.diffusion_kwargs = dict()
        self.eval().requires_grad_(False)
        self.to(self.device)
        self.state_size = self.img_shape[0] * self.img_shape[1] * self.img_shape[2]
        # Rademacher distribution is better than Gaussian distribution
        self.noise = torch.randint(size=(self.state_size,), low=0, high=2, device=self.device) * 2 - 1

    def diffusion_forward(self, x: torch.Tensor, t: int):
        assert len(x.shape) == 2, "x should be N, D"
        N = x.shape[0]
        diffusion = torch.sqrt(self.beta[t]).view(1, 1).repeat(N, x.shape[1])
        drift = -0.5 * self.beta[t] * x
        return drift, diffusion

    def reverse_diffusion_forward(self, x: Tensor, t: int, return_score=False):
        N = x.shape[0]
        tensor_t = torch.zeros((N,), device=self.device, dtype=torch.int) + t
        forward_drift, forward_diffusion = self.diffusion_forward(x, t)
        diffusion = forward_diffusion
        y = self.diffusion_kwargs["y"] if "y" in self.diffusion_kwargs else None
        if y is None or self.cfg == 0:
            pre = self.unet(x.view(N, *self.img_shape), tensor_t, y)[:, :3, :, :].view(N, -1)
        else:
            pre_unconditional = self.unet(x.view(N, *self.img_shape), tensor_t)[:, :3, :, :].view(N, -1)
            pre_conditional = self.unet(x.view(N, *self.img_shape), tensor_t, y)[:, :3, :, :].view(N, -1)
            pre = (1 + self.cfg) * pre_conditional - self.cfg * pre_unconditional
        score = -pre / torch.sqrt(1 - self.alpha_bar[t])
        drift = forward_drift - 0.5 * diffusion**2 * score
        if return_score:
            return drift, score
        return drift

    def skilling_hutchinson_trace_estimator(self, t, x, estimation_times=1):
        expanded_x = x.repeat(estimation_times, 1)
        if "y" in self.diffusion_kwargs:
            ori_y = self.diffusion_kwargs["y"].clone()
            self.diffusion_kwargs["y"] = self.diffusion_kwargs["y"].repeat(estimation_times)
        with torch.enable_grad():
            expanded_x.requires_grad_(True)
            f = self.reverse_diffusion_forward(expanded_x, min(round(float(t) * self.T), self.T - 1))
            noise = self.noise
            loss = torch.sum(noise * f)  # N
            loss.backward()
            grad = expanded_x.grad.clone()
            trace = torch.sum(grad * noise, dim=1)  # N
            expanded_x.requires_grad_(False)
            expanded_x.grad = None
        if "y" in self.diffusion_kwargs:
            self.diffusion_kwargs["y"] = ori_y
        f, trace = f.view(estimation_times, *x.shape), trace.view(estimation_times, -1)
        f = f.mean(0)
        trace = trace.mean(0)
        return torch.cat([f, trace.unsqueeze(1)], dim=1)  # N, state_size+1

    def huanran_approximator(self, t, x):
        f, score = self.reverse_diffusion_forward(x, min(round(float(t) * self.T), self.T - 1), return_score=True)
        dlikelihood_dt = torch.sum((score * f), dim=1, keepdim=True)
        return torch.cat([f, dlikelihood_dt], dim=1), score, f  # N, state_size+1

    def f(self, t: Tensor or int, x: Tensor) -> Tensor:
        """
        :param t:
        :param x: N, state_size+1. Because we concatenate the trace of f_\theta(x) here
        """
        x = x[:, :-1]
        if self.mode == "Skilling-Hutchinson trace estimator":
            result = self.skilling_hutchinson_trace_estimator(t, x)
        else:
            result, _, _ = self.huanran_approximator(t, x)
        return result

    def compute_likelihood(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        :param x: N, D
        return N
        """
        x = self.transform(x)
        batch_size = x.shape[0]
        x = x.view((batch_size, self.state_size))
        x = torch.cat([x, torch.zeros((batch_size, 1), device=x.device)], dim=1)
        t_eval = torch.linspace(1e-5, 1.0 - 1e-5, 2, device=self.device)
        x = torchdiffeq.odeint(self.f, x, t_eval, rtol=1e-5, atol=1e-5, method="dopri5", options=dict(step_size=0.001))
        x = x[-1]  # N, self.state_size + 1
        delta = x[:, -1]
        latent = x[:, :-1]
        input_dim = self.state_size
        latent_log_likelihood = -torch.sum(latent**2, dim=1) / 2 - input_dim * math.log(2 * math.pi) / 2
        likelihood: Tensor = delta + latent_log_likelihood
        bpd = -likelihood / self.state_size / math.log(2) + (8 - 1)
        return bpd, likelihood, latent

    def forward_one_instance(self, x: Tensor):
        """
        :param x: 1, *img_shape
        """
        self.diffusion_kwargs["y"] = torch.arange(self.num_classes, device=self.device)
        bpd, likelihoods, _ = self.compute_likelihood(x.view(x.shape[0], -1).expand(self.num_classes, self.state_size))
        self.diffusion_kwargs.clear()
        return -bpd

    def forward(self, x: Tensor):
        """
        :param x: batch_size, *img_shape
        """
        xs = x.split(1, dim=0)
        result = []
        for now_x in xs:
            result.append(self.forward_one_instance(now_x))
        return torch.stack(result)

    @torch.no_grad()
    def sample(self, latent: Tensor = None, batch_size=64):
        if latent is None:
            latent = torch.randn(batch_size, *self.img_shape)
        batch_size = latent.shape[0]
        x = latent.view(batch_size, self.state_size).to(self.device)
        x = torch.cat([x, torch.zeros((batch_size, 1), device=self.device)], dim=1)
        t_eval = torch.linspace(1 - 1e-3, 0 - 1e-3 + 1e-5, 2, device=self.device)
        x = torchdiffeq.odeint(self.f, x, t_eval, rtol=1e-5, atol=1e-5, method="dopri5", options=dict(step_size=0.001))
        x = x[-1]
        x = x[:, :-1]
        x = x.view(batch_size, *self.img_shape)
        return self.convert(x)

    @staticmethod
    def convert(x: Tensor) -> Tensor:
        return (x + 1) / 2
