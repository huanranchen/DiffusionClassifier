import torch
from torch import nn, Tensor
import numpy as np
from .BaseSampler import BaseSampler


class EDMStochasticSampler(BaseSampler):
    def __init__(self, unet: nn.Module,
                 *args, **kwargs,
                 ):
        super(EDMStochasticSampler, self).__init__(unet, *args, **kwargs)

    def sample(self, batch_size=64, img_shape=(3, 32, 32),
               y=None, randn_like=torch.randn_like,
               num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
               S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, ):
        latents = torch.randn((batch_size, *img_shape)).cuda()
        # Adjust noise levels based on what's supported by the network. Here I set it as the default of EDMPrecondNet
        # sigma_min and sigma_max
        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float, device=latents.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([self.unet.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

        # Main sampling loop.
        x_next = latents.to(torch.float) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
            x_cur = x_next
            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = self.unet.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
            # Euler step.
            denoised = self.unet(x_hat, t_hat, y).to(torch.float)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.unet(x_next, t_next, y).to(torch.float)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        x_next = x_next / 2 + 0.5
        return torch.clamp(x_next, min=0, max=1)

    def purify(self, x: Tensor, y=None,
               randn_like=torch.randn_like,
               num_steps=18, sigma_min=0.002, sigma_max=0.5, rho=7,
               S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
               add_noise=True):
        # y = torch.zeros((x.shape[0], ), device=x.device)
        x = (x - 0.5) * 2
        if add_noise:
            x = x + torch.randn_like(x) * sigma_max
        # Adjust noise levels based on what's supported by the network. Here I set it as the default of EDMPrecondNet
        # sigma_min and sigma_max
        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float, device=x.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([self.unet.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

        # Main sampling loop.
        x_next = x.to(torch.float)
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
            x_cur = x_next
            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = self.unet.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
            # Euler step.
            denoised = self.unet(x_hat, t_hat, y).to(torch.float)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur
            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.unet(x_next, t_next, y).to(torch.float)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        x_next = x_next / 2 + 0.5
        return torch.clamp(x_next, min=0, max=1)
