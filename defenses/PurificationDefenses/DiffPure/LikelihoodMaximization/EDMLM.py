import torch
from torch import nn
from utils.seed import set_seed
from .LikelihoodMaximizationBase import LikelihoodMaximizationBase
from .utils import *
from optimizer import DifferentiableAdam


class EDMEulerIntegralLM(LikelihoodMaximizationBase):
    def __init__(self, edm_dc: nn.Module, *args, **kwargs):
        super(EDMEulerIntegralLM, self).__init__(edm_dc, *args, **kwargs)

    def optimize_back(self, *args, **kwargs):
        return self.likelihood_maximization_T1(*args, **kwargs)

    @torch.enable_grad()
    def likelihood_maximization_T1(
        self,
        x: Tensor,
        iter_step=5,
        eps=100,
        t_min=0.4,
        t_max=0.6,
        lr=0.1,
    ) -> Tensor:
        set_seed(1)
        x = self.transform(x)
        ori_x = x.clone()
        xs = list(x.split(1, dim=0))
        for i in xs:
            i.requires_grad_(True)
        optimizer = torch.optim.Adam(xs, lr=lr)
        for step in range(iter_step):
            x = torch.cat(xs, dim=0)
            t = self.uniform_noise((x.shape[0],), begin=t_min, end=t_max)
            epsilon = torch.randn_like(ori_x)
            noised = x + t.view(-1, 1, 1, 1) * epsilon
            denoised = self.dc.unet(noised, t)
            loss = torch.nn.functional.mse_loss(denoised, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for i, xi in enumerate(xs):
                    l2_clamp(xi, ori_x[i].unsqueeze(0), eps=eps * 2, inplace=True)
        for i in xs:
            i.requires_grad_(False)
        x = torch.cat(xs, dim=0)
        print((torch.norm((x - ori_x).view(ori_x.shape[0], -1), dim=1) / 2).mean())
        x = (x + 1) / 2
        x = torch.clamp(x, min=0, max=1)
        return x.detach()

    @torch.enable_grad()
    def likelihood_maximization_T1_create_graph(
        self,
        x: Tensor,
        iter_step=5,
        t_min=0.4,
        t_max=0.6,
    ) -> Tensor:
        assert x.shape[0] == 1, "Batch size should be 1"
        second_order_mode = x.requires_grad
        # set_seed(1)
        ori_x = x.detach().clone()
        # print(second_order_mode, x.requires_grad)
        if not second_order_mode:
            x.requires_grad_(True)
        optimizer = DifferentiableAdam([x], lr=0.05)
        for step in range(iter_step):
            now_x = self.transform(x)
            t = self.uniform_noise((now_x.shape[0],), begin=t_min, end=t_max)
            epsilon = torch.randn_like(ori_x)
            noised = now_x + t.view(-1, 1, 1, 1) * epsilon
            denoised = self.dc.unet(noised, t)
            loss = torch.nn.functional.mse_loss(denoised, now_x)
            grad = torch.autograd.grad(loss, x, create_graph=second_order_mode)[0]
            x.grad = grad
            x = optimizer.step()[0]["params"][0]
        # print((torch.norm((x - ori_x).view(ori_x.shape[0], -1), dim=1)).mean())
        if not second_order_mode:
            x = x.detach()
        x = torch.clamp(x, min=0, max=1)
        return x

    @torch.enable_grad()
    def likelihood_maximization_in_expectation(
        self, x: Tensor, iter_step=5, create_graph=False, eps=0.5, norm="L2"
    ) -> Tensor:
        x = (x - 0.5) * 2
        if not create_graph:
            x = x.detach().clone()  # do not need preserve computational graph
        momentum = torch.zeros_like(x)
        ori_x = x.clone()  # for clamp
        step_size = eps / iter_step
        x.requires_grad = True
        for step in range(iter_step):
            self.dc.unet_loss_with_grad(x)
            grad = x.grad.clone()
            if norm == "Linf":
                momentum = momentum - grad / torch.norm(grad, p=1)
            elif norm == "L2":
                momentum = momentum - grad / torch.norm(grad, p=2)
            x.requires_grad = False
            with torch.no_grad():
                if norm == "Linf":
                    x = x + step_size * momentum.sign()
                elif norm == "L2":
                    x = x + step_size * momentum
                x = clamp(x)
                if norm == "Linf":
                    x = clamp(x, ori_x - eps, ori_x + eps)
                elif norm == "L2":
                    x = l2_clamp(x, ori_x, eps=eps * 2, inplace=False)
        x.grad = None
        if create_graph:
            return x
        x.requires_grad = False
        x = (x + 1) / 2
        return x.detach()

    def one_step_denoise(self, x: Tensor, normalize=True, sigma=0.5) -> Tensor:
        """
        x: In range (0, 1)
        """
        x = (x - 0.5) * 2 if normalize else x
        x0 = self.unet(x, sigma=torch.zeros((x.shape[0],), device=x.device) + sigma)
        x0 = x0 / 2 + 0.5 if normalize else x0
        return x0

    def uniform_noise(self, *args, begin: float = 0.0, end: float = 1.0, **kwargs):
        x = torch.rand(*args, **kwargs, device=self.device)
        x = x * (end - begin) + begin
        return x
