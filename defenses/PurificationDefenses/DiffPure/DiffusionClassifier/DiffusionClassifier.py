import torch
from torch import nn, Tensor
from ..model import get_unet
from ..sampler import DiffusionSde
from torch.autograd import Function
from typing import Tuple, List
from tqdm import tqdm
from .DiffusionClassifierBase import DiffusionClassifierSingleHeadBase


class DiffusionClassifier(DiffusionClassifierSingleHeadBase):
    def __init__(
        self,
        unet: nn.Module = None,
        beta: Tensor = torch.linspace(0.1 / 1000, 20 / 1000, 1000),
        num_classes=10,
        ts: Tensor = torch.arange(1000),
        optimize_eps=0.5,
        *args,
        **kwargs
    ):
        unet = get_unet(mode="cifar")[0] if unet is None else unet
        super(DiffusionClassifier, self).__init__(unet, *args, **kwargs)
        self.unet = unet
        alpha = 1 - beta
        self.alpha_bar = alpha.cumprod(dim=0).to(self.device)
        self.T = 1000
        self.ts = ts.to(self.device)
        self._init()

        # storing
        self.num_classes = num_classes
        self.unet_criterion = nn.MSELoss()
        self.optimize_eps = optimize_eps

    def partial(self, x: Tensor, class_id: int or None = None, coefficient: int = 1, **kwargs):
        """
        d logit(x, y) / dx
        :param x: in range [0, 1]
        :param class_id:
        :param coefficient:
        :return: d logit(x, y) / dx
        """
        return self.unet_loss_with_grad(x, class_id, self.ts, coefficient, **kwargs)

    @torch.no_grad()
    def unet_loss_without_grad(
        self,
        x: Tensor,
        y: int or Tensor or None = None,
    ) -> Tensor:
        """
        Calculate the diffusion loss
        :param x: in range [0, 1]
        """
        t = self.ts
        y = torch.tensor([y], device=self.device) if type(y) is int else y
        y = y.repeat(t.numel()) if y is not None else None
        x = x.repeat(t.numel(), 1, 1, 1)
        x = self.transform(x)
        noise = torch.randn_like(x)
        noised_x = (
            torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1) * x
            + torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1, 1) * noise
        )
        pre = self.unet(noised_x, t, y)[:, :3, :, :]
        target = noise
        loss = self.unet_criterion(pre, target)
        return loss

    @torch.enable_grad()
    def unet_loss_with_grad(
        self, x: Tensor, y: int or None = None, t: Tensor = None, coefficient=1, batchsize=128, create_graph=False
    ):
        """
        :param x: in range [0, 1]
        """
        t = t.split(batchsize, dim=0)
        total_loss = 0
        for tensor_t in t:
            size = tensor_t.shape[0]
            now_x = (self.transform(x)).repeat(size, 1, 1, 1)
            now_y = torch.tensor([y], device=self.device).repeat(size) if y is not None else None
            noise = torch.randn_like(now_x)
            noised_x = (
                torch.sqrt(self.alpha_bar[tensor_t]).view(-1, 1, 1, 1) * now_x
                + torch.sqrt(1 - self.alpha_bar[tensor_t]).view(-1, 1, 1, 1) * noise
            )
            pre = self.unet(noised_x, tensor_t, now_y)[:, :3, :, :]
            target = noise
            loss = self.unet_criterion(pre, target)
            loss = loss * coefficient * tensor_t.shape[0] / batchsize
            loss.backward(create_graph=create_graph)
            # grad = torch.autograd.grad(loss, x, create_graph=create_graph)
            total_loss += loss.item()
        total_loss = total_loss / (1000 / batchsize)
        x.grad = x.grad / (1000 / batchsize)
        return total_loss

    @torch.enable_grad()
    def optimize_back(
        self, x: Tensor, y: Tensor or int = None, iter_step=10, create_graph=False, norm="L2"
    ) -> Tuple[Tensor]:
        """
        batchsize = 1
        For security, do not support inplace anymore.
        """
        x = x.detach().clone()
        x.requires_grad = True
        optimizer = torch.optim.Adam([x], lr=0.1)
        for _ in range(iter_step):
            t = torch.randint(low=0, high=1000, size=(1,), device=self.device)
            optimizer.zero_grad()
            self.unet_loss_with_grad(x, y, t)
            print(x.grad)
            optimizer.step()
        x.requires_grad = False
        return x
        # eps = self.optimize_eps
        # eps = 5
        # if not create_graph:
        #     x = x.detach().clone()  # do not need preserve computational graph
        # momentum = torch.zeros_like(x)
        # ori_x = x.clone()  # for clamp
        # # step_size = eps / iter_step
        # step_size = 0.01
        # for step in range(iter_step):
        #     if not create_graph:
        #         x.requires_grad = True
        #         # t = self.ts
        #         t = torch.randint(low=0, high=1000, size=(1,), device=self.device)
        #         self.unet_loss_with_grad(x, y, t)
        #         grad = x.grad.clone()
        #         if norm == 'Linf':
        #             momentum = momentum - grad / torch.norm(grad, p=1)
        #         elif norm == 'L2':
        #             momentum = momentum - grad / torch.norm(grad, p=2)
        #         x.requires_grad = False
        #         with torch.no_grad():
        #             if norm == 'Linf':
        #                 x = x + step_size * momentum.sign()
        #             elif norm == 'L2':
        #                 x = x + step_size * momentum
        #             x = clamp(x)
        #             if norm == 'Linf':
        #                 x = clamp(x, ori_x - eps, ori_x + eps)
        #             elif norm == 'L2':
        #                 difference = x - ori_x
        #                 distance = torch.norm(difference.view(difference.shape[0], -1), p=2, dim=1)
        #                 mask = distance > eps
        #                 if torch.sum(mask) > 0:
        #                     difference[mask] = difference[mask] / distance[mask].view(torch.sum(mask), 1, 1, 1) * eps
        #                     x = ori_x + difference
        #     else:  # second order derivative, note that do not modify any attribute, keep graph
        #         t = torch.arange(self.noise_scale, device=self.device)
        #         self.unet_loss_with_grad(x, y, t, create_graph=True)
        #         grad = x.grad.clone()
        #         x = x - step_size * grad.sign()
        #         x = clamp(x)
        #         x = clamp(x, ori_x - eps, ori_x + eps)
        # x.grad = None
        # if create_graph:
        #     return x
        # return x.detach()

    @torch.no_grad()
    def delta(self, x: Tensor, y: int or Tensor or None = None) -> Tensor:
        """

        :param x: in range [0, 1]
        :param y: int value.
        :param repeat_time: only 1. Don't need more.
        :return:
        """
        # diffusion training loss
        # t = torch.arange(start=0, end=self.noise_scale, device=self.device)
        t = torch.randint(low=20, high=980, size=(1,), device=self.device)
        tensor_t = t
        if type(y) is int:
            y = torch.tensor([y], device=self.device)
        if y is not None:
            y = y.repeat(t.shape[0])
        x = x.repeat(t.shape[0], 1, 1, 1)
        x = self.transform(x)
        # start
        weight = 1 - self.alpha_bar
        weight = weight[tensor_t]
        noise = torch.randn_like(x)
        noised_x = (
            torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1) * x
            + torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1, 1) * noise
        )
        if y is not None:
            pre = (1 + 100) * self.unet(noised_x, tensor_t, y)[:, :3, :, :] - 100 * self.unet(noised_x, tensor_t)[
                :, :3, :, :
            ]
        else:
            pre = self.unet(noised_x, tensor_t)[:, :3, :, :]
        result = pre - noise
        result = result.permute(1, 2, 3, 0) @ weight
        return result.unsqueeze(0)

    @torch.no_grad()
    def generation(
        self,
        class_id: int or None = None,
        total_images=1,  # total generation configuration
        lr=2e-2,
        iter_each_sample=1000,  # sampling schedules
        img_shape=(1, 32, 32),  # specific generation configuration
    ) -> List[Tensor]:
        results = []
        for _ in range(total_images):
            x = torch.randn(1, *img_shape, device=self.device)
            x = x * 0.5 + 0.5
            x.requires_grad = True
            optimizer = torch.optim.Adam([x], lr=lr)
            for _ in tqdm(range(iter_each_sample)):
                optimizer.zero_grad()
                print(self.partial(x, class_id, batchsize=128, coefficient=1))
                optimizer.step()
            x.grad = None
            x.requires_grad = False
            x = torch.clamp(x, min=0, max=1)
            results.append(x)
        return results


class DiffusionClassifierFunction(Function):
    """
    batchsize should be 1
    """

    bpda = False

    @staticmethod
    def forward(ctx, x: Tensor):
        assert x.shape[0] == 1, "batch size should be 1"
        x = x.detach()  # because we will do some attribute modification
        if DiffusionClassifierFunction.bpda:
            x = DiffusionClassifierFunction.classifier.optimize_back(x, y=0)
        x.requires_grad = True
        logit = []
        dlogit_dx = []
        for class_id in range(DiffusionClassifierFunction.classifier.num_classes):
            x.grad = None
            with torch.enable_grad():
                logit.append(DiffusionClassifierFunction.classifier.partial(x, class_id, 1))
                grad = x.grad.clone()
                dlogit_dx.append(grad)
            x.grad = None
        logit = torch.tensor(logit, device=torch.device("cuda")).unsqueeze(0)  # 1, num_classes
        logit = logit * -1
        ctx.dlogit_dx = [i * -1 for i in dlogit_dx]
        return logit

    @staticmethod
    def backward(ctx, grad_logit, lower_bound=True):
        """
        :param ctx:
        :param grad_logit: 1, num_classes
        :return:
        """
        dlogit_dx = ctx.dlogit_dx
        dlogit_dx = torch.stack(dlogit_dx)  # num_classes, *x_shape
        dlogit_dx = dlogit_dx.permute(1, 2, 3, 4, 0)  # *x_shape, num_classes
        if lower_bound:
            max_grad = torch.max(torch.abs(grad_logit))
            grad_logit[:, 0] = grad_logit[:, 0] + 0
        grad = dlogit_dx @ grad_logit.squeeze()
        return grad


class DiffusionClassifierSecondOrderFunction(Function):
    """
    direct attack
    """

    @staticmethod
    def forward(ctx, x: Tensor):
        x = x.clone().detach()
        original_x = x.clone()
        ctx.original_x = original_x
        assert x.shape[0] == 1, "batch size should be 1"
        assert x.is_leaf, "x must be leaf variable"
        # optimize back
        x = DiffusionClassifierSecondOrderFunction.classifier.optimize_back(x, y=0)
        x.requires_grad = True
        logit = []
        dlogit_dx = []
        for class_id in range(DiffusionClassifierSecondOrderFunction.classifier.num_classes):
            x.grad = None
            with torch.enable_grad():
                logit.append(DiffusionClassifierSecondOrderFunction.classifier.partial(x, class_id, 1))
                grad = x.grad.clone()
                dlogit_dx.append(grad)
            x.grad = None
        logit = torch.tensor(logit, device=torch.device("cuda")).unsqueeze(0)  # 1, num_classes
        logit = logit * -1
        ctx.dlogit_dx = [i * -1 for i in dlogit_dx]
        return logit

    @staticmethod
    def backward(ctx, grad_logit):
        """
        :param ctx:
        :param grad_logit: 1, num_classes
        :return:
        """
        dlogit_dx = ctx.dlogit_dx
        dlogit_dx = torch.stack(dlogit_dx)  # num_classes, *x_shape
        dlogit_dx = dlogit_dx.permute(1, 2, 3, 4, 0)  # *x_shape, num_classes
        grad = dlogit_dx @ grad_logit.squeeze()  # *x.shape
        # get the d x_hat / d_x
        original_x = ctx.original_x
        original_x.requires_grad = True
        with torch.enable_grad():
            x = DiffusionClassifierSecondOrderFunction.classifier.optimize_back(
                original_x,
                y=0,
                create_graph=True,
            )
            x.backward(grad.detach())
        grad = original_x.grad
        return grad


class RobustDiffusionClassifier(nn.Module):
    def __init__(self, bpda=False, likelihood_maximization=False, diffpure=False, second_order=False, *args, **kwargs):
        """

        :param bpda:  BPDA attack?
        :param likelihood_maximization: See our paper for detail
        :param diffpure: DiffPure before diffusion classifier.
        :param second_order: get the exact gradient
        """
        super(RobustDiffusionClassifier, self).__init__()
        assert not (bpda and second_order), "When direct get gradient, you cannot use bpda"
        self.function = DiffusionClassifierSecondOrderFunction if second_order else DiffusionClassifierFunction
        self.function.classifier = DiffusionClassifier(*args, **kwargs)
        self.unet = self.function.classifier.unet
        self.diffusion_classifier = self.function.classifier
        self.function.bpda = bpda
        self.likelihood_maximization = likelihood_maximization
        self.diffpure = DiffusionSde(grad_checkpoint=True) if diffpure else None
        self._init()

    def _init(self):
        try:
            self.unet.load_state_dict(torch.load("./ema_1100.pt"))
        except:
            print(
                "Please provide us checkpoint of a conditional diffusion model. "
                "Now the parameter of diffusion model is random."
            )
        self.eval().to(torch.device("cuda")).requires_grad_(False)

    def forward(self, x):
        if self.diffpure is not None:
            x = self.diffpure(x)
        if x.requires_grad is False:  # eval mode, prediction
            return self.diffusion_classifier.forward(x, likelihood_maximization=self.likelihood_maximization)
        # crafting adversarial patches, requires_grad mode
        return self.function.apply(x)
