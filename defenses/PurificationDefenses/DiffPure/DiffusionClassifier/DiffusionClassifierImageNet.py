import torch
from torch import nn
from defenses.PurificationDefenses.DiffPure.model import get_unet
from torch import Tensor
from defenses.PurificationDefenses.DiffPure.sampler.sde import DiffusionSde
from defenses.PurificationDefenses.DiffPure.utils import clamp
from torch.autograd import Function
from typing import Tuple, List, Any
from tqdm import tqdm

__all__ = ['DiffusionAsClassifierImageNetWraped']

__IGNORED_CLASS_LOGIT__ = - 999


class DiffusionAsClassifierImageNet(nn.Module):
    def __init__(self,
                 unet: nn.Module = None,
                 beta: Tensor = None,
                 target_class: Tuple[int] = (0, 100, 200, 300, 400, 500, 600, 700, 800, 900),
                 ts: Tensor = torch.arange(1000, device=torch.device('cuda')),
                 num_classes: int = 1000,
                 remain_ignored_class_logit=False,
                 ):
        self.remain_ignored_class_logit = remain_ignored_class_logit
        super(DiffusionAsClassifierImageNet, self).__init__()
        self.device = torch.device('cuda')
        self.tensor_target_class = torch.tensor(target_class, device=self.device)
        if unet is None:
            unet, beta, img_shape = get_unet(mode='imagenet')
        if beta is None:
            beta = torch.linspace(0.1 / 1000, 20 / 1000, 1000, device=self.device)
        self.unet = unet
        alpha = (1 - beta)
        self.alpha_bar = alpha.cumprod(dim=0).to(self.device)
        self.T = 1000
        self.init()

        # storing
        self.target_class = list(target_class)
        self.target_class_tensor = torch.tensor(self.target_class, device=self.device)  # num_classes
        self.unet_criterion = nn.MSELoss()
        self.ts = ts
        self.num_classes = num_classes

    def init(self):
        self.eval().requires_grad_(False)
        self.to(self.device)
        self.transform = lambda x: (x - 0.5) * 2

    def get_one_instance_prediction(self, x: Tensor, optimize=False) -> Tensor:
        """
        :param x: 1, C, H, D
        :return:
        """
        if optimize:
            x = self.optimize_back(x, y=torch.tensor([0], device=self.device))
            # x = self.optimize_back(x)
        loss = []
        for class_id in self.target_class:
            loss.append(self.unet_loss_without_grad(x, class_id))
        loss = torch.tensor(loss, device=self.device)
        print(loss)
        loss = loss * -1  # convert into logit where greatest is the target
        return loss

    def forward(self, x: Tensor, optimize=False) -> Tensor:
        '''
        :param x: N, C, H, D
        :return: N
        '''
        xs = x.split(1)  # 1, C, H, D
        y = []
        for now_x in xs:
            y.append(self.get_one_instance_prediction(now_x, optimize=optimize))
        y = torch.stack(y)  # N, target_class
        result = torch.zeros((y.shape[0], self.num_classes), device=self.device) - __IGNORED_CLASS_LOGIT__
        result[:, self.target_class_tensor] = y
        if not self.remain_ignored_class_logit:
            result = torch.cat([i[:, self.tensor_target_class] for i in result.split(1, dim=0)], dim=0)
        return result

    def partial(self, x: Tensor, class_id: int or None = None, coefficient: int = 1):
        """
        d loss(x, y) / dx
        :param x: in range [0, 1]
        :param class_id:
        :param coefficient:
        :return: d loss(x, y) / dx
        """
        return self.unet_loss_with_grad(x,
                                        class_id,
                                        torch.arange(self.noise_scale, device=self.device),
                                        coefficient)

    @torch.no_grad()
    def unet_loss_without_grad(self, x: Tensor,
                               y: int or None = None,
                               coefficient=1,
                               batchsize=32):
        """
        :param x: in range [0, 1]
        :return:
        """
        t = self.ts
        t = t.split(batchsize, dim=0)
        total_loss = 0
        for tensor_t in t:
            size = tensor_t.shape[0]
            now_x = (self.transform(x)).repeat(size, 1, 1, 1)
            if y is not None:
                now_y = torch.tensor([y], device=self.device).repeat(size)
            noise = torch.randn_like(now_x)
            noised_x = torch.sqrt(self.alpha_bar[tensor_t]).view(-1, 1, 1, 1) * now_x + \
                       torch.sqrt(1 - self.alpha_bar[tensor_t]).view(-1, 1, 1, 1) * noise
            if y is not None:
                pre = self.unet(noised_x, tensor_t, now_y)[:, :3, :, :]
            else:
                pre = self.unet(noised_x, tensor_t)[:, :3, :, :]
            target = noise
            loss = self.unet_criterion(pre, target)
            loss = loss * coefficient * tensor_t.shape[0] / batchsize
            total_loss += loss
        total_loss = total_loss / (1000 / batchsize)
        return total_loss

    @torch.enable_grad()
    def unet_loss_with_grad(self,
                            x: Tensor,
                            y: int or None = None,
                            t: Tensor = None,
                            coefficient=1,
                            batchsize=4,
                            create_graph=False):
        """
        :param x: in range [0, 1]
        :return:
        """
        t = t.split(batchsize, dim=0)
        total_loss = 0
        for tensor_t in t:
            size = tensor_t.shape[0]
            now_x = (self.transform(x)).repeat(size, 1, 1, 1)
            if y is not None:
                now_y = torch.tensor([y], device=self.device).repeat(size)
            noise = torch.randn_like(now_x)
            noised_x = torch.sqrt(self.alpha_bar[tensor_t]).view(-1, 1, 1, 1) * now_x + \
                       torch.sqrt(1 - self.alpha_bar[tensor_t]).view(-1, 1, 1, 1) * noise
            if y is not None:
                pre = self.unet(noised_x, tensor_t, now_y)[:, :3, :, :]
            else:
                pre = self.unet(noised_x, tensor_t)[:, :3, :, :]
            target = noise
            loss = self.unet_criterion(pre, target)
            loss = loss * coefficient * tensor_t.shape[0] / batchsize
            loss.backward(create_graph=create_graph)
            total_loss += loss
        total_loss = total_loss / (1000 / batchsize)
        x.grad = x.grad / (1000 / batchsize)
        return total_loss

    @torch.enable_grad()
    def optimize_back(self, x: Tensor, y: Tensor or None = None,
                      eps=8 / 255, iter_step=10,
                      create_graph=False, ) -> Tuple[Tensor]:
        """
        batchsize = 1
        For security, do not support inplace anymore.
        """
        if not create_graph:
            x = x.detach().clone()  # do not need preserve computational graph
        momentum = torch.zeros_like(x)
        ori_x = x.clone()  # for clamp
        step_size = eps / iter_step
        losses = []
        for step in range(iter_step):
            if not create_graph:
                x.requires_grad = True
                t = torch.arange(self.noise_scale, device=self.device)
                losses.append(self.unet_loss_with_grad(x, y, t).item())
                grad = x.grad.clone()
                momentum = momentum - grad / torch.norm(grad, p=1)
                x.requires_grad = False
                with torch.no_grad():
                    x = x + step_size * momentum.sign()
                    x = clamp(x)
                    x = clamp(x, ori_x - eps, ori_x + eps)
            else:  # second order derivative, note that do not modify any attribute, keep graph
                t = torch.arange(self.noise_scale, device=self.device)
                losses.append(self.unet_loss_with_grad(x, y, t, create_graph=True).item())
                # print(x.grad)
                grad = x.grad.clone()
                x = x - step_size * grad.sign()
                x = clamp(x)
                x = clamp(x, ori_x - eps, ori_x + eps)
        x.grad = None
        if create_graph:
            return x
        return x.detach()

    @torch.no_grad()
    def generation(self, class_id: int or None = None, total_images=1,  # total generation configuration
                   step_size=100, noise_step_size=0.00025, iter_each_sample=1000,  # sampling schedules
                   img_shape=(3, 32, 32)  # specific generation configuration
                   ) -> List[Tensor]:
        results = []
        for _ in range(total_images):
            x = torch.randn(1, *img_shape, device=self.device)
            x = x * 0.5 + 0.5
            x.requires_grad = True
            optimizer = torch.optim.Adam([x], lr=2e-2)
            for _ in tqdm(range(iter_each_sample)):
                optimizer.zero_grad()
                print(self.partial(x, class_id))
                optimizer.step()
            x.grad = None
            x.requires_grad = False
            x = torch.clamp(x, min=0, max=1)
            results.append(x)
        return results


#
class DiffusionAsClassifierImageNetFunction(Function):
    """
    batchsize should be 1
    """
    classifier = None
    bpda_optimize_back = False

    @staticmethod
    def forward(ctx: Any, *args, **kwargs):
        x = args[0]
        target_class_tensor = DiffusionAsClassifierImageNetFunction.classifier.target_class_tensor
        ctx.target_class_tensor = target_class_tensor
        assert x.shape[0] == 1, 'batch size should be 1'
        x = x.detach()  # because we will do some attribute modification
        if DiffusionAsClassifierImageNetFunction.bpda_optimize_back:
            x = DiffusionAsClassifierImageNetFunction.classifier.optimize_back(x,
                                                                               y=torch.tensor([0]).cuda()
                                                                               )
        x.requires_grad = True
        logit = []
        dlogit_dx = []
        for class_id in DiffusionAsClassifierImageNetFunction.classifier.target_class:
            x.grad = None
            with torch.enable_grad():
                logit.append(DiffusionAsClassifierImageNetFunction.classifier.partial(x,
                                                                                      class_id, 1))
                grad = x.grad.clone()
                dlogit_dx.append(grad)
            x.grad = None
        logit = torch.tensor(logit, device=torch.device('cuda')).unsqueeze(0)  # 1, num_classes
        print(logit)
        logit = logit * -1
        ctx.dlogit_dx = [i * -1 for i in dlogit_dx]
        result = torch.zeros((1, 1000)).cuda() - 999
        result[:, target_class_tensor] = logit
        return result

    @staticmethod
    def backward(ctx: Any, grad_logit, lower_bound=False):
        """
        :param ctx:
        :param grad_logit: 1, num_classes
        :return:
        """
        grad_logit = grad_logit[:, ctx.target_class_tensor]
        dlogit_dx = ctx.dlogit_dx
        dlogit_dx = torch.stack(dlogit_dx)  # num_classes, *x_shape
        dlogit_dx = dlogit_dx.permute(1, 2, 3, 4, 0)  # *x_shape, num_classes
        # if lower_bound:
        #     max_grad = torch.max(torch.abs(grad_logit))
        #     grad_logit[:, 0] = grad_logit[:, 0] + 10
        grad = dlogit_dx @ grad_logit.squeeze()
        return grad


class DiffusionAsClassifierImageNetWraped(nn.Module):
    def __init__(self,
                 bpda_optimize_back=False,
                 optimize=False,
                 diffpure=False,
                 target_class: Tuple[int, ...] = (0, 100, 200, 300, 400, 500, 600, 700, 800, 900),
                 remain_ignored_class_logit=False,
                 ):
        super(DiffusionAsClassifierImageNetWraped, self).__init__()
        DiffusionAsClassifierImageNetFunction.classifier = DiffusionAsClassifierImageNet(
            target_class=target_class,
            remain_ignored_class_logit=remain_ignored_class_logit,
        )
        self.unet = DiffusionAsClassifierImageNetFunction.classifier.unet
        self.knnclassifier = DiffusionAsClassifierImageNetFunction.classifier
        if bpda_optimize_back:
            DiffusionAsClassifierImageNetFunction.bpda_optimize_back = True
        self.optimize = optimize
        self.diffpure = None
        if diffpure:
            self.diffpure = DiffusionSde(grad_checkpoint=True)

        self.eval().to(torch.device('cuda')).requires_grad_(False)

    def forward(self, x):
        if self.diffpure is not None:
            x = self.diffpure(x)
        if x.requires_grad is False:  # eval mode, prediction
            logit = self.knnclassifier.forward(x, optimize=self.optimize)
        else:
            # crafting adversarial patches, requires_grad mode
            logit = DiffusionAsClassifierImageNetFunction.apply(x)
        return logit
