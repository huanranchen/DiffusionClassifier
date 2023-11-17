import torch
from torch import nn, Tensor
from torch.autograd import Function
from typing import Tuple, Any, List, Callable
import random


class DiffusionClassifierSingleHeadBase(nn.Module):
    def __init__(
        self,
        unet: nn.Module,
        device: torch.device = torch.device("cuda"),
        transform: Callable = lambda x: (x - 0.5) * 2,
        num_classes: int = 10,
        target_class: List[int] = None,
        share_noise: bool = False,
        precision: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.unet = unet
        self.device = device
        self.transform = transform
        self.precision = precision
        self._init()
        self.target_class = target_class if target_class is not None else list(range(num_classes))
        self.num_classes = num_classes
        self.share_noise = share_noise

    def _init(self):
        self.eval().requires_grad_(False)
        self.to(self.device)
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                module.weight.data = module.weight.data.to(self.precision)
                if module.bias is not None:
                    module.bias.data = module.bias.data.to(self.precision)

    def get_one_instance_prediction(self, x: Tensor) -> Tensor:
        """
        :param x: 1, C, H, D
        :return D
        """
        loss = []
        g = torch.Generator(device=self.device) if self.share_noise else torch.cuda.default_generators[0]
        seed = random.randint(1, 99999)
        for class_id in self.target_class:
            if self.share_noise:
                g.manual_seed(seed)
            loss.append(self.unet_loss_without_grad(x, class_id, generator=g))
        loss = torch.tensor(loss, device=self.device)
        loss = loss * -1  # convert into logit where greatest is the target
        return loss

    def forward(self, x: Tensor) -> Tensor:
        xs = x.split(1)  # 1, C, H, D
        y = []
        for now_x in xs:
            y.append(self.get_one_instance_prediction(now_x))
        y = torch.stack(y)  # N, num_classes
        return y

    def unet_loss_without_grad(
        self, x: Tensor, y: int or Tensor = None, generator: torch.Generator = torch.default_generator
    ) -> float:
        pass


class DiffusionClassifierSingleHeadBaseFunction(Function):
    """
    batchsize should be 1
    """

    classifier = None

    @staticmethod
    def forward(ctx: Any, *args, **kwargs):
        classifier = DiffusionClassifierSingleHeadBaseFunction.classifier
        x = args[0]
        target_class_tensor = DiffusionClassifierSingleHeadBaseFunction.classifier.target_class
        ctx.target_class_tensor = target_class_tensor
        assert x.shape[0] == 1, "batch size should be 1"
        x = x.detach()  # because we will do some attribute modification
        x.requires_grad = True
        logit = []
        dlogit_dx = []
        share_noise = classifier.share_noise
        g = torch.Generator(device=x.device) if share_noise else torch.cuda.default_generators[0]
        seed = random.randint(1, 99999)
        if share_noise:
            g.manual_seed(seed)
        for class_id in classifier.target_class:
            x.grad = None
            with torch.enable_grad():
                logit.append(classifier.unet_loss_with_grad(x, class_id, generator=g))
                grad = x.grad.clone()
                dlogit_dx.append(grad)
            x.grad = None
        logit = torch.tensor(logit, device=torch.device("cuda")).unsqueeze(0)  # 1, num_classes
        logit = logit * -1
        ctx.dlogit_dx = [i * -1 for i in dlogit_dx]
        result = logit
        return result

    @staticmethod
    def backward(ctx: Any, grad_logit, lower_bound=False):
        """
        :param ctx:
        :param grad_logit: 1, num_classes
        :return:
        """
        dlogit_dx = ctx.dlogit_dx
        dlogit_dx = torch.stack(dlogit_dx)  # num_classes, *x_shape
        dlogit_dx = dlogit_dx.permute(1, 2, 3, 4, 0)  # *x_shape, num_classes
        grad = dlogit_dx @ grad_logit.squeeze()
        return grad


class DiffusionClassifierSingleHeadBaseWraped(nn.Module):
    def __init__(self, dc: DiffusionClassifierSingleHeadBase):
        super(DiffusionClassifierSingleHeadBaseWraped, self).__init__()
        DiffusionClassifierSingleHeadBaseFunction.classifier = dc
        self.unet = dc.unet
        self.dc = dc
        self.eval().to(torch.device("cuda")).requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        if x.requires_grad is False:  # eval mode, prediction
            logit = self.dc.forward(x)
        else:
            # crafting adversarial patches, requires_grad mode
            logit = DiffusionClassifierSingleHeadBaseFunction.apply(x)
        return logit
