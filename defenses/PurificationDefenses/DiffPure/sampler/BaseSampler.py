import torch
from torch import nn, Tensor
from abc import abstractmethod
from ..model import get_unet
from torch.utils.checkpoint import checkpoint


def define_checkpointed_unet(unet):
    class SubstituteUnet(torch.nn.Module):
        def __init__(self):
            super(SubstituteUnet, self).__init__()
            for attr in list(dir(unet)):
                if not attr.startswith('__') and not attr.startswith('_') and attr != 'forward':
                    setattr(self, attr, getattr(unet, attr))
            self.unet = unet

        def forward(self, *args, **kwargs):
            x = checkpoint(self.unet, *args, **kwargs)
            return x

    return SubstituteUnet()


class BaseSampler(nn.Module):
    def __init__(self, unet: nn.Module = None,
                 device=torch.device('cuda'),
                 grad_checkpoint=False):
        super(BaseSampler, self).__init__()
        unet = unet if unet is not None else get_unet()[0]
        self.unet = define_checkpointed_unet(unet) if grad_checkpoint else unet
        self.device = device
        self._model_init()

    def _model_init(self):
        self.eval().requires_grad_(False).to(self.device)

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass

    @abstractmethod
    def purify(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.purify(*args, **kwargs)
