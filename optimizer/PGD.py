import torch
from torch.optim import Optimizer


class PGD(Optimizer):
    def __init__(self, params, lr=5e-3, maximum=True, epsilon=16 / 255):
        dampening = 0
        weight_decay = 0
        nesterov = False
        maximize = False
        momentum = 0
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, maximize=maximize)
        super(PGD, self).__init__(params, defaults)
        self.maximum = maximum
        self.lr = lr
        self.original_params = [p.clone() for p in params]
        self.epsilon = epsilon

    @torch.no_grad()
    def step(self, closure=None):
        if self.maximum:
            for group in self.param_groups:
                for i, p in enumerate(group['params']):
                    if p.grad is not None:
                        p.add_(self.lr * p.grad.sign())
                        p.clamp_(min=self.original_params[i] - self.epsilon,
                                 max=self.original_params[i] + self.epsilon)
        else:
            for group in self.param_groups:
                for i, p in enumerate(group['params']):
                    if p.grad is not None:
                        p.add_(-self.lr * p.grad.sign())
                        p.clamp_(min=self.original_params[i] - self.epsilon,
                                 max=self.original_params[i] + self.epsilon)
