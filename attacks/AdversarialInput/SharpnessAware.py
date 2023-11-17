import torch
from attacks.utils import *
from torch import nn
from typing import Callable, List
from .AdversarialInputBase import AdversarialInputAttacker


class MI_SAM(AdversarialInputAttacker):
    def __init__(self, model: List[nn.Module],
                 total_step: int = 10, random_start: bool = False,
                 step_size: float = 16 / 255 / 5,
                 criterion: Callable = nn.CrossEntropyLoss(),
                 targeted_attack=False,
                 mu: float = 1,
                 reverse_step_size: float = 16 / 255 / 15,
                 *args, **kwargs
                 ):
        self.models = model
        self.random_start = random_start
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        super(MI_SAM, self).__init__(model, *args, **kwargs)
        self.reverse_step_size = reverse_step_size

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(self, x, y, ):
        N = x.shape[0]
        original_x = x.clone()
        momentum = torch.zeros_like(x)
        if self.random_start:
            x = self.perturb(x)

        for _ in range(self.total_step):
            # --------------------------------------------------------------------------------#
            # first step
            ori_x = x.clone()
            x.requires_grad = True
            logit = 0
            for model in self.models:
                logit += model(x.to(model.device)).to(x.device)
            loss = self.criterion(logit, y)
            loss.backward()
            grad = x.grad
            x.requires_grad = False
            if self.targerted_attack:
                pass
            else:
                x -= self.reverse_step_size * grad.sign()
            # x.grad = None
            # --------------------------------------------------------------------------------#
            # second step
            x.requires_grad = True
            logit = 0
            for model in self.models:
                logit += model(x.to(model.device)).to(x.device)
            loss = self.criterion(logit, y)
            loss.backward()
            grad = x.grad.clone()
            x.requires_grad = False
            # update
            if self.targerted_attack:
                momentum = self.mu * momentum - grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                x = ori_x + self.step_size * momentum.sign()
            else:
                momentum = self.mu * momentum + grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                x = ori_x + self.step_size * momentum.sign()
            x = self.clamp(x, original_x)
        return x


class BIM_SAM(AdversarialInputAttacker):
    def __init__(self, model: List[nn.Module],
                 total_step: int = 10, random_start: bool = False,
                 step_size: float = 16 / 255 / 5,
                 criterion: Callable = nn.CrossEntropyLoss(),
                 targeted_attack=False,
                 mu: float = 1,
                 reverse_step_size: float = 16 / 255 / 15,
                 *args, **kwargs
                 ):
        self.models = model
        self.random_start = random_start
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        super(BIM_SAM, self).__init__(model, *args, **kwargs)
        self.reverse_step_size = reverse_step_size

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(self, x, y, ):
        N = x.shape[0]
        original_x = x.clone()
        if self.random_start:
            x = self.perturb(x)

        for _ in range(self.total_step):
            # --------------------------------------------------------------------------------#
            # first step
            ori_x = x.clone()
            x.requires_grad = True
            logit = 0
            for model in self.models:
                logit += model(x.to(model.device)).to(x.device)
            loss = self.criterion(logit, y)
            loss.backward()
            grad = x.grad
            x.requires_grad = False
            if self.targerted_attack:
                pass
            else:
                x -= self.reverse_step_size * grad.sign()
            # x.grad = None
            # --------------------------------------------------------------------------------#
            # second step
            x.requires_grad = True
            logit = 0
            for model in self.models:
                logit += model(x.to(model.device)).to(x.device)
            loss = self.criterion(logit, y)
            loss.backward()
            grad = x.grad.clone()
            x.requires_grad = False
            # update
            if self.targerted_attack:
                update = - grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                x = ori_x + self.step_size * update.sign()
            else:
                update = grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                x = ori_x + self.step_size * update.sign()
            x = self.clamp(x, original_x)
        return x


class MI_RAP(AdversarialInputAttacker):
    def __init__(self, model: List[nn.Module],
                 total_step: int = 400,
                 random_start: bool = False,
                 step_size: float = 2 / 255,
                 criterion: Callable = nn.CrossEntropyLoss(),
                 targeted_attack=False,
                 mu: float = 1,
                 reverse_step_size: float = 2 / 255,
                 reverse_step=4,
                 reverse_start_step=100,
                 *args, **kwargs
                 ):
        self.models = model
        self.random_start = random_start
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        super(MI_RAP, self).__init__(model, *args, **kwargs)
        self.reverse_step_size = reverse_step_size
        self.reverse_step = reverse_step
        self.reverse_start_step = reverse_start_step

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(self, x, y, ):
        N = x.shape[0]
        original_x = x.clone()
        momentum = torch.zeros_like(x)
        if self.random_start:
            x = self.perturb(x)

        for iteration in range(self.total_step):
            # --------------------------------------------------------------------------------#
            # first step
            reversed_x = x.clone()
            if iteration >= self.reverse_step_size:
                for _ in range(self.reverse_step):
                    reversed_x.requires_grad = True
                    logit = 0
                    for model in self.models:
                        logit += model(reversed_x.to(model.device)).to(reversed_x.device)
                    loss = self.criterion(logit, y)
                    loss.backward()
                    grad = reversed_x.grad
                    reversed_x.requires_grad = False
                    if self.targerted_attack:
                        reversed_x = reversed_x + self.reverse_step_size * grad.sign()
                    else:
                        reversed_x = reversed_x - self.reverse_step_size * grad.sign()
            # x.grad = None
            # --------------------------------------------------------------------------------#
            # second step
            reversed_x.requires_grad = True
            logit = 0
            for model in self.models:
                logit += model(reversed_x.to(model.device)).to(reversed_x.device)
            loss = self.criterion(logit, y)
            loss.backward()
            grad = reversed_x.grad
            reversed_x.requires_grad = False
            # update
            if self.targerted_attack:
                momentum = self.mu * momentum - grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                x += self.step_size * momentum.sign()
            else:
                momentum = self.mu * momentum + grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                x += self.step_size * momentum.sign()
            x = self.clamp(x, original_x)
        return x
