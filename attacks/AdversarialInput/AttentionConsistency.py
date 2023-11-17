import torch
from attacks.utils import *
from torch import nn
from torch.nn import functional as F
from typing import Callable, List
from .AdversarialInputBase import AdversarialInputAttacker


class MI_AttentionConsistency(AdversarialInputAttacker):
    """
    1.简单版本 让attention map趋近于一致
    2.让attention map的更新方向趋近于一致
    """

    def __init__(
        self,
        model: List[nn.Module],
        total_step: int = 10,
        random_start: bool = False,
        step_size: float = 16 / 255 / 10,
        criterion: Callable = nn.CrossEntropyLoss(),
        targeted_attack=False,
        mu: float = 1,
        interpolate_size=(10, 10),
        *args,
        **kwargs
    ):
        self.random_start = random_start
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        super(MI_AttentionConsistency, self).__init__(model, *args, **kwargs)
        self.interpolate_size = interpolate_size

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(self, x, y):
        N = x.shape[0]
        original_x = x.clone()
        momentum = torch.zeros_like(x)
        if self.random_start:
            x = self.perturb(x)

        for _ in range(self.total_step):
            x.requires_grad = True
            logits, attentions = 0, []
            for model in self.models:
                logit, attention = model(x.to(model.device))
                logits = logits + logit.to(self.device)
                attention = torch.stack(list(attention), dim=1).mean(2)  # bs, layers, sequence, sequence
                attention = F.interpolate(attention, self.interpolate_size, mode="bilinear")
                attention = attention[:, :12, :, :]
                attentions.append(attention.to(self.device))
            # attentions = torch.cat(attentions, dim=1)  # bs, models*layers, sequence, sequence
            attentions = torch.stack(attentions)  # models, bs, layers, sequence, sequence
            # attentions = attentions.view(-1, *attentions.shape[2:])
            M, B, L, S1, S2 = attentions.shape
            attentions = attentions.view(M, B*L, S1, S2).permute(1, 0, 2, 3)
            attentions = attentions[:, :, 1, :]  # bs, layers, sequence
            B, L, S = attentions.shape
            attn_loss = torch.mean((attentions.view(B, L, 1, S) - attentions.view(B, 1, L, S)) ** 2)
            loss = self.criterion(logits, y) - 100 * attn_loss
            # print(loss.item()+attn_loss.item(), attn_loss.item())
            loss.backward()
            grad = x.grad
            x.requires_grad = False
            # update
            if self.targerted_attack:
                momentum = self.mu * momentum - grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                x += self.step_size * momentum.sign()
            else:
                momentum = self.mu * momentum + grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                x += self.step_size * momentum.sign()
            x = self.clamp(x, original_x)
        print('-'*50)
        return x
