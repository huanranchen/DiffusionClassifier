from typing import Callable, List, Iterable
from torch import nn
from attacks.utils import *
from .PerturbationObject import Perturbation
from .SequentialAttacker import SequentialAttacker


class ParallelAttacker(SequentialAttacker):
    '''
    please set your learning rate in optimizer
    set data augmentation in your loader.
    '''

    def __init__(self, *args, **kwargs,
                 ):
        super(ParallelAttacker, self).__init__(*args, **kwargs)

    def attack(self,
               loader: DataLoader or Iterable,
               total_iter_step: int = 10,
               is_clamp=True):
        iter_step = 0
        while True:
            for x, y in loader:
                x = x + self.perturbation.perturbation
                if is_clamp:
                    x = clamp(x)
                x = self.transform(x)

                loss = 0
                for model in self.models:
                    loss += self.criterion(model(x.to(model.device)), y.to(model.device))
                self.perturbation.zero_grad()
                loss.backward()
                self.perturbation.step()

                iter_step += 1
                if iter_step > total_iter_step:
                    self.perturbation.requires_grad(False)
                    return self.perturbation
