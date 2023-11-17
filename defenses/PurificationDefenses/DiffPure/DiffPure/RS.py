from .diffpure import DiffusionPure
import torch


class DiffPureForRandomizedSmoothing(DiffusionPure):
    def __init__(self, *args, sigma=0.25, **kwargs):
        """
        explanation for computing t:
        Sigma need to be multiplied by 2 because RS add noise at [0, 1] while diffusion working at [-1, 1]
        Because alpha_t = 1 / (sigma ** 2 + 1), we can compute t by:
        alpha[t] >= 1 / (sigma ** 2 + 1) > alpha[t+1]
        """
        super(DiffPureForRandomizedSmoothing, self).__init__(*args, **kwargs)
        self.t = torch.max(self.diffusion.alpha_bar < 1 / ((sigma * 2) ** 2 + 1), dim=0)[1] - 1
        print(self.t)

    def forward(self, x, *args, **kwargs):
        x = self.pre_transforms(x)
        x = self.diffusion(x, *args, noise_level=(self.t + 1), scale=True, add_noise=False, **kwargs)
        x = self.post_transforms(x)
        x = self.model(x)
        return x
