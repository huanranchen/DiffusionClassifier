import torch
from torch import nn, Tensor


class OptimalDiffusionClassifier(nn.Module):
    def __init__(self,
                 loader,
                 beta=torch.linspace(0.1 / 1000, 20 / 1000, 1000),
                 num_classes=10,
                 device=torch.device('cuda')):
        super(OptimalDiffusionClassifier, self).__init__()
        self.device = device
        self.num_classes = num_classes
        alpha = (1 - beta)
        self.T = alpha.numel()
        self.alpha_bar = alpha.cumprod(dim=0).to(self.device)
        self.sigma = torch.sqrt(1 - self.alpha_bar)
        self.loader = loader
        self._init()

    def _init(self):
        xs, ys = [], []
        for x, y in self.loader:
            xs.append(x.to(self.device))
            ys.append(y.to(self.device))
        self.xs, self.ys = torch.cat(xs, dim=0), torch.cat(ys, dim=0)
        print('Finish initialize optimal diffusion classifier!!!')
        print('-' * 100)

    def forward_one_logit(self, x, y: int) -> Tensor:
        """
        Here, we still do not need to Monte Carlo epsilon
        :param x: 1, C, H, D
        :param y: 1
        :return:
        """
        mask = self.ys == y
        xs = self.xs[mask]  # N, C, H, D
        delta = x - xs  # N, C, H, D

        repeated_x = x.repeat(self.T, 1, 1, 1)
        xts = torch.sqrt(self.alpha_bar).view(-1, 1, 1, 1) * repeated_x + \
              self.sigma.view(-1, 1, 1, 1) * torch.randn_like(repeated_x)  # T, C, H, D
        softmax_inner = xts - torch.sqrt(self.alpha_bar).view(1, -1, 1, 1, 1) \
                        * xs.unsqueeze(1).repeat(1, self.T, 1, 1, 1)  # N, T, C, H, D
        N, T, C, H, D = softmax_inner.shape
        softmax_inner = torch.sum(softmax_inner.view(N, T, -1) ** 2, dim=2)  # N, T
        softmaxs = torch.softmax(- softmax_inner / (2 * self.sigma.view(1, -1)), dim=0)  # N, T
        softmaxs = softmaxs.permute(1, 0)  # T, N

        delta = delta.view(1, N, C * H * D)
        inner = softmaxs.view(T, N, 1) * delta  # T, N, C*H*D
        inner = torch.sum(inner.reshape(T, N * C * H * D) ** 2, dim=1)  # T
        result = torch.mean(inner * self.alpha_bar / self.sigma ** 2)
        return result

    def get_one_instance_prediction(self, x) -> Tensor:
        logit = []
        for i in range(self.num_classes):
            logit.append(self.forward_one_logit(x, i))
        logit = torch.stack(logit)
        # print(logit)
        return logit * -1

    def forward(self, x) -> Tensor:
        xs = x.split(1)  # 1, C, H, D
        y = []
        for now_x in xs:
            y.append(self.get_one_instance_prediction(now_x))
        y = torch.stack(y)  # N, num_classes
        return y
