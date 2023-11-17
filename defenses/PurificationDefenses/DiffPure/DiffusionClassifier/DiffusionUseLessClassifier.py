import torch
from torch import nn
from defenses.PurificationDefenses.DiffPure.model import get_unet
from torch import Tensor
from defenses.PurificationDefenses.DiffPure.sampler.sde import DiffusionSde
from defenses.PurificationDefenses.DiffPure.utils import clamp, L2_each_instance
from models.unets.score_sde.models.layers import get_timestep_embedding


class EmbeddingOptimizationClassifier(nn.Module):
    def __init__(self, unet: nn.Module = None, beta: Tensor = None,
                 num_classes=10):
        super(EmbeddingOptimizationClassifier, self).__init__()
        self.device = torch.device('cuda')
        if unet is None:
            unet, beta, img_shape = get_unet()
        if beta is None:
            beta = torch.linspace(0.1 / 1000, 20 / 1000, 1000, device=self.device)
        self.unet = unet
        self.beta = beta
        self.alpha = (1 - beta)
        self.alpha_bar = self.alpha.cumprod(dim=0).to(self.device)
        self.T = 1000

        # storing
        self.num_classes = num_classes
        self.unet_criterion = nn.MSELoss()
        self.sde = DiffusionSde(unet=self.unet, beta=self.beta, img_shape=(3, 32, 32))

        self.init()

    def init(self):
        self.get_unet_class_embedding()
        self.eval().requires_grad_(False)
        self.to(self.device)

    @torch.enable_grad()
    def optimize_embedding(self, x: Tensor, embedding: Tensor, iter_time=100, samples=64) -> Tensor:
        """
        :param x: 1, C, H, D
        :param embedding: 1, K
        :return:
        """
        embedding.requires_grad = True
        optimizer = torch.optim.Adam([embedding], lr=4e-4)
        x = x.repeat(samples, 1, 1, 1)
        for _ in range(iter_time):
            # # reconstruct loss
            # p = self.sde.purify(x, given_embedding=embedding)
            # loss = self.unet_criterion(p, x)

            # training loss
            t = torch.randint(10, (samples,), device=self.device)
            tensor_t = t
            noise = torch.randn_like(x)
            noised_x = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1) * x + \
                       torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1, 1) * noise
            pre = self.unet(noised_x, tensor_t, given_embedding=embedding)[:, :3, :, :]
            target = noise
            loss = self.unet_criterion(pre, target)
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        embedding.requires_grad = False
        return embedding

    def get_one_instance_prediction(self, x: Tensor) -> Tensor:
        """
        :param x: 1, C, H, D
        :return:
        """
        embedding = torch.randn((1, 4 * self.unet.nf), device=self.device)
        embedding = self.optimize_embedding(x, embedding).squeeze()
        # self.class_embedding = self.class_embedding / torch.norm(self.class_embedding, dim=1).view(-1, 1)
        # embedding = embedding / torch.norm(embedding)
        # score = self.class_embedding @ embedding
        score = ((self.class_embedding - embedding) ** 2).sum(1)

        print(score)
        predict = torch.min(score, dim=0)[1]
        return predict

    def forward(self, x: Tensor) -> Tensor:
        '''
        :param x: N, C, H, D
        :return: N
        '''
        # x = self.attack(x, y=torch.tensor([3], device=self.device))
        xs = x.split(1)  # 1, C, H, D
        y = []
        for now_x in xs:
            y.append(self.get_one_instance_prediction(now_x))
        y = torch.tensor(y, device=self.device)
        return y

    def get_unet_class_embedding(self):
        x = torch.arange(self.num_classes, device=self.device)
        sin_embedding = get_timestep_embedding(x, self.unet.nf)
        class_temb = self.unet.class_embedding_linear2(
            self.unet.act(self.unet.class_embedding_linear1(sin_embedding)))  # num_classes, D
        self.class_embedding = class_temb


class KminLossClassifier(nn.Module):
    def __init__(self, unet: nn.Module = None, beta: Tensor = None,
                 num_classes=10):
        super(KminLossClassifier, self).__init__()
        self.device = torch.device('cuda')
        if unet is None:
            unet, beta, img_shape = get_unet()
        if beta is None:
            beta = torch.linspace(0.1 / 1000, 20 / 1000, 1000, device=self.device)
        self.unet = unet
        self.beta = beta
        self.alpha = (1 - beta)
        self.alpha_bar = self.alpha.cumprod(dim=0).to(self.device)
        self.T = 1000
        self.init()

        # storing
        self.num_classes = num_classes
        self.unet_criterion = L2_each_instance
        self.sde = DiffusionSde(unet=self.unet, beta=self.beta, img_shape=(3, 32, 32))

    def init(self):
        self.eval().requires_grad_(False)
        self.to(self.device)

    def get_loss(self, x: Tensor, y: int, repeat_time=1, samples=512) -> Tensor:
        loss = 0
        for _ in range(repeat_time):
            # diffusion reconstruction loss
            x = x.repeat(samples, 1, 1, 1)
            y = torch.tensor([y], device=self.device).repeat(samples)
            loss += self.unet_criterion(self.sde.purify(x, y), x)
        # loss /= repeat_time
        return loss

    def get_one_instance_prediction(self, x: Tensor, k=10, samples=512) -> Tensor:
        """
        :param x: 1, C, H, D
        :return:
        """
        loss = []
        for class_id in range(self.num_classes):
            loss.append(self.get_loss(x, class_id, samples=samples))
        loss = torch.stack(loss)  # num_classes, samples
        # print(loss.shape, loss)
        loss = loss.view(-1)
        _, indices = torch.sort(loss, descending=False)  # from small to big
        print(indices)
        k_min = indices[:k] // samples
        print(k_min)
        vote = torch.mode(k_min)[0]
        predict = vote
        return predict

    def forward(self, x: Tensor) -> Tensor:
        '''
        :param x: N, C, H, D
        :return: N
        '''
        # x = self.attack(x, y=torch.tensor([3], device=self.device))
        xs = x.split(1)  # 1, C, H, D
        y = []
        for now_x in xs:
            y.append(self.get_one_instance_prediction(now_x))
        y = torch.tensor(y, device=self.device)
        return y

    @torch.enable_grad()
    def attack(self, x: Tensor, y: Tensor, samples=64, step=500,
               step_size=8 / 255 / 10) -> Tensor:
        """

        :param x: 1, C, H, D
        :param y: 1,
        :param step:
        :param samples:
        :param step_size:
        :return:
        """
        ori_x = x.clone()
        for _ in range(step):
            x.requires_grad = True
            repeated_x = x.repeat(samples, 1, 1, 1)
            loss = self.unet_criterion(self.sde.purify(repeated_x, y.repeat(samples)), repeated_x)
            loss.backward()
            # print(loss)
            grad = x.grad
            x.requires_grad = False
            with torch.no_grad():
                x += grad.sign() * step_size
                x = clamp(x)
                x = clamp(x, ori_x - 8 / 255, ori_x + 8 / 255)
        return x
