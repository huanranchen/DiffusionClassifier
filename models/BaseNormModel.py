import torch
from torchvision import transforms
from torch import nn, Tensor
from typing import Tuple, List
import random
import numpy as np


__IGNORED_CLASS_LOGIT__ = -999


def set_seed(seed=1):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BaseNormModel(nn.Module):
    """
    **kwargs aims to unify with "pretrained=True"
    """

    def __init__(
        self,
        model: nn.Module,
        transform=transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        **kwargs,
    ):
        super(BaseNormModel, self).__init__()
        self.model = model
        self.transforms = transform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = self.transforms(x)
        return self.model(x)


class Identity(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        **kwargs,
    ):
        super(Identity, self).__init__()
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        return self.model(x)


class ClassSelectionModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        target_class: Tuple[int, ...] = (0, 100, 200, 300, 400, 500, 600, 700, 800, 900),
        remain_ignored_class_logit=False,
        device=torch.device("cuda"),
    ):
        super(ClassSelectionModel, self).__init__()
        self.model = model
        self.target_class = torch.tensor(target_class, device=device)
        self.remain_ignored_class_logit = remain_ignored_class_logit

    def forward(self, x):
        logit = self.model(x)
        if not self.remain_ignored_class_logit:
            result = torch.cat([i[:, self.target_class] for i in logit.split(1, dim=0)], dim=0)
        else:
            raise NotImplementedError
        return result


class RegionClassSelectionModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        target_class: Tuple[Tensor],
        remain_ignored_class_logit=False,
        device=torch.device("cuda"),
    ):
        super(RegionClassSelectionModel, self).__init__()
        self.model = model
        self.target_class = [now_class.to(device) for now_class in target_class]
        self.remain_ignored_class_logit = remain_ignored_class_logit

    def forward(self, x):
        logit = self.model(x)  # N, num_classes
        if not self.remain_ignored_class_logit:
            result = torch.stack([logit[:, now_class].mean(1) for now_class in self.target_class], dim=1)
        else:
            raise NotImplementedError
        return result

    @staticmethod
    def restricted_imagenet_target_class():
        return tuple(
            (
                torch.arange(start=151, end=269),
                torch.arange(start=281, end=286),
                torch.arange(start=30, end=33),
                torch.arange(start=33, end=38),
                torch.arange(start=80, end=101),
                torch.arange(start=365, end=383),
                torch.arange(start=389, end=398),
                torch.arange(start=118, end=122),
                torch.arange(start=300, end=320),
            )
        )


class FixPositionPatchModel(nn.Module):
    def __init__(self, base_model: nn.Module, device=torch.device("cuda")):
        super(FixPositionPatchModel, self).__init__()
        self.base_model = base_model
        self.images = None
        self.patch = None
        self.patch_position = None
        self.device = device

    def set_patch(self, patch: Tensor, position: Tuple[Tuple[int, ...], Tuple[int, ...]] = None):
        self.patch = patch
        self.patch_position = position if position is not None else self.patch_position

    @torch.no_grad()
    def initialize_patch(
        self,
        mode: str = "randn",
        position: Tuple[Tuple[int, ...], Tuple[int, ...]] = ((0, 0, 0), (3, 14, 14)),
    ):
        self.patch_position = position
        patch_size = (
            position[1][0] - position[0][0],
            position[1][1] - position[0][1],
            position[1][2] - position[0][2],
        )
        if mode == "randn":
            self.patch = torch.randn(*patch_size, device=self.device) / 2 + 0.5
        else:
            self.patch = torch.rand(size=patch_size, device=self.device)
        self.patch = torch.clamp(self.patch, min=0, max=1).unsqueeze(0)  # 1, C, H, D

    def set_images(self, images: Tensor):
        assert len(images.shape) == 4, "images should be N, C, H, D"
        self.images = images.to(self.device)

    @staticmethod
    def add_patch_to_image(patch: Tensor, image: Tensor, position: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> Tensor:
        image = image.clone()
        image[
            :, position[0][0] : position[1][0], position[0][1] : position[1][1], position[0][2] : position[1][2]
        ] = patch
        return image

    def forward(self, patch: Tensor) -> Tensor:
        x = self.add_patch_to_image(patch.squeeze(), self.images, self.patch_position)
        x = self.base_model(x)
        return x


class LogitEnsemble(nn.Module):
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n = len(self.models)
        self.eval().requires_grad_(False)

    def forward(self, *args, **kwargs) -> Tensor:
        logit = 0
        for model in self.models:
            logit = logit + model(*args, **kwargs)
        logit = logit / self.n
        return logit


class ImageModel(nn.Module):
    def __init__(self, x: Tensor):
        super().__init__()
        self.param = nn.Parameter(x)


class FixNoiseModel(nn.Module):
    """
    Note that this will also fix the noise of subsequent models and operations.
    """
    def __init__(self, m: nn.Module, seed: int = None):
        super().__init__()
        self.m = m
        self.seed = random.randint(0, 9999) if seed is None else seed

    def forward(self, *args, **kwargs):
        set_seed(self.seed)
        return self.m(*args, **kwargs)
