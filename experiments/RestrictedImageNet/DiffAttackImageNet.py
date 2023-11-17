import torch
from attacks import BIM
from data import get_someset_loader, get_NIPS17_loader, get_CIFAR10_test
from tester import test_robustness, test_acc
from defenses import DiffusionPure
from utils.seed import set_seed
from models import BaseNormModel, resnet50, RegionClassSelectionModel
from models.unets.guided_diffusion import get_guided_diffusion_unet
from torchvision import transforms

set_seed(1)

loader = get_someset_loader(
    "./resources/RestrictedImageNet256",
    "./resources/RestrictedImageNet256/gt.npy",
    batch_size=1,
    transform=transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    ),
)

device = torch.device("cuda")
unet = get_guided_diffusion_unet(resolution=128)

diffpure = DiffusionPure(
    mode="sde",
    grad_checkpoint=True,
    unet=unet,
    model=RegionClassSelectionModel(
        BaseNormModel(resnet50(pretrained=True)),
        target_class=RegionClassSelectionModel.restricted_imagenet_target_class(),
    ),
    img_shape=(3, 128, 128),
    post_transforms=transforms.Resize((256, 256)),
)
diffpure.eval().requires_grad_(False)

loader = [item for i, item in enumerate(loader) if 0 <= i < 32]

attacker = BIM([diffpure], step_size=1 / 255, total_step=80,
               eot_step=20, epsilon=4 / 255, norm="Linf", eot_batch_size=4)
test_robustness(
    attacker,
    loader,
    [diffpure],
)
