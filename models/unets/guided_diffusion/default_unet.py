from .script_util import create_model_and_diffusion
from .config import model_and_diffusion_defaults
import torch

__checkpoint_path__ = "./resources/checkpoints/guided_diffusion/256x256_diffusion"


def get_guided_diffusion_unet(pretrained=True, resolution=256, cond=False):
    config = model_and_diffusion_defaults()
    path = __checkpoint_path__ + "_uncond.pt" if not cond else __checkpoint_path__ + ".pt"
    if cond is False:
        config["class_cond"] = False
    if resolution == 128:
        config["image_size"] = 128
    elif resolution == 64:
        config["image_size"] = 64
    model, _ = create_model_and_diffusion(**config)
    if pretrained:
        model.load_state_dict(torch.load(path.replace("256", str(resolution))))

    class GuidedDiffusionMeanModel(torch.nn.Module):
        def __init__(self):
            super(GuidedDiffusionMeanModel, self).__init__()
            self.model = model

        def forward(self, *args, **kwargs):
            return self.model(*args, **kwargs)[:, :3, :, :]

    return GuidedDiffusionMeanModel()
