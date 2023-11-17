import torch
from .script_util import create_classifier


def create_256x256_classifier(pretrained=True):
    defaults = dict(
        image_size=256,
        classifier_use_fp16=False,
        classifier_width=128,
        classifier_depth=2,
        classifier_attention_resolutions="32,16,8",  # 16
        classifier_use_scale_shift_norm=True,  # False
        classifier_resblock_updown=True,  # False
        classifier_pool="attention",
    )
    classifier = create_classifier(**defaults)
    if pretrained:
        classifier.load_state_dict(
            torch.load('./resources/checkpoints/guided_diffusion/256x256_classifier.pt')
        )
    return classifier
