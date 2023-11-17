import torch
from .edm_nets import EDMPrecond


def get_edm_cifar_cond(pretrained=True, **kwargs):
    network_kwargs = dict(
        model_type="SongUNet",
        embedding_type="positional",
        encoder_type="standard",
        decoder_type="standard",
        channel_mult_noise=1,
        resample_filter=[1, 1],
        model_channels=128,
        channel_mult=[2, 2, 2],
        augment_dim=9,
    )
    network_kwargs.update(kwargs)
    model = EDMPrecond(img_resolution=32, img_channels=3, label_dim=10, **network_kwargs)
    if pretrained:
        model.load_state_dict(torch.load("./resources/checkpoints/EDM/edm_cifar_cond.pt"))
    return model


def get_edm_cifar_uncond(pretrained=True, **kwargs):
    network_kwargs = dict(
        model_type="SongUNet",
        embedding_type="positional",
        encoder_type="standard",
        decoder_type="standard",
        channel_mult_noise=1,
        resample_filter=[1, 1],
        model_channels=128,
        channel_mult=[2, 2, 2],
        augment_dim=9,
    )
    network_kwargs.update(kwargs)
    model = EDMPrecond(img_resolution=32, img_channels=3, label_dim=0, **network_kwargs)
    if pretrained:
        model.load_state_dict(torch.load("./resources/checkpoints/EDM/edm_cifar_uncond_vp.pt"))
    return model


def get_edm_cifar_unet(pretrained=True, cond=True, dataset="cifar100", **kwargs):
    assert dataset in ["cifar100", "cifar10"]
    network_kwargs = dict(
        model_type="SongUNet",
        embedding_type="positional",
        encoder_type="standard",
        decoder_type="standard",
        channel_mult_noise=1,
        resample_filter=[1, 1],
        model_channels=128,
        channel_mult=[2, 2, 2],
        augment_dim=9,
    )
    network_kwargs.update(kwargs)
    num_classes = int(dataset.split("cifar")[1]) if cond else 0
    model = EDMPrecond(img_resolution=32, img_channels=3, label_dim=num_classes, **network_kwargs)
    if pretrained:
        if dataset == "cifar100":
            name = "edm_cifar100_cond_vp.pt" if cond else "edm_cifar100_uncond_vp.pt"
        else:
            name = "edm_cifar_cond_vp.pt" if cond else "edm_cifar_uncond_vp.pt"
        model.load_state_dict(torch.load("./resources/checkpoints/EDM/" + name))
    return model


def get_edm_imagenet_64x64_cond(pretrained=True, **kwargs):
    network_kwargs = dict(model_type="DhariwalUNet", model_channels=192, channel_mult=[1, 2, 3, 4])
    network_kwargs.update(kwargs)
    model = EDMPrecond(img_resolution=64, img_channels=3, label_dim=1000, **network_kwargs)
    if pretrained:
        model.load_state_dict(torch.load("./resources/checkpoints/EDM/edm-imagenet-64x64-cond.pt"))
    return model
