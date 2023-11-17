

my_config = {
    "attention_resolutions": '32,16,8',
    "class_cond": False,
    "diffusion_steps": 1000,
    "rescale_timesteps": True,
    "timestep_respacing": '1000',  # Modify this value to decrease the number of timesteps.
    "image_size": 256,
    "learn_sigma": True,
    "noise_schedule": 'linear',
    "num_channels": 256,
    "num_head_channels": 64,
    "num_res_blocks": 2,
    "resblock_updown": True,
    "use_fp16": False,
    "use_scale_shift_norm": True,
}



def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
    )
    res.update(diffusion_defaults())
    res.update(my_config)
    return res


