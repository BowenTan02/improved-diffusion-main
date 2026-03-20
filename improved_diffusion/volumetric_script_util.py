"""
Utilities for creating 3D (T,H,W) UNet models and diffusion processes.
"""

import argparse
from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import UNetModel

def volumetric_model_and_diffusion_defaults():
    """
    Defaults for 3D flux volumes.
    """
    return dict(
        sequence_length=50,     # T dimension
        height=256,
        width=512,
        num_channels=64,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="32,16,8",  # downsample factors at which to apply attention
        channel_mult="1,2,2,4",
        dropout=0.0,
        learn_sigma=False,
        sigma_small=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        use_checkpoint=False,
        use_scale_shift_norm=True,
    )


def create_volumetric_model_and_diffusion(
    sequence_length,
    height,
    width,
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    channel_mult,
    dropout,
    learn_sigma,
    sigma_small,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
):
    model = create_volumetric_model(
        sequence_length=sequence_length,
        height=height,
        width=width,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        learn_sigma=learn_sigma,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        channel_mult=channel_mult,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def create_volumetric_model(
    sequence_length,
    height,
    width,
    num_channels,
    num_res_blocks,
    learn_sigma,
    use_checkpoint,
    attention_resolutions,
    channel_mult,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
):
    """
    Create a 3D UNet model for temporal-spatial flux data.

    dims=3 => Conv3d / attention over flattened (T*H*W).
    Downsample operates on spatial dims (stride (1,2,2)), preserving time length.
    """
    # Parse channel multipliers
    if isinstance(channel_mult, str):
        channel_mult = tuple(int(x) for x in channel_mult.split(","))

    # Parse attention resolutions: these are downsample factors (ds) where attention is applied.
    # ds starts at 1 and doubles per spatial downsample (H,W).
    if isinstance(attention_resolutions, str):
        attention_ds = [int(res) for res in attention_resolutions.split(",")]
    else:
        attention_ds = attention_resolutions

    return UNetModel(
        in_channels=1,
        model_channels=num_channels,
        out_channels=(1 if not learn_sigma else 2),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        conv_resample=True,
        dims=3,  # KEY: 3D convolutions
        num_classes=None,
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

