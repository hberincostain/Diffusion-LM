import argparse
import inspect

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .transformer_model2 import TransformerNetModel2

def model_and_diffusion_defaults():
    """
    Defaults for text diffusion model training.
    """
    return dict(
        image_size=8,  # sequence length
        num_channels=128,  # hidden size
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.1,
        learn_sigma=False,
        sigma_small=False,
        class_cond=False,  # Added missing class_cond parameter
        diffusion_steps=1000,
        noise_schedule="cosine",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,  # Added missing parameter
        use_checkpoint=False,
        use_scale_shift_norm=True,
        model_arch='transformer',
        in_channel=16,
        out_channel=16,
        training_mode='e2e',
        vocab_size=821,
        config_name='bert-base-uncased',
        experiment_mode='lm',
        logits_mode=1,
    )

def create_model_and_diffusion(
    image_size,
    class_cond,  # Added class_cond parameter
    learn_sigma,
    sigma_small,  # Added sigma_small parameter
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    model_arch,
    in_channel,
    out_channel,
    training_mode,
    vocab_size,
    config_name,
    experiment_mode,
    logits_mode,
    **kwargs,
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        in_channel=in_channel,
        out_channel=out_channel,
        vocab_size=vocab_size,
        config_name=config_name,
        experiment_mode=experiment_mode,
        logits_mode=logits_mode,
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
        model_arch=model_arch,
        training_mode=training_mode,
    )
    return model, diffusion

def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    in_channel,
    out_channel,
    vocab_size,
    config_name,
    experiment_mode,
    logits_mode,
):
    channel_mult = (1, 2, 2, 2)
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return TransformerNetModel2(
        in_channels=in_channel,
        model_channels=num_channels,
        out_channels=(out_channel if not learn_sigma else out_channel*2),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(None if not class_cond else None),  # Modified class conditioning
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        config_name=config_name,
        training_mode='e2e',
        vocab_size=vocab_size,
        experiment_mode=experiment_mode,
        logits_mode=logits_mode,
    )

def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="cosine",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=True,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    model_arch='transformer',
    training_mode='e2e',
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    
    # Set loss type based on training mode
    if training_mode == 'e2e':
        loss_type = gd.LossType.E2E_KL if use_kl else gd.LossType.E2E_MSE
    else:
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
        model_mean_type=gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X,
        model_var_type=(
            (gd.ModelVarType.FIXED_LARGE if not sigma_small else gd.ModelVarType.FIXED_SMALL)
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        model_arch=model_arch,
        training_mode=training_mode,
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
