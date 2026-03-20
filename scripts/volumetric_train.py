"""
Train a diffusion model on 3D flux volumes (T, H, W).
"""

import argparse

from improved_diffusion import dist_util, logger
from improved_diffusion.volumetric_datasets import load_volumetric_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.volumetric_script_util import (
    volumetric_model_and_diffusion_defaults,
    create_volumetric_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.log_dir)

    logger.log("creating 3D volumetric model and diffusion...")
    model, diffusion = create_volumetric_model_and_diffusion(
        **args_to_dict(args, volumetric_model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    logger.log("Model configuration:")
    logger.log(f"  - Sequence length (T): {args.sequence_length}")
    logger.log(f"  - Height, Width: {args.height}, {args.width}")
    logger.log(f"  - Model channels: {args.num_channels}")
    logger.log(f"  - Channel multipliers: {args.channel_mult}")
    logger.log(f"  - Num res blocks: {args.num_res_blocks}")
    logger.log(f"  - Attention resolutions (ds): {args.attention_resolutions}")
    logger.log(f"  - Diffusion steps: {args.diffusion_steps}")
    logger.log(f"  - Noise schedule: {args.noise_schedule}")

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_volumetric_data(
        data_path=args.data_path,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        height=args.height,
        width=args.width,
        deterministic=False,
        normalize=args.normalize,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_path="",
        log_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=2,          # start conservatively; 3D is memory heavy
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        normalize=False,       # flux already in linear scale
    )
    defaults.update(volumetric_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

