"""
Train a diffusion model on 1D temporal signals.
"""

import argparse
import os

from improved_diffusion import dist_util, logger
from improved_diffusion.temporal_datasets import load_temporal_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.temporal_script_util import (
    temporal_model_and_diffusion_defaults,
    create_temporal_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.log_dir)

    logger.log("creating 1D temporal model and diffusion...")
    model, diffusion = create_temporal_model_and_diffusion(
        **args_to_dict(args, temporal_model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    
    # Log model architecture info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    logger.log(f"Model configuration:")
    logger.log(f"  - Sequence length: {args.sequence_length}")
    logger.log(f"  - Model channels: {args.num_channels}")
    logger.log(f"  - Channel multipliers: {args.channel_mult}")
    logger.log(f"  - Num res blocks: {args.num_res_blocks}")
    logger.log(f"  - Attention resolutions: {args.attention_resolutions}")
    logger.log(f"  - Diffusion steps: {args.diffusion_steps}")
    logger.log(f"  - Noise schedule: {args.noise_schedule}")
    
    if args.track_best_checkpoint:
        logger.log(f"Best checkpoint tracking enabled: monitoring '{args.best_checkpoint_metric}'")
    
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_temporal_data(
        data_path=args.data_path,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
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
        track_best_checkpoint=args.track_best_checkpoint,
        best_checkpoint_metric=args.best_checkpoint_metric,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_path="",
        log_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=16,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        normalize=True,
        track_best_checkpoint=True,
        best_checkpoint_metric="loss",  # Options: loss, mse, mse_q0, mse_q1, mse_q2, mse_q3
    )
    defaults.update(temporal_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

