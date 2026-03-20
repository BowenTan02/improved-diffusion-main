"""
Generate samples from a trained 1D temporal diffusion model.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from improved_diffusion import dist_util, logger
from improved_diffusion.temporal_script_util import (
    temporal_model_and_diffusion_defaults,
    create_temporal_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.log_dir)

    logger.log("creating 1D temporal model and diffusion...")
    model, diffusion = create_temporal_model_and_diffusion(
        **args_to_dict(args, temporal_model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    
    # Load model checkpoint
    if not args.model_path:
        raise ValueError("Must specify --model_path")
    
    logger.log(f"loading model from {args.model_path}...")
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.eval()

    logger.log("sampling...")
    all_sequences = []
    all_labels = []
    
    # Calculate how many samples per GPU
    num_samples_per_gpu = args.num_samples // dist.get_world_size()
    if dist.get_rank() == 0:
        num_samples_per_gpu += args.num_samples % dist.get_world_size()
    
    logger.log(f"Generating {num_samples_per_gpu} samples on this GPU...")
    
    num_generated = 0
    while num_generated < num_samples_per_gpu:
        batch_size = min(args.batch_size, num_samples_per_gpu - num_generated)
        
        model_kwargs = {}
        
        logger.log(f"Generating batch {num_generated // args.batch_size + 1}...")
        
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (batch_size, 1, args.sequence_length),  # [B, C=1, T]
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            progress=True,
        )
        
        # Move to CPU and convert to numpy
        sample = sample.cpu().numpy()
        
        all_sequences.append(sample)
        num_generated += batch_size
        logger.log(f"Generated {num_generated}/{num_samples_per_gpu} samples")

    # Concatenate all samples
    arr = np.concatenate(all_sequences, axis=0)
    arr = arr[:num_samples_per_gpu]  # Ensure exact count
    
    # Save samples
    if dist.get_rank() == 0:
        shape_str = f"{args.num_samples}x{args.sequence_length}"
        out_path = os.path.join(args.output_dir, f"samples_{shape_str}.npz")
        logger.log(f"Saving samples to {out_path}")
        os.makedirs(args.output_dir, exist_ok=True)
        np.savez(out_path, samples=arr)
        
        # Also save as .pt for easy loading
        th.save(th.from_numpy(arr), os.path.join(args.output_dir, f"samples_{shape_str}.pt"))
        logger.log(f"Saved {arr.shape[0]} samples with shape {arr.shape}")

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        model_path="",
        output_dir="./samples",
        log_dir="",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        timestep_respacing="",
    )
    defaults.update(temporal_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

