# Improved Diffusion for 1D Temporal Signals

A diffusion model framework for generating 1D temporal sequences, adapted from [improved-diffusion](https://github.com/openai/improved-diffusion). Supports unconditional generation, DDIM fast sampling, multi-GPU training, and integration with posterior sampling for inverse problems.

## Research Application

Designed for photon imaging with point process likelihood:

```
Log-flux data (N samples, T timesteps)
  → Train diffusion prior (this code)
    → Integrate with DiffPIR framework
      → Flux estimation from photon arrivals
```

## Installation

```bash
pip install torch numpy blobfile mpi4py tqdm

# Or install as a package
pip install -e .
```

## Quick Start

### 1. Verify Setup

```bash
python scripts/test_temporal_model.py
```

### 2. Generate Data

```bash
python generate_spad_flux_dataset.py
```

### 3. Train

```bash
python scripts/temporal_train.py \
    --data_path data/new_log_flux_dataset.pt \
    --sequence_length 1024 \
    --num_channels 64 \
    --channel_mult "1,2,3,4" \
    --num_res_blocks 2 \
    --attention_resolutions "500,250" \
    --batch_size 64 \
    --lr 1e-4 \
    --diffusion_steps 1000 \
    --noise_schedule linear \
    --save_interval 10000 \
    --lr_anneal_steps 200000 \
    --normalize False \
    --log_dir log
```

### 4. Generate Samples

```bash
# Standard DDPM sampling (1000 steps, best quality)
python scripts/temporal_sample.py \
    --model_path logs/model100000.pt \
    --num_samples 1000

# Fast DDIM sampling (50 steps, ~50x speedup)
python scripts/temporal_sample.py \
    --model_path logs/model100000.pt \
    --num_samples 1000 \
    --use_ddim True \
    --timestep_respacing "50"
```

## Using Your Own Data

Prepare a `.pt` file with shape `[N, T]` (samples x timesteps):

```python
import torch as th
import numpy as np

data = np.load('your_data.npy')  # Shape: [N, T]
th.save(th.from_numpy(data.astype(np.float32)), 'my_data.pt')
```

Data is automatically normalized to `[-1, 1]` during loading. To disable: `--normalize False`.

## Architecture

The model uses a 1D UNet (`dims=1`) with the following default configuration:


| Parameter               | Default   | Description                                 |
| ----------------------- | --------- | ------------------------------------------- |
| `sequence_length`       | 2000      | Temporal dimension                          |
| `num_channels`          | 64        | Base channel count                          |
| `channel_mult`          | (1,2,3,4) | Per-resolution multipliers                  |
| `attention_resolutions` | 500, 250  | Where self-attention is applied (see below) |
| `diffusion_steps`       | 1000      | DDPM timesteps                              |
| `noise_schedule`        | linear    | Beta schedule type                          |


**Shape flow:** `[B, 1, T]` → Conv1d → ResBlocks + Attention → Downsample/Upsample → `[B, 1, T]`

### Choosing attention resolutions

The `attention_resolutions` parameter specifies **spatial resolutions** (not downsample factors). Internally, `temporal_script_util.py` converts them to downsample factors via `ds = sequence_length // resolution`. The UNet applies attention wherever `ds` matches one of these factors.

With `channel_mult=(1,2,3,4)`, the UNet has 3 downsamples, producing these `ds` values:


| Level | ds  | Resolution (T=2000) | Resolution (T=1024) |
| ----- | --- | ------------------- | ------------------- |
| 0     | 1   | 2000                | 1024                |
| 1     | 2   | 1000                | 512                 |
| 2     | 4   | 500                 | 256                 |
| 3     | 8   | 250                 | 128                 |


**You must set `attention_resolutions` to values from the Resolution column for your sequence length**, otherwise attention layers will never activate and the model will lack long-range modeling capacity.


| Sequence Length | Recommended `attention_resolutions` |
| --------------- | ----------------------------------- |
| 1024            | `"256,128"`                         |
| 2000            | `"500,250"`                         |
| 4096            | `"1024,512"`                        |


## Advanced Usage

### Resume Training

```bash
python scripts/temporal_train.py \
    --data_path data.pt \
    --resume_checkpoint logs/model050000.pt
```

### Multi-GPU Training

```bash
mpiexec -n 4 python scripts/temporal_train.py \
    --data_path data.pt \
    --batch_size 64
```

### Memory Optimization

```bash
--use_fp16 True          # Mixed precision
--use_checkpoint True    # Gradient checkpointing
--batch_size 4           # Reduce batch size
```

## Sampling Speed vs Quality


| Method | Steps | Relative Speed | Quality   |
| ------ | ----- | -------------- | --------- |
| DDPM   | 1000  | 1x             | Best      |
| DDIM   | 250   | ~4x            | Very Good |
| DDIM   | 50    | ~20x           | Good      |


## Project Structure

```
improved-diffusion-main/
├── improved_diffusion/
│   ├── temporal_datasets.py       # 1D temporal data loader
│   ├── temporal_script_util.py    # 1D model/diffusion utilities
│   ├── gaussian_diffusion.py      # Core diffusion logic
│   ├── unet.py                    # UNet architecture (supports dims=1)
│   ├── train_util.py              # Training loop
│   └── ...
├── scripts/
│   ├── temporal_train.py          # Training entry point
│   ├── temporal_sample.py         # Sampling entry point
│   ├── test_temporal_model.py     # Test suite
│   └── complete_workflow_example.py  # End-to-end demo
├── generate_spad_flux_dataset.py  # SPAD flux data generation
└── setup.py
```

## Troubleshooting

**Samples look flat or lack structure:** Check that `attention_resolutions` actually matches your sequence length (see table above). If the values don't correspond to real downsample levels, attention never activates and the model has no long-range capacity.

**Samples look noisy:** Train longer (loss should still be decreasing), increase model size (`--num_channels 128`), or try cosine schedule (`--noise_schedule cosine`).

**Training too slow:** Use a smaller model (`--num_channels 32`), reduce attention resolutions to one level, enable FP16, or use multi-GPU.

**Import errors:** Run `pip install torch numpy blobfile mpi4py tqdm` or `pip install -e .`.