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

### 2. Create Example Data

```bash
python scripts/complete_workflow_example.py
```

### 3. Train

```bash
python scripts/temporal_train.py \
    --data_path data/log_flux_train.pt \
    --sequence_length 256 \
    --num_channels 64 \
    --channel_mult "1,2,3,4" \
    --num_res_blocks 2 \
    --attention_resolutions "32,16" \
    --batch_size 16 \
    --lr 1e-4 \
    --diffusion_steps 1000 \
    --noise_schedule linear \
    --save_interval 10000 \
    --lr_anneal_steps 200000
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

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sequence_length` | 256 | Temporal dimension |
| `num_channels` | 64 | Base channel count |
| `channel_mult` | (1,2,3,4) | Per-resolution multipliers |
| `attention_resolutions` | 32, 16 | Where self-attention is applied |
| `diffusion_steps` | 1000 | DDPM timesteps |
| `noise_schedule` | linear | Beta schedule type |

**Shape flow:** `[B, 1, 256]` → Conv1d → ResBlocks + Attention → Downsample/Upsample → `[B, 1, 256]`

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

| Method | Steps | Relative Speed | Quality |
|--------|-------|----------------|---------|
| DDPM | 1000 | 1x | Best |
| DDIM | 250 | ~4x | Very Good |
| DDIM | 50 | ~20x | Good |

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

**Samples look noisy:** Train longer (loss should still be decreasing), increase model size (`--num_channels 128`), or try cosine schedule (`--noise_schedule cosine`).

**Training too slow:** Use a smaller model (`--num_channels 32`), reduce attention (`--attention_resolutions "16"`), enable FP16, or use multi-GPU.

**Import errors:** Run `pip install torch numpy blobfile mpi4py tqdm` or `pip install -e .`.
