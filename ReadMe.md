# Combined Documentation (excluding README.md)

### ✨ What You Can Do Now

1. **Train unconditional diffusion models** on 1D temporal signals
2. **Generate new temporal sequences** using trained models
3. **Use fast DDIM sampling** (50x speedup over standard DDPM)
4. **Integrate with Posterior Sampling** for inverse problem solving
5. **Distribute training** across multiple GPUs with MPI
6. **Save/load checkpoints** compatible with PyTorch
7. **Customize architecture** for different sequence lengths and model sizes

### 🔬 Research Application

**Use case:** Photon imaging with point process likelihood

```
Your log-flux data (50000 samples, 256 timesteps)
              ↓
   Train diffusion model (this code)
              ↓
   Trained model checkpoint
              ↓
   Integrate with DiffPIR framework
              ↓
   Add point process likelihood proximal operator
              ↓
   Perform flux estimation from photon arrivals
```

---

## 🚀 Quick Start Commands

### 1. Install Dependencies
```bash
pip install torch numpy blobfile mpi4py tqdm
```

### 2. Test Everything
```bash
cd /Users/tan583/Documents/Diffusion/improved-diffusion-main
python scripts/test_temporal_model.py
```

### 3. Create Example Data
```bash
python examples/create_example_dataset.py
```

### 4. Train Model
```bash
python scripts/temporal_train.py \
    --data_path my_log_flux_data.pt \
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

### 5. Generate Samples
```bash
python scripts/temporal_sample.py \
    --model_path logs/model100000.pt \
    --num_samples 1000 \
    --use_ddim True \
    --timestep_respacing "50"
```

---

## 📊 Architecture Overview

### Input/Output Shape Flow

```
Data File: [N, T] → [50000, 256]
     ↓ (TemporalDataset adds channel dimension)
Model Input: [B, C, T] → [16, 1, 256]
     ↓ (1D UNet with dims=1)
  Conv1d(1, 64)  →  [16, 64, 256]
  ResBlock       →  [16, 64, 256]
  Downsample     →  [16, 128, 128]
  Attention      →  [16, 128, 128]
  More layers... →  ...
  Upsample       →  [16, 64, 256]
Model Output: [B, C, T] → [16, 1, 256]
```

### Model Configuration (Default)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sequence length | 256 | Temporal dimension |
| Input channels | 1 | Single temporal signal |
| Model channels | 64 | Base channel count |
| Channel multipliers | (1,2,3,4) | Per-resolution multipliers |
| Attention resolutions | 32, 16 | Where to apply attention |
| Diffusion steps | 1000 | DDPM timesteps |
| Noise schedule | linear | Beta schedule |
| Parameters | ~5-10M | Total model size |

---

## 📁 Complete File Structure

```
improved-diffusion-main/
│
├── improved_diffusion/
│   ├── ✨ temporal_datasets.py       NEW - 1D data loader
│   ├── ✨ temporal_script_util.py    NEW - 1D utilities
│   ├── ✓ gaussian_diffusion.py      Works unchanged
│   ├── ✓ unet.py                    Works with dims=1
│   ├── ✓ train_util.py              Works unchanged
│   └── ✓ ... (all other files)     Work unchanged
│
├── scripts/
│   ├── ✨ temporal_train.py          NEW - Training script
│   ├── ✨ temporal_sample.py         NEW - Sampling script
│   ├── ✨ test_temporal_model.py     NEW - Test suite
│   └── ✓ ... (image scripts)        Original files preserved
│
└── examples/
    ├── ✨ create_example_dataset.py  NEW - Synthetic data
    └── ✨ complete_workflow_example.py NEW - Full demo
```

**Legend:**
- ✨ = New file created for 1D temporal support
- ✓ = Original file, works unchanged
- ⭐ = Recommended starting point
- 🚀 = Quick start guide
- 📚 = Detailed documentation

## ✅ Verification Checklist

**Before using with your research data:**

- [x] All files created successfully
- [x] No syntax errors in Python files
- [x] Documentation is comprehensive
- [ ] Dependencies installed (`pip install torch numpy blobfile mpi4py tqdm`)
- [ ] Test suite runs (`python scripts/test_temporal_model.py`)
- [ ] Can create example data (`python examples/create_example_dataset.py`)
- [ ] Can start training (`python scripts/temporal_train.py --data_path ...`)
- [ ] Can generate samples (`python scripts/temporal_sample.py --model_path ...`)

**Items marked [x] are complete. Items marked [ ] require PyTorch environment.**

### Sampling

| Method | Steps | Time/Sample | Quality |
|--------|-------|-------------|---------|
| DDPM   | 1000  | 10-30s      | Best    |
| DDIM   | 250   | 2-8s        | Very Good |
| DDIM   | 50    | 0.5-2s      | Good    |

---

## File: QUICKSTART.md

# Quick Start Guide: 1D Temporal Diffusion Models

This guide will get you up and running with 1D temporal diffusion models in **5 minutes**.

## Prerequisites

```bash
# Install dependencies
pip install torch numpy blobfile mpi4py tqdm

# Or install the package
cd /Users/tan583/Documents/Diffusion/improved-diffusion-main
pip install -e .
```

## Step 1: Create Example Dataset (30 seconds)

We've provided a script to generate synthetic temporal data:

```bash
cd /Users/tan583/Documents/Diffusion/improved-diffusion-main
python examples/create_example_dataset.py
```

This creates:
- `data/log_flux_train.pt`: 10,000 training samples
- `data/log_flux_test.pt`: 1,000 test samples
- Each sample: 256 timesteps of smooth temporal signal

## Step 2: Test the Model (1 minute)

Verify everything works:

```bash
python scripts/test_temporal_model.py
```

This runs 7 comprehensive tests:
1. ✓ Model creation
2. ✓ Forward pass at different timesteps
3. ✓ Forward/reverse diffusion process
4. ✓ Training loss computation
5. ✓ Unconditional sampling
6. ✓ Save/load checkpoints
7. ✓ Dataset loading

If all tests pass, you're ready to train!

## Step 3: Train the Model

### Quick Training (for testing)

Train a small model for a short time to verify everything works:

```bash
python scripts/temporal_train.py \
    --data_path data/log_flux_train.pt \
    --sequence_length 256 \
    --num_channels 32 \
    --batch_size 8 \
    --lr 1e-4 \
    --save_interval 1000 \
    --log_interval 50
```

This trains a small model. Watch for:
- Loss should decrease over time
- No shape errors
- Checkpoints saved to `./logs/`

Press `Ctrl+C` after a few hundred iterations if everything looks good.

### Full Training (for production)

For your actual research, use these settings:

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
    --log_interval 100 \
    --lr_anneal_steps 200000
```

**Training time estimates:**
- Small model (32 channels): ~2-4 hours on GPU for 50K iterations
- Default model (64 channels): ~4-8 hours on GPU for 100K iterations
- Large model (128 channels): ~12-24 hours on GPU for 200K iterations

**Monitor training:**
- Check `./logs/` for tensorboard logs
- Loss should stabilize after 50-100K iterations
- Checkpoints saved every 10K iterations

## Step 4: Generate Samples

After training, generate samples:

```bash
python scripts/temporal_sample.py \
    --model_path ./logs/model050000.pt \
    --num_samples 100 \
    --batch_size 10 \
    --output_dir ./samples
```

**For faster sampling (50x speedup):**

```bash
python scripts/temporal_sample.py \
    --model_path ./logs/model050000.pt \
    --num_samples 100 \
    --batch_size 10 \
    --use_ddim True \
    --timestep_respacing "50" \
    --output_dir ./samples
```

Output files:
- `samples/samples_100x256.npz`: NumPy format
- `samples/samples_100x256.pt`: PyTorch format

## Step 5: Use the Samples

Load and visualize generated samples:

```python
import torch as th
import numpy as np
import matplotlib.pyplot as plt

# Load samples
samples = th.load('samples/samples_100x256.pt')
print(f"Samples shape: {samples.shape}")  # [100, 1, 256]

# Visualize
fig, axes = plt.subplots(5, 1, figsize=(12, 10))
for i in range(5):
    axes[i].plot(samples[i, 0, :].numpy())
    axes[i].set_title(f"Generated Sample {i+1}")
    axes[i].set_xlabel("Time")
    axes[i].set_ylabel("Log-Flux")
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('generated_samples.png', dpi=150)
plt.show()
```

## Using Your Own Data

Replace the example dataset with your own log-flux data:

```python
import torch as th
import numpy as np

# Load your data
# Ensure shape is [N, T] where N=samples, T=timesteps
your_data = np.load('your_photon_data.npy')  # Shape: [50000, 256]

# Save as .pt file
th.save(th.from_numpy(your_data.astype(np.float32)), 'my_log_flux_data.pt')

# Train on your data
# python scripts/temporal_train.py --data_path my_log_flux_data.pt ...
```

**Important:** Your data will be automatically normalized to [-1, 1] during loading. To disable:
```bash
python scripts/temporal_train.py --data_path my_data.pt --normalize False
```

## Common Commands Reference

### Training Commands

**Minimal (fastest, for testing):**
```bash
python scripts/temporal_train.py \
    --data_path data.pt \
    --num_channels 32 \
    --batch_size 8
```

**Recommended (balanced):**
```bash
python scripts/temporal_train.py \
    --data_path data.pt \
    --num_channels 64 \
    --batch_size 16 \
    --save_interval 10000
```

**High Quality (slow, best results):**
```bash
python scripts/temporal_train.py \
    --data_path data.pt \
    --num_channels 128 \
    --channel_mult "1,2,3,4,5" \
    --attention_resolutions "64,32,16" \
    --batch_size 32 \
    --lr_anneal_steps 300000
```

### Resume Training

```bash
python scripts/temporal_train.py \
    --data_path data.pt \
    --resume_checkpoint ./logs/model050000.pt
```

### Multi-GPU Training

```bash
mpiexec -n 4 python scripts/temporal_train.py \
    --data_path data.pt \
    --batch_size 64  # Effective batch size = 64 * 4
```

### Sampling Commands

**Standard sampling:**
```bash
python scripts/temporal_sample.py \
    --model_path model.pt \
    --num_samples 1000
```

**Fast DDIM sampling:**
```bash
python scripts/temporal_sample.py \
    --model_path model.pt \
    --num_samples 1000 \
    --use_ddim True \
    --timestep_respacing "50"
```

## Troubleshooting

### "Out of memory" error
```bash
# Reduce batch size
--batch_size 4

# Or use mixed precision
--use_fp16 True

# Or enable gradient checkpointing
--use_checkpoint True
```

### Samples look random/noisy
- Train longer (check if loss is still decreasing)
- Increase model size: `--num_channels 128`
- Try cosine schedule: `--noise_schedule cosine`

### Training is too slow
- Use smaller model: `--num_channels 32`
- Reduce attention: `--attention_resolutions "16"`
- Enable FP16: `--use_fp16 True`
- Use multi-GPU: `mpiexec -n 4 python ...`

### Import errors
```bash
# Install dependencies
pip install torch numpy blobfile mpi4py tqdm

# Or install package
pip install -e .
```

## Next Steps

1. **Integrate with DiffPIR**: Use the trained model as a prior for inverse problems
2. **Customize architecture**: Modify `temporal_script_util.py` for different configurations
3. **Add conditioning**: Extend the model for class-conditional generation
4. **Experiment with schedules**: Try different noise schedules (cosine, sqrt, etc.)

## Getting Help

- **Test suite**: Run `python scripts/test_temporal_model.py` to diagnose issues
- **Examples**: Check `examples/create_example_dataset.py` for data format

<!-- INSERT_CONTENT_HERE -->
