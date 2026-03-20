"""
Test script to verify 1D temporal diffusion model works correctly.

This script:
1. Creates a small synthetic dataset
2. Initializes the model
3. Tests forward pass with various timesteps
4. Tests training step
5. Tests sampling
6. Verifies all shapes are correct
"""

import torch as th
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from improved_diffusion.temporal_script_util import (
    create_temporal_model_and_diffusion,
)
from improved_diffusion import gaussian_diffusion as gd


def test_model_creation():
    """Test that we can create a 1D temporal model."""
    print("\n" + "="*80)
    print("TEST 1: Model Creation")
    print("="*80)
    
    model, diffusion = create_temporal_model_and_diffusion(
        sequence_length=256,
        num_channels=64,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="32,16",
        channel_mult="1,2,3,4",
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
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created successfully")
    print(f"  Total parameters: {total_params:,}")
    
    return model, diffusion


def test_forward_pass(model, diffusion):
    """Test forward pass with different batch sizes and timesteps."""
    print("\n" + "="*80)
    print("TEST 2: Forward Pass")
    print("="*80)
    
    batch_size = 4
    sequence_length = 256
    
    # Create synthetic input [B, C=1, T]
    x = th.randn(batch_size, 1, sequence_length)
    print(f"✓ Created input tensor with shape: {x.shape}")
    
    # Test at different timesteps
    timesteps = [0, 250, 500, 750, 999]
    
    for t_val in timesteps:
        t = th.tensor([t_val] * batch_size)
        output = model(x, t)
        
        expected_shape = (batch_size, 1, sequence_length)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"✓ Forward pass at t={t_val:4d}: output shape {output.shape}")
    
    print(f"✓ All forward passes successful")


def test_diffusion_process(model, diffusion):
    """Test forward and reverse diffusion."""
    print("\n" + "="*80)
    print("TEST 3: Diffusion Process")
    print("="*80)
    
    batch_size = 4
    sequence_length = 256
    
    # Create clean signal
    x_start = th.randn(batch_size, 1, sequence_length)
    print(f"✓ Created clean signal: {x_start.shape}")
    
    # Test forward diffusion (add noise)
    t = th.randint(0, diffusion.num_timesteps, (batch_size,))
    noise = th.randn_like(x_start)
    x_t = diffusion.q_sample(x_start, t, noise=noise)
    
    print(f"✓ Forward diffusion (q_sample) successful")
    print(f"  Input range: [{x_start.min():.3f}, {x_start.max():.3f}]")
    print(f"  Noisy range: [{x_t.min():.3f}, {x_t.max():.3f}]")
    
    # Test reverse diffusion (denoise one step)
    with th.no_grad():
        out = diffusion.p_sample(model, x_t, t)
        x_denoised = out['sample']
    
    print(f"✓ Reverse diffusion (p_sample) successful")
    print(f"  Denoised range: [{x_denoised.min():.3f}, {x_denoised.max():.3f}]")


def test_training_loss(model, diffusion):
    """Test training loss computation."""
    print("\n" + "="*80)
    print("TEST 4: Training Loss")
    print("="*80)
    
    batch_size = 4
    sequence_length = 256
    
    # Create training batch
    x_start = th.randn(batch_size, 1, sequence_length)
    t = th.randint(0, diffusion.num_timesteps, (batch_size,))
    
    # Compute losses
    losses = diffusion.training_losses(model, x_start, t)
    
    print(f"✓ Training loss computed successfully")
    print(f"  Loss keys: {list(losses.keys())}")
    print(f"  Loss shape: {losses['loss'].shape}")
    print(f"  Mean loss: {losses['loss'].mean():.4f}")
    
    # Test backward pass
    loss = losses['loss'].mean()
    loss.backward()
    
    print(f"✓ Backward pass successful")
    
    # Check gradients exist
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total = sum(1 for p in model.parameters())
    print(f"✓ Gradients computed: {has_grad}/{total} parameters")
    
    # Zero gradients for next test
    model.zero_grad()


def test_sampling(model, diffusion):
    """Test unconditional sampling."""
    print("\n" + "="*80)
    print("TEST 5: Unconditional Sampling")
    print("="*80)
    
    batch_size = 2
    sequence_length = 256
    
    model.eval()
    with th.no_grad():
        # Use fewer steps for faster testing
        # We'll use DDIM with respaced timesteps
        from improved_diffusion.respace import SpacedDiffusion, space_timesteps
        
        # Create a faster diffusion with only 50 steps
        fast_diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(1000, "50"),
            betas=diffusion.betas,
            model_mean_type=diffusion.model_mean_type,
            model_var_type=diffusion.model_var_type,
            loss_type=diffusion.loss_type,
            rescale_timesteps=diffusion.rescale_timesteps,
        )
        
        print(f"  Using {len(fast_diffusion.timestep_map)} timesteps for faster sampling")
        
        samples = fast_diffusion.p_sample_loop(
            model,
            (batch_size, 1, sequence_length),
            progress=False,
        )
    
    print(f"✓ Sampling successful")
    print(f"  Sample shape: {samples.shape}")
    print(f"  Sample range: [{samples.min():.3f}, {samples.max():.3f}]")
    
    model.train()
    return samples


def test_save_load(model, diffusion):
    """Test model saving and loading."""
    print("\n" + "="*80)
    print("TEST 6: Save and Load")
    print("="*80)
    
    # Save model
    save_path = "/tmp/test_temporal_model.pt"
    th.save(model.state_dict(), save_path)
    print(f"✓ Model saved to {save_path}")
    
    # Create new model
    new_model, _ = create_temporal_model_and_diffusion(
        sequence_length=256,
        num_channels=64,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="32,16",
        channel_mult="1,2,3,4",
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
    
    # Load weights
    new_model.load_state_dict(th.load(save_path))
    print(f"✓ Model loaded successfully")
    
    # Verify outputs match
    x = th.randn(2, 1, 256)
    t = th.tensor([100, 200])
    
    model.eval()
    new_model.eval()
    
    with th.no_grad():
        out1 = model(x, t)
        out2 = new_model(x, t)
    
    diff = (out1 - out2).abs().max()
    print(f"✓ Output difference after reload: {diff:.10f}")
    assert diff < 1e-6, "Outputs don't match after reload!"
    
    # Cleanup
    os.remove(save_path)


def test_dataset_compatibility():
    """Test that we can create and load a dataset."""
    print("\n" + "="*80)
    print("TEST 7: Dataset Creation and Loading")
    print("="*80)
    
    # Create synthetic dataset
    num_samples = 100
    sequence_length = 256
    
    # Generate synthetic log-flux data (smooth curves)
    data = []
    for i in range(num_samples):
        # Create a smooth signal using sum of sinusoids
        t = np.linspace(0, 4*np.pi, sequence_length)
        signal = (
            np.sin(t) + 
            0.5 * np.sin(2*t + np.random.randn()) + 
            0.25 * np.sin(4*t + np.random.randn())
        )
        data.append(signal)
    
    data = np.array(data, dtype=np.float32)
    print(f"✓ Created synthetic dataset: {data.shape}")
    
    # Save as .pt file
    dataset_path = "/tmp/test_temporal_data.pt"
    th.save(th.from_numpy(data), dataset_path)
    print(f"✓ Saved dataset to {dataset_path}")
    
    # Load using our dataset class
    from improved_diffusion.temporal_datasets import TemporalDataset
    
    dataset = TemporalDataset(
        data_path=dataset_path,
        sequence_length=sequence_length,
        normalize=True,
        shard=0,
        num_shards=1,
    )
    
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    
    # Test data loading
    sample, kwargs = dataset[0]
    print(f"✓ Sample shape: {sample.shape} (expected [1, {sequence_length}])")
    assert sample.shape == (1, sequence_length), f"Wrong shape: {sample.shape}"
    
    # Cleanup
    os.remove(dataset_path)


def main():
    print("\n" + "#"*80)
    print("# 1D TEMPORAL DIFFUSION MODEL - COMPREHENSIVE TEST SUITE")
    print("#"*80)
    
    # Set random seed for reproducibility
    th.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Run all tests
        model, diffusion = test_model_creation()
        test_forward_pass(model, diffusion)
        test_diffusion_process(model, diffusion)
        test_training_loss(model, diffusion)
        samples = test_sampling(model, diffusion)
        test_save_load(model, diffusion)
        test_dataset_compatibility()
        
        # Summary
        print("\n" + "#"*80)
        print("# ALL TESTS PASSED! ✓")
        print("#"*80)
        print("\nThe 1D temporal diffusion model is working correctly!")
        print("\nNext steps:")
        print("  1. Prepare your log-flux dataset as a .pt file with shape [N, 256]")
        print("  2. Run training: python scripts/temporal_train.py --data_path <your_data>.pt")
        print("  3. Generate samples: python scripts/temporal_sample.py --model_path <checkpoint>.pt")
        print()
        
        return True
        
    except Exception as e:
        print("\n" + "#"*80)
        print(f"# TEST FAILED: {e}")
        print("#"*80)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

