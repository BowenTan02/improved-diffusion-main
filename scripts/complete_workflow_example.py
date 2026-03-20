"""
Complete workflow example for 1D temporal diffusion models.

This script demonstrates the entire pipeline:
1. Create/load data
2. Initialize model
3. Train (simplified)
4. Generate samples
5. Evaluate results

This is for demonstration purposes. For actual training, use scripts/temporal_train.py
"""

import numpy as np
import torch as th
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def step1_create_data():
    """Step 1: Create synthetic temporal dataset."""
    print("\n" + "="*80)
    print("STEP 1: Create Synthetic Dataset")
    print("="*80)
    
    num_samples = 100
    sequence_length = 256
    
    # Generate smooth temporal signals
    signals = []
    for i in range(num_samples):
        t = np.linspace(0, 4*np.pi, sequence_length)
        signal = (
            np.sin(t) + 
            0.5 * np.sin(2*t + np.random.randn()) +
            0.25 * np.sin(4*t + np.random.randn()) +
            np.random.randn() * 0.1
        )
        signals.append(signal)
    
    data = np.array(signals, dtype=np.float32)
    
    # Save as .pt file
    os.makedirs('./temp_data', exist_ok=True)
    data_path = './temp_data/example_data.pt'
    th.save(th.from_numpy(data), data_path)
    
    print(f"✓ Created {num_samples} temporal signals")
    print(f"  Shape: {data.shape}")
    print(f"  Saved to: {data_path}")
    
    return data_path, sequence_length


def step2_create_model(sequence_length):
    """Step 2: Initialize temporal diffusion model."""
    print("\n" + "="*80)
    print("STEP 2: Initialize Model")
    print("="*80)
    
    from improved_diffusion.temporal_script_util import create_temporal_model_and_diffusion
    
    # Create a small model for quick demonstration
    model, diffusion = create_temporal_model_and_diffusion(
        sequence_length=sequence_length,
        num_channels=32,  # Small model for demo
        num_res_blocks=2,
        num_heads=2,
        num_heads_upsample=-1,
        attention_resolutions="32,16",
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
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Architecture: 1D UNet with attention at resolutions 32, 16")
    
    return model, diffusion


def step3_train_model(model, diffusion, data_path, num_iterations=100):
    """Step 3: Train model (simplified demonstration)."""
    print("\n" + "="*80)
    print("STEP 3: Train Model (Demo - Only 100 Iterations)")
    print("="*80)
    print("Note: For real training, use scripts/temporal_train.py")
    
    from improved_diffusion.temporal_datasets import TemporalDataset
    from torch.utils.data import DataLoader
    import torch.optim as optim
    
    # Load dataset
    dataset = TemporalDataset(
        data_path=data_path,
        sequence_length=256,
        normalize=True,
        shard=0,
        num_shards=1,
    )
    
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    model.train()
    losses = []
    
    for iteration in range(num_iterations):
        for batch, _ in loader:
            # Random timesteps
            t = th.randint(0, diffusion.num_timesteps, (batch.shape[0],))
            
            # Compute loss
            loss_dict = diffusion.training_losses(model, batch, t)
            loss = loss_dict['loss'].mean()
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if iteration % 20 == 0:
                print(f"  Iteration {iteration:3d}/{num_iterations}: Loss = {loss.item():.4f}")
            
            break  # One batch per iteration for demo
    
    print(f"✓ Training complete")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Average loss: {np.mean(losses):.4f}")
    
    return model


def step4_generate_samples(model, diffusion, num_samples=10):
    """Step 4: Generate samples from trained model."""
    print("\n" + "="*80)
    print("STEP 4: Generate Samples")
    print("="*80)
    
    from improved_diffusion.respace import SpacedDiffusion, space_timesteps
    
    # Use faster diffusion with only 50 steps for demo
    fast_diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(1000, "50"),
        betas=diffusion.betas,
        model_mean_type=diffusion.model_mean_type,
        model_var_type=diffusion.model_var_type,
        loss_type=diffusion.loss_type,
        rescale_timesteps=diffusion.rescale_timesteps,
    )
    
    print(f"  Generating {num_samples} samples using 50-step DDIM...")
    
    model.eval()
    with th.no_grad():
        samples = fast_diffusion.p_sample_loop(
            model,
            (num_samples, 1, 256),
            progress=False,
        )
    
    print(f"✓ Generated {num_samples} samples")
    print(f"  Shape: {samples.shape}")
    print(f"  Range: [{samples.min():.3f}, {samples.max():.3f}]")
    
    # Save samples
    os.makedirs('./temp_data', exist_ok=True)
    sample_path = './temp_data/generated_samples.pt'
    th.save(samples, sample_path)
    print(f"  Saved to: {sample_path}")
    
    return samples


def step5_visualize_results(samples):
    """Step 5: Visualize generated samples."""
    print("\n" + "="*80)
    print("STEP 5: Visualize Results")
    print("="*80)
    
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(5, 1, figsize=(12, 10))
        
        for i in range(min(5, samples.shape[0])):
            axes[i].plot(samples[i, 0, :].numpy())
            axes[i].set_title(f"Generated Sample {i+1}")
            axes[i].set_xlabel("Time")
            axes[i].set_ylabel("Value")
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        os.makedirs('./temp_data', exist_ok=True)
        plot_path = './temp_data/generated_samples.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to: {plot_path}")
        
        # Also show statistics
        print(f"\nSample Statistics:")
        print(f"  Mean: {samples.mean():.3f}")
        print(f"  Std:  {samples.std():.3f}")
        print(f"  Min:  {samples.min():.3f}")
        print(f"  Max:  {samples.max():.3f}")
        
    except ImportError:
        print("  matplotlib not available, skipping visualization")


def step6_save_load_model(model):
    """Step 6: Demonstrate save/load functionality."""
    print("\n" + "="*80)
    print("STEP 6: Save and Load Model")
    print("="*80)
    
    # Save model
    os.makedirs('./temp_data', exist_ok=True)
    model_path = './temp_data/trained_model.pt'
    th.save(model.state_dict(), model_path)
    print(f"✓ Model saved to: {model_path}")
    
    # Load model (demonstrate reload)
    from improved_diffusion.temporal_script_util import create_temporal_model_and_diffusion
    
    new_model, _ = create_temporal_model_and_diffusion(
        sequence_length=256,
        num_channels=32,
        num_res_blocks=2,
        num_heads=2,
        num_heads_upsample=-1,
        attention_resolutions="32,16",
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
    
    new_model.load_state_dict(th.load(model_path))
    print(f"✓ Model reloaded successfully")
    
    # Verify
    x = th.randn(1, 1, 256)
    t = th.tensor([100])
    
    model.eval()
    new_model.eval()
    
    with th.no_grad():
        out1 = model(x, t)
        out2 = new_model(x, t)
    
    diff = (out1 - out2).abs().max()
    print(f"  Output difference: {diff:.10f} (should be ~0)")


def cleanup():
    """Clean up temporary files."""
    print("\n" + "="*80)
    print("CLEANUP")
    print("="*80)
    
    import shutil
    
    if os.path.exists('./temp_data'):
        shutil.rmtree('./temp_data')
        print("✓ Cleaned up temporary files")


def main():
    print("\n" + "#"*80)
    print("# COMPLETE WORKFLOW EXAMPLE: 1D TEMPORAL DIFFUSION")
    print("#"*80)
    print("\nThis demonstrates the full pipeline from data to samples.")
    print("For production use, please use the dedicated scripts in scripts/")
    
    # Set random seed
    th.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Run complete workflow
        data_path, seq_len = step1_create_data()
        model, diffusion = step2_create_model(seq_len)
        model = step3_train_model(model, diffusion, data_path, num_iterations=100)
        samples = step4_generate_samples(model, diffusion, num_samples=10)
        step5_visualize_results(samples)
        step6_save_load_model(model)
        
        print("\n" + "#"*80)
        print("# WORKFLOW COMPLETE! ✓")
        print("#"*80)
        
        print("\n📁 Generated Files:")
        print("  ./temp_data/example_data.pt       - Training data")
        print("  ./temp_data/trained_model.pt      - Trained model checkpoint")
        print("  ./temp_data/generated_samples.pt  - Generated samples")
        print("  ./temp_data/generated_samples.png - Visualization")
        
        print("\n🚀 Next Steps:")
        print("  1. For serious training: python scripts/temporal_train.py --data_path <your_data>.pt")
        print("  2. For sampling: python scripts/temporal_sample.py --model_path <checkpoint>.pt")
        print("  3. See QUICKSTART.md for detailed instructions")
        
        print("\n🧹 Cleanup:")
        response = input("  Delete temporary files? [y/N]: ")
        if response.lower() == 'y':
            cleanup()
        else:
            print("  Temporary files kept in ./temp_data/")
        
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

