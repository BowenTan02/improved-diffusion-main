"""
Generate synthetic 1D photon flux training data for SPAD imaging.

This script creates 50,000 diverse flux functions φ(t) representing various
realistic and synthetic photon emission scenarios for training diffusion models.

Output:
    - flux_dataset.npy: shape [50000, 1, 1024], dtype=float32
    - metadata.json: parameters and pattern types for each sample
"""

import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline, interp1d
from scipy.ndimage import gaussian_filter1d
import json
from tqdm import tqdm
import os

# Configuration
NUM_SAMPLES = 100000
SEQ_LENGTH = 2000
T_MAX = 1.0  # Time range [0, 1] seconds (can be scaled)
DT = T_MAX / SEQ_LENGTH
FLUX_MIN = 0.0  # photons/sec
FLUX_MAX = 100000.0  # photons/sec
SMOOTH_SIGMA = 1.0  # Gaussian smoothing

# Pattern distribution (must sum to 1.0)
PATTERN_DISTRIBUTION = {
    'constant': 0.01,
    'step_function': 0.02,
    'sinusoidal': 0.10,
    'exponential': 0.06,
    'polynomial_spline': 0.10,
    'gaussian_bumps': 0.03,
    'rotating_object': 0.02,
    'gradual_illumination': 0.06,
    'traffic_scene': 0.02,
    'breathing_pattern': 0.02,
    'complex_combination': 0.56,
}

assert abs(sum(PATTERN_DISTRIBUTION.values()) - 1.0) < 1e-6, "Pattern distribution must sum to 1.0"


def normalize_flux(flux, min_flux=FLUX_MIN, max_flux=FLUX_MAX):
    """Normalize flux to realistic photon count range."""
    flux_min = flux.min()
    flux_max = flux.max()
    
    if flux_max > flux_min:
        # Normalize to [0, 1]
        flux_norm = (flux - flux_min) / (flux_max - flux_min)
        # Scale to [min_flux, max_flux]
        flux_scaled = flux_norm * (max_flux - min_flux) + min_flux
    else:
        flux_scaled = np.ones_like(flux) * min_flux
    
    return flux_scaled


def smooth_flux(flux, sigma=SMOOTH_SIGMA):
    """Apply Gaussian smoothing to avoid aliasing."""
    return gaussian_filter1d(flux, sigma = sigma, mode = 'nearest')


# ============================================================================
# Pattern Generation Functions
# ============================================================================

def generate_constant(t):
    """Constant flux: φ(t) = c"""
    c = np.random.uniform(FLUX_MIN, FLUX_MAX)
    flux = np.ones_like(t) * c
    metadata = {'type': 'constant', 'value': float(c)}
    return flux, metadata

def generate_step_function(t):
    """Step function with random number of steps."""
    num_steps = np.random.randint(2, 8)
    step_positions = np.sort(np.random.uniform(0, T_MAX, num_steps - 1))
    step_positions = np.concatenate([[0], step_positions, [T_MAX]])
    
    flux = np.zeros_like(t)
    amplitudes = []
    
    for i in range(num_steps):
        amplitude = np.random.uniform(FLUX_MIN, FLUX_MAX)
        amplitudes.append(float(amplitude))
        mask = (t >= step_positions[i]) & (t < step_positions[i + 1])
        flux[mask] = amplitude
    
    metadata = {
        'type': 'step_function',
        'num_steps': num_steps,
        'amplitudes': amplitudes,
        'positions': step_positions[1:-1].tolist()
    }
    return flux, metadata


def generate_sinusoidal(t):
    """Sum of sinusoids with random frequencies and amplitudes."""
    num_sines = np.random.randint(1, 6)
    flux = np.zeros_like(t)
    
    frequencies = []
    amplitudes = []
    phases = []
    
    base_level = np.random.uniform(FLUX_MIN, FLUX_MAX / 2)
    
    for _ in range(num_sines):
        freq = np.random.uniform(1, 100)  # Hz
        amp = np.random.uniform(10, FLUX_MAX / 4)
        phase = np.random.uniform(0, 2 * np.pi)
        
        frequencies.append(float(freq))
        amplitudes.append(float(amp))
        phases.append(float(phase))
        
        flux += amp * np.sin(2 * np.pi * freq * t + phase)
    
    flux += base_level
    
    metadata = {
        'type': 'sinusoidal',
        'num_components': num_sines,
        'frequencies': frequencies,
        'amplitudes': amplitudes,
        'phases': phases,
        'base_level': float(base_level)
    }
    return flux, metadata


def generate_exponential(t):
    """Exponential growth or decay: φ(t) = A*exp(±bt)"""
    A = np.random.uniform(FLUX_MIN, FLUX_MAX)
    b = np.random.uniform(-5, 5)
    
    flux = A * np.exp(b * t)
    
    metadata = {
        'type': 'exponential',
        'amplitude': float(A),
        'rate': float(b)
    }
    return flux, metadata


def generate_polynomial_spline(t):
    """Piecewise cubic spline with random control points."""
    num_points = np.random.randint(4, 10)
    control_t = np.sort(np.random.uniform(0, T_MAX, num_points))
    control_flux = np.random.uniform(FLUX_MIN, FLUX_MAX, num_points)
    
    spline = CubicSpline(control_t, control_flux, bc_type='natural')
    flux = spline(t)
    
    metadata = {
        'type': 'polynomial_spline',
        'num_control_points': num_points,
        'control_times': control_t.tolist(),
        'control_values': control_flux.tolist()
    }
    return flux, metadata


def generate_gaussian_bumps(t):
    """Sum of Gaussian peaks at random locations."""
    num_bumps = np.random.randint(2, 11)
    flux = np.ones_like(t) * FLUX_MIN
    
    centers = []
    amplitudes = []
    widths = []
    
    for _ in range(num_bumps):
        center = np.random.uniform(0, T_MAX)
        amplitude = np.random.uniform(FLUX_MIN, FLUX_MAX)
        width = np.random.uniform(0.01, 0.1)
        
        centers.append(float(center))
        amplitudes.append(float(amplitude))
        widths.append(float(width))
        
        flux += amplitude * np.exp(-((t - center) ** 2) / (2 * width ** 2))
    
    metadata = {
        'type': 'gaussian_bumps',
        'num_bumps': num_bumps,
        'centers': centers,
        'amplitudes': amplitudes,
        'widths': widths
    }
    return flux, metadata


# ============================================================================
# Realistic Scene Patterns
# ============================================================================

def generate_rotating_object(t):
    """Periodic flux from rotating object (like lighthouse or rotating fan)."""
    rotation_freq = np.random.uniform(0.5, 20)  # Hz
    num_lobes = np.random.randint(1, 5)  # Number of bright spots
    base_flux = np.random.uniform(FLUX_MIN, FLUX_MIN * 5)
    peak_flux = np.random.uniform(FLUX_MAX / 2, FLUX_MAX)
    
    flux = base_flux + np.zeros_like(t)
    
    for i in range(num_lobes):
        phase = np.random.uniform(0, 2 * np.pi)
        # Sharp peaks when lobe faces detector
        flux += (peak_flux - base_flux) * np.exp(
            -10 * (np.sin(2 * np.pi * rotation_freq * t + phase + i * 2 * np.pi / num_lobes)) ** 2
        )
    
    metadata = {
        'type': 'rotating_object',
        'rotation_frequency': float(rotation_freq),
        'num_lobes': num_lobes,
        'base_flux': float(base_flux),
        'peak_flux': float(peak_flux)
    }
    return flux, metadata


def generate_flickering_light(t):
    """Flickering light source (like candle or faulty LED)."""
    # Base frequency
    base_freq = np.random.uniform(5, 50)
    base_flux = np.random.uniform(FLUX_MIN, FLUX_MAX / 2)
    
    # Multiple frequency components for realistic flicker
    flux = base_flux * np.ones_like(t)
    
    num_components = np.random.randint(3, 8)
    for i in range(num_components):
        freq = base_freq * np.random.uniform(0.5, 3)
        amplitude = np.random.uniform(0.1, 0.4) * base_flux
        phase = np.random.uniform(0, 2 * np.pi)
        flux += amplitude * np.sin(2 * np.pi * freq * t + phase)
    
    # Add random spikes (sudden flickers)
    num_spikes = np.random.randint(0, 5)
    for _ in range(num_spikes):
        spike_t = np.random.uniform(0, T_MAX)
        spike_width = np.random.uniform(0.001, 0.01)
        spike_amp = np.random.uniform(base_flux, FLUX_MAX)
        flux += spike_amp * np.exp(-((t - spike_t) ** 2) / (2 * spike_width ** 2))
    
    metadata = {
        'type': 'flickering_light',
        'base_frequency': float(base_freq),
        'base_flux': float(base_flux),
        'num_components': num_components,
        'num_spikes': num_spikes
    }
    return flux, metadata


def generate_gradual_illumination(t):
    """Gradual illumination change (sunrise/sunset, dimmer adjustment)."""
    change_type = np.random.choice(['linear', 'sigmoid', 'exponential'])
    start_flux = np.random.uniform(FLUX_MIN, FLUX_MAX)
    end_flux = np.random.uniform(FLUX_MIN, FLUX_MAX)
    
    if change_type == 'linear':
        flux = start_flux + (end_flux - start_flux) * (t / T_MAX)
    elif change_type == 'sigmoid':
        midpoint = np.random.uniform(0.3, 0.7) * T_MAX
        steepness = np.random.uniform(5, 20)
        flux = start_flux + (end_flux - start_flux) / (1 + np.exp(-steepness * (t - midpoint)))
    else:  # exponential
        rate = np.random.uniform(-3, 3)
        flux = start_flux + (end_flux - start_flux) * (np.exp(rate * t / T_MAX) - 1) / (np.exp(rate) - 1)
    
    # Add small fluctuations
    noise_level = np.random.uniform(0.01, 0.05) * abs(end_flux - start_flux)
    flux += noise_level * np.random.randn(len(t))
    
    metadata = {
        'type': 'gradual_illumination',
        'change_type': change_type,
        'start_flux': float(start_flux),
        'end_flux': float(end_flux)
    }
    return flux, metadata


def generate_pulsed_laser(t):
    """Pulsed laser or strobe light pattern."""
    pulse_freq = np.random.uniform(1, 100)  # Hz
    duty_cycle = np.random.uniform(0.05, 0.3)
    on_flux = np.random.uniform(FLUX_MAX / 2, FLUX_MAX)
    off_flux = np.random.uniform(FLUX_MIN, FLUX_MIN * 10)
    
    # Square wave
    square = signal.square(2 * np.pi * pulse_freq * t, duty=duty_cycle)
    flux = off_flux + (on_flux - off_flux) * (square + 1) / 2
    
    # Add rise/fall time (realistic laser pulses aren't perfectly square)
    rise_time = np.random.uniform(0.0001, 0.001)
    flux = gaussian_filter1d(flux, sigma=rise_time / DT)
    
    metadata = {
        'type': 'pulsed_laser',
        'pulse_frequency': float(pulse_freq),
        'duty_cycle': float(duty_cycle),
        'on_flux': float(on_flux),
        'off_flux': float(off_flux)
    }
    return flux, metadata


def generate_traffic_scene(t):
    """Simulate traffic scene with passing cars (headlights)."""
    num_cars = np.random.randint(2, 8)
    base_ambient = np.random.uniform(FLUX_MIN, FLUX_MIN * 20)
    flux = base_ambient * np.ones_like(t)
    
    car_times = []
    car_intensities = []
    
    for _ in range(num_cars):
        # Car passing time and speed
        car_t = np.random.uniform(0.1, 0.9) * T_MAX
        car_width = np.random.uniform(0.02, 0.1)  # Duration of passing
        car_intensity = np.random.uniform(FLUX_MAX / 3, FLUX_MAX)
        
        car_times.append(float(car_t))
        car_intensities.append(float(car_intensity))
        
        # Gaussian profile as car passes
        flux += car_intensity * np.exp(-((t - car_t) ** 2) / (2 * car_width ** 2))
    
    # Add street light flicker
    street_light_flux = np.random.uniform(FLUX_MIN * 5, FLUX_MIN * 30)
    street_light_flicker = street_light_flux * (1 + 0.05 * np.sin(2 * np.pi * 60 * t))  # 60 Hz flicker
    flux += street_light_flicker
    
    metadata = {
        'type': 'traffic_scene',
        'num_cars': num_cars,
        'car_times': car_times,
        'car_intensities': car_intensities,
        'base_ambient': float(base_ambient)
    }
    return flux, metadata


def generate_breathing_pattern(t):
    """Biological breathing pattern (for biomedical imaging)."""
    breath_rate = np.random.uniform(0.2, 0.5)  # Hz (12-30 breaths/min)
    heartbeat_rate = np.random.uniform(1.0, 1.5)  # Hz (60-90 bpm)
    
    base_flux = np.random.uniform(FLUX_MIN * 10, FLUX_MAX / 2)
    
    # Breathing component (sinusoidal)
    breath_amp = np.random.uniform(0.1, 0.3) * base_flux
    breath = breath_amp * np.sin(2 * np.pi * breath_rate * t)
    
    # Heartbeat component (sharper peaks)
    heartbeat_amp = np.random.uniform(0.05, 0.15) * base_flux
    heartbeat = heartbeat_amp * signal.square(2 * np.pi * heartbeat_rate * t, duty=0.3)
    heartbeat = gaussian_filter1d(heartbeat, sigma=0.5 / DT)  # Smooth
    
    flux = base_flux + breath + heartbeat
    
    metadata = {
        'type': 'breathing_pattern',
        'breath_rate': float(breath_rate),
        'heartbeat_rate': float(heartbeat_rate),
        'base_flux': float(base_flux)
    }
    return flux, metadata


def generate_complex_combination(t):
    """Complex combination of multiple realistic patterns."""
    # Randomly combine 2-4 patterns
    num_components = np.random.randint(2, 5)
    
    # Available pattern generators (excluding constant and complex_combination itself)
    generators = [
        generate_sinusoidal,
        generate_gaussian_bumps,
        generate_rotating_object,
        generate_flickering_light,
        generate_exponential,
    ]
    
    selected_generators = np.random.choice(generators, size = num_components, replace = False)
    
    flux = np.zeros_like(t)
    component_metadata = []
    
    for gen in selected_generators:
        component_flux, component_meta = gen(t)
        weight = np.random.uniform(0.3, 1.0)
        flux += weight * component_flux
        component_metadata.append({'weight': float(weight), 'component': component_meta})
    
    metadata = {
        'type': 'complex_combination',
        'num_components': num_components,
        'components': component_metadata
    }
    return flux, metadata


# ============================================================================
# Main Generation Function
# ============================================================================

def generate_single_flux(t, pattern_type=None):
    """Generate a single flux function of specified or random type."""
    pattern_generators = {
        'constant': generate_constant,
        'step_function': generate_step_function,
        'sinusoidal': generate_sinusoidal,
        'exponential': generate_exponential,
        'polynomial_spline': generate_polynomial_spline,
        'gaussian_bumps': generate_gaussian_bumps,
        'rotating_object': generate_rotating_object,
        'flickering_light': generate_flickering_light,
        'gradual_illumination': generate_gradual_illumination,
        'pulsed_laser': generate_pulsed_laser,
        'traffic_scene': generate_traffic_scene,
        'breathing_pattern': generate_breathing_pattern,
        'complex_combination': generate_complex_combination,
    }
    
    if pattern_type is None:
        # Sample pattern type according to distribution
        pattern_types = list(PATTERN_DISTRIBUTION.keys())
        probabilities = list(PATTERN_DISTRIBUTION.values())
        pattern_type = np.random.choice(pattern_types, p=probabilities)
    
    # Generate flux
    flux, metadata = pattern_generators[pattern_type](t)
    
    # Ensure non-negative
    flux = np.maximum(flux, 0)
    
    # Normalize to realistic range
    flux = normalize_flux(flux, min_flux = FLUX_MIN, max_flux = FLUX_MAX)
    
    # Apply smoothing
    flux = smooth_flux(flux, sigma = SMOOTH_SIGMA)
    
    # Add metadata
    metadata['flux_min'] = float(flux.min())
    metadata['flux_max'] = float(flux.max())
    metadata['flux_mean'] = float(flux.mean())
    metadata['flux_std'] = float(flux.std())
    
    return flux, metadata


def generate_dataset():
    """Generate complete dataset of flux functions."""
    print("="*80)
    print("Generating SPAD Photon Flux Training Dataset")
    print("="*80)
    print(f"Number of samples: {NUM_SAMPLES}")
    print(f"Sequence length: {SEQ_LENGTH}")
    print(f"Time range: [0, {T_MAX}] seconds")
    print(f"Flux range: [{FLUX_MIN}, {FLUX_MAX}] photons/sec")
    print()
    
    # Time array
    t = np.linspace(0, T_MAX, SEQ_LENGTH)
    
    # Preallocate arrays
    flux_dataset = np.zeros((NUM_SAMPLES, 1, SEQ_LENGTH), dtype=np.float32)
    metadata_list = []
    
    # Pattern statistics
    pattern_counts = {key: 0 for key in PATTERN_DISTRIBUTION.keys()}
    
    # Generate samples
    print("Generating flux functions...")
    for i in tqdm(range(NUM_SAMPLES)):
        flux, metadata = generate_single_flux(t)
        
        # Store
        flux_dataset[i, 0, :] = flux.astype(np.float32)
        metadata['sample_id'] = i
        metadata_list.append(metadata)
        
        # Count patterns
        pattern_counts[metadata['type']] += 1
    
    print("\n" + "="*80)
    print("Dataset Generation Complete!")
    print("="*80)
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Total samples: {NUM_SAMPLES}")
    print(f"  Shape: {flux_dataset.shape}")
    print(f"  Dtype: {flux_dataset.dtype}")
    print(f"  Overall flux range: [{flux_dataset.min():.2f}, {flux_dataset.max():.2f}]")
    print(f"  Overall flux mean: {flux_dataset.mean():.2f}")
    print(f"  Overall flux std: {flux_dataset.std():.2f}")
    
    print("\nPattern Distribution:")
    for pattern_type, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        percentage = 100 * count / NUM_SAMPLES
        expected_percentage = 100 * PATTERN_DISTRIBUTION[pattern_type]
        print(f"  {pattern_type:25s}: {count:6d} ({percentage:5.2f}%, expected {expected_percentage:5.2f}%)")
    
    return flux_dataset, metadata_list


def save_dataset(flux_dataset, metadata_list, output_dir='./data'):
    """Save dataset and metadata to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save flux data
    flux_path = os.path.join(output_dir, 'flux_dataset.npy')
    np.save(flux_path, flux_dataset)
    print(f"\n✓ Saved flux dataset to: {flux_path}")
    print(f"  Size: {os.path.getsize(flux_path) / 1024 / 1024:.2f} MB")
    
    eps = 1e-6  # small floor; adjust if needed
    flux_clamped = np.clip(flux_dataset, 0, None) + eps  # avoid negatives/zeros
    log_flux = np.log1p(flux_clamped)
    log_flux_path = os.path.join(output_dir, 'log_flux_dataset.npy')
    np.save(log_flux_path, log_flux)
    print(f"✓ Saved log flux dataset to: {log_flux_path}")
    print(f"  Size: {os.path.getsize(log_flux_path) / 1024 / 1024:.2f} MB")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump({
            'dataset_info': {
                'num_samples': NUM_SAMPLES,
                'sequence_length': SEQ_LENGTH,
                'time_range': [0, T_MAX],
                'dt': DT,
                'flux_range': [FLUX_MIN, FLUX_MAX],
                'log_flux_range': [float(log_flux.min()), float(log_flux.max())],
                'smooth_sigma': SMOOTH_SIGMA,
                'pattern_distribution': PATTERN_DISTRIBUTION,
            },
            'samples': metadata_list
        }, f, indent=2)
    print(f"✓ Saved metadata to: {metadata_path}")
    
    # Also save as .pt for direct use with training script
    try:
        import torch
        # Remove channel dimension for .pt format: [50000, 1024]
        log_flux_pt = torch.from_numpy(log_flux[:, 0, :])
        log_flux_pt_path = os.path.join(output_dir, 'log_flux_dataset.pt')
        torch.save(log_flux_pt, log_flux_pt_path)
        print(f"✓ Saved PyTorch log flux dataset to: {log_flux_pt_path}")
        print(f"  Shape: {log_flux_pt.shape}")
    except ImportError:
        print("  (PyTorch not available, skipping .pt format)")
    

def visualize_samples(flux_dataset, num_samples=10):
    """Visualize some sample flux functions."""
    try:
        import matplotlib.pyplot as plt
        
        print(f"\nGenerating visualization of {num_samples} samples...")
        
        fig, axes = plt.subplots(num_samples, 1, figsize=(14, 2*num_samples))
        if num_samples == 1:
            axes = [axes]
        
        t = np.linspace(0, T_MAX, SEQ_LENGTH)
        
        for i in range(num_samples):
            idx = np.random.randint(0, flux_dataset.shape[0])
            flux = flux_dataset[idx, 0, :]
            
            axes[i].plot(t, flux, linewidth=0.8)
            axes[i].set_ylabel('Flux\n(photons/s)', fontsize=9)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(0, T_MAX)
            
            if i == num_samples - 1:
                axes[i].set_xlabel('Time (s)', fontsize=10)
        
        plt.tight_layout()
        
        output_dir = './data'
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'flux_samples_visualization.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to: {plot_path}")
        plt.close()
        
    except ImportError:
        print("  (matplotlib not available, skipping visualization)")


def main():
    """Main execution function."""
    # Set random seed for reproducibility (optional)
    # np.random.seed(42)
    
    # Generate dataset
    flux_dataset, metadata_list = generate_dataset()
    
    # Save to disk
    save_dataset(flux_dataset, metadata_list)
    
    # Visualize samples
    visualize_samples(flux_dataset, num_samples=15)
    
    print("\n" + "="*80)
    print("✓ Dataset generation complete!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Train temporal diffusion model:")
    print("     python scripts/temporal_train.py --data_path data/log_flux_dataset.pt")
    print()

if __name__ == "__main__":
    main()

