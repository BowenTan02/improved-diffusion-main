"""
Generate synthetic 1D photon flux training data for SPAD imaging.

This script creates diverse flux functions φ(t) representing various
photon emission scenarios for training diffusion models. The function
catalogue is designed around what is "complex" for SPAD observation:
sudden transitions, sharp edges, and mixtures of smooth and discontinuous
behaviour — not just irregular wiggly curves.

Output:
    - flux_dataset.npy: shape [N, 1, SEQ_LENGTH], dtype=float32
    - metadata.json: parameters and pattern types for each sample
"""

import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
import json
from tqdm import tqdm
import os

# Configuration
NUM_SAMPLES = 2000
SEQ_LENGTH = 2000
T_MAX = 1.0
DT = T_MAX / SEQ_LENGTH
FLUX_MIN = 0.0
FLUX_MAX = 10000.0
SMOOTH_SIGMA = 1.0

PATTERN_DISTRIBUTION = {
    'step_nondecreasing': 0.06,
    'step_general': 0.06,
    'gaussian_bumps': 0.06,
    'exponential': 0.06,
    'wavelet': 0.06,
    'periodic': 0.06,
    'polynomial': 0.06,
    'pulsed_laser': 0.04,
    'flickering_light': 0.04,
    'sigmoid_transition': 0.05,
    'piecewise_linear': 0.05,
    'plateau_with_transients': 0.05,
    'chirp': 0.04,
    'lorentzian_peaks': 0.04,
    'sawtooth': 0.04,
    'constant': 0.01,
    'complex_combination': 0.22,
}

assert abs(sum(PATTERN_DISTRIBUTION.values()) - 1.0) < 1e-6, \
    "Pattern distribution must sum to 1.0"


def normalize_flux(flux, min_flux=FLUX_MIN, max_flux=FLUX_MAX):
    """Normalize flux to realistic photon count range."""
    flux_min = flux.min()
    flux_max = flux.max()
    if flux_max > flux_min:
        flux_norm = (flux - flux_min) / (flux_max - flux_min)
        flux_scaled = flux_norm * (max_flux - min_flux) + min_flux
    else:
        flux_scaled = np.ones_like(flux) * min_flux
    return flux_scaled


def smooth_flux(flux, sigma=SMOOTH_SIGMA):
    """Apply Gaussian smoothing to avoid aliasing."""
    return gaussian_filter1d(flux, sigma=sigma, mode='nearest')


# ============================================================================
# Required Pattern Generators
# ============================================================================

def generate_constant(t):
    """Constant flux: φ(t) = c"""
    c = np.random.uniform(FLUX_MIN, FLUX_MAX)
    flux = np.ones_like(t) * c
    metadata = {'type': 'constant', 'value': float(c)}
    return flux, metadata


def generate_step_nondecreasing(t):
    """Non-decreasing step function — models objects unblocking a light source."""
    num_steps = np.random.randint(2, 10)
    step_positions = np.sort(np.random.uniform(0, T_MAX, num_steps - 1))
    step_positions = np.concatenate([[0], step_positions, [T_MAX]])

    # Generate sorted (non-decreasing) amplitudes
    amplitudes = np.sort(np.random.uniform(FLUX_MIN, FLUX_MAX, num_steps))

    flux = np.zeros_like(t)
    for i in range(num_steps):
        mask = (t >= step_positions[i]) & (t < step_positions[i + 1])
        flux[mask] = amplitudes[i]

    metadata = {
        'type': 'step_nondecreasing',
        'num_steps': num_steps,
        'amplitudes': amplitudes.tolist(),
        'positions': step_positions[1:-1].tolist()
    }
    return flux, metadata


def generate_step_general(t):
    """General step function with arbitrary increases and decreases."""
    num_steps = np.random.randint(2, 10)
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
        'type': 'step_general',
        'num_steps': num_steps,
        'amplitudes': amplitudes,
        'positions': step_positions[1:-1].tolist()
    }
    return flux, metadata


def generate_gaussian_bumps(t):
    """Sum of Gaussian peaks at random locations."""
    num_bumps = np.random.randint(1, 12)
    flux = np.ones_like(t) * FLUX_MIN

    centers, amplitudes, widths = [], [], []
    for _ in range(num_bumps):
        center = np.random.uniform(0, T_MAX)
        amplitude = np.random.uniform(FLUX_MIN, FLUX_MAX)
        width = np.random.uniform(0.005, 0.15)
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


def generate_exponential(t):
    """Exponential growth or decay: φ(t) = A*exp(±bt)"""
    A = np.random.uniform(FLUX_MIN + 1, FLUX_MAX)
    b = np.random.uniform(-8, 8)
    flux = A * np.exp(b * t)
    metadata = {
        'type': 'exponential',
        'amplitude': float(A),
        'rate': float(b)
    }
    return flux, metadata


def generate_wavelet(t):
    """Wavelet-like functions: Ricker (Mexican hat) or Morlet wavelets."""
    wavelet_type = np.random.choice(['ricker', 'morlet'])
    num_wavelets = np.random.randint(1, 6)
    flux = np.zeros_like(t)

    params = []
    for _ in range(num_wavelets):
        center = np.random.uniform(0, T_MAX)
        width = np.random.uniform(0.01, 0.15)
        amp = np.random.uniform(FLUX_MAX * 0.1, FLUX_MAX)
        u = (t - center) / width

        if wavelet_type == 'ricker':
            # Mexican hat: (1 - u^2) * exp(-u^2/2)
            w = (1 - u ** 2) * np.exp(-u ** 2 / 2)
        else:
            # Morlet: cos(5u) * exp(-u^2/2)
            w = np.cos(5 * u) * np.exp(-u ** 2 / 2)

        flux += amp * w
        params.append({'center': float(center), 'width': float(width),
                        'amplitude': float(amp)})

    metadata = {
        'type': 'wavelet',
        'wavelet_type': wavelet_type,
        'num_wavelets': num_wavelets,
        'params': params
    }
    return flux, metadata


def generate_periodic(t):
    """Periodic functions: sine, square wave, triangle, or sawtooth combinations."""
    waveform = np.random.choice(['sine', 'square', 'triangle'])
    num_components = np.random.randint(1, 5)
    base_level = np.random.uniform(FLUX_MIN, FLUX_MAX / 2)
    flux = np.ones_like(t) * base_level

    components = []
    for _ in range(num_components):
        freq = np.random.uniform(1, 80)
        amp = np.random.uniform(10, FLUX_MAX / 4)
        phase = np.random.uniform(0, 2 * np.pi)

        if waveform == 'sine':
            flux += amp * np.sin(2 * np.pi * freq * t + phase)
        elif waveform == 'square':
            duty = np.random.uniform(0.2, 0.8)
            flux += amp * signal.square(2 * np.pi * freq * t + phase, duty=duty)
        elif waveform == 'triangle':
            flux += amp * signal.sawtooth(2 * np.pi * freq * t + phase, width=0.5)

        components.append({'freq': float(freq), 'amp': float(amp),
                            'phase': float(phase)})

    metadata = {
        'type': 'periodic',
        'waveform': waveform,
        'num_components': num_components,
        'base_level': float(base_level),
        'components': components
    }
    return flux, metadata


def generate_polynomial(t):
    """Random polynomial of degree 2–6."""
    degree = np.random.randint(2, 7)
    coeffs = np.random.randn(degree + 1) * FLUX_MAX / 2
    flux = np.polyval(coeffs, t / T_MAX)

    metadata = {
        'type': 'polynomial',
        'degree': degree,
        'coeffs': coeffs.tolist()
    }
    return flux, metadata


def generate_pulsed_laser(t):
    """Pulsed laser or strobe light pattern."""
    pulse_freq = np.random.uniform(1, 100)
    duty_cycle = np.random.uniform(0.05, 0.3)
    on_flux = np.random.uniform(FLUX_MAX / 2, FLUX_MAX)
    off_flux = np.random.uniform(FLUX_MIN, FLUX_MAX * 0.05)

    square = signal.square(2 * np.pi * pulse_freq * t, duty=duty_cycle)
    flux = off_flux + (on_flux - off_flux) * (square + 1) / 2

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


def generate_flickering_light(t):
    """Flickering light source (candle or faulty LED)."""
    base_freq = np.random.uniform(5, 50)
    base_flux = np.random.uniform(FLUX_MAX * 0.2, FLUX_MAX / 2)
    flux = base_flux * np.ones_like(t)

    num_components = np.random.randint(3, 8)
    for _ in range(num_components):
        freq = base_freq * np.random.uniform(0.5, 3)
        amplitude = np.random.uniform(0.1, 0.4) * base_flux
        phase = np.random.uniform(0, 2 * np.pi)
        flux += amplitude * np.sin(2 * np.pi * freq * t + phase)

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


# ============================================================================
# Additional SPAD-relevant Patterns
# ============================================================================

def generate_sigmoid_transition(t):
    """One or more sigmoid (smooth step) transitions — models gradual occlusion
    or reveal events that are the continuous analog of step functions."""
    num_transitions = np.random.randint(1, 5)
    flux = np.zeros_like(t)
    level = np.random.uniform(FLUX_MIN, FLUX_MAX)
    params = []
    for _ in range(num_transitions):
        next_level = np.random.uniform(FLUX_MIN, FLUX_MAX)
        midpoint = np.random.uniform(0.05, 0.95) * T_MAX
        steepness = np.random.uniform(10, 200)  # high → sharp, step-like
        flux += (next_level - level) / (1 + np.exp(-steepness * (t - midpoint)))
        params.append({'from': float(level), 'to': float(next_level),
                        'midpoint': float(midpoint), 'steepness': float(steepness)})
        level = next_level

    flux += params[0]['from']

    metadata = {
        'type': 'sigmoid_transition',
        'num_transitions': num_transitions,
        'params': params
    }
    return flux, metadata


def generate_piecewise_linear(t):
    """Piecewise linear (ramp) functions — constant-rate intensity changes
    between breakpoints, common in controlled illumination experiments."""
    num_knots = np.random.randint(3, 10)
    knot_t = np.sort(np.random.uniform(0, T_MAX, num_knots))
    knot_t[0] = 0.0
    knot_t[-1] = T_MAX
    knot_v = np.random.uniform(FLUX_MIN, FLUX_MAX, num_knots)

    flux = np.interp(t, knot_t, knot_v)

    metadata = {
        'type': 'piecewise_linear',
        'num_knots': num_knots,
        'knot_times': knot_t.tolist(),
        'knot_values': knot_v.tolist()
    }
    return flux, metadata


def generate_plateau_with_transients(t):
    """Long flat plateaus interrupted by sharp transient spikes or dips —
    the quintessential SPAD challenge (most of the signal is easy; the
    transient is where reconstruction quality matters)."""
    base_level = np.random.uniform(FLUX_MIN, FLUX_MAX * 0.6)
    flux = np.ones_like(t) * base_level

    num_transients = np.random.randint(1, 6)
    params = []
    for _ in range(num_transients):
        center = np.random.uniform(0.05, 0.95) * T_MAX
        width = np.random.uniform(0.002, 0.03)
        # Spike up or dip down
        direction = np.random.choice([-1, 1])
        amp = np.random.uniform(FLUX_MAX * 0.3, FLUX_MAX)
        flux += direction * amp * np.exp(-((t - center) ** 2) / (2 * width ** 2))
        params.append({'center': float(center), 'width': float(width),
                        'amplitude': float(direction * amp)})

    metadata = {
        'type': 'plateau_with_transients',
        'base_level': float(base_level),
        'num_transients': num_transients,
        'transients': params
    }
    return flux, metadata


def generate_chirp(t):
    """Chirp signal — frequency-swept sinusoid. Useful because it tests
    the prior's ability to represent varying local frequency content."""
    f0 = np.random.uniform(1, 10)
    f1 = np.random.uniform(30, 150)
    amp = np.random.uniform(FLUX_MAX * 0.3, FLUX_MAX)
    base = np.random.uniform(FLUX_MIN, FLUX_MAX * 0.4)
    method = np.random.choice(['linear', 'quadratic', 'logarithmic'])

    flux = base + amp * signal.chirp(t, f0, T_MAX, f1, method=method)

    metadata = {
        'type': 'chirp',
        'f0': float(f0),
        'f1': float(f1),
        'method': method,
        'amplitude': float(amp),
        'base': float(base)
    }
    return flux, metadata


def generate_lorentzian_peaks(t):
    """Sum of Lorentzian (Cauchy) peaks — heavier tails than Gaussians,
    modelling slower intensity fall-off from scattering events."""
    num_peaks = np.random.randint(1, 8)
    flux = np.ones_like(t) * FLUX_MIN

    params = []
    for _ in range(num_peaks):
        center = np.random.uniform(0, T_MAX)
        gamma = np.random.uniform(0.005, 0.08)  # half-width
        amp = np.random.uniform(FLUX_MAX * 0.1, FLUX_MAX)
        flux += amp * gamma ** 2 / ((t - center) ** 2 + gamma ** 2)
        params.append({'center': float(center), 'gamma': float(gamma),
                        'amplitude': float(amp)})

    metadata = {
        'type': 'lorentzian_peaks',
        'num_peaks': num_peaks,
        'params': params
    }
    return flux, metadata


def generate_sawtooth(t):
    """Sawtooth / asymmetric ramp wave — models linearly-ramping then
    quickly resetting intensity (e.g. scanning or charging behaviour)."""
    freq = np.random.uniform(2, 60)
    # width=1 → rising ramp, width=0 → falling ramp
    width = np.random.uniform(0.0, 1.0)
    amp = np.random.uniform(FLUX_MAX * 0.2, FLUX_MAX)
    base = np.random.uniform(FLUX_MIN, FLUX_MAX * 0.3)
    phase = np.random.uniform(0, 2 * np.pi)

    flux = base + amp * (signal.sawtooth(2 * np.pi * freq * t + phase, width=width) + 1) / 2

    metadata = {
        'type': 'sawtooth',
        'frequency': float(freq),
        'width': float(width),
        'amplitude': float(amp),
        'base': float(base)
    }
    return flux, metadata


# ============================================================================
# Complex Combination
# ============================================================================

def generate_complex_combination(t):
    """Complex combination of multiple patterns."""
    num_components = np.random.randint(2, 5)

    generators = [
        generate_step_nondecreasing,
        generate_step_general,
        generate_gaussian_bumps,
        generate_exponential,
        generate_wavelet,
        generate_periodic,
        generate_polynomial,
        generate_pulsed_laser,
        generate_flickering_light,
        generate_sigmoid_transition,
        generate_piecewise_linear,
        generate_plateau_with_transients,
        generate_chirp,
        generate_lorentzian_peaks,
        generate_sawtooth,
    ]

    selected_generators = np.random.choice(generators, size=num_components, replace=False)

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
# Main Generation
# ============================================================================

def generate_single_flux(t, pattern_type=None):
    """Generate a single flux function of specified or random type."""
    pattern_generators = {
        'constant': generate_constant,
        'step_nondecreasing': generate_step_nondecreasing,
        'step_general': generate_step_general,
        'gaussian_bumps': generate_gaussian_bumps,
        'exponential': generate_exponential,
        'wavelet': generate_wavelet,
        'periodic': generate_periodic,
        'polynomial': generate_polynomial,
        'pulsed_laser': generate_pulsed_laser,
        'flickering_light': generate_flickering_light,
        'sigmoid_transition': generate_sigmoid_transition,
        'piecewise_linear': generate_piecewise_linear,
        'plateau_with_transients': generate_plateau_with_transients,
        'chirp': generate_chirp,
        'lorentzian_peaks': generate_lorentzian_peaks,
        'sawtooth': generate_sawtooth,
        'complex_combination': generate_complex_combination,
    }

    if pattern_type is None:
        pattern_types = list(PATTERN_DISTRIBUTION.keys())
        probabilities = list(PATTERN_DISTRIBUTION.values())
        pattern_type = np.random.choice(pattern_types, p=probabilities)

    flux, metadata = pattern_generators[pattern_type](t)

    # Ensure non-negative
    flux = np.maximum(flux, 0)

    # Normalize to realistic range
    flux = normalize_flux(flux, min_flux=FLUX_MIN, max_flux=FLUX_MAX)

    # Apply smoothing
    flux = smooth_flux(flux, sigma=SMOOTH_SIGMA)

    metadata['flux_min'] = float(flux.min())
    metadata['flux_max'] = float(flux.max())
    metadata['flux_mean'] = float(flux.mean())
    metadata['flux_std'] = float(flux.std())

    return flux, metadata


def generate_dataset():
    """Generate complete dataset of flux functions."""
    print("=" * 80)
    print("Generating SPAD Photon Flux Training Dataset")
    print("=" * 80)
    print(f"Number of samples: {NUM_SAMPLES}")
    print(f"Sequence length: {SEQ_LENGTH}")
    print(f"Time range: [0, {T_MAX}] seconds")
    print(f"Flux range: [{FLUX_MIN}, {FLUX_MAX}] photons/sec")
    print()

    t = np.linspace(0, T_MAX, SEQ_LENGTH)
    flux_dataset = np.zeros((NUM_SAMPLES, 1, SEQ_LENGTH), dtype=np.float32)
    metadata_list = []
    pattern_counts = {key: 0 for key in PATTERN_DISTRIBUTION.keys()}

    print("Generating flux functions...")
    for i in tqdm(range(NUM_SAMPLES)):
        flux, metadata = generate_single_flux(t)
        flux_dataset[i, 0, :] = flux.astype(np.float32)
        metadata['sample_id'] = i
        metadata_list.append(metadata)
        pattern_counts[metadata['type']] += 1

    print("\n" + "=" * 80)
    print("Dataset Generation Complete!")
    print("=" * 80)

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
        print(f"  {pattern_type:30s}: {count:6d} ({percentage:5.2f}%, expected {expected_percentage:5.2f}%)")

    return flux_dataset, metadata_list


def save_dataset(flux_dataset, metadata_list, output_dir='./data'):
    """Save dataset and metadata to disk."""
    os.makedirs(output_dir, exist_ok=True)

    flux_path = os.path.join(output_dir, 'flux_dataset.npy')
    np.save(flux_path, flux_dataset)
    print(f"\n✓ Saved flux dataset to: {flux_path}")
    print(f"  Size: {os.path.getsize(flux_path) / 1024 / 1024:.2f} MB")

    eps = 1e-6
    flux_clamped = np.clip(flux_dataset, 0, None) + eps
    log_flux = np.log1p(flux_clamped)
    log_flux_path = os.path.join(output_dir, 'log_flux_dataset.npy')
    np.save(log_flux_path, log_flux)
    print(f"✓ Saved log flux dataset to: {log_flux_path}")
    print(f"  Size: {os.path.getsize(log_flux_path) / 1024 / 1024:.2f} MB")

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

    try:
        import torch
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

        fig, axes = plt.subplots(num_samples, 1, figsize=(14, 2 * num_samples))
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
    flux_dataset, metadata_list = generate_dataset()
    save_dataset(flux_dataset, metadata_list)
    visualize_samples(flux_dataset, num_samples=15)

    print("\n" + "=" * 80)
    print("✓ Dataset generation complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Train temporal diffusion model:")
    print("     python scripts/temporal_train.py --data_path data/log_flux_dataset.pt")
    print()


if __name__ == "__main__":
    main()
