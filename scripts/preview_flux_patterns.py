"""
Preview flux patterns before generating full dataset.

This script shows examples of all 13 pattern types without generating
the full 50,000 sample dataset. Useful for understanding the data before
committing to full generation.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Import pattern generators from main script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the pattern generation functions
try:
    from generate_spad_flux_dataset import (
        generate_constant,
        generate_step_function,
        generate_sinusoidal,
        generate_exponential,
        generate_polynomial_spline,
        generate_gaussian_bumps,
        generate_rotating_object,
        generate_flickering_light,
        generate_gradual_illumination,
        generate_pulsed_laser,
        generate_traffic_scene,
        generate_breathing_pattern,
        generate_complex_combination,
        ensure_positive,
        normalize_flux,
        smooth_flux,
        SEQ_LENGTH,
        T_MAX,
    )
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    print("Warning: Could not import from generate_spad_flux_dataset.py")
    print("Make sure the script is in the same directory.")


def preview_all_patterns():
    """Generate and visualize one example of each pattern type."""
    
    if not IMPORTS_AVAILABLE:
        print("Error: Required imports not available.")
        return
    
    # Time array
    t = np.linspace(0, T_MAX, SEQ_LENGTH)
    
    # Pattern generators
    patterns = [
        ('Constant', generate_constant),
        ('Step Function', generate_step_function),
        ('Sinusoidal', generate_sinusoidal),
        ('Exponential', generate_exponential),
        ('Polynomial Spline', generate_polynomial_spline),
        ('Gaussian Bumps', generate_gaussian_bumps),
        ('Rotating Object', generate_rotating_object),
        ('Flickering Light', generate_flickering_light),
        ('Gradual Illumination', generate_gradual_illumination),
        ('Pulsed Laser', generate_pulsed_laser),
        ('Traffic Scene', generate_traffic_scene),
        ('Breathing Pattern', generate_breathing_pattern),
        ('Complex Combination', generate_complex_combination),
    ]
    
    # Create figure
    n_patterns = len(patterns)
    fig, axes = plt.subplots(n_patterns, 1, figsize=(14, 2.5*n_patterns))
    
    print("\n" + "="*80)
    print("Generating Preview of All Flux Pattern Types")
    print("="*80 + "\n")
    
    for i, (name, generator) in enumerate(patterns):
        # Generate pattern
        flux, metadata = generator(t)
        
        # Apply processing (same as full generation)
        flux = ensure_positive(flux)
        flux = normalize_flux(flux)
        flux = smooth_flux(flux)
        
        # Plot
        axes[i].plot(t, flux, linewidth=1.0, color='#2E86AB')
        axes[i].set_ylabel('Flux\n(photons/s)', fontsize=9)
        axes[i].set_title(f'{i+1}. {name}', fontsize=11, fontweight='bold', loc='left')
        axes[i].grid(True, alpha=0.3, linestyle='--')
        axes[i].set_xlim(0, T_MAX)
        
        # Add statistics as text
        stats_text = f"Min: {flux.min():.0f}  Max: {flux.max():.0f}  Mean: {flux.mean():.0f}"
        axes[i].text(0.98, 0.95, stats_text, transform=axes[i].transAxes,
                    fontsize=8, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Print metadata
        print(f"{i+1:2d}. {name:25s} - {metadata['type']}")
        
        if i == n_patterns - 1:
            axes[i].set_xlabel('Time (s)', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    output_path = 'flux_patterns_preview.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved preview to: {output_path}")
    
    # Show
    plt.show()
    
    print("\n" + "="*80)
    print("Preview Complete!")
    print("="*80)
    print("\nTo generate the full dataset (50,000 samples):")
    print("  python examples/generate_spad_flux_dataset.py")
    print()


def preview_pattern_variety(pattern_name, num_examples=5):
    """Show variety within a single pattern type."""
    
    if not IMPORTS_AVAILABLE:
        print("Error: Required imports not available.")
        return
    
    # Time array
    t = np.linspace(0, T_MAX, SEQ_LENGTH)
    
    # Map pattern names to generators
    pattern_map = {
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
    
    if pattern_name not in pattern_map:
        print(f"Error: Unknown pattern '{pattern_name}'")
        print(f"Available patterns: {list(pattern_map.keys())}")
        return
    
    generator = pattern_map[pattern_name]
    
    # Create figure
    fig, axes = plt.subplots(num_examples, 1, figsize=(14, 2*num_examples))
    if num_examples == 1:
        axes = [axes]
    
    print(f"\n{'='*80}")
    print(f"Showing {num_examples} variations of: {pattern_name}")
    print('='*80 + '\n')
    
    for i in range(num_examples):
        # Generate pattern
        flux, metadata = generator(t)
        
        # Apply processing
        flux = ensure_positive(flux)
        flux = normalize_flux(flux)
        flux = smooth_flux(flux)
        
        # Plot
        axes[i].plot(t, flux, linewidth=1.0, color='#A23B72')
        axes[i].set_ylabel('Flux\n(photons/s)', fontsize=9)
        axes[i].set_title(f'Variation {i+1}', fontsize=10, fontweight='bold', loc='left')
        axes[i].grid(True, alpha=0.3, linestyle='--')
        axes[i].set_xlim(0, T_MAX)
        
        # Statistics
        stats_text = f"Min: {flux.min():.0f}  Max: {flux.max():.0f}  Mean: {flux.mean():.0f}"
        axes[i].text(0.98, 0.95, stats_text, transform=axes[i].transAxes,
                    fontsize=8, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        if i == num_examples - 1:
            axes[i].set_xlabel('Time (s)', fontsize=10)
        
        # Print some metadata
        print(f"Variation {i+1}: ", end='')
        for key, value in list(metadata.items())[:3]:
            if key != 'type':
                print(f"{key}={value}", end='  ')
        print()
    
    plt.tight_layout()
    
    # Save
    output_path = f'flux_pattern_{pattern_name}_variations.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved to: {output_path}")
    
    plt.show()


def compare_realistic_patterns():
    """Compare the realistic scene patterns side-by-side."""
    
    if not IMPORTS_AVAILABLE:
        print("Error: Required imports not available.")
        return
    
    # Time array
    t = np.linspace(0, T_MAX, SEQ_LENGTH)
    
    # Realistic patterns
    realistic_patterns = [
        ('Rotating Object', generate_rotating_object),
        ('Flickering Light', generate_flickering_light),
        ('Gradual Illumination', generate_gradual_illumination),
        ('Pulsed Laser', generate_pulsed_laser),
        ('Traffic Scene', generate_traffic_scene),
        ('Breathing Pattern', generate_breathing_pattern),
        ('Complex Combination', generate_complex_combination),
    ]
    
    fig, axes = plt.subplots(len(realistic_patterns), 1, figsize=(14, 2*len(realistic_patterns)))
    
    print("\n" + "="*80)
    print("Realistic SPAD Scene Patterns")
    print("="*80 + "\n")
    
    for i, (name, generator) in enumerate(realistic_patterns):
        flux, metadata = generator(t)
        flux = ensure_positive(flux)
        flux = normalize_flux(flux)
        flux = smooth_flux(flux)
        
        axes[i].plot(t, flux, linewidth=1.0, color='#F18F01')
        axes[i].set_ylabel('Flux\n(photons/s)', fontsize=9)
        axes[i].set_title(f'{name} 🌟', fontsize=11, fontweight='bold', loc='left')
        axes[i].grid(True, alpha=0.3, linestyle='--')
        axes[i].set_xlim(0, T_MAX)
        
        stats_text = f"Range: [{flux.min():.0f}, {flux.max():.0f}]"
        axes[i].text(0.98, 0.95, stats_text, transform=axes[i].transAxes,
                    fontsize=8, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        if i == len(realistic_patterns) - 1:
            axes[i].set_xlabel('Time (s)', fontsize=10)
        
        print(f"{name:25s} - Physical model with realistic dynamics")
    
    plt.tight_layout()
    
    output_path = 'flux_realistic_patterns.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved to: {output_path}")
    
    plt.show()
    
    print("\n" + "="*80)
    print("These patterns represent physically-motivated scenarios")
    print("="*80 + "\n")


def main():
    """Main preview function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preview SPAD flux patterns')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'realistic', 'variety'],
                       help='Preview mode: all patterns, realistic only, or variations')
    parser.add_argument('--pattern', type=str, default='rotating_object',
                       help='Pattern name for variety mode')
    parser.add_argument('--num', type=int, default=5,
                       help='Number of variations to show')
    
    args = parser.parse_args()
    
    if args.mode == 'all':
        preview_all_patterns()
    elif args.mode == 'realistic':
        compare_realistic_patterns()
    elif args.mode == 'variety':
        preview_pattern_variety(args.pattern, args.num)


if __name__ == "__main__":
    # Check if matplotlib is available
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required for previewing patterns")
        print("Install with: pip install matplotlib")
        exit(1)
    
    main()

