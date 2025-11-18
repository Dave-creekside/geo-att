#!/usr/bin/env python3
"""
Visualize geometry evolution during training.
Creates timeline plots showing hyperbolic/euclidean/spherical percentage over epochs.
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_geometry_data(experiment_dir):
    """
    Load geometry data from experiment logs.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        List of geometry data dictionaries
    """
    geometry_file = Path(experiment_dir) / 'logs' / 'geometry.jsonl'
    
    if not geometry_file.exists():
        raise FileNotFoundError(f"Geometry log not found: {geometry_file}")
    
    data = []
    with open(geometry_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    return data


def plot_geometry_timeline(geometry_data, output_path=None, show=True):
    """
    Create stacked area chart of geometry evolution.
    
    Args:
        geometry_data: List of geometry dictionaries
        output_path: Path to save figure (optional)
        show: Whether to display figure
    """
    # Extract data
    epochs = [d['epoch'] for d in geometry_data]
    steps = [d['step'] for d in geometry_data]
    
    h_pcts = [d['geometry_percentages']['hyperbolic'] for d in geometry_data]
    e_pcts = [d['geometry_percentages']['euclidean'] for d in geometry_data]
    s_pcts = [d['geometry_percentages']['spherical'] for d in geometry_data]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Stacked area chart
    ax1.fill_between(epochs, 0, h_pcts, alpha=0.7, label='Hyperbolic', color='#e74c3c')
    ax1.fill_between(epochs, h_pcts, np.array(h_pcts) + np.array(e_pcts), 
                     alpha=0.7, label='Euclidean', color='#3498db')
    ax1.fill_between(epochs, np.array(h_pcts) + np.array(e_pcts), 100, 
                     alpha=0.7, label='Spherical', color='#2ecc71')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Percentage of Heads (%)', fontsize=12)
    ax1.set_title('Geometry Distribution Evolution (Stacked Area)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Plot 2: Individual line plots
    ax2.plot(epochs, h_pcts, 'o-', label='Hyperbolic', color='#e74c3c', linewidth=2, markersize=4)
    ax2.plot(epochs, e_pcts, 's-', label='Euclidean', color='#3498db', linewidth=2, markersize=4)
    ax2.plot(epochs, s_pcts, '^-', label='Spherical', color='#2ecc71', linewidth=2, markersize=4)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Percentage of Heads (%)', fontsize=12)
    ax2.set_title('Geometry Distribution Evolution (Line Plot)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # Detect if there's a significant Euclidean shift
    if len(e_pcts) > 10:
        max_e = max(e_pcts)
        if max_e > 15:  # Significant Euclidean presence
            max_e_idx = e_pcts.index(max_e)
            max_e_epoch = epochs[max_e_idx]
            ax2.axvline(max_e_epoch, color='red', linestyle='--', alpha=0.5, linewidth=2)
            ax2.text(max_e_epoch, max_e + 5, f'Peak E: {max_e:.1f}%\nEpoch {max_e_epoch}',
                    ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to: {output_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_curvature_statistics(geometry_data, output_path=None, show=True):
    """
    Plot curvature statistics over time.
    
    Args:
        geometry_data: List of geometry dictionaries
        output_path: Path to save figure
        show: Whether to display figure
    """
    epochs = [d['epoch'] for d in geometry_data]
    means = [d['curvature_stats']['mean'] for d in geometry_data]
    stds = [d['curvature_stats']['std'] for d in geometry_data]
    mins = [d['curvature_stats']['min'] for d in geometry_data]
    maxs = [d['curvature_stats']['max'] for d in geometry_data]
    medians = [d['curvature_stats']['median'] for d in geometry_data]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot mean with std shading
    ax.plot(epochs, means, 'b-', label='Mean', linewidth=2)
    ax.fill_between(epochs, 
                    np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    alpha=0.3, label='±1 Std Dev')
    
    # Plot median
    ax.plot(epochs, medians, 'g--', label='Median', linewidth=1.5)
    
    # Plot min/max
    ax.plot(epochs, mins, 'r:', label='Min', linewidth=1, alpha=0.7)
    ax.plot(epochs, maxs, 'r:', label='Max', linewidth=1, alpha=0.7)
    
    # Add horizontal lines for geometry boundaries
    ax.axhline(y=-0.1, color='gray', linestyle='--', alpha=0.3)
    ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.3)
    ax.text(epochs[-1], -0.1, 'Hyperbolic', ha='right', va='bottom', fontsize=9)
    ax.text(epochs[-1], 0.1, 'Spherical', ha='right', va='top', fontsize=9)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Curvature Value', fontsize=12)
    ax.set_title('Curvature Statistics Over Training', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to: {output_path}")
    
    if show:
        plt.show()
    
    return fig


def print_summary(geometry_data):
    """Print summary statistics."""
    if not geometry_data:
        print("No geometry data found.")
        return
    
    print("\n" + "="*70)
    print("GEOMETRY EVOLUTION SUMMARY")
    print("="*70)
    
    # Initial vs Final
    initial = geometry_data[0]
    final = geometry_data[-1]
    
    print(f"\nInitial (Epoch {initial['epoch']}):")
    print(f"  Hyperbolic: {initial['geometry_percentages']['hyperbolic']:.1f}%")
    print(f"  Euclidean:  {initial['geometry_percentages']['euclidean']:.1f}%")
    print(f"  Spherical:  {initial['geometry_percentages']['spherical']:.1f}%")
    
    print(f"\nFinal (Epoch {final['epoch']}):")
    print(f"  Hyperbolic: {final['geometry_percentages']['hyperbolic']:.1f}%")
    print(f"  Euclidean:  {final['geometry_percentages']['euclidean']:.1f}%")
    print(f"  Spherical:  {final['geometry_percentages']['spherical']:.1f}%")
    
    # Changes
    h_change = final['geometry_percentages']['hyperbolic'] - initial['geometry_percentages']['hyperbolic']
    e_change = final['geometry_percentages']['euclidean'] - initial['geometry_percentages']['euclidean']
    s_change = final['geometry_percentages']['spherical'] - initial['geometry_percentages']['spherical']
    
    print(f"\nChanges:")
    print(f"  Hyperbolic: {h_change:+.1f}%")
    print(f"  Euclidean:  {e_change:+.1f}%")
    print(f"  Spherical:  {s_change:+.1f}%")
    
    # Detect Euclidean emergence
    e_pcts = [d['geometry_percentages']['euclidean'] for d in geometry_data]
    max_e = max(e_pcts)
    if max_e > 15:
        max_e_idx = e_pcts.index(max_e)
        max_e_epoch = geometry_data[max_e_idx]['epoch']
        print(f"\n⚠️ Significant Euclidean emergence detected!")
        print(f"   Peak: {max_e:.1f}% at epoch {max_e_epoch}")
    
    print("="*70)


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description='Visualize geometry evolution')
    parser.add_argument('--experiment', type=str, required=True,
                       help='Experiment run ID or path to experiment directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save figures (default: experiment/visualizations/)')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display figures (just save)')
    
    args = parser.parse_args()
    
    # Find experiment directory
    if Path(args.experiment).exists():
        experiment_dir = Path(args.experiment)
    elif (Path('experiments') / args.experiment).exists():
        experiment_dir = Path('experiments') / args.experiment
    else:
        # Try to find by run_id
        experiments_dir = Path('experiments')
        matches = list(experiments_dir.glob(f"*{args.experiment}*"))
        if matches:
            experiment_dir = matches[0]
            print(f"Found experiment: {experiment_dir}")
        else:
            print(f"Error: Experiment not found: {args.experiment}")
            return
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = experiment_dir / 'visualizations'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"GEOMETRY EVOLUTION VISUALIZATION")
    print(f"{'='*70}")
    print(f"Experiment: {experiment_dir}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")
    
    # Load data
    print("Loading geometry data...")
    try:
        geometry_data = load_geometry_data(experiment_dir)
        print(f"✓ Loaded {len(geometry_data)} geometry snapshots")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Print summary
    print_summary(geometry_data)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Timeline plot
    timeline_path = output_dir / 'geometry_timeline.png'
    plot_geometry_timeline(geometry_data, output_path=timeline_path, show=not args.no_show)
    
    # Curvature statistics plot
    stats_path = output_dir / 'curvature_statistics.png'
    plot_curvature_statistics(geometry_data, output_path=stats_path, show=not args.no_show)
    
    print(f"\n✅ Visualization complete!")
    print(f"   Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
