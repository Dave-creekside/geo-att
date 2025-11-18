#!/usr/bin/env python3
"""
Deep Geometry Analysis - Advanced metrics from existing geometry logs.
Analyzes geometry.jsonl to extract insights beyond basic H/E/S distribution.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import pearsonr


def load_geometry_log(experiment_dir):
    """Load geometry data from experiment logs."""
    geometry_file = Path(experiment_dir) / 'logs' / 'geometry.jsonl'
    
    if not geometry_file.exists():
        raise FileNotFoundError(f"Geometry log not found: {geometry_file}")
    
    data = []
    with open(geometry_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    return data


def extract_curvature_matrix(geometry_data):
    """
    Extract curvature evolution as matrix.
    
    Returns:
        curvature_matrix: ndarray of shape (n_snapshots, n_layers, n_heads)
        epochs: list of epoch numbers
        steps: list of step numbers
    """
    n_snapshots = len(geometry_data)
    
    # Get dimensions from first snapshot
    first_curvatures = geometry_data[0]['curvatures']
    
    # Curvatures are stored as dict: {"layer_0": [...], "layer_1": [...]}
    layer_keys = sorted([k for k in first_curvatures.keys() if k.startswith('layer_')])
    n_layers = len(layer_keys)
    n_heads = len(first_curvatures[layer_keys[0]])
    
    # Build matrix
    curvature_matrix = np.zeros((n_snapshots, n_layers, n_heads))
    epochs = []
    steps = []
    
    for i, snapshot in enumerate(geometry_data):
        epochs.append(snapshot['epoch'])
        steps.append(snapshot['step'])
        
        curv_dict = snapshot['curvatures']
        for layer_idx, layer_key in enumerate(layer_keys):
            layer_curv = curv_dict[layer_key]
            for head_idx, curv in enumerate(layer_curv):
                curvature_matrix[i, layer_idx, head_idx] = curv
    
    return curvature_matrix, epochs, steps


def classify_geometry(k, threshold=0.1):
    """Classify curvature into H/E/S."""
    if k < -threshold:
        return 'H'
    elif k > threshold:
        return 'S'
    else:
        return 'E'


def analyze_per_layer(curvature_matrix, threshold=0.1):
    """
    Analyze geometry distribution per layer.
    
    Returns:
        dict with per-layer statistics
    """
    n_snapshots, n_layers, n_heads = curvature_matrix.shape
    
    per_layer_stats = {}
    
    for layer in range(n_layers):
        layer_data = curvature_matrix[:, layer, :]  # (n_snapshots, n_heads)
        
        # Final distribution (last snapshot)
        final_curvatures = layer_data[-1, :]
        n_h = np.sum(final_curvatures < -threshold)
        n_e = np.sum(np.abs(final_curvatures) <= threshold)
        n_s = np.sum(final_curvatures > threshold)
        
        per_layer_stats[f'layer_{layer}'] = {
            'mean_curvature': float(np.mean(final_curvatures)),
            'std_curvature': float(np.std(final_curvatures)),
            'final_distribution': {
                'hyperbolic': int(n_h),
                'euclidean': int(n_e),
                'spherical': int(n_s),
                'pct_hyperbolic': float(n_h / n_heads * 100),
                'pct_euclidean': float(n_e / n_heads * 100),
                'pct_spherical': float(n_s / n_heads * 100)
            }
        }
    
    return per_layer_stats


def detect_transitions(curvature_matrix, threshold=0.1):
    """
    Detect when individual heads change geometry type.
    
    Returns:
        list of transition events
    """
    n_snapshots, n_layers, n_heads = curvature_matrix.shape
    
    transitions = []
    
    for layer in range(n_layers):
        for head in range(n_heads):
            head_evolution = curvature_matrix[:, layer, head]
            
            # Track geometry over time
            prev_geom = classify_geometry(head_evolution[0], threshold)
            
            for t in range(1, n_snapshots):
                curr_geom = classify_geometry(head_evolution[t], threshold)
                
                if curr_geom != prev_geom:
                    transitions.append({
                        'snapshot': t,
                        'layer': layer,
                        'head': head,
                        'from': prev_geom,
                        'to': curr_geom,
                        'curvature_before': float(head_evolution[t-1]),
                        'curvature_after': float(head_evolution[t])
                    })
                    prev_geom = curr_geom
    
    return transitions


def compute_temporal_correlations(curvature_matrix):
    """
    Compute autocorrelation for each head to detect oscillation vs convergence.
    
    Returns:
        dict with autocorrelation statistics
    """
    n_snapshots, n_layers, n_heads = curvature_matrix.shape
    
    autocorrelations = np.zeros((n_layers, n_heads))
    
    for layer in range(n_layers):
        for head in range(n_heads):
            series = curvature_matrix[:, layer, head]
            
            # Lag-1 autocorrelation
            if len(series) > 1:
                corr, _ = pearsonr(series[:-1], series[1:])
                autocorrelations[layer, head] = corr
    
    return {
        'mean_autocorr': float(np.mean(autocorrelations)),
        'std_autocorr': float(np.std(autocorrelations)),
        'min_autocorr': float(np.min(autocorrelations)),
        'max_autocorr': float(np.max(autocorrelations)),
        'autocorr_matrix': autocorrelations.tolist()
    }


def cluster_head_behaviors(curvature_matrix):
    """
    Cluster heads by their curvature evolution patterns.
    
    Returns:
        linkage matrix for hierarchical clustering
    """
    n_snapshots, n_layers, n_heads = curvature_matrix.shape
    
    # Reshape to (n_total_heads, n_snapshots)
    head_evolutions = curvature_matrix.transpose(1, 2, 0).reshape(-1, n_snapshots)
    
    # Hierarchical clustering
    Z = linkage(head_evolutions, method='ward')
    
    return Z, head_evolutions


def plot_per_layer_distributions(per_layer_stats, output_path):
    """Plot curvature distributions per layer."""
    n_layers = len(per_layer_stats)
    
    fig, axes = plt.subplots(1, n_layers, figsize=(4*n_layers, 4), sharey=True)
    if n_layers == 1:
        axes = [axes]
    
    for layer_idx, (layer_name, stats) in enumerate(per_layer_stats.items()):
        ax = axes[layer_idx]
        
        dist = stats['final_distribution']
        categories = ['Hyperbolic', 'Euclidean', 'Spherical']
        counts = [dist['hyperbolic'], dist['euclidean'], dist['spherical']]
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        
        bars = ax.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Number of Heads', fontsize=11)
        ax.set_title(f'Layer {layer_idx}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add percentages
        total = sum(counts)
        for bar, count in zip(bars, counts):
            if count > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count}\n({count/total*100:.0f}%)',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('Final Geometry Distribution by Layer', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_transition_heatmap(transitions, n_layers, n_heads, n_snapshots, output_path):
    """Plot heatmap showing when transitions occurred."""
    # Create matrix: (n_layers * n_heads) × n_snapshots
    transition_matrix = np.zeros((n_layers * n_heads, n_snapshots))
    
    for trans in transitions:
        head_idx = trans['layer'] * n_heads + trans['head']
        transition_matrix[head_idx, trans['snapshot']] = 1
    
    fig, ax = plt.subplots(figsize=(12, max(6, n_layers * n_heads // 4)))
    
    im = ax.imshow(transition_matrix, aspect='auto', cmap='Reds', interpolation='nearest')
    
    ax.set_xlabel('Training Snapshot', fontsize=12)
    ax.set_ylabel('Head (Layer × Head ID)', fontsize=12)
    ax.set_title('Geometry Transition Events', fontsize=14, fontweight='bold')
    
    # Add layer separators
    for layer in range(1, n_layers):
        ax.axhline(y=layer * n_heads - 0.5, color='blue', linewidth=2, alpha=0.5)
    
    plt.colorbar(im, ax=ax, label='Transition occurred')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_temporal_correlations(autocorr_matrix, output_path):
    """Plot autocorrelation heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(autocorr_matrix, aspect='auto', cmap='RdYlGn', 
                   vmin=-1, vmax=1, interpolation='nearest')
    
    ax.set_xlabel('Head', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title('Temporal Autocorrelation (Lag-1)\nHigh = Stable, Low = Volatile', 
                 fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Autocorrelation')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_head_clustering(Z, head_evolutions, n_layers, n_heads, output_path):
    """Plot dendrogram of head clustering."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create dendrogram
    dendrogram(Z, ax=ax, color_threshold=0.7*max(Z[:,2]))
    
    ax.set_xlabel('Head ID (Layer×Heads + Head)', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    ax.set_title('Hierarchical Clustering of Head Behaviors', fontsize=14, fontweight='bold')
    
    # Add layer labels
    layer_positions = [i * n_heads + n_heads // 2 for i in range(n_layers)]
    layer_labels = [f'L{i}' for i in range(n_layers)]
    ax.set_xticks(layer_positions)
    ax.set_xticklabels(layer_labels)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_curvature_trajectories(curvature_matrix, epochs, output_path, max_heads=20):
    """Plot individual head trajectories (sample for visualization)."""
    n_snapshots, n_layers, n_heads = curvature_matrix.shape
    
    fig, axes = plt.subplots(n_layers, 1, figsize=(12, 4*n_layers), sharex=True)
    if n_layers == 1:
        axes = [axes]
    
    for layer in range(n_layers):
        ax = axes[layer]
        
        # Plot up to max_heads trajectories
        heads_to_plot = min(n_heads, max_heads)
        
        for head in range(heads_to_plot):
            trajectory = curvature_matrix[:, layer, head]
            ax.plot(epochs, trajectory, alpha=0.6, linewidth=1)
        
        # Add geometry boundaries
        ax.axhline(y=-0.1, color='red', linestyle='--', alpha=0.3, label='H/E boundary')
        ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.3, label='E/S boundary')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.2)
        
        ax.set_ylabel('Curvature', fontsize=11)
        ax.set_title(f'Layer {layer} - Individual Head Trajectories', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
    
    axes[-1].set_xlabel('Epoch', fontsize=12)
    plt.suptitle(f'Curvature Evolution (showing {heads_to_plot}/{n_heads} heads per layer)', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def analyze_positional_bias(curvature_matrix, threshold=0.1):
    """Analyze if certain head positions prefer certain geometries."""
    n_snapshots, n_layers, n_heads = curvature_matrix.shape
    
    # Final geometries for each head
    final_curvatures = curvature_matrix[-1, :, :]
    
    # Count geometry type by head position
    position_counts = defaultdict(lambda: {'H': 0, 'E': 0, 'S': 0})
    
    for layer in range(n_layers):
        for head in range(n_heads):
            geom = classify_geometry(final_curvatures[layer, head], threshold)
            position_counts[head][geom] += 1
    
    return dict(position_counts)


def analyze_transition_asymmetry(transitions):
    """Analyze directionality of transitions."""
    transition_matrix = defaultdict(int)
    
    for trans in transitions:
        key = (trans['from'], trans['to'])
        transition_matrix[key] += 1
    
    # Calculate asymmetry scores
    asymmetry = {}
    pairs = [('H', 'E'), ('H', 'S'), ('E', 'S')]
    
    for a, b in pairs:
        forward = transition_matrix.get((a, b), 0)
        reverse = transition_matrix.get((b, a), 0)
        total = forward + reverse
        
        if total > 0:
            asymmetry[f'{a}↔{b}'] = {
                'forward': forward,
                'reverse': reverse,
                'bias': (forward - reverse) / total  # -1 to 1
            }
    
    return asymmetry


def analyze_curvature_drift(curvature_matrix):
    """Analyze how curvatures drift from initial to final values."""
    n_snapshots, n_layers, n_heads = curvature_matrix.shape
    
    initial = curvature_matrix[0, :, :].flatten()
    final = curvature_matrix[-1, :, :].flatten()
    
    drift = final - initial
    
    return {
        'mean_drift': float(np.mean(drift)),
        'std_drift': float(np.std(drift)),
        'max_positive_drift': float(np.max(drift)),
        'max_negative_drift': float(np.min(drift)),
        'heads_increased': int(np.sum(drift > 0.1)),
        'heads_decreased': int(np.sum(drift < -0.1)),
        'heads_stable': int(np.sum(np.abs(drift) <= 0.1)),
        'initial_final_correlation': float(np.corrcoef(initial, final)[0, 1])
    }


def plot_positional_bias(position_counts, n_layers, output_path):
    """Plot heatmap showing geometry preference by head position."""
    n_positions = len(position_counts)
    
    # Build matrix: position × geometry type
    data = np.zeros((n_positions, 3))
    positions = sorted(position_counts.keys())
    
    for i, pos in enumerate(positions):
        data[i, 0] = position_counts[pos]['H']
        data[i, 1] = position_counts[pos]['E']
        data[i, 2] = position_counts[pos]['S']
    
    # Normalize to percentages
    data = data / n_layers * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(data.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    
    ax.set_xticks(range(n_positions))
    ax.set_xticklabels([f'H{i}' for i in positions])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Hyperbolic', 'Euclidean', 'Spherical'])
    
    ax.set_xlabel('Head Position', fontsize=12)
    ax.set_title('Geometry Preference by Head Position\n(% of layers)', 
                 fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(n_positions):
        for j in range(3):
            text = ax.text(i, j, f'{data[i, j]:.0f}%',
                          ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=ax, label='Percentage')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_curvature_drift(curvature_matrix, output_path):
    """Scatter plot of initial vs final curvature."""
    initial = curvature_matrix[0, :, :].flatten()
    final = curvature_matrix[-1, :, :].flatten()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Scatter plot
    ax.scatter(initial, final, alpha=0.6, s=50)
    
    # Add y=x line (no change)
    lim = max(abs(initial).max(), abs(final).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3, label='No change')
    
    # Add geometry boundaries
    ax.axhline(y=-0.1, color='red', linestyle=':', alpha=0.3)
    ax.axhline(y=0.1, color='green', linestyle=':', alpha=0.3)
    ax.axvline(x=-0.1, color='red', linestyle=':', alpha=0.3)
    ax.axvline(x=0.1, color='green', linestyle=':', alpha=0.3)
    
    # Quadrant labels
    ax.text(-lim*0.8, lim*0.8, 'H→S', fontsize=12, alpha=0.5)
    ax.text(lim*0.8, -lim*0.8, 'S→H', fontsize=12, alpha=0.5)
    
    ax.set_xlabel('Initial Curvature', fontsize=12)
    ax.set_ylabel('Final Curvature', fontsize=12)
    ax.set_title('Curvature Drift: Initial vs Final', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_summary_report(per_layer_stats, transitions, temporal_stats, 
                         positional_bias, transition_asymmetry, drift_stats,
                         output_path):
    """Create markdown summary report."""
    
    with open(output_path, 'w') as f:
        f.write("# Deep Geometry Analysis Report\n\n")
        
        # Per-layer summary
        f.write("## Per-Layer Geometry Distribution\n\n")
        for layer_name, stats in per_layer_stats.items():
            layer_num = layer_name.split('_')[1]
            dist = stats['final_distribution']
            f.write(f"### Layer {layer_num}\n\n")
            f.write(f"- **Mean Curvature**: {stats['mean_curvature']:.4f}\n")
            f.write(f"- **Std Curvature**: {stats['std_curvature']:.4f}\n")
            f.write(f"- **Distribution**:\n")
            f.write(f"  - Hyperbolic: {dist['hyperbolic']} ({dist['pct_hyperbolic']:.1f}%)\n")
            f.write(f"  - Euclidean: {dist['euclidean']} ({dist['pct_euclidean']:.1f}%)\n")
            f.write(f"  - Spherical: {dist['spherical']} ({dist['pct_spherical']:.1f}%)\n\n")
        
        # Positional bias
        f.write("## Positional Bias Analysis\n\n")
        f.write("Geometry preferences by head position (across all layers):\n\n")
        for pos in sorted(positional_bias.keys()):
            counts = positional_bias[pos]
            total = sum(counts.values())
            f.write(f"### Head Position {pos}\n\n")
            for geom in ['H', 'E', 'S']:
                pct = counts[geom] / total * 100 if total > 0 else 0
                f.write(f"- **{geom}**: {counts[geom]} ({pct:.0f}%)\n")
            f.write("\n")
        
        # Transition analysis
        f.write("## Geometry Transitions\n\n")
        f.write(f"**Total transitions detected**: {len(transitions)}\n\n")
        
        if transitions:
            # Group by type
            transition_types = defaultdict(int)
            for trans in transitions:
                key = f"{trans['from']}→{trans['to']}"
                transition_types[key] += 1
            
            f.write("### Transition Types:\n\n")
            for trans_type, count in sorted(transition_types.items(), key=lambda x: -x[1]):
                f.write(f"- **{trans_type}**: {count} occurrences\n")
            
            f.write("\n### Transition Asymmetry:\n\n")
            for pair, stats in transition_asymmetry.items():
                f.write(f"**{pair}**:\n")
                f.write(f"- Forward: {stats['forward']}\n")
                f.write(f"- Reverse: {stats['reverse']}\n")
                f.write(f"- Bias: {stats['bias']:.2f} ")
                if abs(stats['bias']) > 0.5:
                    f.write("(strong asymmetry)\n")
                else:
                    f.write("(balanced)\n")
                f.write("\n")
            
            f.write("### Most Active Heads (by transition count):\n\n")
            head_transition_counts = defaultdict(int)
            for trans in transitions:
                head_id = f"L{trans['layer']}H{trans['head']}"
                head_transition_counts[head_id] += 1
            
            for head_id, count in sorted(head_transition_counts.items(), key=lambda x: -x[1])[:10]:
                f.write(f"- **{head_id}**: {count} transitions\n")
        
        # Curvature drift
        f.write("\n## Curvature Drift Analysis\n\n")
        f.write("Change from initial to final curvature:\n\n")
        f.write(f"- **Mean Drift**: {drift_stats['mean_drift']:.4f}\n")
        f.write(f"- **Std Drift**: {drift_stats['std_drift']:.4f}\n")
        f.write(f"- **Max Positive**: {drift_stats['max_positive_drift']:.4f}\n")
        f.write(f"- **Max Negative**: {drift_stats['max_negative_drift']:.4f}\n")
        f.write(f"- **Heads Increased**: {drift_stats['heads_increased']}\n")
        f.write(f"- **Heads Decreased**: {drift_stats['heads_decreased']}\n")
        f.write(f"- **Heads Stable**: {drift_stats['heads_stable']}\n")
        f.write(f"- **Initial-Final Correlation**: {drift_stats['initial_final_correlation']:.3f}\n")
        
        # Temporal analysis
        f.write("\n## Temporal Correlation Analysis\n\n")
        f.write("Lag-1 autocorrelation measures stability:\n")
        f.write("- High (>0.8): Head curvature is stable/converging\n")
        f.write("- Low (<0.5): Head curvature is volatile/oscillating\n\n")
        f.write(f"- **Mean Autocorrelation**: {temporal_stats['mean_autocorr']:.3f}\n")
        f.write(f"- **Std Autocorrelation**: {temporal_stats['std_autocorr']:.3f}\n")
        f.write(f"- **Range**: [{temporal_stats['min_autocorr']:.3f}, {temporal_stats['max_autocorr']:.3f}]\n")
    
    print(f"✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Deep geometry analysis from logs')
    parser.add_argument('--experiment', type=str, required=True,
                       help='Experiment directory or run ID')
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='Curvature classification threshold (default: 0.1)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: experiment/deep_analysis/)')
    
    args = parser.parse_args()
    
    # Find experiment directory
    if Path(args.experiment).exists():
        experiment_dir = Path(args.experiment)
    elif (Path('experiments') / args.experiment).exists():
        experiment_dir = Path('experiments') / args.experiment
    else:
        print(f"Error: Experiment not found: {args.experiment}")
        return
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = experiment_dir / 'deep_analysis'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("DEEP GEOMETRY ANALYSIS")
    print("="*70)
    print(f"Experiment: {experiment_dir}")
    print(f"Output: {output_dir}")
    print("="*70 + "\n")
    
    # Load data
    print("Loading geometry data...")
    try:
        geometry_data = load_geometry_log(experiment_dir)
        print(f"✓ Loaded {len(geometry_data)} snapshots")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Extract curvature matrix
    print("Extracting curvature evolution...")
    curvature_matrix, epochs, steps = extract_curvature_matrix(geometry_data)
    n_snapshots, n_layers, n_heads = curvature_matrix.shape
    print(f"✓ Shape: {n_snapshots} snapshots × {n_layers} layers × {n_heads} heads")
    
    # Analyses
    print("\nRunning analyses...")
    
    print("  1. Per-layer distribution analysis...")
    per_layer_stats = analyze_per_layer(curvature_matrix, args.threshold)
    
    print("  2. Detecting geometry transitions...")
    transitions = detect_transitions(curvature_matrix, args.threshold)
    print(f"     → Found {len(transitions)} transition events")
    
    print("  3. Computing temporal correlations...")
    temporal_stats = compute_temporal_correlations(curvature_matrix)
    print(f"     → Mean autocorr: {temporal_stats['mean_autocorr']:.3f}")
    
    print("  4. Clustering head behaviors...")
    Z, head_evolutions = cluster_head_behaviors(curvature_matrix)
    print(f"     → Clustered {len(head_evolutions)} heads")
    
    print("  5. Analyzing positional bias...")
    positional_bias = analyze_positional_bias(curvature_matrix, args.threshold)
    
    print("  6. Analyzing transition asymmetry...")
    transition_asymmetry = analyze_transition_asymmetry(transitions)
    
    print("  7. Analyzing curvature drift...")
    drift_stats = analyze_curvature_drift(curvature_matrix)
    print(f"     → Mean drift: {drift_stats['mean_drift']:.4f}")
    
    # Save analysis results
    print("\nSaving results...")
    report = {
        'experiment': str(experiment_dir),
        'n_snapshots': n_snapshots,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'per_layer_stats': per_layer_stats,
        'transitions': {
            'total_count': len(transitions),
            'events': transitions[:100]  # Save first 100 for JSON size
        },
        'temporal_stats': temporal_stats
    }
    
    with open(output_dir / 'report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✓ Saved: {output_dir / 'report.json'}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    print("  1. Per-layer distributions...")
    plot_per_layer_distributions(per_layer_stats, plots_dir / 'per_layer_distributions.png')
    
    print("  2. Transition heatmap...")
    plot_transition_heatmap(transitions, n_layers, n_heads, n_snapshots, 
                           plots_dir / 'transition_heatmap.png')
    
    print("  3. Temporal correlations...")
    autocorr_matrix = np.array(temporal_stats['autocorr_matrix'])
    plot_temporal_correlations(autocorr_matrix, plots_dir / 'temporal_correlations.png')
    
    print("  4. Head clustering...")
    plot_head_clustering(Z, head_evolutions, n_layers, n_heads, 
                        plots_dir / 'head_clustering.png')
    
    print("  5. Curvature trajectories...")
    plot_curvature_trajectories(curvature_matrix, epochs, 
                               plots_dir / 'curvature_trajectories.png')
    
    print("  6. Positional bias heatmap...")
    plot_positional_bias(positional_bias, n_layers, plots_dir / 'positional_bias.png')
    
    print("  7. Curvature drift scatter...")
    plot_curvature_drift(curvature_matrix, plots_dir / 'curvature_drift.png')
    
    # Create summary markdown
    print("\nCreating summary report...")
    create_summary_report(per_layer_stats, transitions, temporal_stats,
                         positional_bias, transition_asymmetry, drift_stats,
                         output_dir / 'summary.md')
    
    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n📊 Results saved to: {output_dir}")
    print(f"📈 Visualizations: {plots_dir}")
    print(f"📄 Report: {output_dir / 'report.json'}")
    print(f"📝 Summary: {output_dir / 'summary.md'}")
    print("\n" + "="*70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
