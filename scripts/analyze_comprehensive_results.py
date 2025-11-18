#!/usr/bin/env python3
"""
Analyze and visualize comprehensive research results.
Generates publication-quality figures from run_comprehensive_experiments.py output.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime


def load_results(results_file='comprehensive_results/comprehensive_results.json'):
    """Load comprehensive results from JSON"""
    if not os.path.exists(results_file):
        print(f"Error: Results file not found: {results_file}")
        print("Run 'python run_comprehensive_experiments.py' first!")
        sys.exit(1)
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    return data


def create_universal_pattern_figure(experiments, output_dir):
    """Figure 1: Universal 50/50 Pattern Across All Experiments"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Extract data
    all_data = []
    for exp in experiments:
        if 'geometry' in exp.get('geometric', {}) and exp['geometric']['geometry']:
            geo = exp['geometric']['geometry']
            all_data.append({
                'task': exp['task'],
                'size': exp['model_size'],
                'hyp': geo['pct_hyperbolic'],
                'euc': geo['pct_euclidean'],
                'sph': geo['pct_spherical'],
                'total_heads': geo['n_hyperbolic'] + geo['n_euclidean'] + geo['n_spherical']
            })
    
    if not all_data:
        print("No geometry data found!")
        return
    
    # Panel A: Scatter plot showing all experiments cluster at 50/50
    ax1 = fig.add_subplot(gs[0, 0])
    
    colors = {'tiny': '#FF6B6B', 'small': '#4ECDC4', 'medium': '#45B7D1', 'large': '#96CEB4'}
    markers = {'sst2': 'o', 'mnli': 's', 'wikitext': '^', 'ner': 'D'}
    
    for d in all_data:
        ax1.scatter(d['hyp'], d['sph'], 
                   c=colors.get(d['size'], 'gray'),
                   marker=markers.get(d['task'], 'o'),
                   s=150, alpha=0.7, edgecolors='black', linewidth=1.5,
                   label=f"{d['task']}-{d['size']}" if all_data.index(d) < 1 else "")
    
    # Add 50/50 target
    ax1.axvline(50, color='red', linestyle='--', linewidth=2, alpha=0.5, label='50% line')
    ax1.axhline(50, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax1.plot([50], [50], 'r*', markersize=20, label='Ideal 50/50')
    
    ax1.set_xlabel('Hyperbolic %', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Spherical %', fontsize=12, fontweight='bold')
    ax1.set_title('Universal Convergence to 50/50 Split', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 100])
    ax1.set_ylim([0, 100])
    
    # Add legend for sizes
    from matplotlib.patches import Patch
    size_legend = [Patch(facecolor=colors[s], label=s.capitalize()) for s in ['tiny', 'small', 'medium', 'large']]
    ax1.legend(handles=size_legend, loc='upper left', fontsize=9)
    
    # Panel B: Bar chart by model size
    ax2 = fig.add_subplot(gs[0, 1])
    
    sizes = ['tiny', 'small', 'medium', 'large']
    size_data = {s: {'hyp': [], 'euc': [], 'sph': []} for s in sizes}
    
    for d in all_data:
        size_data[d['size']]['hyp'].append(d['hyp'])
        size_data[d['size']]['euc'].append(d['euc'])
        size_data[d['size']]['sph'].append(d['sph'])
    
    x = np.arange(len(sizes))
    width = 0.25
    
    hyp_means = [np.mean(size_data[s]['hyp']) for s in sizes]
    euc_means = [np.mean(size_data[s]['euc']) for s in sizes]
    sph_means = [np.mean(size_data[s]['sph']) for s in sizes]
    
    ax2.bar(x - width, hyp_means, width, label='Hyperbolic', alpha=0.8, color='#2196F3')
    ax2.bar(x, euc_means, width, label='Euclidean', alpha=0.8, color='#4CAF50')
    ax2.bar(x + width, sph_means, width, label='Spherical', alpha=0.8, color='#FF9800')
    
    ax2.axhline(50, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Geometry by Model Size (Averaged)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.capitalize() for s in sizes])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 100])
    
    # Panel C: Box plot showing consistency
    ax3 = fig.add_subplot(gs[1, 0])
    
    all_hyp = [d['hyp'] for d in all_data]
    all_sph = [d['sph'] for d in all_data]
    all_euc = [d['euc'] for d in all_data]
    
    box_data = [all_hyp, all_euc, all_sph]
    bp = ax3.boxplot(box_data, labels=['Hyperbolic', 'Euclidean', 'Spherical'],
                     patch_artist=True, notch=True)
    
    colors_box = ['#2196F3', '#4CAF50', '#FF9800']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.axhline(50, color='red', linestyle='--', linewidth=2, alpha=0.5, label='50%')
    ax3.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Distribution Consistency Across All Experiments', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim([0, 100])
    
    # Panel D: Summary statistics table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Calculate statistics
    hyp_mean, hyp_std = np.mean(all_hyp), np.std(all_hyp)
    euc_mean, euc_std = np.mean(all_euc), np.std(all_euc)
    sph_mean, sph_std = np.mean(all_sph), np.std(all_sph)
    
    n_experiments = len(all_data)
    n_near_fifty = sum(1 for d in all_data if 45 < d['hyp'] < 55 and 45 < d['sph'] < 55)
    
    table_data = [
        ['Statistic', 'Hyperbolic', 'Euclidean', 'Spherical'],
        ['', '', '', ''],
        ['Mean (%)', f'{hyp_mean:.1f}', f'{euc_mean:.1f}', f'{sph_mean:.1f}'],
        ['Std Dev (%)', f'{hyp_std:.1f}', f'{euc_std:.1f}', f'{sph_std:.1f}'],
        ['Min (%)', f'{min(all_hyp):.1f}', f'{min(all_euc):.1f}', f'{min(all_sph):.1f}'],
        ['Max (%)', f'{max(all_hyp):.1f}', f'{max(all_euc):.1f}', f'{max(all_sph):.1f}'],
        ['', '', '', ''],
        ['Total Experiments', str(n_experiments), '', ''],
        ['Near 50/50 Split', str(n_near_fifty), '', ''],
        ['Percentage', f'{n_near_fifty/n_experiments*100:.0f}%', '', ''],
    ]
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.23, 0.23, 0.23])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white', size=11)
    
    # Style dividers
    for col in range(4):
        table[(1, col)].set_facecolor('#f0f0f0')
        table[(6, col)].set_facecolor('#f0f0f0')
    
    # Highlight 50/50 confirmation
    for col in range(4):
        table[(9, col)].set_facecolor('#C8E6C9' if n_near_fifty/n_experiments > 0.8 else '#FFCDD2')
        table[(9, col)].set_text_props(weight='bold')
    
    ax4.set_title('Statistical Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle('Universal 50/50 Hyperbolic-Spherical Pattern', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    filepath = os.path.join(output_dir, 'universal_pattern.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()


def create_performance_matrix(experiments, output_dir):
    """Figure 2: Performance Heatmap"""
    
    # Create matrix: tasks × sizes
    tasks = ['sst2', 'mnli', 'wikitext', 'ner']
    sizes = ['tiny', 'small', 'medium', 'large']
    
    geo_matrix = np.zeros((len(tasks), len(sizes)))
    std_matrix = np.zeros((len(tasks), len(sizes)))
    
    for exp in experiments:
        if 'best_metric' in exp.get('geometric', {}):
            task_idx = tasks.index(exp['task'])
            size_idx = sizes.index(exp['model_size'])
            geo_matrix[task_idx, size_idx] = exp['geometric']['best_metric']
            std_matrix[task_idx, size_idx] = exp['standard']['best_metric']
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Geometric heatmap
    im1 = ax1.imshow(geo_matrix, cmap='YlGnBu', aspect='auto')
    ax1.set_xticks(range(len(sizes)))
    ax1.set_yticks(range(len(tasks)))
    ax1.set_xticklabels([s.capitalize() for s in sizes])
    ax1.set_yticklabels([t.upper() for t in tasks])
    ax1.set_title('Geometric Model Performance', fontsize=13, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Metric')
    
    # Add values
    for i in range(len(tasks)):
        for j in range(len(sizes)):
            if geo_matrix[i, j] > 0:
                ax1.text(j, i, f'{geo_matrix[i, j]:.2f}',
                        ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Standard heatmap
    im2 = ax2.imshow(std_matrix, cmap='YlOrRd', aspect='auto')
    ax2.set_xticks(range(len(sizes)))
    ax2.set_yticks(range(len(tasks)))
    ax2.set_xticklabels([s.capitalize() for s in sizes])
    ax2.set_yticklabels([t.upper() for t in tasks])
    ax2.set_title('Standard Model Performance', fontsize=13, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Metric')
    
    for i in range(len(tasks)):
        for j in range(len(sizes)):
            if std_matrix[i, j] > 0:
                ax2.text(j, i, f'{std_matrix[i, j]:.2f}',
                        ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Difference heatmap
    diff_matrix = geo_matrix - std_matrix
    im3 = ax3.imshow(diff_matrix, cmap='RdYlGn', aspect='auto', vmin=-0.1, vmax=0.1)
    ax3.set_xticks(range(len(sizes)))
    ax3.set_yticks(range(len(tasks)))
    ax3.set_xticklabels([s.capitalize() for s in sizes])
    ax3.set_yticklabels([t.upper() for t in tasks])
    ax3.set_title('Difference (Geo - Std)', fontsize=13, fontweight='bold')
    plt.colorbar(im3, ax=ax3, label='Difference')
    
    for i in range(len(tasks)):
        for j in range(len(sizes)):
            if diff_matrix[i, j] != 0:
                color = 'white' if abs(diff_matrix[i, j]) > 0.05 else 'black'
                ax3.text(j, i, f'{diff_matrix[i, j]:+.3f}',
                        ha='center', va='center', fontsize=9, fontweight='bold', color=color)
    
    plt.suptitle('Performance Across Tasks and Model Sizes', 
                 fontsize=16, fontweight='bold')
    
    filepath = os.path.join(output_dir, 'performance_matrix.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()


def create_geometry_by_scale(experiments, output_dir):
    """Figure 3: Geometry Distribution Consistency Across Scales"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    sizes = ['tiny', 'small', 'medium', 'large']
    size_labels = ['Tiny\n2 heads', 'Small\n8 heads', 'Medium\n32 heads', 'Large\n72 heads']
    
    for idx, (size, label) in enumerate(zip(sizes, size_labels)):
        ax = axes[idx // 2, idx % 2]
        
        # Get data for this size
        size_exps = [e for e in experiments if e['model_size'] == size 
                     and 'geometry' in e.get('geometric', {})]
        
        if not size_exps:
            continue
        
        tasks_list = []
        hyp_list = []
        euc_list = []
        sph_list = []
        
        for exp in size_exps:
            geo = exp['geometric']['geometry']
            tasks_list.append(exp['task'].upper())
            hyp_list.append(geo['n_hyperbolic'])
            euc_list.append(geo['n_euclidean'])
            sph_list.append(geo['n_spherical'])
        
        x = np.arange(len(tasks_list))
        width = 0.25
        
        ax.bar(x - width, hyp_list, width, label='Hyperbolic', alpha=0.8, color='#2196F3')
        ax.bar(x, euc_list, width, label='Euclidean', alpha=0.8, color='#4CAF50')
        ax.bar(x + width, sph_list, width, label='Spherical', alpha=0.8, color='#FF9800')
        
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'{label}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(tasks_list, fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        total_heads = hyp_list[0] + euc_list[0] + sph_list[0] if hyp_list else 0
        if total_heads > 0:
            ax.axhline(total_heads/2, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.suptitle('Geometry Distribution Across All Scales', 
                 fontsize=16, fontweight='bold')
    
    filepath = os.path.join(output_dir, 'geometry_by_scale.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()


def create_publication_figure(experiments, output_dir):
    """Main Publication Figure: Multi-panel comprehensive view"""
    
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Collect all geometry data
    all_hyp, all_sph, all_euc = [], [], []
    for exp in experiments:
        if 'geometry' in exp.get('geometric', {}):
            geo = exp['geometric']['geometry']
            all_hyp.append(geo['pct_hyperbolic'])
            all_sph.append(geo['pct_spherical'])
            all_euc.append(geo['pct_euclidean'])
    
    if not all_hyp:
        print("No data for publication figure")
        return
    
    # Panel 1: Main finding - scatter
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.scatter(all_hyp, all_sph, s=200, alpha=0.6, c='blue', edgecolors='black', linewidth=2)
    ax1.plot([50], [50], 'r*', markersize=30, label='Ideal 50/50', zorder=10)
    ax1.axvline(50, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax1.axhline(50, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax1.set_xlabel('Hyperbolic %', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Spherical %', fontsize=14, fontweight='bold')
    ax1.set_title('Universal 50/50 Pattern (n=' + str(len(all_hyp)) + ' experiments)', 
                  fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 100])
    ax1.set_ylim([0, 100])
    
    # Panel 2: Statistics
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    stats_data = [
        ['Geometry', 'Mean ± Std'],
        ['', ''],
        ['Hyperbolic', f'{np.mean(all_hyp):.1f}% ± {np.std(all_hyp):.1f}%'],
        ['Euclidean', f'{np.mean(all_euc):.1f}% ± {np.std(all_euc):.1f}%'],
        ['Spherical', f'{np.mean(all_sph):.1f}% ± {np.std(all_sph):.1f}%'],
        ['', ''],
        ['Hyp/Sph Ratio', f'{np.mean(all_hyp)/np.mean(all_sph):.2f}'],
        ['Near 50/50', f'{sum(1 for h,s in zip(all_hyp,all_sph) if 45<h<55 and 45<s<55)}/{len(all_hyp)}'],
    ]
    
    table = ax2.table(cellText=stats_data, cellLoc='left', loc='center',
                     colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax2.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    # Remaining panels: Task-specific results
    tasks = ['sst2', 'mnli', 'wikitext', 'ner']
    task_names = ['SST-2 Sentiment', 'MNLI Entailment', 'WikiText LM', 'WikiANN NER']
    
    for idx, (task, name) in enumerate(zip(tasks, task_names)):
        ax = fig.add_subplot(gs[(idx//2)+1, idx%2])
        
        task_data = [e for e in experiments if e['task'] == task 
                     and 'geometry' in e.get('geometric', {})]
        
        if not task_data:
            continue
        
        sizes_found = []
        hyp_vals = []
        sph_vals = []
        
        for exp in sorted(task_data, key=lambda x: ['tiny','small','medium','large'].index(x['model_size'])):
            geo = exp['geometric']['geometry']
            sizes_found.append(exp['model_size'].capitalize())
            hyp_vals.append(geo['pct_hyperbolic'])
            sph_vals.append(geo['pct_spherical'])
        
        x = np.arange(len(sizes_found))
        width = 0.35
        
        ax.bar(x - width/2, hyp_vals, width, label='Hyperbolic', alpha=0.8, color='#2196F3')
        ax.bar(x + width/2, sph_vals, width, label='Spherical', alpha=0.8, color='#FF9800')
        ax.axhline(50, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='50%')
        
        ax.set_ylabel('Percentage (%)', fontsize=11)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sizes_found, fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 100])
    
    plt.suptitle('Universal Geometry Across All Tasks and Scales', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    filepath = os.path.join(output_dir, 'publication_figure.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()


def main():
    print("="*80)
    print("  COMPREHENSIVE RESULTS ANALYZER")
    print("  Publication-Quality Figures Generator")
    print("="*80)
    
    # Load results
    results_file = 'comprehensive_results/comprehensive_results.json'
    
    if not os.path.exists(results_file):
        print(f"\n✗ Results file not found: {results_file}")
        print("  Run 'python run_comprehensive_experiments.py' first!")
        sys.exit(1)
    
    print(f"\nLoading results from: {results_file}")
    data = load_results(results_file)
    
    experiments = data['experiments']
    print(f"✓ Loaded {len(experiments)} experiments")
    print(f"  Total time: {data.get('total_time_hours', 0):.1f} hours")
    print(f"  Timestamp: {data.get('timestamp', 'unknown')}")
    
    # Create output directory for figures
    figures_dir = 'comprehensive_results/figures'
    os.makedirs(figures_dir, exist_ok=True)
    
    print(f"\nGenerating figures in: {figures_dir}/")
    print("-"*80)
    
    # Generate all figures
    print("\n1. Creating universal pattern figure...")
    create_universal_pattern_figure(experiments, figures_dir)
    
    print("\n2. Creating performance matrix...")
    create_performance_matrix(experiments, figures_dir)
    
    print("\n3. Creating geometry by scale figure...")
    create_geometry_by_scale(experiments, figures_dir)
    
    print("\n4. Creating publication figure...")
    create_publication_figure(experiments, figures_dir)
    
    print("\n" + "="*80)
    print("✓ ALL FIGURES GENERATED!")
    print("="*80)
    print(f"\nOutput directory: {figures_dir}/")
    print("\nGenerated files:")
    print("  1. universal_pattern.png - Main 50/50 finding")
    print("  2. performance_matrix.png - Results heatmap")
    print("  3. geometry_by_scale.png - Scale independence")
    print("  4. publication_figure.png - Combined multi-panel")
    print("\n✓ Ready for publication!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
