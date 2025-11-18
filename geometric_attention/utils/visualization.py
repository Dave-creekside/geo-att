"""
Visualization utilities for geometric attention analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch


def plot_curvature_heatmap(curvature_matrix: np.ndarray, title: str = "Curvatures", 
                           save_path: Optional[str] = None):
    """Plot heatmap of learned curvatures"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(curvature_matrix, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
    ax.set_xlabel('Head', fontsize=11)
    ax.set_ylabel('Layer', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Curvature k')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_geometry_distribution(n_hyp: int, n_euc: int, n_sph: int, 
                               title: str = "Geometry Distribution",
                               save_path: Optional[str] = None):
    """Plot distribution of geometric types"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    labels = ['Hyperbolic', 'Euclidean', 'Spherical']
    counts = [n_hyp, n_euc, n_sph]
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    
    bars = ax.bar(labels, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add percentage labels
    total = sum(counts)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}/{total}\n({count/total*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Number of Heads', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_training_curves(history: Dict[str, List], title: str = "Training History",
                        save_path: Optional[str] = None):
    """Plot training and validation curves"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss plot
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs, history['train_loss'], 'o-', label='Train', linewidth=2, markersize=6)
    axes[0].plot(epochs, history['val_loss'], 's-', label='Val', linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Loss', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy/Perplexity plot
    metric_name = 'Accuracy' if max(history['train_acc']) <= 1.0 else 'Perplexity'
    axes[1].plot(epochs, history['train_acc'], 'o-', label='Train', linewidth=2, markersize=6)
    axes[1].plot(epochs, history['val_acc'], 's-', label='Val', linewidth=2, markersize=6)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel(metric_name, fontsize=11)
    axes[1].set_title(metric_name, fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_comparison_results(results: Dict[str, Dict], save_path: Optional[str] = None):
    """Plot comparison between geometric and standard models"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison
    models = list(results.keys())
    accuracies = [results[m]['best_val_acc'] for m in models]
    colors = ['#2196F3', '#FF9800']
    
    bars1 = axes[0].bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Validation Accuracy', fontsize=11)
    axes[0].set_title('Model Performance', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Training time comparison
    times = [results[m]['training_time'] for m in models]
    bars2 = axes[1].bar(models, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Training Time (seconds)', fontsize=11)
    axes[1].set_title('Training Speed', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add time labels
    for bar, time in zip(bars2, times):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 10,
                    f'{time:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('Geometric vs Standard Transformer Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_attention_weights(attention_weights: torch.Tensor, tokens: List[str] = None,
                          title: str = "Attention Weights", save_path: Optional[str] = None):
    """Visualize attention weights matrix"""
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
    
    if tokens:
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens)
    
    ax.set_xlabel('Keys', fontsize=11)
    ax.set_ylabel('Queries', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Attention Weight')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def create_comprehensive_visualization(sst2_results: Dict, mnli_results: Dict, 
                                      ner_results: Dict = None, lm_results: Dict = None,
                                      save_path: Optional[str] = None):
    """Create comprehensive multi-task visualization"""
    n_tasks = 2 + (1 if ner_results else 0) + (1 if lm_results else 0)
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, n_tasks, hspace=0.3, wspace=0.3)
    
    # Row 1: Curvature heatmaps
    if 'curvature_matrix' in sst2_results:
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(sst2_results['curvature_matrix'], cmap='RdBu_r', 
                         vmin=-2, vmax=2, aspect='auto')
        ax1.set_title('SST-2 Curvatures', fontsize=11, fontweight='bold')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    if 'curvature_matrix' in mnli_results:
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(mnli_results['curvature_matrix'], cmap='RdBu_r', 
                         vmin=-2, vmax=2, aspect='auto')
        ax2.set_title('MNLI Curvatures', fontsize=11, fontweight='bold')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # Row 2: Geometry distributions
    ax_dist = fig.add_subplot(gs[1, :])
    
    tasks = ['SST-2', 'MNLI']
    hyp_counts = [sst2_results.get('n_hyperbolic', 0), mnli_results.get('n_hyperbolic', 0)]
    euc_counts = [sst2_results.get('n_euclidean', 0), mnli_results.get('n_euclidean', 0)]
    sph_counts = [sst2_results.get('n_spherical', 0), mnli_results.get('n_spherical', 0)]
    
    if ner_results:
        tasks.append('NER')
        hyp_counts.append(ner_results.get('n_hyperbolic', 0))
        euc_counts.append(ner_results.get('n_euclidean', 0))
        sph_counts.append(ner_results.get('n_spherical', 0))
    
    x = np.arange(len(tasks))
    width = 0.25
    
    bars1 = ax_dist.bar(x - width, hyp_counts, width, label='Hyperbolic', 
                       alpha=0.8, color='#2196F3')
    bars2 = ax_dist.bar(x, euc_counts, width, label='Euclidean', 
                       alpha=0.8, color='#4CAF50')
    bars3 = ax_dist.bar(x + width, sph_counts, width, label='Spherical', 
                       alpha=0.8, color='#FF9800')
    
    ax_dist.set_ylabel('Count', fontsize=11)
    ax_dist.set_title('Geometry Distribution Across Tasks', fontsize=12, fontweight='bold')
    ax_dist.set_xticks(x)
    ax_dist.set_xticklabels(tasks)
    ax_dist.legend()
    ax_dist.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Geometric Attention Analysis', fontsize=15, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig
