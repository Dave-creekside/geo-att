#!/usr/bin/env python3
"""
Live training monitor - plots current training progress in real-time.
Monitors checkpoint files and updates plots as training progresses.
"""

import os
import sys
import time
import glob
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse


def find_latest_checkpoint(pattern='checkpoints/best_*.pt'):
    """Find the most recent checkpoint matching pattern"""
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    
    # Sort by modification time
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]


def load_checkpoint_history(checkpoint_path):
    """Load training history from checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        history = checkpoint.get('history', {})
        model_name = checkpoint.get('model_name', 'unknown')
        epoch = checkpoint.get('epoch', 0)
        metric = checkpoint.get('metric', 0)
        
        # Extract geometry if available
        geometry = None
        if 'geometry' in checkpoint:
            geometry = checkpoint['geometry']
        
        return {
            'history': history,
            'model_name': model_name,
            'epoch': epoch,
            'metric': metric,
            'geometry': geometry
        }
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None


def plot_training_progress(checkpoint_path, save_plot=False):
    """Create training progress visualization"""
    
    data = load_checkpoint_history(checkpoint_path)
    if not data:
        print("No data to plot")
        return
    
    history = data['history']
    
    if not history or not history.get('train_loss'):
        print("No training history available yet")
        return
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Training Progress: {data['model_name']}\nEpoch {data['epoch']} | Metric: {data['metric']:.4f}",
                 fontsize=14, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1 = axes[0, 0]
    if history.get('train_loss'):
        ax1.plot(epochs, history['train_loss'], 'o-', label='Train', linewidth=2, markersize=6)
    if history.get('val_loss'):
        ax1.plot(epochs, history['val_loss'], 's-', label='Val', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy/Perplexity plot
    ax2 = axes[0, 1]
    metric_name = 'Perplexity' if max(history.get('train_acc', [1])) > 10 else 'Accuracy'
    
    if history.get('train_acc'):
        ax2.plot(epochs, history['train_acc'], 'o-', label='Train', linewidth=2, markersize=6)
    if history.get('val_acc'):
        ax2.plot(epochs, history['val_acc'], 's-', label='Val', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel(metric_name, fontsize=11)
    ax2.set_title(f'{metric_name}', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Progress info
    ax3 = axes[1, 0]
    ax3.axis('off')
    
    current_epoch = len(history['train_loss'])
    best_val = (min if metric_name == 'Perplexity' else max)(history.get('val_acc', [0]))
    
    info_text = f"""
    Model: {data['model_name']}
    
    Current Epoch: {current_epoch}
    Best Val Metric: {best_val:.4f}
    
    Latest Metrics:
      Train Loss: {history['train_loss'][-1]:.4f}
      Val Loss:   {history['val_loss'][-1]:.4f}
      Train {metric_name}: {history['train_acc'][-1]:.4f}
      Val {metric_name}:   {history['val_acc'][-1]:.4f}
    """
    
    ax3.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax3.set_title('Current Status', fontsize=12, fontweight='bold')
    
    # Geometry (if available)
    ax4 = axes[1, 1]
    if data.get('geometry'):
        geo = data['geometry']
        categories = ['Hyperbolic', 'Euclidean', 'Spherical']
        counts = [geo.get('n_hyperbolic', 0), geo.get('n_euclidean', 0), geo.get('n_spherical', 0)]
        colors = ['#2196F3', '#4CAF50', '#FF9800']
        
        bars = ax4.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax4.set_ylabel('Count', fontsize=11)
        ax4.set_title('Learned Geometry Distribution', fontsize=12, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        total = sum(counts)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            if height > 0:
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{count}\n({count/total*100:.0f}%)',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'Geometry data not available\n(will appear after training)',
                ha='center', va='center', fontsize=11)
        ax4.set_xlim([0, 1])
        ax4.set_ylim([0, 1])
        ax4.axis('off')
    
    plt.tight_layout()
    
    if save_plot:
        filepath = 'training_progress.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot: {filepath}")
    
    return fig


def monitor_training(checkpoint_pattern, refresh_interval=30):
    """Monitor training in real-time with auto-refresh"""
    
    print("="*80)
    print("  LIVE TRAINING MONITOR")
    print(f"  Watching: {checkpoint_pattern}")
    print(f"  Refresh: every {refresh_interval} seconds")
    print("  Press Ctrl+C to exit")
    print("="*80)
    
    plt.ion()  # Interactive mode
    
    last_checkpoint = None
    iteration = 0
    
    try:
        while True:
            # Find latest checkpoint
            checkpoint_path = find_latest_checkpoint(checkpoint_pattern)
            
            if checkpoint_path and checkpoint_path != last_checkpoint:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Found checkpoint: {os.path.basename(checkpoint_path)}")
                
                # Clear previous plot
                plt.clf()
                
                # Create new plot
                fig = plot_training_progress(checkpoint_path, save_plot=True)
                
                if fig:
                    plt.draw()
                    plt.pause(0.1)
                    last_checkpoint = checkpoint_path
                    iteration += 1
                    print(f"✓ Updated plot (iteration {iteration})")
            else:
                if iteration == 0:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Waiting for checkpoint...")
                
            # Wait before checking again
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        plt.ioff()
        plt.show()  # Keep final plot open


def main():
    parser = argparse.ArgumentParser(description='Monitor training progress in real-time')
    parser.add_argument('--pattern', type=str, default='checkpoints/best_*.pt',
                       help='Checkpoint file pattern to monitor (default: checkpoints/best_*.pt)')
    parser.add_argument('--refresh', type=int, default=30,
                       help='Refresh interval in seconds (default: 30)')
    parser.add_argument('--once', action='store_true',
                       help='Plot once and exit (no live monitoring)')
    args = parser.parse_args()
    
    if args.once:
        # Plot once from latest checkpoint
        checkpoint_path = find_latest_checkpoint(args.pattern)
        
        if not checkpoint_path:
            print(f"No checkpoints found matching: {args.pattern}")
            sys.exit(1)
        
        print(f"Plotting from: {os.path.basename(checkpoint_path)}")
        plot_training_progress(checkpoint_path, save_plot=True)
        plt.show()
    else:
        # Live monitoring mode
        monitor_training(args.pattern, args.refresh)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
