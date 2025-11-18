"""
Helper utilities for geometric attention transformers.
"""

import torch
import random
import numpy as np
from typing import Optional


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(cuda_id: Optional[int] = None) -> torch.device:
    """Get the appropriate device for computation
    
    Args:
        cuda_id: Specific CUDA device ID to use (0, 1, etc.)
                If None, uses default cuda device
                
    Returns:
        torch.device: The selected device (cuda:X or cpu)
    """
    if torch.cuda.is_available():
        if cuda_id is not None:
            # Check if requested GPU exists
            if cuda_id >= torch.cuda.device_count():
                print(f"Warning: GPU {cuda_id} not found. Found {torch.cuda.device_count()} GPUs.")
                print(f"Falling back to GPU 0")
                return torch.device('cuda:0')
            return torch.device(f'cuda:{cuda_id}')
        return torch.device('cuda')
    else:
        print("CUDA not available, using CPU")
        return torch.device('cpu')


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, loss: float, path: str, additional_info: dict = None):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model: torch.nn.Module, path: str, 
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   device: Optional[torch.device] = None) -> dict:
    """Load model checkpoint"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {path}")
    return checkpoint


def print_model_summary(model: torch.nn.Module, name: str = "Model"):
    """Print model summary"""
    total_params = count_parameters(model)
    print(f"\n{name} Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Total size: {total_params * 4 / 1024 / 1024:.2f} MB (assuming float32)")
    
    # Count layers
    n_layers = sum(1 for _ in model.modules())
    print(f"  Total modules: {n_layers}")


def format_time(seconds: float) -> str:
    """Format time in seconds to readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_geometry_type(curvature: float, threshold: float = 0.1) -> str:
    """Determine geometry type from curvature value"""
    if curvature < -threshold:
        return "hyperbolic"
    elif abs(curvature) <= threshold:
        return "euclidean"
    else:
        return "spherical"


def analyze_geometry_distribution(curvatures: np.ndarray, threshold: float = 0.1) -> dict:
    """Analyze distribution of geometric types in curvatures"""
    flat = curvatures.flatten()
    
    n_hyperbolic = np.sum(flat < -threshold)
    n_euclidean = np.sum(np.abs(flat) <= threshold)
    n_spherical = np.sum(flat > threshold)
    total = len(flat)
    
    return {
        'n_hyperbolic': n_hyperbolic,
        'n_euclidean': n_euclidean,
        'n_spherical': n_spherical,
        'pct_hyperbolic': n_hyperbolic / total * 100,
        'pct_euclidean': n_euclidean / total * 100,
        'pct_spherical': n_spherical / total * 100,
        'total': total
    }


def print_results_table(results: dict):
    """Print formatted results table"""
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key:30s}: {value:.4f}")
        elif isinstance(value, int):
            print(f"  {key:30s}: {value:,}")
        else:
            print(f"  {key:30s}: {value}")
    
    print("="*70)
