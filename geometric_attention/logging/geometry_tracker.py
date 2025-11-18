"""
Geometry tracking utilities for analyzing curvature evolution.
Extracts, analyzes, and formats geometric attention data.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional


def extract_geometry_statistics(curvatures: List[List[float]], 
                                threshold: float = 0.1) -> Dict[str, Any]:
    """
    Extract geometry statistics from curvature data.
    
    Args:
        curvatures: List of lists (per layer, per head) of curvature values
        threshold: Threshold for classifying geometries (default: 0.1)
        
    Returns:
        Dictionary with geometry counts, percentages, and statistics
    """
    # Flatten all curvatures
    flat_curvatures = []
    for layer_curvs in curvatures:
        flat_curvatures.extend(layer_curvs)
    
    flat_curvatures = np.array(flat_curvatures)
    total_heads = len(flat_curvatures)
    
    # Count geometry types
    n_hyperbolic = np.sum(flat_curvatures < -threshold)
    n_euclidean = np.sum(np.abs(flat_curvatures) <= threshold)
    n_spherical = np.sum(flat_curvatures > threshold)
    
    # Calculate percentages
    pct_hyperbolic = 100.0 * n_hyperbolic / total_heads if total_heads > 0 else 0.0
    pct_euclidean = 100.0 * n_euclidean / total_heads if total_heads > 0 else 0.0
    pct_spherical = 100.0 * n_spherical / total_heads if total_heads > 0 else 0.0
    
    # Statistics
    stats = {
        # Counts
        'n_hyperbolic': int(n_hyperbolic),
        'n_euclidean': int(n_euclidean),
        'n_spherical': int(n_spherical),
        'total_heads': int(total_heads),
        
        # Percentages
        'pct_hyperbolic': float(pct_hyperbolic),
        'pct_euclidean': float(pct_euclidean),
        'pct_spherical': float(pct_spherical),
        
        # Curvature statistics
        'curvature_mean': float(np.mean(flat_curvatures)),
        'curvature_std': float(np.std(flat_curvatures)),
        'curvature_min': float(np.min(flat_curvatures)),
        'curvature_max': float(np.max(flat_curvatures)),
        'curvature_median': float(np.median(flat_curvatures)),
        
        # Ratio
        'hyperbolic_spherical_ratio': float(n_hyperbolic / n_spherical) if n_spherical > 0 else 0.0
    }
    
    return stats


def format_curvatures_for_logging(curvatures: List[List[float]]) -> Dict[str, List[float]]:
    """
    Format curvatures for JSON logging.
    
    Args:
        curvatures: List of lists (per layer, per head)
        
    Returns:
        Dictionary with layer keys and curvature lists
    """
    formatted = {}
    for layer_idx, layer_curvs in enumerate(curvatures):
        layer_key = f"layer_{layer_idx}"
        formatted[layer_key] = [float(k) for k in layer_curvs]
    
    return formatted


def detect_geometry_shift(geometry_history: List[Dict[str, float]], 
                         window_size: int = 10,
                         threshold_change: float = 5.0) -> Optional[int]:
    """
    Detect significant shifts in geometry distribution.
    
    Args:
        geometry_history: List of geometry statistics dicts over time
        window_size: Number of epochs to use for smoothing
        threshold_change: Minimum percentage change to consider significant
        
    Returns:
        Epoch index where significant shift occurred, or None
    """
    if len(geometry_history) < window_size * 2:
        return None
    
    euclidean_pcts = [g['pct_euclidean'] for g in geometry_history]
    
    # Smooth with moving average
    smoothed = []
    for i in range(len(euclidean_pcts) - window_size):
        window = euclidean_pcts[i:i+window_size]
        smoothed.append(np.mean(window))
    
    # Find maximum gradient (rate of change)
    if len(smoothed) < 2:
        return None
    
    gradients = np.diff(smoothed)
    max_gradient_idx = np.argmax(np.abs(gradients))
    max_gradient = gradients[max_gradient_idx]
    
    # Check if change is significant
    if abs(max_gradient) > threshold_change:
        return max_gradient_idx + window_size // 2
    
    return None


class GeometryTracker:
    """
    Tracks and analyzes geometry evolution during training.
    """
    
    def __init__(self, threshold: float = 0.1):
        """
        Args:
            threshold: Curvature threshold for geometry classification
        """
        self.threshold = threshold
        self.history = []
        
    def add_snapshot(self, epoch: int, step: int, curvatures: List[List[float]]) -> Dict[str, Any]:
        """
        Add a geometry snapshot and return statistics.
        
        Args:
            epoch: Current epoch
            step: Current step
            curvatures: Curvature data from model
            
        Returns:
            Dictionary with geometry statistics
        """
        stats = extract_geometry_statistics(curvatures, self.threshold)
        stats['epoch'] = epoch
        stats['step'] = step
        
        self.history.append(stats)
        return stats
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """
        Get summary of geometry evolution over training.
        
        Returns:
            Dictionary with evolution statistics
        """
        if not self.history:
            return {}
        
        # Extract time series
        epochs = [h['epoch'] for h in self.history]
        h_pcts = [h['pct_hyperbolic'] for h in self.history]
        e_pcts = [h['pct_euclidean'] for h in self.history]
        s_pcts = [h['pct_spherical'] for h in self.history]
        
        # Detect shift
        shift_epoch = detect_geometry_shift(self.history)
        
        summary = {
            'total_snapshots': len(self.history),
            'epoch_range': [min(epochs), max(epochs)],
            
            # Initial state (first 10% of training)
            'initial_geometry': {
                'hyperbolic': float(np.mean(h_pcts[:len(h_pcts)//10])),
                'euclidean': float(np.mean(e_pcts[:len(e_pcts)//10])),
                'spherical': float(np.mean(s_pcts[:len(s_pcts)//10]))
            },
            
            # Final state (last 10% of training)
            'final_geometry': {
                'hyperbolic': float(np.mean(h_pcts[-len(h_pcts)//10:])),
                'euclidean': float(np.mean(e_pcts[-len(e_pcts)//10:])),
                'spherical': float(np.mean(s_pcts[-len(s_pcts)//10:]))
            },
            
            # Change over training
            'geometry_change': {
                'hyperbolic': float(np.mean(h_pcts[-len(h_pcts)//10:]) - np.mean(h_pcts[:len(h_pcts)//10])),
                'euclidean': float(np.mean(e_pcts[-len(e_pcts)//10:]) - np.mean(e_pcts[:len(e_pcts)//10])),
                'spherical': float(np.mean(s_pcts[-len(s_pcts)//10:]) - np.mean(s_pcts[:len(s_pcts)//10]))
            },
            
            # Phase transition detection
            'phase_transition_epoch': shift_epoch
        }
        
        return summary
    
    def get_current_distribution(self) -> Dict[str, float]:
        """Get most recent geometry distribution."""
        if not self.history:
            return {}
        
        latest = self.history[-1]
        return {
            'hyperbolic': latest['pct_hyperbolic'],
            'euclidean': latest['pct_euclidean'],
            'spherical': latest['pct_spherical']
        }
    
    def get_timeline_data(self) -> Tuple[List[int], Dict[str, List[float]]]:
        """
        Get data formatted for timeline plotting.
        
        Returns:
            (epochs, data_dict) where data_dict has 'hyperbolic', 'euclidean', 'spherical' keys
        """
        epochs = [h['epoch'] for h in self.history]
        data = {
            'hyperbolic': [h['pct_hyperbolic'] for h in self.history],
            'euclidean': [h['pct_euclidean'] for h in self.history],
            'spherical': [h['pct_spherical'] for h in self.history]
        }
        return epochs, data
