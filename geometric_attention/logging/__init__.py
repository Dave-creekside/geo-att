"""
Comprehensive logging system for geometric attention experiments.
Tracks training metrics, geometry evolution, and attention patterns.
"""

from .experiment_logger import ExperimentLogger
from .geometry_tracker import GeometryTracker, extract_geometry_statistics

__all__ = [
    'ExperimentLogger',
    'GeometryTracker',
    'extract_geometry_statistics'
]
