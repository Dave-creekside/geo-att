"""
Geometric Attention: Learnable Curvature Attention for Transformers

This package implements attention mechanisms that operate in different geometric spaces
(hyperbolic, spherical, Euclidean) with learnable curvature parameters.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .models.geometric_attention import (
    GeometricAttentionHead,
    LearnableCurvatureAttention,
    MultiHeadLearnableCurvature
)

from .models.transformers import (
    GeometricTransformer,
    StandardTransformer,
    GeometricTransformerLayer
)

__all__ = [
    "GeometricAttentionHead",
    "LearnableCurvatureAttention",
    "MultiHeadLearnableCurvature",
    "GeometricTransformer",
    "StandardTransformer",
    "GeometricTransformerLayer"
]
