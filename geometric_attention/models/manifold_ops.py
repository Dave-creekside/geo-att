"""
Manifold operations for geometric deep learning.

Implements proper exponential/logarithmic maps and distance functions
for hyperbolic (Poincaré ball), spherical, and Euclidean geometries.

This module provides the mathematical foundation for learning on curved spaces
with correct gradient flow.
"""

import torch
import torch.nn.functional as F
from typing import Optional


# ============================================================================
# Poincaré Ball Operations (Hyperbolic Geometry)
# ============================================================================

def mobius_add(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Möbius addition in the Poincaré ball model.
    
    Formula: x ⊕_c y = [(1 + 2c⟨x,y⟩ + c||y||²)x + (1 - c||x||²)y] / [1 + 2c⟨x,y⟩ + c²||x||²||y||²]
    
    Args:
        x, y: Points on Poincaré ball [*, dim]
        c: Curvature (c < 0 for hyperbolic)
        dim: Dimension along which to compute norms
        
    Returns:
        x ⊕ y on the Poincaré ball
    """
    x2 = torch.sum(x * x, dim=dim, keepdim=True)
    y2 = torch.sum(y * y, dim=dim, keepdim=True)
    xy = torch.sum(x * y, dim=dim, keepdim=True)
    
    numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denominator = 1 + 2 * c * xy + c * c * x2 * y2
    
    # Numerical stability
    denominator = torch.clamp(denominator, min=1e-15)
    
    return numerator / denominator


def mobius_scalar_mul(r: torch.Tensor, x: torch.Tensor, c: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Möbius scalar multiplication in Poincaré ball.
    
    Formula: r ⊗_c x = (1/√(-c)) * tanh(r * arctanh(√(-c)||x||)) * (x / ||x||)
    
    Args:
        r: Scalar multiplier
        x: Point on Poincaré ball [*, dim]
        c: Curvature (c < 0)
        dim: Dimension along which to compute norm
        
    Returns:
        r ⊗ x on the Poincaré ball
    """
    x_norm = torch.norm(x, dim=dim, keepdim=True).clamp(min=1e-15)
    sqrt_c = torch.sqrt(-c)
    
    # arctanh(√(-c)||x||)
    inner = (sqrt_c * x_norm).clamp(min=-1 + 1e-7, max=1 - 1e-7)
    arctanh_val = torch.arctanh(inner)
    
    # tanh(r * arctanh(...))
    tanh_val = torch.tanh(r * arctanh_val)
    
    # Final formula
    return (tanh_val / (sqrt_c * x_norm)) * x


def exp_map_zero(v: torch.Tensor, c: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Exponential map from tangent space at origin to Poincaré ball.
    
    Formula: exp_0(v) = tanh(√(-c)||v||) * v / (√(-c)||v||)
    
    Args:
        v: Tangent vector at origin [*, dim]
        c: Curvature (c < 0)
        dim: Dimension along which to compute norm
        
    Returns:
        Point on Poincaré ball
    """
    v_norm = torch.norm(v, dim=dim, keepdim=True).clamp(min=1e-15)
    sqrt_c = torch.sqrt(-c)
    
    # tanh(√(-c)||v||) * v / (√(-c)||v||)
    tanh_term = torch.tanh(sqrt_c * v_norm)
    
    return tanh_term * v / (sqrt_c * v_norm)


def log_map_zero(x: torch.Tensor, c: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Logarithmic map from Poincaré ball to tangent space at origin.
    
    Formula: log_0(x) = arctanh(√(-c)||x||) * x / (√(-c)||x||)
    
    Args:
        x: Point on Poincaré ball [*, dim]
        c: Curvature (c < 0)
        dim: Dimension along which to compute norm
        
    Returns:
        Tangent vector at origin
    """
    x_norm = torch.norm(x, dim=dim, keepdim=True).clamp(min=1e-15)
    sqrt_c = torch.sqrt(-c)
    
    # Clamp to valid range for arctanh
    inner = (sqrt_c * x_norm).clamp(min=-1 + 1e-7, max=1 - 1e-7)
    arctanh_val = torch.arctanh(inner)
    
    return arctanh_val * x / (sqrt_c * x_norm)


def exp_map(v: torch.Tensor, x: torch.Tensor, c: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Exponential map from tangent space at x to Poincaré ball.
    
    Formula: exp_x(v) = x ⊕_c [tanh(√(-c)||v||_x/2) * v / (√(-c)||v||_x)]
    
    Args:
        v: Tangent vector at point x [*, dim]
        x: Base point on Poincaré ball [*, dim]
        c: Curvature (c < 0)
        dim: Dimension along which to compute norm
        
    Returns:
        Point on Poincaré ball
    """
    v_norm = torch.norm(v, dim=dim, keepdim=True).clamp(min=1e-15)
    sqrt_c = torch.sqrt(-c)
    
    # Second term: tanh(√(-c)||v||/2) * v / (√(-c)||v||)
    second_term = torch.tanh(sqrt_c * v_norm / 2) * v / (sqrt_c * v_norm)
    
    return mobius_add(x, second_term, c, dim=dim)


def log_map(y: torch.Tensor, x: torch.Tensor, c: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Logarithmic map from Poincaré ball to tangent space at x.
    
    Formula: log_x(y) = (2/√(-c)) * arctanh(√(-c)||−x ⊕_c y||) * (−x ⊕_c y) / ||−x ⊕_c y||
    
    Args:
        y: Point on Poincaré ball [*, dim]
        x: Base point on Poincaré ball [*, dim]
        c: Curvature (c < 0)
        dim: Dimension along which to compute norm
        
    Returns:
        Tangent vector at x
    """
    diff = mobius_add(-x, y, c, dim=dim)
    diff_norm = torch.norm(diff, dim=dim, keepdim=True).clamp(min=1e-15)
    sqrt_c = torch.sqrt(-c)
    
    # Clamp for arctanh
    inner = (sqrt_c * diff_norm).clamp(min=-1 + 1e-7, max=1 - 1e-7)
    arctanh_val = torch.arctanh(inner)
    
    return (2 / sqrt_c) * arctanh_val * diff / diff_norm


def project_to_poincare_ball(x: torch.Tensor, c: torch.Tensor, dim: int = -1, eps: float = 1e-5) -> torch.Tensor:
    """
    Project points to Poincaré ball by enforcing ||x|| < 1/√(-c).
    
    Args:
        x: Points to project [*, dim]
        c: Curvature (c < 0)
        dim: Dimension along which to compute norm
        eps: Safety margin from boundary
        
    Returns:
        Projected points on Poincaré ball
    """
    norm = torch.norm(x, dim=dim, keepdim=True).clamp(min=1e-15)
    max_norm = (1.0 - eps) / torch.sqrt(-c)
    
    # Only project if norm exceeds maximum
    cond = norm > max_norm
    projected = x / norm * max_norm
    
    return torch.where(cond, projected, x)


def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute distance between points on Poincaré ball.
    
    Formula: d_c(x,y) = (2/√(-c)) * arctanh(√(-c)||−x ⊕_c y||)
    
    Args:
        x, y: Points on Poincaré ball [*, dim]
        c: Curvature (c < 0)
        dim: Dimension along which to compute distance
        
    Returns:
        Distances [*, 1] or [*] depending on keepdim
    """
    sqrt_c = torch.sqrt(-c)
    
    # Compute ||−x ⊕_c y||
    diff = mobius_add(-x, y, c, dim=dim)
    diff_norm = torch.norm(diff, dim=dim, keepdim=False)
    
    # Clamp for numerical stability
    inner = (sqrt_c * diff_norm).clamp(min=-1 + 1e-7, max=1 - 1e-7)
    arctanh_val = torch.arctanh(inner)
    
    return (2 / sqrt_c) * arctanh_val


# ============================================================================
# Spherical Operations
# ============================================================================

def project_to_sphere(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Project points to unit sphere by normalization.
    
    Args:
        x: Points to project [*, dim]
        dim: Dimension along which to normalize
        
    Returns:
        Normalized points on sphere
    """
    return F.normalize(x, p=2, dim=dim)


def spherical_distance(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute great-circle distance on sphere.
    
    Formula: d(x,y) = (1/√c) * arccos(⟨x,y⟩)
    
    Args:
        x, y: Points on sphere (assumed normalized) [*, dim]
        c: Curvature (c > 0)
        dim: Dimension along which to compute distance
        
    Returns:
        Distances
    """
    # Cosine similarity
    cos_angle = torch.sum(x * y, dim=dim)
    cos_angle = torch.clamp(cos_angle, min=-1.0 + 1e-7, max=1.0 - 1e-7)
    
    # Great circle distance
    angle = torch.acos(cos_angle)
    sqrt_c = torch.sqrt(c)
    
    return angle / sqrt_c


# ============================================================================
# Euclidean Operations
# ============================================================================

def euclidean_distance(x: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute Euclidean distance.
    
    Args:
        x, y: Points [*, dim]
        dim: Dimension along which to compute distance
        
    Returns:
        Distances
    """
    diff = x - y
    return torch.norm(diff, dim=dim)


# ============================================================================
# Unified Interface
# ============================================================================

def unified_exp_map(v: torch.Tensor, x: torch.Tensor, curvature: torch.Tensor, 
                    threshold: float = 0.01, dim: int = -1) -> torch.Tensor:
    """
    Unified exponential map that handles all geometries.
    
    Args:
        v: Tangent vector [*, dim]
        x: Base point [*, dim]
        curvature: Learnable curvature parameter
        threshold: Threshold for near-zero curvature
        dim: Dimension along which to operate
        
    Returns:
        Point on manifold
    """
    if curvature < -threshold:  # Hyperbolic
        result = exp_map(v, x, curvature, dim=dim)
        return project_to_poincare_ball(result, curvature, dim=dim)
    elif curvature > threshold:  # Spherical
        # For spherical, just normalize (simple stereographic projection)
        return project_to_sphere(x + v, dim=dim)
    else:  # Euclidean
        return x + v


def unified_distance(x: torch.Tensor, y: torch.Tensor, curvature: torch.Tensor,
                     threshold: float = 0.01, dim: int = -1) -> torch.Tensor:
    """
    Unified distance function that handles all geometries.
    
    NOTE: Uses Python if statements for correctness. The torch.where() approach
    caused NaN gradients because it computed distances on unprojected points.
    The slight compile overhead is acceptable for numerical stability.
    
    Args:
        x, y: Points [*, dim]
        curvature: Learnable curvature parameter
        threshold: Threshold for near-zero curvature
        dim: Dimension along which to compute distance
        
    Returns:
        Distances
    """
    c = curvature
    
    if c < -threshold:  # Hyperbolic
        return poincare_distance(x, y, c, dim=dim)
    elif c > threshold:  # Spherical
        return spherical_distance(x, y, c, dim=dim)
    else:  # Euclidean
        return euclidean_distance(x, y, dim=dim)


def project_to_manifold(x: torch.Tensor, curvature: torch.Tensor,
                       threshold: float = 0.01, dim: int = -1) -> torch.Tensor:
    """
    Project points to appropriate manifold based on curvature.
    
    NOTE: Uses Python if statements for correctness. The torch.where() approach
    caused NaN gradients in spherical heads during extended training (Test 7).
    The slight compile overhead is acceptable for numerical stability.
    
    Args:
        x: Points [*, dim]
        curvature: Learnable curvature parameter
        threshold: Threshold for near-zero curvature
        dim: Dimension along which to project
        
    Returns:
        Points on manifold
    """
    if curvature < -threshold:  # Hyperbolic
        return project_to_poincare_ball(x, curvature, dim=dim)
    elif curvature > threshold:  # Spherical
        return project_to_sphere(x, dim=dim)
    else:  # Euclidean
        return x  # No projection needed


# ============================================================================
# Utility Functions
# ============================================================================

def check_poincare_point(x: torch.Tensor, c: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Check if points are valid on Poincaré ball (||x|| < 1/√(-c)).
    
    Returns:
        Boolean tensor indicating validity
    """
    norm = torch.norm(x, dim=dim)
    max_norm = 1.0 / torch.sqrt(-c)
    return norm < max_norm


def poincare_to_lorentz(x: torch.Tensor, c: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Convert from Poincaré ball to Lorentz/hyperboloid model.
    
    Formula: (x_0, x_1, ..., x_n) → ((1 + ||x||²) / (1 - ||x||²), 2x / (1 - ||x||²))
    
    Args:
        x: Points on Poincaré ball [*, dim]
        c: Curvature (c < 0)
        
    Returns:
        Points on hyperboloid [*, dim+1]
    """
    x_norm_sq = torch.sum(x * x, dim=dim, keepdim=True)
    denominator = 1 - x_norm_sq
    denominator = torch.clamp(denominator, min=1e-15)
    
    # Time coordinate
    x_0 = (1 + x_norm_sq) / denominator
    
    # Space coordinates
    x_space = 2 * x / denominator
    
    return torch.cat([x_0, x_space], dim=dim)
