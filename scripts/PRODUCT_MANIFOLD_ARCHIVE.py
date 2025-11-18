"""
PRODUCT MANIFOLD ATTENTION - ARCHIVED CODE

This file contains an alternative attention mechanism that was developed
but found to be ineffective in initial tests (prior to causal masking fix).

Status: EXPERIMENTAL / NOT MAINTAINED
Last Active: Pre-October 2025
Causal Masking: NOT IMPLEMENTED (would need updates to work with current system)
Archived: November 2025

==============================================================================
Concept:
==============================================================================

Instead of each head learning a single curvature k (hyperbolic, euclidean, or spherical),
each head computes distances in ALL THREE geometries and learns weights to combine them:

    distance = w_H * dist_hyperbolic + w_E * dist_euclidean + w_S * dist_spherical

Where w_H, w_E, w_S are learnable parameters (softmaxed to sum to 1).

==============================================================================
Why It Was Set Aside:
==============================================================================

- Initial tests showed worse performance than LearnableCurvatureAttention
- More complex and harder to interpret
- Current approach (learnable curvature) works extremely well
- Phase transition discovery (Euclidean emergence) validates current approach
- Adds computational overhead without clear benefits

==============================================================================
Current System (LearnableCurvature) Advantages:
==============================================================================

✅ Simpler: Each head picks ONE geometry via single parameter k
✅ Faster: No need to compute 3 distances per head
✅ Interpretable: Can clearly see which heads are H/E/S
✅ Effective: 29% better perplexity than standard
✅ Discoverable: Found phase transitions (E% increases over training)

==============================================================================
Possible Future Use:
==============================================================================

If you want to revisit this approach with causal masking:

1. Add causal masking support:
   - Update ProductManifoldAttentionHead.forward() to accept mask parameter
   - Apply mask before softmax (see current geometric_attention.py for examples)

2. Create language model variant:
   - ProductManifoldCausalLM (similar to GeometricCausalLM)
   - Use causal masking throughout

3. Test hypothesis:
   - Do weights concentrate [1,0,0] (→ specialization like current system)?
   - Or stay distributed [0.5,0.3,0.2] (→ true geometry blending)?
   - Does it also show phase transitions?
   - Is it worth the added complexity?

4. Compare performance:
   - ProductManifold vs LearnableCurvature
   - Perplexity, generation quality, training time

Note: This code is preserved for reference and potential future experiments.
It represents a valid alternative approach that may be worth revisiting.

==============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple

# NOTE: This code has NOT been updated with causal masking support!
# It would need modifications to work with current language models.

# ============================================================================
# ARCHIVED CLASSES FROM geometric_attention.py
# ============================================================================

class ProductManifoldAttention(nn.Module):
    """Multi-head attention where each head operates in different geometry"""
    
    def __init__(self, dim: int, n_heads: int = 4, head_configs: Optional[List] = None):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # Default: mix of geometries
        if head_configs is None:
            head_configs = [
                {'type': 'hyperbolic', 'curvature': -1.0},
                {'type': 'spherical', 'curvature': 1.0},
                {'type': 'euclidean', 'curvature': 0.0},
            ] * (n_heads // 3 + 1)

        self.heads = nn.ModuleList([
            GeometricAttentionHead(
                self.head_dim,
                manifold_type=config['type'],
                curvature=config['curvature']
            )
            for config in head_configs[:n_heads]
        ])

        self.to_head_spaces = nn.ModuleList([
            nn.Linear(dim, self.head_dim) for _ in range(n_heads)
        ])

        concat_dim = self.head_dim * n_heads
        self.output_proj = nn.Linear(concat_dim, dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List]:
        """x: [batch, seq_len, dim]"""
        batch, seq_len, _ = x.shape

        head_outputs = []
        attention_weights = []

        for head, proj in zip(self.heads, self.to_head_spaces):
            x_head = proj(x)  # [batch, seq_len, head_dim]

            if head.manifold_type == 'hyperbolic':
                # Add time coordinate: t = sqrt(1 + ||x||^2)
                space_norm = torch.sum(x_head ** 2, dim=-1, keepdim=True)
                time_coord = torch.sqrt(1 + space_norm + 1e-8)
                x_manifold = torch.cat([time_coord, x_head], dim=-1)

            elif head.manifold_type == 'spherical':
                x_manifold = head.manifold.projx(x_head) if hasattr(head, 'manifold') else x_head
            else:
                x_manifold = x_head

            output, attn = head(x_manifold, values=x_head)
            head_outputs.append(output)
            attention_weights.append(attn)

        combined = torch.cat(head_outputs, dim=-1)
        output = self.output_proj(combined)

        return output, attention_weights


# ============================================================================
# ARCHIVED CLASSES FROM transformers.py
# ============================================================================

class ProductManifoldTransformerLayer(nn.Module):
    """Transformer layer with product manifold attention"""
    
    def __init__(self, dim: int, n_heads: int = 8, ff_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadProductManifold(dim, n_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        # Self-attention with residual
        attn_out, _, geo_weights = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)

        # Feed-forward with residual
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out

        return x, geo_weights


class MultiHeadProductManifold(nn.Module):
    """Multi-head attention where each head operates in product space"""
    
    def __init__(self, dim: int, n_heads: int = 8):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # Create separate heads
        self.heads = nn.ModuleList([
            ProductManifoldAttentionHead(self.head_dim)
            for _ in range(n_heads)
        ])

        # Output projection
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None, np.ndarray]:
        """
        x: [batch, seq, dim]
        Returns: output, attention_weights, geometry_weights per head
        """
        batch_size, seq_len, dim = x.shape

        # Split into heads
        x_heads = x.view(batch_size, seq_len, self.n_heads, self.head_dim)
        x_heads = x_heads.transpose(1, 2)  # [batch, n_heads, seq, head_dim]

        head_outputs = []
        all_geometry_weights = []

        for i, head in enumerate(self.heads):
            x_head = x_heads[:, i, :, :]  # [batch, seq, head_dim]
            output, _, geo_weights = head(x_head)
            head_outputs.append(output)
            all_geometry_weights.append(geo_weights.detach().cpu().numpy())

        # Concatenate heads
        output = torch.stack(head_outputs, dim=1)  # [batch, n_heads, seq, head_dim]
        output = output.transpose(1, 2)  # [batch, seq, n_heads, head_dim]
        output = output.contiguous().view(batch_size, seq_len, dim)

        # Output projection
        output = self.out_proj(output)

        # Geometry weights: [n_heads, 3] (hyp, euc, sph for each head)
        geometry_weights = np.array(all_geometry_weights)

        return output, None, geometry_weights


class ProductManifoldAttentionHead(nn.Module):
    """
    Single attention head operating in product space ℍ² × 𝕊² × ℝ²
    Each head learns how to weight/combine all three geometries
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.scale = dim ** -0.5

        # Standard QKV projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        # Learnable geometry weights (one set per head)
        # Initialize close to equal (1/3 each)
        self.geometry_weights = nn.Parameter(torch.ones(3) / 3)  # [w_hyp, w_euc, w_sph]

        # Optional: learnable temperature for distance scaling
        self.temperature = nn.Parameter(torch.ones(1))

    def compute_distances(self, q: torch.Tensor, k: torch.Tensor) -> dict:
        """
        Compute distances in all three geometries simultaneously
        q, k: [batch, seq, dim]
        Returns: dict with distances in each geometry
        """
        batch_size, seq_len, dim = q.shape
        eps = 1e-8

        # Euclidean distance (always computed)
        diff = q.unsqueeze(2) - k.unsqueeze(1)  # [batch, seq_q, seq_k, dim]
        euclidean_dist_sq = torch.sum(diff ** 2, dim=-1)
        euclidean_dist = torch.sqrt(euclidean_dist_sq + eps)  # [batch, seq_q, seq_k]

        # Hyperbolic distance (fixed k=-1 for stability)
        k_hyp = -1.0
        q_norm = torch.norm(q, dim=-1, keepdim=True)
        k_norm = torch.norm(k, dim=-1, keepdim=True)

        cosh_arg = 1 + 2 * euclidean_dist_sq / ((1 - q_norm.unsqueeze(2)**2) * (1 - k_norm.unsqueeze(1)**2) + eps)
        cosh_arg = torch.clamp(cosh_arg, min=1.0, max=1e6)
        hyperbolic_dist = torch.acosh(cosh_arg) / torch.sqrt(torch.abs(torch.tensor(k_hyp)) + eps)

        # Spherical distance (fixed k=+1)
        k_sph = 1.0
        q_normalized = q / (torch.norm(q, dim=-1, keepdim=True) + eps)
        k_normalized = k / (torch.norm(k, dim=-1, keepdim=True) + eps)

        cos_angle = torch.sum(q_normalized.unsqueeze(2) * k_normalized.unsqueeze(1), dim=-1)
        cos_angle = torch.clamp(cos_angle, -1.0 + eps, 1.0 - eps)
        spherical_dist = torch.acos(cos_angle) / torch.sqrt(torch.tensor(k_sph) + eps)

        return {
            'hyperbolic': hyperbolic_dist,
            'euclidean': euclidean_dist,
            'spherical': spherical_dist
        }

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [batch, seq, dim]
        Returns: output, attention_weights, learned_weights
        """
        batch_size, seq_len, dim = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Compute distances in all geometries
        distances = self.compute_distances(q, k)

        # Normalize geometry weights with softmax
        weights = F.softmax(self.geometry_weights, dim=0)
        w_hyp, w_euc, w_sph = weights[0], weights[1], weights[2]

        # Combine distances with learned weights
        combined_distance = (
            w_hyp * distances['hyperbolic'] +
            w_euc * distances['euclidean'] +
            w_sph * distances['spherical']
        )

        # Convert distance to similarity (negative distance, scaled)
        similarity = -combined_distance * self.temperature

        # Attention weights
        attention_weights = F.softmax(similarity, dim=-1)

        # Apply attention
        output = torch.matmul(attention_weights, v)

        return output, attention_weights, weights


class ProductManifoldTransformer(nn.Module):
    """Full transformer with product manifold attention"""
    
    def __init__(self, vocab_size: int, dim: int = 768, n_layers: int = 6, 
                 n_heads: int = 12, max_seq_len: int = 128, n_classes: int = 2, 
                 dropout: float = 0.1):
        super().__init__()
        self.dim = dim

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)

        self.layers = nn.ModuleList([
            ProductManifoldTransformerLayer(dim, n_heads, dim*4, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(dim, n_classes)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, List]:
        x = self.token_emb(input_ids)
        x = x + self.pos_emb[:, :x.size(1), :]
        x = self.dropout(x)

        all_geometry_weights = []

        for layer in self.layers:
            x, geo_weights = layer(x)
            all_geometry_weights.append(geo_weights)

        x = self.norm(x)
        pooled = x[:, 0, :]
        logits = self.classifier(pooled)

        return logits, all_geometry_weights


# ==============================================================================
# TO RESURRECT THIS CODE:
# ==============================================================================
#
# 1. Add to geometric_attention/models/experimental/__init__.py
# 2. Update with causal masking (see language_models.py for pattern)
# 3. Create ProductManifoldCausalLM variant
# 4. Add to training scripts as optional model type
# 5. Compare against current GeometricCausalLM
#
# ==============================================================================
