"""
Transformer architectures using geometric attention.
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Optional, List, Tuple
from .geometric_attention import MultiHeadLearnableCurvature, OptimizedGeometricAttention


class GeometricTransformerLayer(nn.Module):
    """Single transformer layer with learnable curvature attention"""
    
    def __init__(self, dim: int, n_heads: int = 8, ff_dim: int = 2048, 
                 dropout: float = 0.1, use_optimized: bool = False):
        super().__init__()
        
        if use_optimized:
            # Use optimized version with multiple heads
            self.attention = OptimizedMultiHeadGeometricAttention(dim, n_heads)
        else:
            self.attention = MultiHeadLearnableCurvature(dim, n_heads)
            
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[float]]:
        # Self-attention with residual
        attn_out, attentions, curvatures = self.attention(self.norm1(x), mask=mask)
        x = x + self.dropout(attn_out)

        # Feed-forward with residual
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out

        return x, curvatures


class GeometricTransformer(nn.Module):
    """Full transformer with learnable curvature attention"""
    
    def __init__(self, vocab_size: int, dim: int = 256, n_layers: int = 4, 
                 n_heads: int = 8, max_seq_len: int = 128, n_classes: int = 2, 
                 dropout: float = 0.1, use_optimized: bool = False):
        super().__init__()
        self.dim = dim

        # Token + positional embeddings
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)

        # Transformer layers
        self.layers = nn.ModuleList([
            GeometricTransformerLayer(dim, n_heads, dim*4, dropout, use_optimized)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        # Classification head
        self.classifier = nn.Linear(dim, n_classes)

    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List]:
        # Embed tokens
        x = self.token_emb(input_ids)  # [batch, seq, dim]
        x = x + self.pos_emb[:, :x.size(1), :]
        x = self.dropout(x)

        # Track curvatures across layers
        all_curvatures = []

        # Apply transformer layers
        for layer in self.layers:
            x, curvatures = layer(x)
            all_curvatures.append(curvatures)

        x = self.norm(x)

        # Pool: take [CLS] token (first token)
        pooled = x[:, 0, :]

        # Classify
        logits = self.classifier(pooled)

        return logits, all_curvatures


class StandardTransformer(nn.Module):
    """Baseline standard transformer for comparison"""
    
    def __init__(self, vocab_size: int, dim: int = 256, n_layers: int = 4, 
                 n_heads: int = 8, max_seq_len: int = 128, n_classes: int = 2, 
                 dropout: float = 0.1):
        super().__init__()
        self.dim = dim

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)

        # Standard transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(dim, n_classes)

    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, None]:
        x = self.token_emb(input_ids)
        x = x + self.pos_emb[:, :x.size(1), :]
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.norm(x)

        pooled = x[:, 0, :]
        logits = self.classifier(pooled)

        return logits, None


class GeometricTransformerNER(nn.Module):
    """Geometric transformer for token classification"""
    
    def __init__(self, vocab_size: int, dim: int = 768, n_layers: int = 6, 
                 n_heads: int = 12, max_seq_len: int = 128, n_labels: int = 7, 
                 dropout: float = 0.1):
        super().__init__()
        self.dim = dim

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)

        self.layers = nn.ModuleList([
            GeometricTransformerLayer(dim, n_heads, dim*4, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
        # Token classification head
        self.classifier = TokenClassificationHead(dim, n_labels, dropout)

    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List]:
        x = self.token_emb(input_ids)
        x = x + self.pos_emb[:, :x.size(1), :]
        x = self.dropout(x)

        all_curvatures = []
        for layer in self.layers:
            x, curvatures = layer(x)
            all_curvatures.append(curvatures)

        x = self.norm(x)
        logits = self.classifier(x)

        return logits, all_curvatures


class StandardTransformerNER(nn.Module):
    """Standard transformer for token classification"""
    
    def __init__(self, vocab_size: int, dim: int = 768, n_layers: int = 6, 
                 n_heads: int = 12, max_seq_len: int = 128, n_labels: int = 7, 
                 dropout: float = 0.1):
        super().__init__()
        self.dim = dim

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = TokenClassificationHead(dim, n_labels, dropout)

    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, None]:
        x = self.token_emb(input_ids)
        x = x + self.pos_emb[:, :x.size(1), :]
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.norm(x)
        logits = self.classifier(x)

        return logits, None


class TokenClassificationHead(nn.Module):
    """Token-level classification for NER"""
    
    def __init__(self, dim: int, n_labels: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(dim, n_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        logits = self.classifier(x)  # [batch, seq_len, n_labels]
        return logits


class OptimizedMultiHeadGeometricAttention(nn.Module):
    """Optimized multi-head attention with batched operations"""
    
    def __init__(self, dim: int, n_heads: int = 8):
        super().__init__()
        assert dim % n_heads == 0

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # Use ModuleList for heads
        self.heads = nn.ModuleList([
            OptimizedGeometricAttention(self.head_dim)
            for _ in range(n_heads)
        ])

        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, None, List[float]]:
        batch_size, seq_len, dim = x.shape

        # Split into heads
        x_heads = x.view(batch_size, seq_len, self.n_heads, self.head_dim)
        x_heads = x_heads.transpose(1, 2)  # [batch, n_heads, seq, head_dim]

        # Process heads (could be parallelized further)
        head_outputs = []
        curvatures = []

        for i, head in enumerate(self.heads):
            x_head = x_heads[:, i, :, :]
            output, _, k_val = head(x_head, mask=mask)
            head_outputs.append(output)
            curvatures.append(k_val)

        # Concatenate
        output = torch.stack(head_outputs, dim=1)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        output = self.out_proj(output)

        return output, None, curvatures


class OptimizedGeometricTransformer(nn.Module):
    """Optimized full geometric transformer"""
    
    def __init__(self, vocab_size: int, dim: int = 768, n_layers: int = 6, 
                 n_heads: int = 12, max_seq_len: int = 128, n_classes: int = 2, 
                 dropout: float = 0.1):
        super().__init__()
        self.dim = dim

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)

        # Use optimized layers
        self.layers = nn.ModuleList([
            GeometricTransformerLayer(dim, n_heads, dim*4, dropout, use_optimized=True)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(dim, n_classes)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, List]:
        x = self.token_emb(input_ids)
        x = x + self.pos_emb[:, :x.size(1), :]
        x = self.dropout(x)

        all_curvatures = []
        for layer in self.layers:
            x, curvatures = layer(x)
            all_curvatures.append(curvatures)

        x = self.norm(x)
        pooled = x[:, 0, :]
        logits = self.classifier(pooled)

        return logits, all_curvatures
