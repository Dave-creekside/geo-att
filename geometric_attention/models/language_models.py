"""
Language models using geometric attention.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from .transformers import GeometricTransformerLayer


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create causal attention mask (upper triangular).
    
    Args:
        seq_len: Sequence length
        device: Device to create mask on
        
    Returns:
        Boolean mask of shape [seq_len, seq_len] where True = masked position
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask


class GeometricCausalLM(nn.Module):
    """Geometric transformer for causal language modeling"""
    
    def __init__(self, vocab_size: int, dim: int = 768, n_layers: int = 6, 
                 n_heads: int = 12, max_seq_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.dim = dim

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)

        # Use geometric layers
        self.layers = nn.ModuleList([
            GeometricTransformerLayer(dim, n_heads, dim*4, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        # Language modeling head (predict next token)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Tie weights with embedding (common practice)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], List]:
        x = self.token_emb(input_ids)
        x = x + self.pos_emb[:, :x.size(1), :]
        x = self.dropout(x)

        # Create causal mask for autoregressive generation
        seq_len = x.size(1)
        causal_mask = create_causal_mask(seq_len, x.device)

        all_curvatures = []
        for layer in self.layers:
            x, curvatures = layer(x, mask=causal_mask)
            all_curvatures.append(curvatures)

        x = self.norm(x)
        logits = self.lm_head(x)  # [batch, seq, vocab_size]

        loss = None
        if labels is not None:
            # Flatten for loss computation
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return logits, loss, all_curvatures


class StandardCausalLM(nn.Module):
    """Standard transformer for causal language modeling"""
    
    def __init__(self, vocab_size: int, dim: int = 768, n_layers: int = 6, 
                 n_heads: int = 12, max_seq_len: int = 128, dropout: float = 0.1):
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
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        x = self.token_emb(input_ids)
        x = x + self.pos_emb[:, :x.size(1), :]
        x = self.dropout(x)

        # Create causal mask for TransformerEncoder (uses additive mask, not boolean)
        seq_len = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return logits, loss, None


class TinyGeometricLM(nn.Module):
    """Minimal geometric language model for testing"""
    
    def __init__(self, vocab_size: int, dim: int = 128, n_layers: int = 1, 
                 n_heads: int = 2, max_seq_len: int = 128, dropout: float = 0.1):
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
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], List]:
        x = self.token_emb(input_ids)
        x = x + self.pos_emb[:, :x.size(1), :]
        x = self.dropout(x)

        # Create causal mask for autoregressive generation
        seq_len = x.size(1)
        causal_mask = create_causal_mask(seq_len, x.device)

        all_curvatures = []
        for layer in self.layers:
            x, curvatures = layer(x, mask=causal_mask)
            all_curvatures.append(curvatures)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return logits, loss, all_curvatures


class TinyStandardLM(nn.Module):
    """Minimal standard language model for comparison"""
    
    def __init__(self, vocab_size: int, dim: int = 128, n_layers: int = 1, 
                 n_heads: int = 2, max_seq_len: int = 128, dropout: float = 0.1):
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
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        x = self.token_emb(input_ids)
        x = x + self.pos_emb[:, :x.size(1), :]
        x = self.dropout(x)

        # Create causal mask for TransformerEncoder
        seq_len = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return logits, loss, None


class SmallGeometricLM(nn.Module):
    """Small geometric language model"""
    
    def __init__(self, vocab_size: int, dim: int = 256, n_layers: int = 2, 
                 n_heads: int = 4, max_seq_len: int = 128, dropout: float = 0.3):
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
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], List]:
        x = self.token_emb(input_ids)
        x = x + self.pos_emb[:, :x.size(1), :]
        x = self.dropout(x)

        # Create causal mask for autoregressive generation
        seq_len = x.size(1)
        causal_mask = create_causal_mask(seq_len, x.device)

        all_curvatures = []
        for layer in self.layers:
            x, curvatures = layer(x, mask=causal_mask)
            all_curvatures.append(curvatures)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return logits, loss, all_curvatures


class SmallStandardLM(nn.Module):
    """Small standard language model"""
    
    def __init__(self, vocab_size: int, dim: int = 256, n_layers: int = 2, 
                 n_heads: int = 4, max_seq_len: int = 128, dropout: float = 0.3):
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
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        x = self.token_emb(input_ids)
        x = x + self.pos_emb[:, :x.size(1), :]
        x = self.dropout(x)

        # Create causal mask for TransformerEncoder
        seq_len = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return logits, loss, None


class LargeGeometricLM(nn.Module):
    """Large geometric language model"""
    
    def __init__(self, vocab_size: int, dim: int = 1024, n_layers: int = 6, 
                 n_heads: int = 16, max_seq_len: int = 128, dropout: float = 0.1):
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
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], List]:
        x = self.token_emb(input_ids)
        x = x + self.pos_emb[:, :x.size(1), :]
        x = self.dropout(x)

        # Create causal mask for autoregressive generation
        seq_len = x.size(1)
        causal_mask = create_causal_mask(seq_len, x.device)

        all_curvatures = []
        for layer in self.layers:
            x, curvatures = layer(x, mask=causal_mask)
            all_curvatures.append(curvatures)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return logits, loss, all_curvatures


class LargeStandardLM(nn.Module):
    """Large standard language model"""
    
    def __init__(self, vocab_size: int, dim: int = 1024, n_layers: int = 6, 
                 n_heads: int = 16, max_seq_len: int = 128, dropout: float = 0.1):
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
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        x = self.token_emb(input_ids)
        x = x + self.pos_emb[:, :x.size(1), :]
        x = self.dropout(x)

        # Create causal mask for TransformerEncoder
        seq_len = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return logits, loss, None
