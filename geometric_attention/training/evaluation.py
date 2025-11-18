"""
Evaluation utilities for various tasks.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Tuple


def evaluate(model: nn.Module, loader, criterion, device: torch.device) -> Tuple[float, float]:
    """Evaluate classification model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch.get('label', batch.get('labels')).to(device)

            outputs = model(input_ids)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
                
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


def evaluate_lm(model: nn.Module, loader, device: torch.device) -> Tuple[float, float]:
    """Evaluate language model"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, labels=labels)
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                _, loss, _ = outputs
            else:
                loss = outputs[1] if isinstance(outputs, tuple) else outputs

            # Count non-padding tokens
            n_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return avg_loss, perplexity


def evaluate_ner(model: nn.Module, loader, criterion, device: torch.device) -> Tuple[float, float]:
    """Evaluate NER model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            # Reshape for loss: [batch*seq, n_labels] and [batch*seq]
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            total_loss += loss.item()

            # Calculate accuracy (ignoring -100 labels)
            preds = logits.argmax(dim=-1)
            mask = labels != -100
            correct += ((preds == labels) & mask).sum().item()
            total += mask.sum().item()

    return total_loss / len(loader), correct / total if total > 0 else 0


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_text(model: nn.Module, tokenizer, prompt: str, 
                  max_length: int = 50, temperature: float = 1.0, 
                  top_k: int = 50, top_p: float = 0.9, 
                  repetition_penalty: float = 1.2,
                  device: torch.device = None) -> str:
    """Generate text autoregressively with improved sampling
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Input text prompt
        max_length: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k filtering (0 to disable)
        top_p: Nucleus sampling threshold (0 to disable)
        repetition_penalty: Penalty for repeating tokens (>1.0 discourages repetition)
        device: Device to use
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    model.to(device)

    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated = input_ids.clone()
    
    # Track generated tokens for repetition penalty
    past_tokens = set()

    with torch.no_grad():
        for step in range(max_length):
            # Get logits for next token
            outputs = model(generated)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
                
            next_token_logits = logits[0, -1, :].clone()
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in past_tokens:
                    next_token_logits[token_id] /= repetition_penalty
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply nucleus (top-p) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample from distribution
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Check for EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # Add to past tokens for repetition penalty
            past_tokens.add(next_token.item())
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

            # Stop if we hit max sequence length
            if generated.size(1) >= 128:
                break

    # Decode
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return text


def analyze_curvatures(model: nn.Module, loader, device: torch.device = None) -> dict:
    """Analyze learned curvatures from a geometric model"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        # Get a sample batch
        sample = next(iter(loader))
        input_ids = sample['input_ids'][:4].to(device)
        
        # Get model outputs with curvatures
        outputs = model(input_ids)
        if isinstance(outputs, tuple) and len(outputs) >= 3:
            _, all_curvatures = outputs[0], outputs[2]
        else:
            return {}
    
    # Convert to numpy array
    import numpy as np
    curv_matrix = np.array(all_curvatures)
    flat_curvatures = curv_matrix.flatten()
    
    # Calculate geometry distribution
    n_hyperbolic = np.sum(flat_curvatures < -0.1)
    n_euclidean = np.sum(np.abs(flat_curvatures) <= 0.1)
    n_spherical = np.sum(flat_curvatures > 0.1)
    
    total = len(flat_curvatures)
    
    return {
        'curvature_matrix': curv_matrix,
        'flat_curvatures': flat_curvatures,
        'n_hyperbolic': n_hyperbolic,
        'n_euclidean': n_euclidean,
        'n_spherical': n_spherical,
        'pct_hyperbolic': n_hyperbolic / total * 100,
        'pct_euclidean': n_euclidean / total * 100,
        'pct_spherical': n_spherical / total * 100
    }
