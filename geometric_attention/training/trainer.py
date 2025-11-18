"""
Training loops and utilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import time
import os
from datetime import datetime
from typing import Dict, Optional, Tuple, Any


def train_epoch(model: nn.Module, loader, optimizer, criterion, 
                device: torch.device, desc: str = "Training") -> Tuple[float, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=desc)
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        labels = batch.get('label', batch.get('labels')).to(device)

        optimizer.zero_grad()
        
        # Handle different model outputs
        outputs = model(input_ids)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
            
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.3f}'})

    return total_loss / len(loader), correct / total


def train_epoch_lm(model: nn.Module, loader, optimizer, 
                   scheduler: Optional[LambdaLR], device: torch.device, 
                   desc: str = "Training", clip_grad: float = 1.0) -> Tuple[float, float]:
    """Train language model for one epoch"""
    model.train()
    epoch_loss = 0
    epoch_tokens = 0

    pbar = tqdm(loader, desc=desc)
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        
        # Language model forward pass
        outputs = model(input_ids, labels=labels)
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            _, loss, _ = outputs
        else:
            loss = outputs[1] if isinstance(outputs, tuple) else outputs

        loss.backward()

        # Gradient clipping
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        # Track loss
        n_tokens = (labels != -100).sum().item()
        epoch_loss += loss.item() * n_tokens
        epoch_tokens += n_tokens

        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ppl': f'{torch.exp(loss):.2f}',
            'lr': f'{current_lr:.2e}'
        })

    train_loss = epoch_loss / epoch_tokens if epoch_tokens > 0 else 0
    train_ppl = torch.exp(torch.tensor(train_loss)).item()

    return train_loss, train_ppl


def train_epoch_ner(model: nn.Module, loader, optimizer, criterion, 
                    scheduler: Optional[LambdaLR], device: torch.device,
                    desc: str = "Training", clip_grad: float = 1.0) -> Tuple[float, float]:
    """Train NER model for one epoch"""
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=desc)
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        
        outputs = model(input_ids)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
            
        # Reshape for loss: [batch*seq, n_labels] and [batch*seq]
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()

        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        # Track loss and accuracy
        n_tokens = (labels != -100).sum().item()
        epoch_loss += loss.item() * n_tokens
        
        # Calculate accuracy
        preds = logits.argmax(dim=-1)
        mask = labels != -100
        correct += ((preds == labels) & mask).sum().item()
        total += mask.sum().item()

        current_lr = optimizer.param_groups[0]['lr'] if optimizer else 0
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}' if total > 0 else '0.000',
            'lr': f'{current_lr:.2e}'
        })

    train_loss = epoch_loss / total if total > 0 else 0
    train_acc = correct / total if total > 0 else 0

    return train_loss, train_acc


def create_optimizer(model: nn.Module, learning_rate: float = 5e-5, 
                      weight_decay: float = 0.01) -> optim.Optimizer:
    """Create AdamW optimizer"""
    return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def create_lr_scheduler(optimizer: optim.Optimizer, num_warmup_steps: int, 
                        num_training_steps: int) -> LambdaLR:
    """Create linear warmup and decay scheduler"""
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, 
            float(num_training_steps - current_step) / 
            float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    """General trainer class with checkpoint saving"""
    
    def __init__(self, model: nn.Module, device: torch.device = None, 
                 checkpoint_dir: str = 'checkpoints', model_name: str = 'model',
                 use_compile: bool = False, model_config: Dict = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.best_metric = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.use_compile = use_compile
        self.model_config = model_config  # Store for checkpoint saving
        
        # Apply torch.compile() if requested and available
        if use_compile:
            if hasattr(torch, 'compile'):
                # Enable TF32 for Ampere GPUs (3090, 4090, A100, etc.)
                torch.set_float32_matmul_precision('high')
                print("✓ Enabled TensorFloat32 precision for ~1.3x speedup")
                
                print("Compiling model with torch.compile()...")
                print("  First forward pass will be slow (JIT compilation)")
                print("  Subsequent passes will be ~1.9x faster")
                print("  Expected combined speedup: ~2.5x")
                
                # Compile with settings to avoid recompilation
                self.model = torch.compile(
                    self.model, 
                    mode='reduce-overhead',
                    fullgraph=False  # Allow graph breaks to avoid recompilation
                )
                print("✓ Model compiled successfully!")
            else:
                print("⚠️ torch.compile() not available (requires PyTorch 2.0+)")
                print("   Continuing without compilation...")
        
        # Create checkpoint directory if it doesn't exist
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, epoch: int, optimizer: optim.Optimizer, 
                       metric: float, history: Dict, 
                       is_best: bool = False, prefix: str = "checkpoint",
                       model_config: Dict = None) -> str:
        """Save model checkpoint with timestamp, metric, and architecture"""
        if not self.checkpoint_dir:
            return None
            
        # Format metric for filename
        metric_str = f"{metric:.4f}".replace('.', '')
        
        # Create filename with timestamp and metric
        if is_best:
            filename = f"best_{self.model_name}_{self.timestamp}_metric{metric_str}.pt"
        else:
            filename = f"{prefix}_{self.model_name}_epoch{epoch}_{self.timestamp}.pt"
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        # Use stored model_config from __init__, or fallback to extraction
        if model_config is None:
            model_config = self.model_config
        
        # Last resort: try to extract from model (won't work with compiled models)
        if model_config is None and hasattr(self.model, 'dim'):
            model_config = {
                'dim': self.model.dim,
                'n_layers': len(self.model.layers) if hasattr(self.model, 'layers') else 0,
                'n_heads': getattr(self.model, 'n_heads', 0)
            }
        
        # Save checkpoint with architecture info
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metric': metric,
            'history': history,
            'timestamp': self.timestamp,
            'model_name': self.model_name,
            'model_config': model_config  # Save architecture
        }
        
        torch.save(checkpoint, filepath)
        return filepath
    
    def load_checkpoint(self, filepath: str, optimizer: Optional[optim.Optimizer] = None) -> Dict:
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint
        
    def train(self, train_loader, val_loader, n_epochs: int = 3, 
              learning_rate: float = 5e-5, weight_decay: float = 0.01,
              warmup_steps: int = 0, task_type: str = 'classification',
              save_best: bool = True, save_final: bool = True) -> Dict[str, Any]:
        """Train the model with checkpoint saving
        
        Args:
            save_best: Save checkpoint when validation metric improves
            save_final: Save checkpoint at the end of training
        """
        
        # Setup optimizer
        optimizer = create_optimizer(self.model, learning_rate, weight_decay)
        
        # Setup scheduler if warmup is needed
        scheduler = None
        if warmup_steps > 0:
            total_steps = n_epochs * len(train_loader)
            scheduler = create_lr_scheduler(optimizer, warmup_steps, total_steps)
        
        # Setup loss function
        if task_type == 'classification' or task_type == 'ner':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = None  # Language modeling uses built-in loss
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        start_time = time.time()
        
        for epoch in range(n_epochs):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{n_epochs}")
            print(f"{'='*70}")
            
            # Train
            if task_type == 'classification':
                train_loss, train_acc = train_epoch(
                    self.model, train_loader, optimizer, criterion, 
                    self.device, desc=f"Epoch {epoch+1}"
                )
            elif task_type == 'language_modeling':
                train_loss, train_acc = train_epoch_lm(
                    self.model, train_loader, optimizer, scheduler, 
                    self.device, desc=f"Epoch {epoch+1}"
                )
            elif task_type == 'ner':
                train_loss, train_acc = train_epoch_ner(
                    self.model, train_loader, optimizer, criterion, 
                    scheduler, self.device, desc=f"Epoch {epoch+1}"
                )
            
            # Validate
            from .evaluation import evaluate, evaluate_lm, evaluate_ner
            
            if task_type == 'classification':
                val_loss, val_acc = evaluate(self.model, val_loader, criterion, self.device)
            elif task_type == 'language_modeling':
                val_loss, val_acc = evaluate_lm(self.model, val_loader, self.device)
            elif task_type == 'ner':
                val_loss, val_acc = evaluate_ner(self.model, val_loader, criterion, self.device)
            
            # Store results
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"  Train Loss: {train_loss:.4f}, Train Acc/PPL: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}, Val Acc/PPL:   {val_acc:.4f}")
            
            # Analyze geometry for geometric models
            if 'geometric' in self.model_name.lower():
                from .evaluation import analyze_curvatures
                try:
                    curv_results = analyze_curvatures(self.model, val_loader, self.device)
                    if curv_results:
                        print(f"  Geometry:   {curv_results['pct_hyperbolic']:.1f}% H, "
                              f"{curv_results['pct_euclidean']:.1f}% E, "
                              f"{curv_results['pct_spherical']:.1f}% S")
                except Exception:
                    pass  # Silently skip if geometry analysis fails
            
            # Save best checkpoint
            if save_best:
                # Determine if this is the best model
                is_best = False
                current_metric = val_acc
                
                # For language modeling, lower perplexity is better
                if task_type == 'language_modeling':
                    if self.best_metric is None or current_metric < self.best_metric:
                        is_best = True
                        self.best_metric = current_metric
                # For classification/NER, higher accuracy is better  
                else:
                    if self.best_metric is None or current_metric > self.best_metric:
                        is_best = True
                        self.best_metric = current_metric
                
                if is_best:
                    checkpoint_path = self.save_checkpoint(
                        epoch + 1, optimizer, current_metric, history, is_best=True
                    )
                    if checkpoint_path:
                        print(f"  ✓ Saved best checkpoint: {os.path.basename(checkpoint_path)}")
        
        training_time = time.time() - start_time
        print(f"\nTraining complete in {training_time:.1f}s")
        
        # Save final checkpoint
        if save_final and self.checkpoint_dir:
            final_metric = history['val_acc'][-1] if history['val_acc'] else 0.0
            checkpoint_path = self.save_checkpoint(
                n_epochs, optimizer, final_metric, history, 
                is_best=False, prefix="final"
            )
            if checkpoint_path:
                print(f"✓ Saved final checkpoint: {os.path.basename(checkpoint_path)}")
        
        return history
