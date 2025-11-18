#!/usr/bin/env python3
"""
Test script for comprehensive logging system.
Trains a small model and validates all logging functionality.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import AutoTokenizer

from geometric_attention.models.language_models import (
    TinyGeometricLM, SmallGeometricLM, GeometricCausalLM, LargeGeometricLM
)
from geometric_attention.data import load_wikitext_dataset, LanguageModelingDataset, create_data_loader
from geometric_attention.logging import ExperimentLogger
from geometric_attention.utils import set_seed, get_device


def create_lr_scheduler(optimizer, num_warmup_steps, num_training_steps):
    """Create linear warmup and decay scheduler."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / 
            float(max(1, num_training_steps - num_warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda)


def train_with_logging(model, train_loader, val_loader, logger, 
                      n_epochs=50, device=None, start_epoch=0, 
                      saved_optimizer_state=None, saved_scheduler_state=None):
    """
    Train model with comprehensive logging.
    
    Args:
        model: Language model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        logger: ExperimentLogger instance
        n_epochs: Number of epochs (target total epochs)
        device: Device to train on
        start_epoch: Starting epoch (for resumption)
        saved_optimizer_state: Saved optimizer state dict (for resumption)
        saved_scheduler_state: Saved scheduler state dict (for resumption)
    """
    if device is None:
        device = get_device(cuda_id=1)
    
    model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    
    # Load optimizer state if resuming
    if saved_optimizer_state:
        optimizer.load_state_dict(saved_optimizer_state)
        print(f"✓ Optimizer state loaded")
    
    total_steps = n_epochs * len(train_loader)
    warmup_steps = min(500, total_steps // 10)
    scheduler = create_lr_scheduler(optimizer, warmup_steps, total_steps)
    
    # Load scheduler state if resuming
    if saved_scheduler_state:
        scheduler.load_state_dict(saved_scheduler_state)
        print(f"✓ Scheduler state loaded")
    
    if saved_optimizer_state or saved_scheduler_state:
        print()  # Blank line after state loading messages
    
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n{'='*70}")
    print(f"Training Configuration:")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Total Steps: {total_steps}")
    print(f"  Warmup Steps: {warmup_steps}")
    print(f"  Optimizer: AdamW (lr=5e-4)")
    print(f"  Logging Frequency: Every {logger.geometry_log_freq} steps")
    print(f"{'='*70}\n")
    
    best_val_ppl = float('inf')
    
    for epoch in range(start_epoch, n_epochs):
        # Training
        model.train()
        epoch_loss = 0
        epoch_tokens = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass - captures curvatures!
            logits, loss, curvatures = model(input_ids, labels=labels)
            
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Track metrics
            n_tokens = (labels != -100).sum().item()
            epoch_loss += loss.item() * n_tokens
            epoch_tokens += n_tokens
            
            # HIGH-FREQUENCY LOGGING: Every step
            logger.log_training_step(
                epoch=epoch,
                step=step,
                loss=loss.item(),
                perplexity=torch.exp(loss).item(),
                learning_rate=optimizer.param_groups[0]['lr'],
                grad_norm=grad_norm.item()
            )
            
            # MEDIUM-FREQUENCY LOGGING: Geometry (every N steps)
            logger.log_geometry(
                epoch=epoch,
                step=step,
                curvatures=curvatures
            )
            
            # Update progress bar
            current_ppl = torch.exp(loss).item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ppl': f'{current_ppl:.1f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # Validation
        model.eval()
        val_loss = 0
        val_tokens = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                logits, loss, curvatures = model(input_ids, labels=labels)
                
                n_tokens = (labels != -100).sum().item()
                val_loss += loss.item() * n_tokens
                val_tokens += n_tokens
        
        # Calculate epoch metrics
        train_loss_avg = epoch_loss / epoch_tokens
        train_ppl = torch.exp(torch.tensor(train_loss_avg)).item()
        
        val_loss_avg = val_loss / val_tokens
        val_ppl = torch.exp(torch.tensor(val_loss_avg)).item()
        
        # Log epoch summary
        logger.log_epoch_summary(
            epoch=epoch,
            train_loss=train_loss_avg,
            train_ppl=train_ppl,
            val_loss=val_loss_avg,
            val_ppl=val_ppl
        )
        
        # Print epoch results
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train PPL: {train_ppl:.2f}")
        print(f"  Val PPL:   {val_ppl:.2f}")
        
        # Print current geometry distribution
        current_geom = logger.geometry_tracker.get_current_distribution()
        if current_geom:
            print(f"  Geometry:  {current_geom['hyperbolic']:.1f}% H, "
                  f"{current_geom['euclidean']:.1f}% E, "
                  f"{current_geom['spherical']:.1f}% S")
        
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = logger.save_checkpoint(
                    epoch=epoch + 1,
                    model_state=model.state_dict(),
                    metrics={'val_ppl': val_ppl, 'train_ppl': train_ppl},
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict()
                )
                print(f"  ✓ Checkpoint saved: {checkpoint_path}")
        
        # Track best model
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            # Save best checkpoint
            checkpoint_path = logger.save_checkpoint(
                epoch=epoch + 1,
                model_state=model.state_dict(),
                metrics={'val_ppl': val_ppl, 'train_ppl': train_ppl, 'is_best': True},
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict()
            )
    
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"  Best Validation PPL: {best_val_ppl:.2f}")
    print(f"{'='*70}\n")


def main():
    """Run logging system test."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train geometric model with comprehensive logging')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs to train (default: 50)')
    parser.add_argument('--dim', type=int, default=128,
                       help='Model dimension (default: 128)')
    parser.add_argument('--n-layers', type=int, default=None,
                       help='Number of layers (default: auto based on dim)')
    parser.add_argument('--n-heads', type=int, default=None,
                       help='Number of attention heads (default: auto based on dim)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--compile', action='store_true',
                       help='Use torch.compile() for speedup')
    parser.add_argument('--full-dataset', action='store_true',
                       help='Use full WikiText-2 dataset (default: subset of 5000 samples)')
    parser.add_argument('--resume-from-checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--start-epoch', type=int, default=0,
                       help='Starting epoch number (set automatically if resuming)')
    parser.add_argument('--dataset-name', type=str, default=None,
                       help='HuggingFace dataset name (e.g., "Creekside/logic", "wikitext")')
    parser.add_argument('--dataset-config', type=str, default=None,
                       help='Dataset config/subset (optional)')
    
    args = parser.parse_args()
    
    # Auto-select layers and heads based on dimension
    if args.n_layers is None:
        if args.dim <= 128:
            args.n_layers = 1
        elif args.dim <= 256:
            args.n_layers = 2
        elif args.dim <= 512:
            args.n_layers = 4
        else:
            args.n_layers = 6
    
    if args.n_heads is None:
        if args.dim <= 128:
            args.n_heads = 2
        elif args.dim <= 256:
            args.n_heads = 4
        elif args.dim <= 512:
            args.n_heads = 8
        else:
            args.n_heads = 12
    
    # Determine dataset name for display
    if args.dataset_name:
        dataset_for_display = args.dataset_name
    elif args.full_dataset:
        dataset_for_display = 'Full WikiText-2'
    else:
        dataset_for_display = 'WikiText-2 Subset (5000 samples)'
    
    print("\n" + "="*70)
    if args.resume_from_checkpoint:
        print("RESUMING TRAINING FROM CHECKPOINT")
    else:
        print("COMPREHENSIVE LOGGING SYSTEM - GEOMETRIC MODEL TRAINING")
    print("="*70)
    
    if args.resume_from_checkpoint:
        # Show clear resumption info
        print(f"\n  Resuming from: Epoch {args.start_epoch if args.start_epoch else '?'}")
        print(f"  Training to:   Epoch {args.epochs} (+{args.epochs - (args.start_epoch if args.start_epoch else 0)} additional)")
        print(f"  Model:         {args.dim}d, {args.n_layers}L, {args.n_heads}H")
        print(f"  Dataset:       {dataset_for_display}")
        print(f"  Batch Size:    {args.batch_size}")
        print(f"  torch.compile: {'Yes' if args.compile else 'No'}")
    else:
        # Show normal training info
        print(f"\n  Model:         {args.dim}d, {args.n_layers}L, {args.n_heads}H")
        print(f"  Epochs:        {args.epochs}")
        print(f"  Dataset:       {dataset_for_display}")
        print(f"  Batch Size:    {args.batch_size}")
        print(f"  torch.compile: {'Yes' if args.compile else 'No'}")
    
    print("="*70 + "\n")
    
    # Set seed
    set_seed(42)
    
    # Device
    device = get_device(cuda_id=1)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device.index)}\n")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Determine dataset to use
    if args.dataset_name:
        # Load custom HuggingFace dataset
        from datasets import load_dataset
        
        print(f"Loading dataset '{args.dataset_name}'...")
        if args.dataset_config:
            print(f"  Config: {args.dataset_config}")
            raw_data = load_dataset(args.dataset_name, args.dataset_config)
        else:
            raw_data = load_dataset(args.dataset_name)
        
        # Auto-detect or create splits
        train_split = 'train' if 'train' in raw_data else list(raw_data.keys())[0]
        val_split = None
        for split_name in ['validation', 'valid', 'val', 'dev', 'test']:
            if split_name in raw_data:
                val_split = split_name
                break
        
        if not val_split:
            print("Creating 90/10 train/val split...")
            split_data = raw_data[train_split].train_test_split(test_size=0.1, seed=42)
            train_data_raw = split_data['train']
            val_data_raw = split_data['test']
        else:
            train_data_raw = raw_data[train_split]
            val_data_raw = raw_data[val_split]
        
        # Interactive field selection (same as main.py)
        sample = train_data_raw[0]
        fields = [k for k in sample.keys() if isinstance(sample[k], str)]
        
        print("\n" + "="*70)
        print("📋 DATASET FIELDS PREVIEW")
        print("="*70)
        
        for i, field in enumerate(fields, 1):
            value = str(sample[field])
            display_value = value if len(value) <= 100 else value[:97] + "..."
            print(f"  [{i}] {field} ({len(value)} chars):")
            print(f"      \"{display_value}\"")
            print()
        
        print("Select field(s) to use for training:")
        print("  - Single field: Enter number (e.g., '3')")
        print("  - Multiple fields: Enter numbers separated by spaces (e.g., '1 3')")
        print()
        
        field_choice = input("Your choice: ").strip()
        
        # Parse selection
        selected_fields = []
        try:
            indices = [int(x) - 1 for x in field_choice.split()]
            selected_fields = [fields[i] for i in indices if 0 <= i < len(fields)]
            
            if not selected_fields:
                print("⚠️ No valid fields selected, using auto-detect")
            else:
                print(f"✓ Selected fields: {', '.join(selected_fields)}")
        except (ValueError, IndexError):
            print("⚠️ Invalid selection, using auto-detect")
            selected_fields = []
        
        # Fallback to auto-detect if needed
        if not selected_fields:
            for field in ['text', 'content', 'sentence', 'document', 'passage', 'answer']:
                if field in sample:
                    selected_fields = [field]
                    print(f"✓ Auto-detected text field: '{field}'")
                    break
        
        # Apply field mapping
        if len(selected_fields) == 1:
            train_data_raw = train_data_raw.map(lambda x: {'text': x[selected_fields[0]]})
            val_data_raw = val_data_raw.map(lambda x: {'text': x[selected_fields[0]]})
        else:
            print(f"\nCombining {len(selected_fields)} fields with '\\n\\n' separator")
            def combine_fields(example):
                return {'text': "\n\n".join([example[f] for f in selected_fields if f in example])}
            train_data_raw = train_data_raw.map(combine_fields)
            val_data_raw = val_data_raw.map(combine_fields)
        
        # Show final sample
        final_sample = train_data_raw[0]['text']
        print("\n" + "="*70)
        print("📋 FINAL TRAINING SAMPLE PREVIEW")
        print("="*70)
        print(f"  Fields: {', '.join(selected_fields)}")
        print(f"  Length: {len(final_sample)} characters")
        print(f"\n  Sample:")
        display_sample = final_sample if len(final_sample) <= 200 else final_sample[:197] + "..."
        print(f"  \"{display_sample}\"")
        print("="*70 + "\n")
        
        train_dataset = LanguageModelingDataset(train_data_raw, tokenizer, max_length=128)
        val_dataset = LanguageModelingDataset(val_data_raw, tokenizer, max_length=128)
        
        dataset_display_name = args.dataset_name
    else:
        # Default: Load WikiText-2
        print("Loading WikiText-2 dataset...")
        raw_data = load_wikitext_dataset("wikitext-2-raw-v1")
        
        # Create datasets
        if args.full_dataset:
            train_dataset = LanguageModelingDataset(raw_data['train'], tokenizer, max_length=128)
        else:
            train_dataset = LanguageModelingDataset(
                raw_data['train'].select(range(min(5000, len(raw_data['train'])))),
                tokenizer,
                max_length=128
            )
        
        val_dataset = LanguageModelingDataset(raw_data['validation'], tokenizer, max_length=128)
        dataset_display_name = 'wikitext-2-raw-v1'
    
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}\n")
    
    # Create data loaders
    train_loader = create_data_loader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = create_data_loader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Select appropriate model class based on size
    if args.dim <= 128:
        model_class = TinyGeometricLM
        model_name = 'TinyGeometricLM'
    elif args.dim <= 256:
        model_class = SmallGeometricLM
        model_name = 'SmallGeometricLM'
    elif args.dim <= 512:
        model_class = GeometricCausalLM
        model_name = 'GeometricCausalLM'
    else:
        model_class = LargeGeometricLM
        model_name = 'LargeGeometricLM'
    
    # Initialize model
    print(f"Initializing {model_name} ({args.dim}d, {args.n_layers} layers, {args.n_heads} heads)...")
    model = model_class(
        vocab_size=tokenizer.vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=128,
        dropout=0.1
    )
    
    # Checkpoint resumption
    saved_optimizer_state = None
    saved_scheduler_state = None
    start_epoch = 0
    experiment_dir = None
    
    if args.resume_from_checkpoint:
        from pathlib import Path
        print(f"\n{'='*70}")
        print("RESUMING FROM CHECKPOINT")
        print(f"{'='*70}")
        
        checkpoint = torch.load(args.resume_from_checkpoint, map_location='cpu')
        
        # Handle compiled model keys (_orig_mod. prefix)
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        
        saved_optimizer_state = checkpoint.get('optimizer_state_dict')
        saved_scheduler_state = checkpoint.get('scheduler_state_dict')
        start_epoch = checkpoint.get('epoch', 0)
        experiment_dir = Path(args.resume_from_checkpoint).parent.parent
        
        print(f"✓ Model state loaded from epoch {start_epoch}")
        print(f"✓ Experiment directory: {experiment_dir}")
        if saved_optimizer_state:
            print(f"✓ Optimizer state available for loading")
        if saved_scheduler_state:
            print(f"✓ Scheduler state available for loading")
        print(f"{'='*70}\n")
    
    # Apply torch.compile if requested
    if args.compile:
        if hasattr(torch, 'compile'):
            print("Compiling model with torch.compile()...")
            torch.set_float32_matmul_precision('high')
            model = torch.compile(model, mode='reduce-overhead')
            print("✓ Model compiled!")
        else:
            print("⚠️ torch.compile() not available (requires PyTorch 2.0+)")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")
    
    # Initialize logger
    config = {
        'model': {
            'type': model_name,
            'vocab_size': tokenizer.vocab_size,
            'dim': args.dim,
            'n_layers': args.n_layers,
            'n_heads': args.n_heads,
            'max_seq_len': 128,
            'dropout': 0.1,
            'total_params': total_params,
            'compiled': args.compile
        },
        'training': {
            'dataset': dataset_display_name,
            'full_dataset': args.full_dataset,
            'n_epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': 5e-4,
            'weight_decay': 0.01,
            'warmup_steps': 500,
            'gradient_clip': 1.0
        },
        'hardware': {
            'device': str(device),
            'gpu_name': torch.cuda.get_device_name(device.index) if device.type == 'cuda' else 'CPU'
        }
    }
    
    # Create experiment name based on config
    exp_name = f'geometric_{args.dim}d_{args.n_layers}L_{args.n_heads}H_{args.epochs}ep'
    
    # Determine resume parameters
    resume_dir = None
    start_step = 0
    if args.resume_from_checkpoint:
        resume_dir = str(experiment_dir)
        start_step = start_epoch * len(train_loader)  # Approximate
    
    # Use context manager for automatic cleanup
    with ExperimentLogger(
        experiment_name=exp_name,
        config=config,
        geometry_log_freq=10,  # Log geometry every 10 steps
        resume_from_dir=resume_dir,
        start_epoch=start_epoch,
        start_step=start_step
    ) as logger:
        
        # Train model
        train_with_logging(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            logger=logger,
            n_epochs=args.epochs,
            device=device,
            start_epoch=start_epoch,
            saved_optimizer_state=saved_optimizer_state,
            saved_scheduler_state=saved_scheduler_state
        )
    
    print("\n✅ Logging system test complete!")
    print(f"\nExperiment data saved to: {logger.experiment_dir}")
    print("\nNext steps:")
    print(f"  1. Visualize: python plot_geometry_evolution.py --experiment {logger.run_id}")
    print(f"  2. Check logs in {logger.experiment_dir}/logs/")
    print(f"  3. Analyze geometry.jsonl for phase transitions")


if __name__ == "__main__":
    main()
