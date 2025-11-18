#!/usr/bin/env python3
"""
Example training script for Geometric Attention models.
"""

import torch
import argparse
from transformers import AutoTokenizer
from geometric_attention.models import GeometricTransformer, StandardTransformer
from geometric_attention.data import SST2Dataset, load_glue_dataset, create_data_loader
from geometric_attention.training import Trainer
from geometric_attention.utils import set_seed, get_device, print_model_summary
from geometric_attention.utils.visualization import plot_training_curves, plot_curvature_heatmap


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Geometric Attention models on SST-2')
    parser.add_argument('--dim', type=int, default=768, 
                       help='Model dimension (default: 768)')
    parser.add_argument('--epochs', type=int, default=10, 
                       help='Number of training epochs (default: 10)')
    args = parser.parse_args()
    
    # Set configuration - matching original experiment settings
    config = {
        'model_dim': args.dim,      # From CLI argument
        'n_layers': 6,         # Increased from 2
        'n_heads': 12,         # Increased from 4
        'n_epochs': args.epochs,    # From CLI argument
        'batch_size': 16,      # Reduced from 32 for better gradients
        'learning_rate': 3e-5,
        'seed': 42,
        'use_full_dataset': False,  # Set to True to use full dataset
        'train_samples': 10000,  # If not using full dataset
        'val_samples': 1000     # If not using full dataset
    }
    
    # Set seed for reproducibility
    set_seed(config['seed'])
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA is available. Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. Will use CPU.")
    
    # Get device - explicitly use GPU 1 (your 3090)
    device = get_device(cuda_id=1)  # Use GPU 1 (your RTX 3090)
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(device.index)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024**3:.2f} GB")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Load data
    print("Loading SST-2 dataset...")
    sst2 = load_glue_dataset("sst2")
    
    # Create datasets - use more data for meaningful results
    if config['use_full_dataset']:
        print("Using full dataset")
        train_dataset = SST2Dataset(sst2['train'], tokenizer, max_length=128)
        val_dataset = SST2Dataset(sst2['validation'], tokenizer, max_length=128)
    else:
        print(f"Using subset: {config['train_samples']} train, {config['val_samples']} validation")
        train_dataset = SST2Dataset(sst2['train'][:config['train_samples']], tokenizer, max_length=128)
        val_dataset = SST2Dataset(sst2['validation'][:config['val_samples']], tokenizer, max_length=128)
    
    # Create data loaders
    train_loader = create_data_loader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = create_data_loader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize models
    print("\n" + "="*70)
    print("Initializing models...")
    print("="*70)
    
    # Geometric model
    geometric_model = GeometricTransformer(
        vocab_size=tokenizer.vocab_size,
        dim=config['model_dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        n_classes=2,
        dropout=0.1
    )
    print_model_summary(geometric_model, "Geometric Transformer")
    
    # Standard model for comparison
    standard_model = StandardTransformer(
        vocab_size=tokenizer.vocab_size,
        dim=config['model_dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        n_classes=2,
        dropout=0.1
    )
    print_model_summary(standard_model, "Standard Transformer")
    
    # Train geometric model
    print("\n" + "="*70)
    print("Training Geometric Transformer...")
    print("="*70)
    
    geo_trainer = Trainer(geometric_model, device=device)
    geo_history = geo_trainer.train(
        train_loader,
        val_loader,
        n_epochs=config['n_epochs'],
        learning_rate=config['learning_rate'],
        task_type='classification'
    )
    
    # Train standard model
    print("\n" + "="*70)
    print("Training Standard Transformer...")
    print("="*70)
    
    std_trainer = Trainer(standard_model, device=device)
    std_history = std_trainer.train(
        train_loader,
        val_loader,
        n_epochs=config['n_epochs'],
        learning_rate=config['learning_rate'],
        task_type='classification'
    )
    
    # Compare results
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    
    geo_best_acc = max(geo_history['val_acc'])
    std_best_acc = max(std_history['val_acc'])
    
    print(f"Geometric Model - Best Val Acc: {geo_best_acc:.4f}")
    print(f"Standard Model  - Best Val Acc: {std_best_acc:.4f}")
    print(f"Difference: {(geo_best_acc - std_best_acc):.4f}")
    
    # Analyze learned geometries
    from geometric_attention.training.evaluation import analyze_curvatures
    print("\n" + "="*70)
    print("ANALYZING LEARNED GEOMETRIES")
    print("="*70)
    
    curvature_results = analyze_curvatures(geometric_model, val_loader, device)
    if curvature_results:
        print(f"Hyperbolic heads: {curvature_results['n_hyperbolic']} ({curvature_results['pct_hyperbolic']:.1f}%)")
        print(f"Euclidean heads:  {curvature_results['n_euclidean']} ({curvature_results['pct_euclidean']:.1f}%)")
        print(f"Spherical heads:  {curvature_results['n_spherical']} ({curvature_results['pct_spherical']:.1f}%)")
    
    print("\n" + "="*70)
    print("Training complete! Check the plots directory for visualizations.")
    print("="*70)


if __name__ == "__main__":
    main()
