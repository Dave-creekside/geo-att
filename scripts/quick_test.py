#!/usr/bin/env python3
"""
Quick test script to verify everything works with improved settings.
Uses smaller subset but proper configuration.
"""

import torch
import argparse
from transformers import AutoTokenizer
from geometric_attention.models import GeometricTransformer, StandardTransformer
from geometric_attention.data import SST2Dataset, load_glue_dataset, create_data_loader
from geometric_attention.training import Trainer
from geometric_attention.utils import set_seed, get_device, print_model_summary
from geometric_attention.training.evaluation import analyze_curvatures


def main():
    print("\n" + "="*80)
    print("QUICK TEST - Verifying Geometric Attention Setup")
    print("="*80)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Quick test of Geometric Attention setup')
    parser.add_argument('--dim', type=int, default=512, 
                       help='Model dimension (default: 512)')
    parser.add_argument('--epochs', type=int, default=2, 
                       help='Number of training epochs (default: 2)')
    args = parser.parse_args()
    
    # Smaller config for quick test
    config = {
        'model_dim': args.dim,      # From CLI argument
        'n_layers': 4,         # Enough to see patterns
        'n_heads': 8,          # 32 total heads
        'n_epochs': args.epochs,    # From CLI argument
        'batch_size': 32,
        'learning_rate': 3e-5,
        'train_samples': 5000,  # Enough to learn
        'val_samples': 500,
        'seed': 42,
        'gpu_id': 1  # RTX 3090
    }
    
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*80)
    
    # Set seed
    set_seed(config['seed'])
    
    # Setup device
    device = get_device(cuda_id=config['gpu_id'])
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device.index)}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Load data
    print("Loading SST-2 dataset...")
    sst2 = load_glue_dataset("sst2")
    
    # Create datasets
    train_dataset = SST2Dataset(sst2['train'][:config['train_samples']], tokenizer, max_length=128)
    val_dataset = SST2Dataset(sst2['validation'][:config['val_samples']], tokenizer, max_length=128)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = create_data_loader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = create_data_loader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Initialize models
    print("\n" + "="*80)
    print("Testing Geometric Transformer...")
    print("="*80)
    
    geometric_model = GeometricTransformer(
        vocab_size=tokenizer.vocab_size,
        dim=config['model_dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        n_classes=2,
        dropout=0.1
    )
    print_model_summary(geometric_model, "Geometric Transformer")
    
    # Quick training
    geo_trainer = Trainer(geometric_model, device=device)
    geo_history = geo_trainer.train(
        train_loader,
        val_loader,
        n_epochs=config['n_epochs'],
        learning_rate=config['learning_rate'],
        task_type='classification'
    )
    
    # Test standard model
    print("\n" + "="*80)
    print("Testing Standard Transformer...")
    print("="*80)
    
    standard_model = StandardTransformer(
        vocab_size=tokenizer.vocab_size,
        dim=config['model_dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        n_classes=2,
        dropout=0.1
    )
    print_model_summary(standard_model, "Standard Transformer")
    
    # Quick training
    std_trainer = Trainer(standard_model, device=device)
    std_history = std_trainer.train(
        train_loader,
        val_loader,
        n_epochs=config['n_epochs'],
        learning_rate=config['learning_rate'],
        task_type='classification'
    )
    
    # Results
    print("\n" + "="*80)
    print("QUICK TEST RESULTS")
    print("="*80)
    
    geo_acc = max(geo_history['val_acc'])
    std_acc = max(std_history['val_acc'])
    
    print(f"Geometric Model - Best Val Acc: {geo_acc:.4f}")
    print(f"Standard Model  - Best Val Acc: {std_acc:.4f}")
    print(f"Difference: {(geo_acc - std_acc):.4f}")
    
    # Analyze geometries
    print("\n" + "="*80)
    print("GEOMETRY DISTRIBUTION")
    print("="*80)
    
    curvature_results = analyze_curvatures(geometric_model, val_loader, device)
    if curvature_results:
        total_heads = config['n_layers'] * config['n_heads']
        print(f"Total heads: {total_heads}")
        print(f"Hyperbolic: {curvature_results['n_hyperbolic']} ({curvature_results['pct_hyperbolic']:.1f}%)")
        print(f"Euclidean:  {curvature_results['n_euclidean']} ({curvature_results['pct_euclidean']:.1f}%)")
        print(f"Spherical:  {curvature_results['n_spherical']} ({curvature_results['pct_spherical']:.1f}%)")
        
        # Check for expected 50/50 split
        hyp_ratio = curvature_results['n_hyperbolic'] / max(curvature_results['n_spherical'], 1)
        print(f"\nHyperbolic/Spherical Ratio: {hyp_ratio:.2f} (expected ~1.0)")
    
    print("\n" + "="*80)
    
    if geo_acc > 0.6 and std_acc > 0.6:
        print("✓ TEST PASSED - Models are learning properly!")
        print("  Both models achieved >60% accuracy on 5k samples")
        print("  Ready for full experiments!")
    else:
        print("⚠ TEST WARNING - Low accuracy detected")
        print("  Consider checking your environment or increasing training")
    
    print("="*80)
    
    print("\nTo run full experiments:")
    print("  1. Full SST-2:    python train_sst2_full.py")
    print("  2. Language Model: python train_language_model.py")
    print("  3. Updated example: python train_example.py")


if __name__ == "__main__":
    main()
