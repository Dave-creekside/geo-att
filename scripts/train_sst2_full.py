#!/usr/bin/env python3
"""
Full SST-2 training script for Geometric Attention models.
Replicates original experiments with complete dataset.
"""

import torch
import time
import argparse
from transformers import AutoTokenizer
from geometric_attention.models import GeometricTransformer, StandardTransformer
from geometric_attention.data import SST2Dataset, load_glue_dataset, create_data_loader
from geometric_attention.training import Trainer
from geometric_attention.utils import set_seed, get_device, print_model_summary
from geometric_attention.utils.visualization import plot_training_curves, plot_geometry_distribution
from geometric_attention.training.evaluation import analyze_curvatures
import matplotlib.pyplot as plt


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Geometric Attention on full SST-2 dataset')
    parser.add_argument('--dim', type=int, default=768, 
                       help='Model dimension (default: 768)')
    parser.add_argument('--epochs', type=int, default=5, 
                       help='Number of training epochs (default: 5)')
    parser.add_argument('--compile', action='store_true',
                       help='Use torch.compile() for ~1.9x speedup (PyTorch 2.0+)')
    args = parser.parse_args()
    
    # Configuration matching original experiments
    config = {
        'task': 'SST-2',
        'model_dim': args.dim,      # From CLI argument
        'n_layers': 6,
        'n_heads': 12,
        'n_epochs': args.epochs,    # From CLI argument
        'batch_size': 32,  # Can use larger batch size with full dataset
        'learning_rate': 2e-5,
        'warmup_steps': 500,
        'seed': 42,
        'max_seq_len': 128,
        'gpu_id': 1  # Use GPU 1 (RTX 3090)
    }
    
    print("="*80)
    print(f"SST-2 Full Dataset Experiment - Geometric vs Standard Transformer")
    print("="*80)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*80)
    
    # Set seed for reproducibility
    set_seed(config['seed'])
    
    # Setup device
    if torch.cuda.is_available():
        print(f"\nCUDA is available. Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    device = get_device(cuda_id=config['gpu_id'])
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(device.index)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024**3:.2f} GB")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Load full SST-2 dataset
    print("\nLoading full SST-2 dataset...")
    sst2 = load_glue_dataset("sst2")
    
    # Create datasets with full data
    train_dataset = SST2Dataset(sst2['train'], tokenizer, max_length=config['max_seq_len'])
    val_dataset = SST2Dataset(sst2['validation'], tokenizer, max_length=config['max_seq_len'])
    
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    
    # Create data loaders
    train_loader = create_data_loader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=4  # Use multiple workers for faster data loading
    )
    val_loader = create_data_loader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=4
    )
    
    # ========================================================================
    # TRAIN GEOMETRIC MODEL FIRST (to save VRAM)
    # ========================================================================
    print("\n" + "="*80)
    print("Initializing & Training Geometric Transformer...")
    print("="*80)
    
    geometric_model = GeometricTransformer(
        vocab_size=tokenizer.vocab_size,
        dim=config['model_dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        max_seq_len=config['max_seq_len'],
        n_classes=2,
        dropout=0.1
    )
    print_model_summary(geometric_model, "Geometric Transformer")
    
    start_time = time.time()
    geo_trainer = Trainer(geometric_model, device=device, model_name="geometric_sst2",
                         use_compile=args.compile)
    geo_history = geo_trainer.train(
        train_loader,
        val_loader,
        n_epochs=config['n_epochs'],
        learning_rate=config['learning_rate'],
        warmup_steps=config['warmup_steps'],
        task_type='classification'
    )
    geo_time = time.time() - start_time
    
    # Free memory before loading standard model
    print("\nFreeing GPU memory...")
    del geo_trainer
    geometric_model = geometric_model.cpu()
    torch.cuda.empty_cache()
    print(f"✓ GPU memory cleared")
    
    # ========================================================================
    # TRAIN STANDARD MODEL SECOND
    # ========================================================================
    print("\n" + "="*80)
    print("Initializing & Training Standard Transformer...")
    print("="*80)
    
    standard_model = StandardTransformer(
        vocab_size=tokenizer.vocab_size,
        dim=config['model_dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        max_seq_len=config['max_seq_len'],
        n_classes=2,
        dropout=0.1
    )
    print_model_summary(standard_model, "Standard Transformer")
    
    start_time = time.time()
    std_trainer = Trainer(standard_model, device=device, model_name="standard_sst2",
                         use_compile=args.compile)
    std_history = std_trainer.train(
        train_loader,
        val_loader,
        n_epochs=config['n_epochs'],
        learning_rate=config['learning_rate'],
        warmup_steps=config['warmup_steps'],
        task_type='classification'
    )
    std_time = time.time() - start_time
    
    # Move geometric back for analysis
    geometric_model = geometric_model.to(device)
    
    # Results comparison
    print("\n" + "="*80)
    print("FINAL RESULTS - SST-2 FULL DATASET")
    print("="*80)
    
    geo_best_acc = max(geo_history['val_acc'])
    std_best_acc = max(std_history['val_acc'])
    
    print(f"Geometric Transformer:")
    print(f"  Best Validation Accuracy: {geo_best_acc:.4f}")
    print(f"  Final Train Loss: {geo_history['train_loss'][-1]:.4f}")
    print(f"  Training Time: {geo_time/60:.1f} minutes")
    
    print(f"\nStandard Transformer:")
    print(f"  Best Validation Accuracy: {std_best_acc:.4f}")
    print(f"  Final Train Loss: {std_history['train_loss'][-1]:.4f}")
    print(f"  Training Time: {std_time/60:.1f} minutes")
    
    print(f"\nImprovement: {(geo_best_acc - std_best_acc)*100:.2f}%")
    print(f"Speed Factor: {std_time/geo_time:.2f}x")
    
    # Analyze learned geometries
    print("\n" + "="*80)
    print("LEARNED GEOMETRY DISTRIBUTION")
    print("="*80)
    
    curvature_results = analyze_curvatures(geometric_model, val_loader, device)
    if curvature_results:
        n_hyp = curvature_results['n_hyperbolic']
        n_euc = curvature_results['n_euclidean']
        n_sph = curvature_results['n_spherical']
        total = n_hyp + n_euc + n_sph
        
        print(f"Total heads: {total} ({config['n_layers']} layers × {config['n_heads']} heads)")
        print(f"\nGeometry Distribution:")
        print(f"  Hyperbolic: {n_hyp:3d}/{total} ({curvature_results['pct_hyperbolic']:5.1f}%)")
        print(f"  Euclidean:  {n_euc:3d}/{total} ({curvature_results['pct_euclidean']:5.1f}%)")
        print(f"  Spherical:  {n_sph:3d}/{total} ({curvature_results['pct_spherical']:5.1f}%)")
        
        # Expected ~50/50 hyperbolic-spherical split
        print(f"\nHyperbolic/Spherical Ratio: {n_hyp}/{n_sph} = {n_hyp/max(n_sph,1):.2f}")
        
        # Create visualization
        try:
            plot_geometry_distribution(n_hyp, n_euc, n_sph, 
                                     title="SST-2 Learned Geometry Distribution",
                                     save_path="sst2_geometry_distribution.png")
        except:
            pass
    
    # Save training curves
    try:
        plot_training_curves(geo_history, title="Geometric Transformer - SST-2",
                           save_path="sst2_geometric_curves.png")
        plot_training_curves(std_history, title="Standard Transformer - SST-2", 
                           save_path="sst2_standard_curves.png")
    except:
        pass
    
    print("\n" + "="*80)
    print("SST-2 experiment complete!")
    print("="*80)
    
    # Save results for later analysis
    import json
    results = {
        'task': 'SST-2',
        'config': config,
        'geometric': {
            'best_val_acc': float(geo_best_acc),
            'training_time': geo_time,
            'n_hyperbolic': int(n_hyp),
            'n_euclidean': int(n_euc),
            'n_spherical': int(n_sph)
        },
        'standard': {
            'best_val_acc': float(std_best_acc),
            'training_time': std_time
        }
    }
    
    with open('sst2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to sst2_results.json")


if __name__ == "__main__":
    main()
