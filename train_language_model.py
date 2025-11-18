#!/usr/bin/env python3
"""
Language Modeling training script for Geometric Attention models.
Tests on WikiText-2 dataset to demonstrate universal geometry patterns.
"""

import torch
import time
import argparse
from transformers import AutoTokenizer
from geometric_attention.models.language_models import GeometricCausalLM, StandardCausalLM
from geometric_attention.data import LanguageModelingDataset, load_wikitext_dataset, create_data_loader
from geometric_attention.training import Trainer
from geometric_attention.utils import set_seed, get_device, print_model_summary
from geometric_attention.utils.visualization import plot_training_curves, plot_geometry_distribution
from geometric_attention.training.evaluation import analyze_curvatures, generate_text
import matplotlib.pyplot as plt


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Geometric Attention on WikiText-2 language modeling')
    parser.add_argument('--dim', type=int, default=512, 
                       help='Model dimension (default: 512)')
    parser.add_argument('--epochs', type=int, default=5, 
                       help='Number of training epochs (default: 5)')
    parser.add_argument('--compile', action='store_true',
                       help='Use torch.compile() for ~1.9x speedup (PyTorch 2.0+)')
    args = parser.parse_args()
    
    # Configuration for language modeling - improved defaults
    config = {
        'task': 'Language Modeling',
        'dataset': 'wikitext-2',
        'model_dim': args.dim,      # From CLI argument
        'n_layers': 6,              # Increased from 4
        'n_heads': 8,
        'n_epochs': args.epochs,    # From CLI argument
        'batch_size': 16,
        'learning_rate': 1e-4,      # Reduced from 3e-4 for better convergence
        'warmup_steps': 1000,
        'seed': 42,
        'max_seq_len': 256,
        'gpu_id': 1  # Use GPU 1 (RTX 3090)
    }
    
    print("="*80)
    print(f"Language Modeling Experiment - WikiText-2")
    print(f"Geometric vs Standard Causal LM")
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
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load WikiText-2 dataset
    print("\nLoading WikiText-2 dataset...")
    wikitext = load_wikitext_dataset("wikitext-2-raw-v1")
    
    # Create datasets
    train_dataset = LanguageModelingDataset(
        wikitext['train'], 
        tokenizer, 
        max_length=config['max_seq_len']
    )
    val_dataset = LanguageModelingDataset(
        wikitext['validation'], 
        tokenizer, 
        max_length=config['max_seq_len']
    )
    
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    
    # Create data loaders
    train_loader = create_data_loader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=2
    )
    val_loader = create_data_loader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=2
    )
    
    # ========================================================================
    # TRAIN GEOMETRIC MODEL FIRST
    # ========================================================================
    print("\n" + "="*80)
    print("Initializing & Training Geometric Causal Language Model...")
    print("="*80)
    
    geometric_model = GeometricCausalLM(
        vocab_size=tokenizer.vocab_size,
        dim=config['model_dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        max_seq_len=config['max_seq_len'],
        dropout=0.1
    )
    print_model_summary(geometric_model, "Geometric Causal LM")
    
    start_time = time.time()
    geo_trainer = Trainer(geometric_model, device=device, model_name="geometric_lm", 
                         use_compile=args.compile)
    geo_history = geo_trainer.train(
        train_loader,
        val_loader,
        n_epochs=config['n_epochs'],
        learning_rate=config['learning_rate'],
        warmup_steps=config['warmup_steps'],
        task_type='language_modeling'
    )
    geo_time = time.time() - start_time
    
    # Free up memory before training standard model
    print("\nFreeing GPU memory...")
    del geo_trainer
    geometric_model = geometric_model.cpu()  # Move to CPU before deleting
    torch.cuda.empty_cache()
    print(f"✓ GPU memory cleared")
    
    # ========================================================================
    # TRAIN STANDARD MODEL SECOND
    # ========================================================================
    print("\n" + "="*80)
    print("Initializing & Training Standard Causal Language Model...")
    print("="*80)
    
    standard_model = StandardCausalLM(
        vocab_size=tokenizer.vocab_size,
        dim=config['model_dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        max_seq_len=config['max_seq_len'],
        dropout=0.1
    )
    print_model_summary(standard_model, "Standard Causal LM")
    
    start_time = time.time()
    std_trainer = Trainer(standard_model, device=device, model_name="standard_lm",
                         use_compile=args.compile)
    std_history = std_trainer.train(
        train_loader,
        val_loader,
        n_epochs=config['n_epochs'],
        learning_rate=config['learning_rate'],
        warmup_steps=config['warmup_steps'],
        task_type='language_modeling'
    )
    std_time = time.time() - start_time
    
    # Move back to device for final analysis
    geometric_model = geometric_model.to(device)
    
    # Results comparison
    print("\n" + "="*80)
    print("FINAL RESULTS - LANGUAGE MODELING")
    print("="*80)
    
    # For language modeling, lower perplexity is better
    geo_best_ppl = min(geo_history['val_acc'])  # 'val_acc' contains perplexity for LM
    std_best_ppl = min(std_history['val_acc'])
    
    print(f"Geometric Causal LM:")
    print(f"  Best Validation Perplexity: {geo_best_ppl:.2f}")
    print(f"  Final Train Loss: {geo_history['train_loss'][-1]:.4f}")
    print(f"  Training Time: {geo_time/60:.1f} minutes")
    
    print(f"\nStandard Causal LM:")
    print(f"  Best Validation Perplexity: {std_best_ppl:.2f}")
    print(f"  Final Train Loss: {std_history['train_loss'][-1]:.4f}")
    print(f"  Training Time: {std_time/60:.1f} minutes")
    
    print(f"\nPerplexity Improvement: {(std_best_ppl - geo_best_ppl):.2f} (lower is better)")
    print(f"Relative Improvement: {((std_best_ppl - geo_best_ppl)/std_best_ppl)*100:.1f}%")
    
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
                                     title="Language Modeling Geometry Distribution",
                                     save_path="lm_geometry_distribution.png")
        except:
            pass
    
    # Generate sample text
    print("\n" + "="*80)
    print("SAMPLE GENERATIONS")
    print("="*80)
    
    prompts = [
        "The meaning of life is",
        "In the beginning",
        "Once upon a time",
    ]
    
    print("\nGeometric Model Generations:")
    for prompt in prompts:
        try:
            generated = generate_text(geometric_model, tokenizer, prompt, 
                                    max_length=30, temperature=0.8, device=device)
            print(f"  Prompt: '{prompt}'")
            print(f"  Generated: {generated}\n")
        except:
            pass
    
    print("\nStandard Model Generations:")
    for prompt in prompts:
        try:
            generated = generate_text(standard_model, tokenizer, prompt, 
                                    max_length=30, temperature=0.8, device=device)
            print(f"  Prompt: '{prompt}'")
            print(f"  Generated: {generated}\n")
        except:
            pass
    
    # Save training curves
    try:
        plot_training_curves(geo_history, title="Geometric Causal LM - WikiText-2",
                           save_path="lm_geometric_curves.png")
        plot_training_curves(std_history, title="Standard Causal LM - WikiText-2", 
                           save_path="lm_standard_curves.png")
    except:
        pass
    
    print("\n" + "="*80)
    print("Language Modeling experiment complete!")
    print("="*80)
    
    # Save results for later analysis
    import json
    results = {
        'task': 'Language Modeling - WikiText-2',
        'config': config,
        'geometric': {
            'best_val_perplexity': float(geo_best_ppl),
            'training_time': geo_time,
            'n_hyperbolic': int(n_hyp) if 'n_hyp' in locals() else 0,
            'n_euclidean': int(n_euc) if 'n_euc' in locals() else 0,
            'n_spherical': int(n_sph) if 'n_sph' in locals() else 0
        },
        'standard': {
            'best_val_perplexity': float(std_best_ppl),
            'training_time': std_time
        }
    }
    
    with open('lm_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to lm_results.json")


if __name__ == "__main__":
    main()
