#!/usr/bin/env python3
"""
Geometric Attention Training System - Interactive CLI Shell
Complete system for training, inference, and analysis.
"""

import os
import sys
import torch
import json
import time
import glob
import subprocess
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer
from geometric_attention.models import GeometricTransformer, StandardTransformer
from geometric_attention.models.language_models import (
    GeometricCausalLM, StandardCausalLM,
    TinyGeometricLM, SmallGeometricLM, LargeGeometricLM
)
from geometric_attention.models.transformers import GeometricTransformerNER, StandardTransformerNER
from geometric_attention.data import (
    SST2Dataset, MNLIDataset, NERDataset, LanguageModelingDataset,
    load_glue_dataset, load_wikitext_dataset, load_wikiann_dataset,
    create_data_loader
)
from geometric_attention.training import Trainer
from geometric_attention.utils import set_seed, get_device, print_model_summary
from geometric_attention.training.evaluation import analyze_curvatures
from geometric_attention.dialogue import ConversationManager, ResponseGenerator


# Global state for loaded model (enables Continue Training feature)
LOADED_MODEL_STATE = {
    'loaded': False,
    'checkpoint_path': None,
    'dim': None,
    'n_layers': None,
    'n_heads': None,
    'epochs_trained': None,
    'dataset': None,
    'model_type': None,  # 'geometric' or 'standard'
    'val_ppl': None
}


# Model configuration presets
MODEL_CONFIGS = {
    '1': {'name': 'Tiny', 'dim': 128, 'n_layers': 1, 'n_heads': 2, 'dropout': 0.1},
    '2': {'name': 'Small', 'dim': 256, 'n_layers': 2, 'n_heads': 4, 'dropout': 0.1},
    '3': {'name': 'Medium', 'dim': 512, 'n_layers': 4, 'n_heads': 8, 'dropout': 0.1},
    '4': {'name': 'Large', 'dim': 768, 'n_layers': 6, 'n_heads': 12, 'dropout': 0.1},
    '5': {'name': 'XLarge', 'dim': 1024, 'n_layers': 8, 'n_heads': 16, 'dropout': 0.1},
}


def safe_input(prompt):
    """
    Input wrapper that allows 'exit' to cancel and return to previous menu.
    
    Args:
        prompt: The input prompt to display
        
    Returns:
        User input string, or None if user typed 'exit'
    """
    value = input(prompt).strip()
    if value.lower().startswith('exit'):
        print("\n↩ Returning to previous menu...")
        return None
    return value


def print_header():
    """Print main header with loaded model status"""
    os.system('clear' if os.name != 'nt' else 'cls')
    print("\n" + "="*80)
    print("  GEOMETRIC ATTENTION TRAINING SYSTEM")
    print("  Learnable Curvature Transformers for NLP")
    print("="*80)
    
    # Show loaded model info if any
    if LOADED_MODEL_STATE['loaded']:
        print("\n📦 LOADED MODEL:")
        model_type = LOADED_MODEL_STATE['model_type'].upper() if LOADED_MODEL_STATE['model_type'] else 'UNKNOWN'
        dim = LOADED_MODEL_STATE['dim'] or '?'
        layers = LOADED_MODEL_STATE['n_layers'] or '?'
        heads = LOADED_MODEL_STATE['n_heads'] or '?'
        epochs = LOADED_MODEL_STATE['epochs_trained'] or '?'
        dataset = LOADED_MODEL_STATE['dataset'] or 'unknown'
        ppl = LOADED_MODEL_STATE['val_ppl']
        
        print(f"  {model_type} | {dim}d, {layers}L, {heads}H | "
              f"Epoch {epochs} | {dataset}" + 
              (f" | PPL {ppl:.1f}" if ppl else ""))
        print("="*80)


def select_model_config():
    """Interactive model configuration selection"""
    print("\nSELECT MODEL CONFIGURATION:")
    print("-" * 50)
    for key, config in MODEL_CONFIGS.items():
        print(f"  [{key}] {config['name']:8s} - {config['dim']} dim, "
              f"{config['n_layers']} layers, {config['n_heads']} heads")
    print(f"  [6] Custom - Enter your own values")
    print()
    
    while True:
        choice = safe_input("Enter choice [1-6]: ")
        if choice is None:
            return None
        
        if choice in MODEL_CONFIGS:
            return MODEL_CONFIGS[choice]
        elif choice == '6':
            print("\nCUSTOM CONFIGURATION:")
            try:
                dim_input = safe_input("  Model dimension (e.g., 512): ")
                if dim_input is None:
                    return None
                dim = int(dim_input)
                
                layers_input = safe_input("  Number of layers (e.g., 4): ")
                if layers_input is None:
                    return None
                n_layers = int(layers_input)
                
                heads_input = safe_input("  Number of heads (e.g., 8): ")
                if heads_input is None:
                    return None
                n_heads = int(heads_input)
                
                dropout_input = safe_input("  Dropout rate (e.g., 0.1): ")
                if dropout_input is None:
                    return None
                dropout = float(dropout_input)
                
                return {
                    'name': 'Custom',
                    'dim': dim,
                    'n_layers': n_layers,
                    'n_heads': n_heads,
                    'dropout': dropout
                }
            except ValueError:
                print("Invalid input. Please try again.\n")
        else:
            print("Invalid choice. Please try again.\n")


def get_model_name():
    """Get model name from user"""
    print("\nMODEL NAME:")
    print("-" * 50)
    default_name = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    name = input(f"Enter model name (default: {default_name}): ").strip()
    return name if name else default_name


def select_task_type():
    """Select task type"""
    print("\nSELECT TASK TYPE:")
    print("-" * 50)
    print("  [1] Classification (Sentiment, NLI, etc.)")
    print("  [2] Language Modeling (Causal LM for conversations)")
    print("  [3] Named Entity Recognition (Token classification)")
    print()
    
    while True:
        choice = safe_input("Enter choice [1-3]: ")
        if choice is None:
            return None
        
        if choice == '1':
            return 'classification'
        elif choice == '2':
            return 'language_modeling'
        elif choice == '3':
            return 'ner'
        else:
            print("Invalid choice. Please try again.\n")


def select_dataset_source(task_type):
    """Select dataset source"""
    print("\nSELECT DATASET SOURCE:")
    print("-" * 50)
    print("  [1] HuggingFace Dataset")
    print("  [2] Local Dataset (from datasets/ folder)")
    print()
    
    while True:
        choice = safe_input("Enter choice [1-2]: ")
        if choice is None:
            return None
        
        if choice == '1':
            return select_huggingface_dataset(task_type)
        elif choice == '2':
            return select_local_dataset(task_type)
        else:
            print("Invalid choice. Please try again.\n")


def select_huggingface_dataset(task_type):
    """Select from HuggingFace datasets"""
    print("\nHUGGINGFACE DATASETS:")
    print("-" * 50)
    
    if task_type == 'classification':
        print("  [1] SST-2 (Sentiment)")
        print("  [2] MNLI (NLI)")
        print("  [3] Custom HuggingFace Dataset")
        choice = safe_input("\nEnter choice [1-3]: ")
        if choice is None:
            return None
        
        if choice == '1':
            return {'type': 'huggingface', 'name': 'sst2', 'loader': 'glue'}
        elif choice == '2':
            return {'type': 'huggingface', 'name': 'mnli', 'loader': 'glue'}
        elif choice == '3':
            return select_custom_huggingface_dataset(task_type)
            
    elif task_type == 'language_modeling':
        print("  [1] WikiText-2")
        print("  [2] WikiText-103")
        print("  [3] Custom HuggingFace Dataset")
        choice = safe_input("\nEnter choice [1-3]: ")
        if choice is None:
            return None
        
        if choice == '1':
            return {'type': 'huggingface', 'name': 'wikitext-2-raw-v1', 'loader': 'wikitext'}
        elif choice == '2':
            return {'type': 'huggingface', 'name': 'wikitext-103-raw-v1', 'loader': 'wikitext'}
        elif choice == '3':
            return select_custom_huggingface_dataset(task_type)
            
    elif task_type == 'ner':
        return {'type': 'huggingface', 'name': 'en', 'loader': 'wikiann'}
    
    return None


def select_custom_huggingface_dataset(task_type):
    """Select a custom HuggingFace dataset"""
    print("\nCUSTOM HUGGINGFACE DATASET:")
    print("-" * 50)
    print("Examples:")
    print("  - Simple: 'wikitext', 'imdb', 'bookcorpus'")
    print("  - With org: 'Creekside/logic', 'bigcode/the-stack'")
    print("  - With config: 'wikitext-2-raw-v1'")
    print()
    
    dataset_name = safe_input("Enter HuggingFace dataset name: ")
    if not dataset_name:
        print("No dataset name provided.")
        return None
    
    # Optional config/subset
    config_name = safe_input("Enter config/subset (optional, press Enter to skip): ")
    if config_name == '':
        config_name = None
    
    # Optional text field name
    if task_type == 'language_modeling':
        print("\nText field name (the field containing the text data)")
        text_field = safe_input("Enter field name (default: auto-detect): ")
        if text_field == '':
            text_field = None
    else:
        text_field = None
    
    return {
        'type': 'huggingface',
        'name': dataset_name,
        'config': config_name,
        'text_field': text_field,
        'loader': 'custom'
    }


def select_local_dataset(task_type):
    """Select from local datasets"""
    datasets_dir = 'datasets'
    
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)
        print(f"\nCreated {datasets_dir}/ directory.")
        print("Please add your dataset files and run again.")
        return None
    
    files = [f for f in os.listdir(datasets_dir) 
             if not f.startswith('.') and f != 'README.md' and f.endswith(('.json', '.jsonl', '.txt'))]
    
    if not files:
        print(f"\nNo datasets found in {datasets_dir}/")
        print("Add .json, .jsonl, or .txt files and try again.")
        return None
    
    print(f"\nLOCAL DATASETS:")
    print("-" * 50)
    for i, file in enumerate(files, 1):
        print(f"  [{i}] {file}")
    print()
    
    while True:
        choice_str = safe_input(f"Enter choice [1-{len(files)}]: ")
        if choice_str is None:
            return None
        
        try:
            choice = int(choice_str)
            if 1 <= choice <= len(files):
                filepath = os.path.join(datasets_dir, files[choice - 1])
                return {'type': 'local', 'path': filepath, 'name': files[choice - 1]}
        except ValueError:
            pass
        print("Invalid choice. Please try again.\n")


def get_training_config():
    """Get training configuration"""
    print("\nTRAINING CONFIGURATION:")
    print("-" * 50)
    
    try:
        epochs = int(input("Number of epochs (default 5): ").strip() or "5")
        batch_size = int(input("Batch size (default 32): ").strip() or "32")
        learning_rate = float(input("Learning rate (default 3e-5): ").strip() or "3e-5")
        
        return {
            'n_epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
    except ValueError:
        print("Invalid input, using defaults")
        return {'n_epochs': 5, 'batch_size': 32, 'learning_rate': 3e-5}


def load_dataset_from_source(dataset_info, task_type, tokenizer, max_length=128):
    """Load dataset from source"""
    
    if dataset_info['type'] == 'huggingface':
        if dataset_info['loader'] == 'glue':
            raw_data = load_glue_dataset(dataset_info['name'])
            if dataset_info['name'] == 'sst2':
                return (SST2Dataset(raw_data['train'], tokenizer, max_length),
                       SST2Dataset(raw_data['validation'], tokenizer, max_length))
            elif dataset_info['name'] == 'mnli':
                return (MNLIDataset(raw_data['train'], tokenizer, max_length),
                       MNLIDataset(raw_data['validation_matched'], tokenizer, max_length))
                
        elif dataset_info['loader'] == 'wikitext':
            raw_data = load_wikitext_dataset(dataset_info['name'])
            return (LanguageModelingDataset(raw_data['train'], tokenizer, max_length),
                   LanguageModelingDataset(raw_data['validation'], tokenizer, max_length))
            
        elif dataset_info['loader'] == 'wikiann':
            raw_data = load_wikiann_dataset(dataset_info['name'])
            return (NERDataset(raw_data['train'], tokenizer, max_length),
                   NERDataset(raw_data['validation'], tokenizer, max_length))
        
        elif dataset_info['loader'] == 'custom':
            # Load custom HuggingFace dataset
            from datasets import load_dataset
            
            print(f"\nLoading dataset '{dataset_info['name']}'...")
            if dataset_info.get('config'):
                print(f"  Config: {dataset_info['config']}")
            
            try:
                # Load dataset with optional config
                if dataset_info.get('config'):
                    raw_data = load_dataset(dataset_info['name'], dataset_info['config'])
                else:
                    raw_data = load_dataset(dataset_info['name'])
                
                print(f"✓ Dataset loaded successfully!")
                
                # Determine splits (try common names)
                train_split = None
                val_split = None
                
                for split_name in ['train', 'training']:
                    if split_name in raw_data:
                        train_split = split_name
                        break
                
                for split_name in ['validation', 'valid', 'val', 'dev', 'test']:
                    if split_name in raw_data:
                        val_split = split_name
                        break
                
                if not train_split or not val_split:
                    print(f"\nAvailable splits: {list(raw_data.keys())}")
                    train_split = input("Enter train split name: ").strip()
                    val_split = input("Enter validation split name (press Enter if none): ").strip()
                
                # Handle datasets with only train split
                if not val_split or val_split == '':
                    print("\n⚠️ No validation split found. Creating split from train data...")
                    split_choice = input("Split ratio - [1] 90/10, [2] 80/20 (default: 1): ").strip() or '1'
                    
                    if split_choice == '2':
                        split_ratio = 0.8  # 80/20
                        print("✓ Using 80/20 train/val split")
                    else:
                        split_ratio = 0.9  # 90/10
                        print("✓ Using 90/10 train/val split")
                    
                    full_train = raw_data[train_split]
                    split_idx = int(len(full_train) * split_ratio)
                    
                    # Create train/val from single split using HuggingFace's train_test_split
                    split_data = full_train.train_test_split(test_size=1.0-split_ratio, seed=42)
                    train_data_raw = split_data['train']
                    val_data_raw = split_data['test']
                    
                    print(f"✓ Created splits: {len(train_data_raw)} train, {len(val_data_raw)} val")
                else:
                    print(f"✓ Using splits: train='{train_split}', val='{val_split}'")
                    train_data_raw = raw_data[train_split]
                    val_data_raw = raw_data[val_split]
                
                # Field selection for language modeling
                selected_fields = []
                if task_type == 'language_modeling':
                    # Show field preview
                    sample = train_data_raw[0]
                    fields = [k for k in sample.keys() if isinstance(sample[k], str)]
                    
                    print("\n" + "="*70)
                    print("📋 DATASET FIELDS PREVIEW")
                    print("="*70)
                    
                    for i, field in enumerate(fields, 1):
                        value = str(sample[field])
                        # Truncate long values
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
                    try:
                        indices = [int(x) - 1 for x in field_choice.split()]
                        selected_fields = [fields[i] for i in indices if 0 <= i < len(fields)]
                        
                        if not selected_fields:
                            print("⚠️ No valid fields selected, using auto-detect")
                            selected_fields = None
                        else:
                            print(f"✓ Selected fields: {', '.join(selected_fields)}")
                    except (ValueError, IndexError):
                        print("⚠️ Invalid selection, using auto-detect")
                        selected_fields = None
                    
                    # Fallback to auto-detect if no selection
                    if not selected_fields:
                        for field in ['text', 'content', 'sentence', 'document', 'passage', 'answer']:
                            if field in sample:
                                selected_fields = [field]
                                print(f"✓ Auto-detected text field: '{field}'")
                                break
                
                # Convert to appropriate dataset format
                if task_type == 'language_modeling':
                    # Combine selected fields
                    if selected_fields and len(selected_fields) > 0:
                        if len(selected_fields) == 1:
                            # Single field - simple mapping
                            field_name = selected_fields[0]
                            train_data = train_data_raw.map(lambda x: {'text': x[field_name]})
                            val_data = val_data_raw.map(lambda x: {'text': x[field_name]})
                        else:
                            # Multiple fields - combine with separator
                            print(f"\nCombining {len(selected_fields)} fields with '\\n\\n' separator")
                            
                            def combine_fields(example):
                                combined = "\n\n".join([example[f] for f in selected_fields if f in example])
                                return {'text': combined}
                            
                            train_data = train_data_raw.map(combine_fields)
                            val_data = val_data_raw.map(combine_fields)
                        
                        # Show final sample preview
                        final_sample = train_data[0]['text']
                        print("\n" + "="*70)
                        print("📋 FINAL TRAINING SAMPLE PREVIEW")
                        print("="*70)
                        print(f"  Fields: {', '.join(selected_fields)}")
                        print(f"  Length: {len(final_sample)} characters")
                        print(f"\n  Sample:")
                        # Show first 200 chars
                        display_sample = final_sample if len(final_sample) <= 200 else final_sample[:197] + "..."
                        print(f"  \"{display_sample}\"")
                        print("="*70 + "\n")
                    else:
                        # Fallback to raw data
                        train_data = train_data_raw
                        val_data = val_data_raw
                    
                    return (LanguageModelingDataset(train_data, tokenizer, max_length),
                           LanguageModelingDataset(val_data, tokenizer, max_length))
                
                elif task_type == 'classification':
                    # Use SST2Dataset as generic wrapper
                    return (SST2Dataset(raw_data[train_split], tokenizer, max_length),
                           SST2Dataset(raw_data[val_split], tokenizer, max_length))
                
                else:  # NER
                    return (NERDataset(raw_data[train_split], tokenizer, max_length),
                           NERDataset(raw_data[val_split], tokenizer, max_length))
                
            except Exception as e:
                print(f"\n✗ Error loading dataset: {e}")
                import traceback
                traceback.print_exc()
                return None, None
    
    elif dataset_info['type'] == 'local':
        filepath = dataset_info['path']
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.jsonl':
            data = []
            with open(filepath, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
        elif ext == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
        elif ext in ['.txt', '.text']:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            data = [{'text': line.strip()} for line in lines if line.strip()]
        
        # Split 80/20
        split_idx = int(len(data) * 0.8)
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        if task_type == 'language_modeling':
            return (LanguageModelingDataset(train_data, tokenizer, max_length),
                   LanguageModelingDataset(val_data, tokenizer, max_length))
        else:
            return (SST2Dataset(train_data, tokenizer, max_length),
                   SST2Dataset(val_data, tokenizer, max_length))
    
    return None, None


def train_model_with_logging(model, model_name, model_type, train_loader, val_loader,
                            n_epochs, learning_rate, task_type, model_config, dataset_info,
                            device, use_compile, tokenizer):
    """
    Train a model with comprehensive logging using ExperimentLogger.
    
    Returns:
        tuple: (history dict, training time in seconds)
    """
    from geometric_attention.logging import ExperimentLogger
    from geometric_attention.training.evaluation import evaluate, evaluate_lm, evaluate_ner
    import torch.nn as nn
    import torch.optim as optim
    from transformers import get_cosine_schedule_with_warmup
    from tqdm import tqdm
    
    # Prepare configuration for logger
    total_params = sum(p.numel() for p in model.parameters())
    
    # Determine dataset name for display
    if dataset_info['type'] == 'huggingface':
        dataset_name = dataset_info.get('name', 'unknown')
    else:
        dataset_name = dataset_info.get('name', 'local_dataset')
    
    # Determine n_classes for config
    n_classes = 3 if dataset_info.get('name') == 'mnli' else 2
    
    config = {
        'model': {
            'type': model_type,
            'vocab_size': tokenizer.vocab_size,
            'dim': model_config['dim'],
            'n_layers': model_config['n_layers'],
            'n_heads': model_config['n_heads'],
            'dropout': model_config['dropout'],
            'n_classes': n_classes if task_type == 'classification' else None,
            'n_labels': 7 if task_type == 'ner' else None,
            'total_params': total_params,
            'compiled': use_compile
        },
        'training': {
            'dataset': dataset_name,
            'task_type': task_type,
            'n_epochs': n_epochs,
            'batch_size': train_loader.batch_size,
            'learning_rate': learning_rate,
            'weight_decay': 0.01,
            'warmup_steps': 0,
            'gradient_clip': 1.0
        },
        'hardware': {
            'device': str(device),
            'gpu_name': torch.cuda.get_device_name(device.index) if device.type == 'cuda' else 'CPU'
        }
    }
    
    # Create experiment name
    task_short = {'classification': 'cls', 'language_modeling': 'lm', 'ner': 'ner'}.get(task_type, 'task')
    dataset_short = dataset_name.split('/')[-1] if '/' in dataset_name else dataset_name
    dataset_short = dataset_short.replace('-raw-v1', '')  # Simplify wikitext names
    exp_name = f"{model_type}_{model_config['dim']}d_{model_config['n_layers']}L_{model_config['n_heads']}H_{task_short}_{dataset_short}"
    
    # Move model to device
    model.to(device)
    
    # Apply torch.compile if requested
    if use_compile:
        if hasattr(torch, 'compile'):
            torch.set_float32_matmul_precision('high')
            print("Compiling model with torch.compile()...")
            print("  First forward pass will be slow (JIT compilation)")
            model = torch.compile(model, mode='reduce-overhead', fullgraph=False)
            print("✓ Model compiled!")
        else:
            print("⚠️ torch.compile() not available (requires PyTorch 2.0+)")
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Setup learning rate scheduler with warmup and cosine decay
    total_steps = n_epochs * len(train_loader)
    warmup_steps = min(500, total_steps // 10)  # 10% warmup or 500 steps, whichever is smaller
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Update config with warmup info
    config['training']['warmup_steps'] = warmup_steps
    config['training']['total_steps'] = total_steps
    config['training']['scheduler'] = 'cosine_with_warmup'
    
    # Setup criterion
    if task_type in ['classification', 'ner']:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = None
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    start_time = time.time()
    
    # Use ExperimentLogger with context manager
    with ExperimentLogger(
        experiment_name=exp_name,
        config=config,
        geometry_log_freq=10,  # Log geometry every 10 steps
        enable_attention_logging=False
    ) as logger:
        
        best_metric = None
        
        for epoch in range(n_epochs):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{n_epochs}")
            print(f"{'='*70}")
            
            # Training
            model.train()
            epoch_loss = 0
            epoch_tokens = 0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for step, batch in enumerate(pbar):
                input_ids = batch['input_ids'].to(device)
                
                if task_type == 'language_modeling':
                    labels = batch['labels'].to(device)
                else:
                    labels = batch.get('label', batch.get('labels')).to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                if task_type == 'language_modeling':
                    # Language model returns (logits, loss, curvatures)
                    outputs = model(input_ids, labels=labels)
                    if isinstance(outputs, tuple) and len(outputs) >= 2:
                        logits, loss = outputs[0], outputs[1]
                        curvatures = outputs[2] if len(outputs) >= 3 else None
                    else:
                        loss = outputs
                        logits = None
                        curvatures = None
                else:
                    outputs = model(input_ids)
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                        curvatures = outputs[1] if len(outputs) >= 2 else None
                    else:
                        logits = outputs
                        curvatures = None
                    
                    if task_type == 'ner':
                        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                    else:
                        loss = criterion(logits, labels)
                
                loss.backward()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()  # Update learning rate
                
                # Track metrics
                if task_type == 'language_modeling':
                    n_tokens = (labels != -100).sum().item()
                    epoch_loss += loss.item() * n_tokens
                    epoch_tokens += n_tokens
                else:
                    epoch_loss += loss.item()
                    if task_type == 'ner':
                        preds = logits.argmax(dim=-1)
                        mask = labels != -100
                        correct += ((preds == labels) & mask).sum().item()
                        total += mask.sum().item()
                    else:
                        preds = logits.argmax(dim=1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)
                
                # Log training step
                logger.log_training_step(
                    epoch=epoch,
                    step=step,
                    loss=loss.item(),
                    perplexity=torch.exp(loss).item() if task_type == 'language_modeling' else None,
                    learning_rate=optimizer.param_groups[0]['lr'],
                    grad_norm=grad_norm.item()
                )
                
                # Log geometry if geometric model
                if model_type == 'geometric' and curvatures is not None:
                    logger.log_geometry(epoch=epoch, step=step, curvatures=curvatures)
                
                # Update progress bar
                if task_type == 'language_modeling':
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'ppl': f'{torch.exp(loss).item():.1f}'
                    })
                else:
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{correct/total:.3f}' if total > 0 else '0.000'
                    })
            
            # Calculate epoch metrics
            if task_type == 'language_modeling':
                train_loss = epoch_loss / epoch_tokens if epoch_tokens > 0 else 0
                train_acc = torch.exp(torch.tensor(train_loss)).item()  # PPL
            else:
                train_loss = epoch_loss / len(train_loader)
                train_acc = correct / total if total > 0 else 0
            
            # Validation
            if task_type == 'classification':
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            elif task_type == 'language_modeling':
                val_loss, val_acc = evaluate_lm(model, val_loader, device)
            else:  # ner
                val_loss, val_acc = evaluate_ner(model, val_loader, criterion, device)
            
            # Store results
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Log epoch summary
            logger.log_epoch_summary(
                epoch=epoch,
                train_loss=train_loss,
                train_ppl=train_acc if task_type == 'language_modeling' else 0,
                val_loss=val_loss,
                val_ppl=val_acc if task_type == 'language_modeling' else 0
            )
            
            print(f"  Train Loss: {train_loss:.4f}, Train {'PPL' if task_type == 'language_modeling' else 'Acc'}: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}, Val {'PPL' if task_type == 'language_modeling' else 'Acc'}:   {val_acc:.4f}")
            
            # Show geometry distribution for geometric models
            if model_type == 'geometric':
                current_geom = logger.geometry_tracker.get_current_distribution()
                if current_geom:
                    print(f"  Geometry:   {current_geom['hyperbolic']:.1f}% H, "
                          f"{current_geom['euclidean']:.1f}% E, "
                          f"{current_geom['spherical']:.1f}% S")
            
            # Save best checkpoint
            is_best = False
            current_metric = val_acc
            
            if task_type == 'language_modeling':
                # Lower perplexity is better
                if best_metric is None or current_metric < best_metric:
                    is_best = True
                    best_metric = current_metric
            else:
                # Higher accuracy is better
                if best_metric is None or current_metric > best_metric:
                    is_best = True
                    best_metric = current_metric
            
            if is_best:
                checkpoint_path = logger.save_checkpoint(
                    epoch=epoch + 1,
                    model_state=model.state_dict(),
                    metrics={'val_metric': val_acc, 'train_metric': train_acc, 'is_best': True}
                )
                print(f"  ✓ Saved best checkpoint: {os.path.basename(checkpoint_path)}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = logger.save_checkpoint(
                    epoch=epoch + 1,
                    model_state=model.state_dict(),
                    metrics={'val_metric': val_acc, 'train_metric': train_acc}
                )
    
    training_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Training complete in {training_time:.1f}s ({training_time/60:.1f} min)")
    print(f"{'='*70}")
    
    return history, training_time


def run_training_workflow(model_config, model_name, task_type, dataset_info, train_config, use_compile=True):
    """Execute the full training workflow with comprehensive logging"""
    
    set_seed(42)
    device = get_device(cuda_id=1)
    
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device.index)}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    if task_type == 'language_modeling':
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset, val_dataset = load_dataset_from_source(
        dataset_info, task_type, tokenizer, max_length=128
    )
    
    if train_dataset is None:
        print("Failed to load dataset.")
        return
    
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    
    # Create loaders
    train_loader = create_data_loader(train_dataset, batch_size=train_config['batch_size'],
                                     shuffle=True, num_workers=2)
    val_loader = create_data_loader(val_dataset, batch_size=train_config['batch_size'],
                                   shuffle=False, num_workers=2)
    
    # Determine n_classes
    n_classes = 3 if dataset_info.get('name') == 'mnli' else 2
    
    # ========================================================================
    # TRAIN GEOMETRIC MODEL FIRST
    # ========================================================================
    print("\n" + "="*80)
    print("Initializing & Training Geometric Model...")
    print("="*80)
    
    if task_type == 'classification':
        geo_model = GeometricTransformer(
            vocab_size=tokenizer.vocab_size, dim=model_config['dim'],
            n_layers=model_config['n_layers'], n_heads=model_config['n_heads'],
            n_classes=n_classes, dropout=model_config['dropout']
        )
    elif task_type == 'language_modeling':
        geo_model = GeometricCausalLM(
            vocab_size=tokenizer.vocab_size, dim=model_config['dim'],
            n_layers=model_config['n_layers'], n_heads=model_config['n_heads'],
            dropout=model_config['dropout']
        )
    else:
        geo_model = GeometricTransformerNER(
            vocab_size=tokenizer.vocab_size, dim=model_config['dim'],
            n_layers=model_config['n_layers'], n_heads=model_config['n_heads'],
            n_labels=7, dropout=model_config['dropout']
        )
    
    print_model_summary(geo_model, "Geometric Model")
    
    # Train geometric model with full logging
    geo_history, geo_time = train_model_with_logging(
        model=geo_model,
        model_name=f"geometric_{model_name}",
        model_type='geometric',
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=train_config['n_epochs'],
        learning_rate=train_config['learning_rate'],
        task_type=task_type,
        model_config=model_config,
        dataset_info=dataset_info,
        device=device,
        use_compile=use_compile,
        tokenizer=tokenizer
    )
    
    # Free memory before loading standard model
    print("\nFreeing GPU memory...")
    del geo_model
    torch.cuda.empty_cache()
    print("✓ GPU memory cleared")
    
    # ========================================================================
    # TRAIN STANDARD MODEL SECOND
    # ========================================================================
    print("\n" + "="*80)
    print("Initializing & Training Standard Model...")
    print("="*80)
    
    if task_type == 'classification':
        std_model = StandardTransformer(
            vocab_size=tokenizer.vocab_size, dim=model_config['dim'],
            n_layers=model_config['n_layers'], n_heads=model_config['n_heads'],
            n_classes=n_classes, dropout=model_config['dropout']
        )
    elif task_type == 'language_modeling':
        std_model = StandardCausalLM(
            vocab_size=tokenizer.vocab_size, dim=model_config['dim'],
            n_layers=model_config['n_layers'], n_heads=model_config['n_heads'],
            dropout=model_config['dropout']
        )
    else:
        std_model = StandardTransformerNER(
            vocab_size=tokenizer.vocab_size, dim=model_config['dim'],
            n_layers=model_config['n_layers'], n_heads=model_config['n_heads'],
            n_labels=7, dropout=model_config['dropout']
        )
    
    print_model_summary(std_model, "Standard Model")
    
    # Train standard model with full logging
    std_history, std_time = train_model_with_logging(
        model=std_model,
        model_name=f"standard_{model_name}",
        model_type='standard',
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=train_config['n_epochs'],
        learning_rate=train_config['learning_rate'],
        task_type=task_type,
        model_config=model_config,
        dataset_info=dataset_info,
        device=device,
        use_compile=use_compile,
        tokenizer=tokenizer
    )
    
    # Results
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    
    geo_best = (min if task_type == 'language_modeling' else max)(geo_history['val_acc'])
    std_best = (min if task_type == 'language_modeling' else max)(std_history['val_acc'])
    
    print(f"\nGeometric Model: {geo_best:.4f} ({geo_time/60:.1f} min)")
    print(f"Standard Model:  {std_best:.4f} ({std_time/60:.1f} min)")
    print(f"Improvement: {(geo_best - std_best):.4f}")


def main_menu():
    """Display main menu and get choice"""
    print("\n" + "-"*80)
    print("MAIN MENU")
    print("-"*80)
    print("  [1] Train New Model")
    print("  [2] Load & Analyze Checkpoint")
    print("  [3] Interactive Chat/Inference")
    
    # Show Continue Training option (grayed out if no model loaded)
    if LOADED_MODEL_STATE['loaded']:
        print("  [4] Continue Training from Loaded Model")
    else:
        print("  [4] Continue Training (requires loaded model)")
    
    print("  [5] Compare Checkpoints")
    print("  [6] Geometry Analysis")
    print("  [7] Exit")
    print()
    
    choice = input("Select option [1-7]: ").strip()
    return choice


def geometry_analysis_menu():
    """Sub-menu for geometry analysis options"""
    print("\n" + "="*80)
    print("GEOMETRY ANALYSIS")
    print("="*80)
    print("\n  [1] Analyze Existing Checkpoint")
    print("      • Load checkpoint and show H/E/S distribution")
    print("      • Curvature statistics and visualizations")
    print("      • Detect 50/50 pattern")
    print()
    print("  [2] Train with Geometry Logging")
    print("      • Train model with detailed geometry tracking")
    print("      • Log evolution every 10 steps")
    print("      • Auto-generate visualizations")
    print()
    print("  [3] Visualize Experiment Results")
    print("      • Select completed experiment")
    print("      • Generate geometry evolution plots")
    print("      • Show training curves and statistics")
    print()
    print("  [4] Deep Geometry Analysis")
    print("      • Advanced metrics from existing logs")
    print("      • Per-layer analysis, transitions, clustering")
    print("      • Temporal correlations and head behaviors")
    print()
    print("  [0] Back to Main Menu")
    print()
    
    choice = input("Select option [0-4]: ").strip()
    return choice


def select_checkpoint(title="Select Checkpoint"):
    """Browse and select a checkpoint file - two-level menu for experiments"""
    
    # Group checkpoints by experiment (only experiments/ directory)
    experiments = {}
    
    # Search experiments/*/checkpoints/
    if os.path.exists('experiments'):
        for exp_dir in Path('experiments').iterdir():
            if exp_dir.is_dir():
                ckpt_dir = exp_dir / 'checkpoints'
                if ckpt_dir.exists():
                    ckpts = list(ckpt_dir.glob('*.pt'))
                    if ckpts:
                        experiments[exp_dir.name] = {
                            'path': str(exp_dir),
                            'checkpoints': [str(c) for c in ckpts],
                            'mtime': max(os.path.getmtime(str(c)) for c in ckpts)
                        }
    
    if not experiments:
        print("\n⚠️ No checkpoints found in experiments/ directory!")
        print("Tip: Train a model first using menu option [1] or [6]→[2]")
        return None
    
    # Sort experiments by modification time (newest first)
    sorted_experiments = sorted(experiments.items(), key=lambda x: x[1]['mtime'], reverse=True)
    
    # STEP 1: Select experiment (show 5 most recent)
    print(f"\n{title}:")
    print("-" * 80)
    print("Step 1: Select Experiment")
    print("-" * 80)
    
    display_experiments = sorted_experiments[:5]
    for i, (exp_name, exp_data) in enumerate(display_experiments, 1):
        num_ckpts = len(exp_data['checkpoints'])
        mtime = datetime.fromtimestamp(exp_data['mtime'])
        print(f"  [{i}] {exp_name}")
        print(f"      {num_ckpts} checkpoints | Last modified: {mtime.strftime('%Y-%m-%d %H:%M')}")
    
    if len(sorted_experiments) > 5:
        print(f"  [6] Show all {len(sorted_experiments)} experiments")
    print("  [0] Cancel")
    
    print()
    while True:
        try:
            choice = input(f"Select experiment [0-{min(6, len(sorted_experiments))}]: ").strip()
            choice_int = int(choice)
            
            if choice_int == 0:
                return None
            elif choice_int == 6 and len(sorted_experiments) > 5:
                # Show all experiments
                print("\nAll Experiments:")
                print("-" * 80)
                for i, (exp_name, exp_data) in enumerate(sorted_experiments, 1):
                    num_ckpts = len(exp_data['checkpoints'])
                    print(f"  [{i}] {exp_name} ({num_ckpts} checkpoints)")
                
                choice = int(input(f"\nSelect experiment [1-{len(sorted_experiments)}] or 0 to cancel: "))
                if choice == 0:
                    return None
                if 1 <= choice <= len(sorted_experiments):
                    selected_exp = sorted_experiments[choice - 1]
                    break
            elif 1 <= choice_int <= min(5, len(sorted_experiments)):
                selected_exp = display_experiments[choice_int - 1]
                break
            else:
                print("Invalid choice. Try again.")
        except ValueError:
            print("Invalid input. Try again.")
    
    exp_name, exp_data = selected_exp
    
    # STEP 2: Select checkpoint from experiment
    checkpoints = sorted(exp_data['checkpoints'], key=os.path.getmtime, reverse=True)
    
    print(f"\nStep 2: Select Checkpoint from '{exp_name}'")
    print("-" * 80)
    
    for i, ckpt in enumerate(checkpoints, 1):
        filename = os.path.basename(ckpt)
        size_mb = os.path.getsize(ckpt) / (1024 * 1024)
        mtime = datetime.fromtimestamp(os.path.getmtime(ckpt))
        print(f"  [{i}] {filename}")
        print(f"      {size_mb:.1f} MB | {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print()
    while True:
        try:
            choice = int(input(f"Select checkpoint [1-{len(checkpoints)}] or 0 to cancel: ").strip())
            if choice == 0:
                return None
            if 1 <= choice <= len(checkpoints):
                return checkpoints[choice - 1]
        except ValueError:
            pass
        print("Invalid choice. Try again.\n")


def train_new_model():
    """Train new model with full workflow"""
    print("\n" + "="*80)
    print("TRAIN NEW MODEL")
    print("="*80)
    
    # 1. Select model configuration
    model_config = select_model_config()
    if model_config is None:
        return
    print(f"\n✓ Selected: {model_config['name']} "
          f"({model_config['dim']}d, {model_config['n_layers']}L, {model_config['n_heads']}H)")
    
    # 2. Get model name
    model_name = get_model_name()
    print(f"✓ Model name: {model_name}")
    
    # 3. Select task type
    task_type = select_task_type()
    if task_type is None:
        return
    print(f"✓ Task type: {task_type}")
    
    # 4. Select dataset
    dataset_info = select_dataset_source(task_type)
    if dataset_info is None:
        return
    print(f"✓ Dataset: {dataset_info.get('name', dataset_info.get('path', 'unknown'))}")
    
    # 5. Get training config
    train_config = get_training_config()
    print(f"✓ Training: {train_config['n_epochs']} epochs, "
          f"batch size {train_config['batch_size']}, "
          f"lr {train_config['learning_rate']}")
    
    # 6. Confirm
    print("\n" + "="*80)
    print("CONFIGURATION SUMMARY")
    print("="*80)
    print(f"Model:        {model_name}")
    print(f"Architecture: {model_config['name']} ({model_config['dim']}d, "
          f"{model_config['n_layers']}L, {model_config['n_heads']}H)")
    print(f"Task:         {task_type}")
    print(f"Dataset:      {dataset_info.get('name', dataset_info.get('path'))}")
    print(f"Epochs:       {train_config['n_epochs']}")
    print(f"Batch Size:   {train_config['batch_size']}")
    print(f"Learning Rate: {train_config['learning_rate']}")
    print("="*80)
    
    confirm = input("\nStart training? [y/n]: ").strip().lower()
    if confirm != 'y':
        print("Training cancelled.")
        input("\nPress Enter to return to main menu...")
        return
    
    # Ask about torch.compile (default to yes)
    compile_input = input("\nUse torch.compile() for ~2x speedup? [Y/n]: ").strip().lower()
    use_compile = compile_input != 'n'  # Anything except 'n' = yes
    
    if use_compile:
        print("✓ Will use torch.compile() (first 2 epochs slower due to JIT compilation)")
    else:
        print("✓ torch.compile() disabled")
    
    # Run training
    try:
        run_training_workflow(model_config, model_name, task_type, dataset_info, train_config, use_compile)
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to return to main menu...")


def load_and_analyze_checkpoint():
    """Load checkpoint and populate global state (menu option [2])"""
    print("\n" + "="*80)
    print("LOAD & ANALYZE CHECKPOINT")
    print("="*80)
    
    # Select checkpoint
    checkpoint_path = select_checkpoint("Select Model Checkpoint")
    if not checkpoint_path:
        input("\nPress Enter to return to main menu...")
        return
    
    print(f"\nLoading checkpoint: {os.path.basename(checkpoint_path)}")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        input("\nPress Enter to return to main menu...")
        return
    
    # Extract metadata
    epoch = checkpoint.get('epoch', '?')
    metric = checkpoint.get('metric')
    
    # Detect model type from state_dict
    state_dict_keys = list(checkpoint.get('model_state_dict', {}).keys())
    is_geometric = any('curvature_raw' in k or 'attention.heads' in k or 'to_heads' in k 
                      for k in state_dict_keys)
    
    # Get architecture (multiple methods)
    dim, n_layers, n_heads = None, None, None
    
    # Method 1: From checkpoint
    if 'model_config' in checkpoint and checkpoint['model_config']:
        model_cfg = checkpoint['model_config']
        dim = model_cfg.get('dim')
        n_layers = model_cfg.get('n_layers')
        n_heads = model_cfg.get('n_heads')
    
    # Method 2: From experiment config.json
    if not dim:
        exp_dir = Path(checkpoint_path).parent.parent
        config_file = exp_dir / 'config.json'
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = json.load(f)
                    dim = config.get('model', {}).get('dim')
                    n_layers = config.get('model', {}).get('n_layers')
                    n_heads = config.get('model', {}).get('n_heads')
            except:
                pass
    
    # Method 3: Parse from directory name
    if not dim:
        import re
        exp_name = Path(checkpoint_path).parent.parent.name
        match = re.search(r'(\d+)d_(\d+)L_(\d+)H', exp_name)
        if match:
            dim, n_layers, n_heads = map(int, match.groups())
    
    # Extract dataset info
    dataset = 'unknown'
    if 'config' in checkpoint:
        dataset = checkpoint['config'].get('training', {}).get('dataset', 'unknown')
    else:
        # Try from experiment config
        exp_dir = Path(checkpoint_path).parent.parent
        config_file = exp_dir / 'config.json'
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = json.load(f)
                    dataset = config.get('training', {}).get('dataset', 'unknown')
            except:
                pass
    
    # POPULATE GLOBAL STATE
    LOADED_MODEL_STATE['loaded'] = True
    LOADED_MODEL_STATE['checkpoint_path'] = checkpoint_path
    LOADED_MODEL_STATE['dim'] = dim
    LOADED_MODEL_STATE['n_layers'] = n_layers
    LOADED_MODEL_STATE['n_heads'] = n_heads
    LOADED_MODEL_STATE['epochs_trained'] = epoch
    LOADED_MODEL_STATE['dataset'] = dataset
    LOADED_MODEL_STATE['model_type'] = 'geometric' if is_geometric else 'standard'
    LOADED_MODEL_STATE['val_ppl'] = metric if isinstance(metric, (int, float)) else None
    
    # Display summary
    print("\n" + "="*80)
    print("CHECKPOINT SUMMARY")
    print("="*80)
    print(f"  Model Type: {'Geometric' if is_geometric else 'Standard'}")
    print(f"  Architecture: {dim}d, {n_layers}L, {n_heads}H")
    print(f"  Epoch: {epoch}")
    print(f"  Dataset: {dataset}")
    if isinstance(metric, (int, float)):
        print(f"  Validation Metric: {metric:.4f}")
    print(f"  Checkpoint: {checkpoint_path}")
    print("="*80)
    
    print("\n✅ Model loaded into global state!")
    print("   You can now use [3] Chat/Inference or [4] Continue Training")
    
    input("\nPress Enter to return to main menu...")


def load_model_and_chat():
    """Interactive chat/inference mode (menu option [3])"""
    print("\n" + "="*80)
    print("INTERACTIVE CHAT/INFERENCE")
    print("="*80)
    
    checkpoint_path = None
    is_geometric = None
    
    # Check if model already loaded
    if LOADED_MODEL_STATE['loaded']:
        print("\n📦 Currently Loaded Model:")
        model_type = LOADED_MODEL_STATE['model_type'].upper()
        print(f"  {model_type} | {LOADED_MODEL_STATE['dim']}d, "
              f"{LOADED_MODEL_STATE['n_layers']}L, {LOADED_MODEL_STATE['n_heads']}H")
        print(f"  Epoch {LOADED_MODEL_STATE['epochs_trained']} | {LOADED_MODEL_STATE['dataset']}")
        
        print("\n[1] Use loaded model (quick start)")
        print("[2] Select different checkpoint")
        print("[0] Cancel")
        
        choice = input("\nChoice [0-2]: ").strip()
        
        if choice == '0':
            input("\nPress Enter to return to main menu...")
            return
        elif choice == '1':
            # Use loaded model
            checkpoint_path = LOADED_MODEL_STATE['checkpoint_path']
            is_geometric = LOADED_MODEL_STATE['model_type'] == 'geometric'
            print(f"\n✓ Using loaded model: {os.path.basename(checkpoint_path)}")
        else:
            # Select different checkpoint
            checkpoint_path = select_checkpoint("Select Model Checkpoint")
            if not checkpoint_path:
                input("\nPress Enter to return to main menu...")
                return
    else:
        # No model loaded, must select
        print("\n⚠️ No model currently loaded.")
        checkpoint_path = select_checkpoint("Select Model Checkpoint")
        if not checkpoint_path:
            input("\nPress Enter to return to main menu...")
            return
    
    print(f"\nLoading checkpoint: {os.path.basename(checkpoint_path)}")
    
    # Load checkpoint metadata
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
        
        # Handle metric formatting safely
        metric = checkpoint.get('metric')
        if metric is not None and isinstance(metric, (int, float)):
            print(f"✓ Best metric: {metric:.4f}")
        else:
            print(f"✓ Best metric: Not available")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        input("\nPress Enter to return to main menu...")
        return
    
    # Determine model type from state_dict structure (more reliable than model_name)
    state_dict_keys = list(checkpoint.get('model_state_dict', {}).keys())
    
    # Geometric models have unique keys like 'curvature_raw', 'attention.heads', 'to_heads'
    is_geometric = any('curvature_raw' in k or 'attention.heads' in k or 'to_heads' in k 
                      for k in state_dict_keys)
    
    # Fallback: check model_name if state_dict detection unclear
    if not is_geometric and 'geometric' in checkpoint.get('model_name', '').lower():
        is_geometric = True
    
    print(f"✓ Model type: {'Geometric' if is_geometric else 'Standard'}")
    
    # Ask for task type
    print("\nWhat task was this model trained for?")
    print("  [1] Classification")
    print("  [2] Language Modeling (for chat)")
    print("  [3] NER")
    task_choice = input("Enter choice [1-3]: ").strip()
    
    if task_choice == '2':
        # Chat mode for language models
        chat_mode(checkpoint_path, is_geometric)
    elif task_choice == '1':
        # Inference mode for classification
        classification_inference(checkpoint_path, is_geometric)
    else:
        print("NER inference not yet implemented")
        input("\nPress Enter to return to main menu...")


def chat_mode(checkpoint_path, is_geometric):
    """Enhanced interactive chat mode using ConversationManager and ResponseGenerator"""
    print("\n" + "="*80)
    print("ENHANCED CHAT MODE")
    print("="*80)
    
    # Load model
    device = get_device(cuda_id=1)
    print(f"Loading model on {device}...")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Try multiple methods to get architecture
        dim, n_layers, n_heads = None, None, None
        
        # Method 1: From checkpoint model_config
        if 'model_config' in checkpoint and checkpoint['model_config']:
            model_cfg = checkpoint['model_config']
            dim = model_cfg.get('dim')
            n_layers = model_cfg.get('n_layers')
            n_heads = model_cfg.get('n_heads')
            if dim:
                print(f"✓ Architecture from checkpoint: {dim}d, {n_layers}L, {n_heads}H")
        
        # Method 2: From experiment config.json
        if not dim:
            exp_dir = Path(checkpoint_path).parent.parent
            config_file = exp_dir / 'config.json'
            if config_file.exists():
                try:
                    with open(config_file) as f:
                        config = json.load(f)
                        dim = config.get('model', {}).get('dim')
                        n_layers = config.get('model', {}).get('n_layers')
                        n_heads = config.get('model', {}).get('n_heads')
                    if dim:
                        print(f"✓ Architecture from experiment config: {dim}d, {n_layers}L, {n_heads}H")
                except:
                    pass
        
        # Method 3: Parse from experiment directory name
        if not dim:
            import re
            exp_name = Path(checkpoint_path).parent.parent.name
            # Pattern: geometric_128d_1L_2H_...
            match = re.search(r'(\d+)d_(\d+)L_(\d+)H', exp_name)
            if match:
                dim, n_layers, n_heads = map(int, match.groups())
                print(f"✓ Architecture from directory name: {dim}d, {n_layers}L, {n_heads}H")
        
        # Method 4: Ask user (last resort)
        if not dim:
            print("\nNote: Architecture not found. Please specify:")
            dim = int(input("  Dimension (e.g., 512): ") or "512")
            n_layers = int(input("  Layers (e.g., 4): ") or "4")
            n_heads = int(input("  Heads (e.g., 8): ") or "8")
        
        # Initialize model
        if is_geometric:
            model = GeometricCausalLM(
                vocab_size=tokenizer.vocab_size,
                dim=dim,
                n_layers=n_layers,
                n_heads=n_heads
            )
        else:
            model = StandardCausalLM(
                vocab_size=tokenizer.vocab_size,
                dim=dim,
                n_layers=n_layers,
                n_heads=n_heads
            )
        
        # Load state dict, stripping _orig_mod. prefix if present (from torch.compile)
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        
        print("✓ Model loaded successfully!")
        
        # Create conversation manager and response generator
        conversation_manager = ConversationManager(max_context_length=2048, max_turns=20)
        response_generator = ResponseGenerator(model, tokenizer, device, max_model_length=128)
        
        print("\n" + "-"*80)
        print("ENHANCED CHAT MODE")
        print("-"*80)
        print("Commands:")
        print("  'exit' or 'quit'  - Return to main menu")
        print("  'clear'           - Clear conversation history")
        print("  'save <name>'     - Save conversation")
        print("  'load <name>'     - Load conversation")
        print("  'stats'           - Show conversation statistics")
        print("-"*80 + "\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                break
            
            if user_input.lower() == 'clear':
                conversation_manager.clear()
                print("Conversation cleared.\n")
                continue
            
            if user_input.lower().startswith('save '):
                # Save conversation
                name = user_input[5:].strip() or f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                filepath = f"conversations/{name}.json"
                os.makedirs('conversations', exist_ok=True)
                conversation_manager.save_conversation(filepath)
                print(f"✓ Conversation saved to {filepath}\n")
                continue
            
            if user_input.lower().startswith('load '):
                # Load conversation
                name = user_input[5:].strip()
                filepath = f"conversations/{name}.json"
                try:
                    conversation_manager.load_conversation(filepath)
                    print(f"✓ Conversation loaded from {filepath}")
                    print(f"  {len(conversation_manager.turns)} turns loaded\n")
                except Exception as e:
                    print(f"✗ Error loading conversation: {e}\n")
                continue
            
            if user_input.lower() == 'stats':
                # Show statistics
                stats = conversation_manager.get_statistics()
                if stats:
                    print("\nConversation Statistics:")
                    print(f"  Total turns: {stats['total_turns']}")
                    print(f"  User turns: {stats['user_turns']}")
                    print(f"  Assistant turns: {stats['assistant_turns']}")
                    print(f"  Avg user length: {stats['avg_user_length']:.0f} chars")
                    print(f"  Avg assistant length: {stats['avg_assistant_length']:.0f} chars")
                    print()
                else:
                    print("No conversation data yet.\n")
                continue
            
            if not user_input:
                continue
            
            # Generate response using enhanced system
            try:
                response = response_generator.generate_conversation_turn(
                    conversation_manager,
                    user_input,
                    max_length=150,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=1.2
                )
                
                print(f"Model: {response}\n")
                
            except Exception as e:
                print(f"Error generating response: {e}\n")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to return to main menu...")


def classification_inference(checkpoint_path, is_geometric):
    """Inference mode for classification models"""
    print("\n" + "="*80)
    print("CLASSIFICATION INFERENCE")
    print("="*80)
    
    device = get_device(cuda_id=1)
    print(f"Loading model on {device}...")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Try to load architecture from checkpoint
        if 'model_config' in checkpoint and checkpoint['model_config']:
            model_cfg = checkpoint['model_config']
            dim = model_cfg.get('dim', 768)
            n_layers = model_cfg.get('n_layers', 6)
            n_heads = model_cfg.get('n_heads', 12)
            n_classes = model_cfg.get('n_classes', 2)
            print(f"✓ Loaded architecture from checkpoint: {dim}d, {n_layers}L, {n_heads}H")
        else:
            # Fallback: ask user
            print("\nNote: Architecture not in checkpoint. Please specify:")
            dim = int(input("  Dimension (e.g., 768): ") or "768")
            n_layers = int(input("  Layers (e.g., 6): ") or "6")
            n_heads = int(input("  Heads (e.g., 12): ") or "12")
            n_classes = int(input("  Number of classes (e.g., 2): ") or "2")
        
        # Initialize model
        if is_geometric:
            model = GeometricTransformer(
                vocab_size=tokenizer.vocab_size,
                dim=dim,
                n_layers=n_layers,
                n_heads=n_heads,
                n_classes=n_classes
            )
        else:
            model = StandardTransformer(
                vocab_size=tokenizer.vocab_size,
                dim=dim,
                n_layers=n_layers,
                n_heads=n_heads,
                n_classes=n_classes
            )
        
        # Load state dict, stripping _orig_mod. prefix if present (from torch.compile)
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        print("✓ Model loaded successfully!")
        print("\n" + "-"*80)
        print("CLASSIFICATION MODE")
        print("-"*80)
        print("Enter text to classify (type 'exit' to quit)")
        print("-"*80 + "\n")
        
        while True:
            text = input("Text: ").strip()
            
            if text.lower() in ['exit', 'quit', 'q']:
                break
            
            if not text:
                continue
            
            # Tokenize and predict
            try:
                encoding = tokenizer(
                    text,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(device)
                
                with torch.no_grad():
                    outputs = model(input_ids)
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                    
                    probs = torch.softmax(logits, dim=-1)[0]
                    pred = logits.argmax(dim=-1).item()
                
                print(f"Prediction: Class {pred}")
                print(f"Probabilities: {[f'{p:.3f}' for p in probs.cpu().numpy()]}\n")
                
            except Exception as e:
                print(f"Error: {e}\n")
    
    except Exception as e:
        print(f"Error: {e}")
    
    input("\nPress Enter to return to main menu...")


def compare_checkpoints():
    """Compare multiple checkpoints"""
    print("\n" + "="*80)
    print("COMPARE CHECKPOINTS")
    print("="*80)
    
    checkpoints = []
    
    print("\nSelect 2 or more checkpoints to compare")
    print("(Enter 0 when done selecting)\n")
    
    for i in range(10):  # Max 10 checkpoints
        ckpt = select_checkpoint(f"Select Checkpoint #{i+1}")
        if ckpt is None:
            break
        checkpoints.append(ckpt)
        print(f"✓ Added: {os.path.basename(ckpt)}")
        
        if i > 0:  # At least 2 selected
            more = input("\nAdd another checkpoint? [y/n]: ").strip().lower()
            if more != 'y':
                break
    
    if len(checkpoints) < 2:
        print("\nNeed at least 2 checkpoints to compare.")
        input("Press Enter to return to main menu...")
        return
    
    # Load and compare
    print("\n" + "="*80)
    print(f"COMPARISON RESULTS ({len(checkpoints)} models)")
    print("="*80)
    
    results = []
    for ckpt_path in checkpoints:
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            results.append({
                'name': os.path.basename(ckpt_path)[:40],
                'metric': ckpt.get('metric', 0),
                'epoch': ckpt.get('epoch', '?'),
                'model': ckpt.get('model_name', '?')
            })
        except:
            pass
    
    # Print comparison table
    print(f"\n{'Model':<42} {'Metric':>8} {'Epoch':>6} {'Type':>12}")
    print("-"*80)
    for r in results:
        print(f"{r['name']:<42} {r['metric']:>8.4f} {str(r['epoch']):>6} {r['model']:>12}")
    
    # Find best
    if results:
        best = max(results, key=lambda x: x['metric'])
        print("\n" + "="*80)
        print(f"Best Model: {best['name']}")
        print(f"Metric: {best['metric']:.4f}")
        print("="*80)
    
    input("\nPress Enter to return to main menu...")


def analyze_geometry():
    """Analyze geometry distribution of a model"""
    print("\n" + "="*80)
    print("ANALYZE MODEL GEOMETRY")
    print("="*80)
    
    checkpoint_path = select_checkpoint("Select Geometric Model")
    if not checkpoint_path:
        return
    
    # Check experiment directory for 'geometric', not filename
    exp_dir_name = Path(checkpoint_path).parent.parent.name
    if 'geometric' not in exp_dir_name.lower():
        print("\nWarning: This doesn't appear to be a geometric model.")
        print(f"  Experiment: {exp_dir_name}")
        cont = input("Continue anyway? [y/n]: ").strip().lower()
        if cont != 'y':
            return
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print("\n" + "-"*80)
        print("CHECKPOINT INFO")
        print("-"*80)
        print(f"Model: {checkpoint.get('model_name', 'unknown')}")
        print(f"Epoch: {checkpoint.get('epoch', '?')}")
        
        # Safe metric formatting
        metric = checkpoint.get('metric', '?')
        if isinstance(metric, (int, float)):
            print(f"Metric: {metric:.4f}")
        else:
            print(f"Metric: {metric}")
        
        print(f"Timestamp: {checkpoint.get('timestamp', 'unknown')}")
        
        # Try to auto-detect architecture first
        dim, n_layers, n_heads = None, None, None
        
        # Method 1: From checkpoint model_config
        if 'model_config' in checkpoint and checkpoint['model_config']:
            model_cfg = checkpoint['model_config']
            dim = model_cfg.get('dim')
            n_layers = model_cfg.get('n_layers')
            n_heads = model_cfg.get('n_heads')
        
        # Method 2: From experiment config.json
        if not dim:
            exp_dir = Path(checkpoint_path).parent.parent
            config_file = exp_dir / 'config.json'
            if config_file.exists():
                try:
                    with open(config_file) as f:
                        config = json.load(f)
                        dim = config.get('model', {}).get('dim')
                        n_layers = config.get('model', {}).get('n_layers')
                        n_heads = config.get('model', {}).get('n_heads')
                except:
                    pass
        
        # Method 3: Parse from directory name
        if not dim:
            import re
            match = re.search(r'(\d+)d_(\d+)L_(\d+)H', exp_dir_name)
            if match:
                dim, n_layers, n_heads = map(int, match.groups())
        
        # Method 4: Ask user only if all methods failed
        if not dim:
            print("\nArchitecture not found. Please specify:")
            dim = int(input("  Dimension (e.g., 512): ") or "512")
            n_layers = int(input("  Layers (e.g., 4): ") or "4")
            n_heads = int(input("  Heads (e.g., 8): ") or "8")
        else:
            print(f"\n✓ Auto-detected architecture: {dim}d, {n_layers}L, {n_heads}H")
        
        # Load a small amount of data for analysis
        print("\nLoading sample data for analysis...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create a small dummy dataset
        dummy_data = [{'text': f'Sample text {i}'} for i in range(100)]
        dataset = LanguageModelingDataset(dummy_data, tokenizer, max_length=128)
        loader = create_data_loader(dataset, batch_size=16, shuffle=False)
        
        # Initialize and load model
        model = GeometricCausalLM(
            vocab_size=tokenizer.vocab_size,
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads
        )
        
        # Load state dict, stripping _orig_mod. prefix if present (from torch.compile)
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        
        device = get_device(cuda_id=1)
        
        # Analyze curvatures
        print("\nAnalyzing learned geometries...")
        curv_results = analyze_curvatures(model, loader, device)
        
        if curv_results:
            print("\n" + "="*80)
            print("GEOMETRY DISTRIBUTION")
            print("="*80)
            
            n_hyp = curv_results['n_hyperbolic']
            n_euc = curv_results['n_euclidean']
            n_sph = curv_results['n_spherical']
            total = n_hyp + n_euc + n_sph
            
            print(f"\nTotal Attention Heads: {total}")
            print(f"  ({n_layers} layers × {n_heads} heads/layer)")
            print()
            print(f"Hyperbolic: {n_hyp:4d} ({curv_results['pct_hyperbolic']:5.1f}%) ■" + "■"*int(curv_results['pct_hyperbolic']/5))
            print(f"Euclidean:  {n_euc:4d} ({curv_results['pct_euclidean']:5.1f}%) ■" + "■"*int(curv_results['pct_euclidean']/5))
            print(f"Spherical:  {n_sph:4d} ({curv_results['pct_spherical']:5.1f}%) ■" + "■"*int(curv_results['pct_spherical']/5))
            
            if n_sph > 0:
                ratio = n_hyp / n_sph
                print(f"\nHyperbolic/Spherical Ratio: {ratio:.2f}")
                if 0.8 < ratio < 1.2:
                    print("✓ Universal 50/50 pattern detected!")
                else:
                    print("⚠ Deviation from expected 50/50 pattern")
            
            print("\nCurvature Statistics:")
            flat_curv = curv_results['flat_curvatures']
            print(f"  Mean: {flat_curv.mean():.4f}")
            print(f"  Std:  {flat_curv.std():.4f}")
            print(f"  Min:  {flat_curv.min():.4f}")
            print(f"  Max:  {flat_curv.max():.4f}")
        
    except Exception as e:
        print(f"\nError analyzing geometry: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to return to main menu...")


def run_comprehensive_experiments():
    """Run comprehensive research experiment suite"""
    print("\n" + "="*80)
    print("COMPREHENSIVE RESEARCH EXPERIMENTS")
    print("="*80)
    
    print("\nThis will run a systematic test across:")
    print("  • 4 Tasks: SST-2, MNLI, WikiText-2, WikiANN NER")
    print("  • 4 Sizes: Tiny (128d), Small (256d), Medium (512d), Large (768d)")
    print("  • = 16 experiments × 2 models = 32 training runs")
    print(f"\nEstimated time: 8-12 hours")
    print(f"Output: Auto-generated report with all results")
    
    confirm = input("\nStart comprehensive experiments? [y/n]: ").strip().lower()
    if confirm != 'y':
        input("\nPress Enter to return to main menu...")
        return
    
    # Import and run
    try:
        result = subprocess.run([sys.executable, 'run_comprehensive_experiments.py'], 
                               cwd=os.getcwd())
        
        if result.returncode == 0:
            print("\n✓ Experiments completed successfully!")
            print("  Check comprehensive_results/ for output")
        else:
            print(f"\n⚠️ Experiments exited with code {result.returncode}")
            
    except Exception as e:
        print(f"\nError running experiments: {e}")
    
    input("\nPress Enter to return to main menu...")


def visualize_experiment_results():
    """Visualize geometry evolution for a completed experiment"""
    print("\n" + "="*80)
    print("VISUALIZE EXPERIMENT RESULTS")
    print("="*80)
    
    # Find experiments with geometry logs
    experiments = {}
    
    if os.path.exists('experiments'):
        for exp_dir in Path('experiments').iterdir():
            if exp_dir.is_dir():
                geometry_log = exp_dir / 'logs' / 'geometry.jsonl'
                if geometry_log.exists():
                    experiments[exp_dir.name] = {
                        'path': str(exp_dir),
                        'mtime': os.path.getmtime(str(geometry_log))
                    }
    
    if not experiments:
        print("\n⚠️ No experiments with geometry logs found!")
        print("Tip: Train using [6]→[2] to create experiments with geometry tracking")
        input("\nPress Enter to return to menu...")
        return
    
    # Sort by modification time (newest first)
    sorted_experiments = sorted(experiments.items(), key=lambda x: x[1]['mtime'], reverse=True)
    
    # Display experiments
    print("\nAvailable Experiments:")
    print("-" * 80)
    
    for i, (exp_name, exp_data) in enumerate(sorted_experiments[:10], 1):
        mtime = datetime.fromtimestamp(exp_data['mtime'])
        print(f"  [{i}] {exp_name}")
        print(f"      Last updated: {mtime.strftime('%Y-%m-%d %H:%M')}")
    
    if len(sorted_experiments) > 10:
        print(f"\n  ... and {len(sorted_experiments) - 10} more")
    
    print("\n  [0] Cancel")
    
    # Get selection
    print()
    while True:
        try:
            choice = int(input(f"Select experiment [0-{min(10, len(sorted_experiments))}]: ").strip())
            if choice == 0:
                return
            if 1 <= choice <= min(10, len(sorted_experiments)):
                selected_exp = sorted_experiments[choice - 1]
                break
        except ValueError:
            pass
        print("Invalid choice. Try again.\n")
    
    exp_name, exp_data = selected_exp
    experiment_dir = Path(exp_data['path'])
    
    print(f"\n{'='*80}")
    print(f"Generating visualizations for: {exp_name}")
    print(f"{'='*80}\n")
    
    # Call the plot_geometry_evolution.py script
    viz_cmd = [sys.executable, 'plot_geometry_evolution.py', 
              '--experiment', str(experiment_dir)]
    
    try:
        result = subprocess.run(viz_cmd, cwd=os.getcwd())
        
        if result.returncode == 0:
            print(f"\n✅ Visualizations generated successfully!")
            print(f"   Saved to: {experiment_dir}/visualizations/")
        else:
            print(f"\n⚠️ Visualization generation failed with code {result.returncode}")
            
    except Exception as e:
        print(f"\nError generating visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to return to menu...")


def deep_geometry_analysis():
    """Run deep geometry analysis on completed experiment"""
    print("\n" + "="*80)
    print("DEEP GEOMETRY ANALYSIS")
    print("="*80)
    
    # Find experiments with geometry logs (same as visualize_experiment_results)
    experiments = {}
    
    if os.path.exists('experiments'):
        for exp_dir in Path('experiments').iterdir():
            if exp_dir.is_dir():
                geometry_log = exp_dir / 'logs' / 'geometry.jsonl'
                if geometry_log.exists():
                    experiments[exp_dir.name] = {
                        'path': str(exp_dir),
                        'mtime': os.path.getmtime(str(geometry_log))
                    }
    
    if not experiments:
        print("\n⚠️ No experiments with geometry logs found!")
        print("Tip: Train using [6]→[2] to create experiments with geometry tracking")
        input("\nPress Enter to return to menu...")
        return
    
    # Sort by modification time (newest first)
    sorted_experiments = sorted(experiments.items(), key=lambda x: x[1]['mtime'], reverse=True)
    
    # Display experiments
    print("\nAvailable Experiments:")
    print("-" * 80)
    
    for i, (exp_name, exp_data) in enumerate(sorted_experiments[:10], 1):
        mtime = datetime.fromtimestamp(exp_data['mtime'])
        print(f"  [{i}] {exp_name}")
        print(f"      Last updated: {mtime.strftime('%Y-%m-%d %H:%M')}")
    
    if len(sorted_experiments) > 10:
        print(f"\n  ... and {len(sorted_experiments) - 10} more")
    
    print("\n  [0] Cancel")
    
    # Get selection
    print()
    while True:
        try:
            choice = int(input(f"Select experiment [0-{min(10, len(sorted_experiments))}]: ").strip())
            if choice == 0:
                return
            if 1 <= choice <= min(10, len(sorted_experiments)):
                selected_exp = sorted_experiments[choice - 1]
                break
        except ValueError:
            pass
        print("Invalid choice. Try again.\n")
    
    exp_name, exp_data = selected_exp
    experiment_dir = Path(exp_data['path'])
    
    print(f"\n{'='*80}")
    print(f"Running deep analysis for: {exp_name}")
    print(f"{'='*80}\n")
    
    # Call analyze_geometry_deep.py script
    analysis_cmd = [sys.executable, 'analyze_geometry_deep.py', 
                   '--experiment', str(experiment_dir)]
    
    try:
        result = subprocess.run(analysis_cmd, cwd=os.getcwd())
        
        if result.returncode == 0:
            print(f"\n✅ Deep analysis completed successfully!")
            print(f"   Results: {experiment_dir}/deep_analysis/")
            print(f"   Report: {experiment_dir}/deep_analysis/summary.md")
        else:
            print(f"\n⚠️ Analysis failed with code {result.returncode}")
            
    except Exception as e:
        print(f"\nError running deep analysis: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to return to menu...")


def run_logging_experiment():
    """Run experiment with comprehensive geometry logging"""
    print("\n" + "="*80)
    print("LOGGING EXPERIMENT - GEOMETRY EVOLUTION TRACKING")
    print("="*80)
    
    print("\nThis trains a geometric model with detailed logging to track")
    print("curvature evolution and detect phase transitions (e.g., Euclidean")
    print("emergence around epoch 90-100).")
    print("\nNote: Only geometric models supported (not standard models)")
    
    # Configuration
    config = {
        'dim': 128,
        'epochs': 200,
        'full_dataset': True,
        'compile': True
    }
    
    # Interactive configuration
    print("\n" + "-"*80)
    print("CONFIGURATION")
    print("-"*80)
    
    # Select dimension
    print("\nModel Dimension:")
    print("  [1] 128d (Tiny - 1L, 2H) - Fast, good for testing")
    print("  [2] 256d (Small - 2L, 4H) - Balanced")
    print("  [3] 512d (Medium - 4L, 8H) - Better quality")
    print("  [4] 768d (Large - 6L, 12H) - Best quality")
    
    dim_choice = input("\nSelect dimension [1-4] (default: 1): ").strip() or '1'
    dim_map = {'1': 128, '2': 256, '3': 512, '4': 768}
    config['dim'] = dim_map.get(dim_choice, 128)
    
    # Select epochs
    print("\nNumber of Epochs:")
    print("  [1] 50  - Quick test (~20-30 min)")
    print("  [2] 100 - Medium run (~1 hour)")
    print("  [3] 200 - Capture phase transition (~2-3 hours)")
    print("  [4] Custom - Enter your own")
    
    epoch_choice = input("\nSelect epochs [1-4] (default: 3): ").strip() or '3'
    epoch_map = {'1': 50, '2': 100, '3': 200}
    if epoch_choice in epoch_map:
        config['epochs'] = epoch_map[epoch_choice]
    elif epoch_choice == '4':
        try:
            config['epochs'] = int(input("  Enter number of epochs: "))
        except ValueError:
            config['epochs'] = 200
    else:
        config['epochs'] = 200
    
    # Full dataset?
    full_input = input("\nUse full WikiText-2 dataset? [Y/n]: ").strip().lower()
    config['full_dataset'] = full_input != 'n'
    
    # torch.compile?
    compile_input = input("\nUse torch.compile() for ~2x speedup? [Y/n]: ").strip().lower()
    config['compile'] = compile_input != 'n'
    
    # Summary
    print("\n" + "="*80)
    print("CONFIGURATION SUMMARY")
    print("="*80)
    print(f"  Model: {config['dim']}d")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Dataset: {'Full WikiText-2' if config['full_dataset'] else 'Subset (5000 samples)'}")
    print(f"  torch.compile: {'Yes' if config['compile'] else 'No'}")
    print("\nFeatures:")
    print("  • Geometry logged every 10 steps")
    print("  • Checkpoints saved every 10 epochs")
    print("  • Phase transition detection")
    print("  • Auto-visualization option")
    print("="*80)
    
    confirm = input("\nStart logging experiment? [y/n]: ").strip().lower()
    if confirm != 'y':
        print("Experiment cancelled.")
        input("\nPress Enter to return to main menu...")
        return
    
    # Build command
    cmd = [sys.executable, 'test_logging_system.py', '--epochs', str(config['epochs']), 
           '--dim', str(config['dim'])]
    
    if config['full_dataset']:
        cmd.append('--full-dataset')
    if config['compile']:
        cmd.append('--compile')
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    
    # Run experiment
    try:
        result = subprocess.run(cmd, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("\n✅ Logging experiment completed successfully!")
            
            # Find the most recent experiment directory
            exp_dirs = sorted(Path('experiments').glob('geometric_*'), key=os.path.getmtime, reverse=True)
            if exp_dirs:
                latest_exp = exp_dirs[0]
                print(f"\nExperiment data: {latest_exp}")
                
                # Offer visualization
                viz_input = input("\nGenerate visualizations now? [Y/n]: ").strip().lower()
                if viz_input != 'n':
                    print("\nGenerating visualizations...")
                    viz_cmd = [sys.executable, 'plot_geometry_evolution.py', 
                              '--experiment', str(latest_exp), '--no-show']
                    viz_result = subprocess.run(viz_cmd, cwd=os.getcwd())
                    
                    if viz_result.returncode == 0:
                        print(f"\n✓ Visualizations saved to: {latest_exp}/visualizations/")
                    else:
                        print("\n⚠️ Visualization generation failed")
        else:
            print(f"\n⚠️ Experiment exited with code {result.returncode}")
            
    except Exception as e:
        print(f"\nError running logging experiment: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to return to main menu...")


def continue_training():
    """Continue training from loaded checkpoint (menu option [4])"""
    if not LOADED_MODEL_STATE['loaded']:
        print("\n⚠️ No model loaded!")
        print("Please use [2] Load & Analyze Checkpoint first.")
        input("\nPress Enter to return to main menu...")
        return
    
    print("\n" + "="*80)
    print("CONTINUE TRAINING FROM CHECKPOINT")
    print("="*80)
    
    # Show loaded model info
    print(f"\n📦 Loaded Model:")
    print(f"  Type: {LOADED_MODEL_STATE['model_type'].upper()}")
    print(f"  Architecture: {LOADED_MODEL_STATE['dim']}d, {LOADED_MODEL_STATE['n_layers']}L, {LOADED_MODEL_STATE['n_heads']}H")
    print(f"  Currently at Epoch: {LOADED_MODEL_STATE['epochs_trained']}")
    print(f"  Dataset: {LOADED_MODEL_STATE['dataset']}")
    
    # Dataset selection
    print("\n" + "-"*80)
    print("DATASET SELECTION")
    print("-"*80)
    print(f"\nCurrent dataset: {LOADED_MODEL_STATE['dataset']}")
    print("\n  [1] Use same dataset")
    print("  [2] Select different dataset")
    
    dataset_choice = input("\nYour choice [1-2] (default: 1): ").strip() or '1'
    
    use_custom_dataset = False
    dataset_info = None
    
    if dataset_choice == '2':
        dataset_info = select_dataset_source('language_modeling')
        if dataset_info is None:
            print("No dataset selected, cancelling.")
            input("\nPress Enter to return to main menu...")
            return
        use_custom_dataset = True
        print(f"✓ Selected dataset: {dataset_info.get('name', 'custom')}")
    
    # Get additional epochs
    print("\n" + "-"*80)
    print("TRAINING CONFIGURATION")
    print("-"*80)
    
    try:
        additional_epochs = int(input(f"\nAdditional epochs to train (default: 50): ").strip() or "50")
    except ValueError:
        additional_epochs = 50
    
    # torch.compile option
    compile_input = input("\nUse torch.compile() for ~2x speedup? [Y/n]: ").strip().lower()
    use_compile = compile_input != 'n'
    
    # Summary
    current_epoch = LOADED_MODEL_STATE['epochs_trained']
    final_epoch = current_epoch + additional_epochs
    
    print("\n" + "="*80)
    print("CONTINUE TRAINING SUMMARY")
    print("="*80)
    print(f"  Starting epoch: {current_epoch}")
    print(f"  Additional epochs: {additional_epochs}")
    print(f"  Final epoch: {final_epoch}")
    print(f"  Dataset: {LOADED_MODEL_STATE['dataset']}")
    print(f"  torch.compile: {'Yes' if use_compile else 'No'}")
    print("\n  Note: Training will continue in same experiment directory")
    print("="*80)
    
    confirm = input("\nContinue training? [y/n]: ").strip().lower()
    if confirm != 'y':
        print("Training cancelled.")
        input("\nPress Enter to return to main menu...")
        return
    
    # Build command for test_logging_system with resume flags
    checkpoint_path = LOADED_MODEL_STATE['checkpoint_path']
    dim = LOADED_MODEL_STATE['dim']
    n_layers = LOADED_MODEL_STATE['n_layers']
    n_heads = LOADED_MODEL_STATE['n_heads']
    
    cmd = [
        sys.executable, 'test_logging_system.py',
        '--resume-from-checkpoint', checkpoint_path,
        '--epochs', str(final_epoch),  # Total target epochs
        '--dim', str(dim),
        '--n-layers', str(n_layers),
        '--n-heads', str(n_heads)
    ]
    
    if use_compile:
        cmd.append('--compile')
    
    # Add dataset parameters
    if use_custom_dataset and dataset_info:
        # Custom dataset selected
        if dataset_info['loader'] == 'custom':
            cmd.extend(['--dataset-name', dataset_info['name']])
            if dataset_info.get('config'):
                cmd.extend(['--dataset-config', dataset_info['config']])
            # Note: Field selection would need to be saved from the selection process
            # For now, fields will auto-detect when loading
        elif dataset_info['loader'] == 'wikitext':
            cmd.append('--full-dataset')
    else:
        # Use same dataset as original training
        dataset = LOADED_MODEL_STATE.get('dataset', 'wikitext-2-raw-v1')
        if 'wikitext' in dataset.lower():
            cmd.append('--full-dataset')
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    print("="*80 + "\n")
    
    try:
        # Run from current directory (already in geometric-attention/)
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print("\n" + "="*80)
            print("✅ Continue training completed successfully!")
            print("="*80)
        else:
            print(f"\n⚠️ Training exited with code {result.returncode}")
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to return to main menu...")


def main():
    """Main CLI shell loop"""
    while True:
        print_header()
        choice = main_menu()
        
        if choice == '1':
            train_new_model()
        elif choice == '2':
            load_and_analyze_checkpoint()
        elif choice == '3':
            load_model_and_chat()
        elif choice == '4':
            continue_training()
        elif choice == '5':
            compare_checkpoints()
        elif choice == '6':
            # Geometry analysis sub-menu
            while True:
                geo_choice = geometry_analysis_menu()
                if geo_choice == '0':
                    break  # Back to main menu
                elif geo_choice == '1':
                    analyze_geometry()
                elif geo_choice == '2':
                    run_logging_experiment()
                elif geo_choice == '3':
                    visualize_experiment_results()
                elif geo_choice == '4':
                    deep_geometry_analysis()
                else:
                    print("\nInvalid choice. Please try again.")
                    time.sleep(1)
        elif choice == '7':
            print("\nExiting. Goodbye!")
            sys.exit(0)
        else:
            print("\nInvalid choice. Please try again.")
            time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
