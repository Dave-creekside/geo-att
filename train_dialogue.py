#!/usr/bin/env python3
"""
Specialized training script for multi-turn dialogue using geometric attention.
Uses ConversationDataset for proper multi-turn handling.
"""

import torch
import time
import argparse
from transformers import AutoTokenizer
from geometric_attention.models.language_models import GeometricCausalLM, StandardCausalLM
from geometric_attention.dialogue import ConversationDataset, create_conversation_splits
from geometric_attention.data import create_data_loader
from geometric_attention.training import Trainer
from geometric_attention.utils import set_seed, get_device, print_model_summary
import json


def load_conversation_data(filepath: str):
    """Load conversations from file"""
    ext = filepath.split('.')[-1].lower()
    
    conversations = []
    
    if ext == 'jsonl':
        with open(filepath, 'r') as f:
            for line in f:
                conversations.append(json.loads(line))
    elif ext == 'json':
        with open(filepath, 'r') as f:
            conversations = json.load(f)
    elif ext in ['txt', 'text']:
        with open(filepath, 'r') as f:
            # Assume each paragraph is a conversation
            content = f.read()
            conversations = [{'text': conv.strip()} for conv in content.split('\n\n') if conv.strip()]
    
    return conversations


def main():
    parser = argparse.ArgumentParser(description='Train dialogue models with geometric attention')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to conversation data file')
    parser.add_argument('--dim', type=int, default=512,
                       help='Model dimension (default: 512)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--compile', action='store_true',
                       help='Use torch.compile() for speedup')
    parser.add_argument('--format', type=str, default='auto',
                       choices=['auto', 'simple', 'chatml', 'alpaca'],
                       help='Conversation format (default: auto)')
    args = parser.parse_args()
    
    print("="*80)
    print("  DIALOGUE TRAINING WITH GEOMETRIC ATTENTION")
    print("="*80)
    
    config = {
        'data_file': args.data,
        'model_dim': args.dim,
        'n_layers': 6,
        'n_heads': 8,
        'n_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': 1e-4,
        'warmup_steps': 500,
        'max_seq_len': 512,  # Longer for dialogue context
        'format': args.format,
        'seed': 42
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Setup
    set_seed(config['seed'])
    device = get_device(cuda_id=1)
    
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device.index)}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load conversation data
    print(f"\nLoading conversations from: {args.data}")
    conversations = load_conversation_data(args.data)
    print(f"✓ Loaded {len(conversations)} conversations")
    
    # Split data
    train_convs, val_convs, test_convs = create_conversation_splits(conversations)
    print(f"  Train: {len(train_convs)}")
    print(f"  Val: {len(val_convs)}")
    print(f"  Test: {len(test_convs)}")
    
    # Create datasets with conversation-specific handling
    print("\nCreating conversation datasets...")
    train_dataset = ConversationDataset(
        train_convs, 
        tokenizer, 
        max_length=config['max_seq_len'],
        format_type=config['format'],
        mask_user_turns=True  # Only train on assistant responses
    )
    val_dataset = ConversationDataset(
        val_convs,
        tokenizer,
        max_length=config['max_seq_len'],
        format_type=config['format'],
        mask_user_turns=True
    )
    
    print(f"✓ Datasets created")
    
    # Create loaders
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
    # TRAIN GEOMETRIC MODEL
    # ========================================================================
    print("\n" + "="*80)
    print("Training Geometric Dialogue Model...")
    print("="*80)
    
    geometric_model = GeometricCausalLM(
        vocab_size=tokenizer.vocab_size,
        dim=config['model_dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        max_seq_len=config['max_seq_len'],
        dropout=0.1
    )
    print_model_summary(geometric_model, "Geometric Dialogue Model")
    
    start_time = time.time()
    geo_trainer = Trainer(
        geometric_model, 
        device=device,
        model_name="geometric_dialogue",
        use_compile=args.compile
    )
    
    geo_history = geo_trainer.train(
        train_loader,
        val_loader,
        n_epochs=config['n_epochs'],
        learning_rate=config['learning_rate'],
        warmup_steps=config['warmup_steps'],
        task_type='language_modeling',
        save_best=True,
        save_final=True
    )
    geo_time = time.time() - start_time
    
    # Free memory
    del geo_trainer
    geometric_model.cpu()
    torch.cuda.empty_cache()
    
    # ========================================================================
    # TRAIN STANDARD MODEL FOR COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("Training Standard Dialogue Model...")
    print("="*80)
    
    standard_model = StandardCausalLM(
        vocab_size=tokenizer.vocab_size,
        dim=config['model_dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        max_seq_len=config['max_seq_len'],
        dropout=0.1
    )
    print_model_summary(standard_model, "Standard Dialogue Model")
    
    start_time = time.time()
    std_trainer = Trainer(
        standard_model,
        device=device,
        model_name="standard_dialogue",
        use_compile=args.compile
    )
    
    std_history = std_trainer.train(
        train_loader,
        val_loader,
        n_epochs=config['n_epochs'],
        learning_rate=config['learning_rate'],
        warmup_steps=config['warmup_steps'],
        task_type='language_modeling',
        save_best=True,
        save_final=True
    )
    std_time = time.time() - start_time
    
    # Move geometric back for testing
    geometric_model.to(device)
    
    # ========================================================================
    # RESULTS
    # ========================================================================
    print("\n" + "="*80)
    print("DIALOGUE TRAINING RESULTS")
    print("="*80)
    
    geo_best_ppl = min(geo_history['val_acc'])
    std_best_ppl = min(std_history['val_acc'])
    
    print(f"\nGeometric Model:")
    print(f"  Best Validation Perplexity: {geo_best_ppl:.2f}")
    print(f"  Training Time: {geo_time/60:.1f} minutes")
    
    print(f"\nStandard Model:")
    print(f"  Best Validation Perplexity: {std_best_ppl:.2f}")
    print(f"  Training Time: {std_time/60:.1f} minutes")
    
    print(f"\nImprovement: {(std_best_ppl - geo_best_ppl):.2f} perplexity (lower is better)")
    
    # Test generation on sample prompts
    print("\n" + "="*80)
    print("SAMPLE DIALOGUE GENERATIONS")
    print("="*80)
    
    from geometric_attention.dialogue import ResponseGenerator
    
    geo_generator = ResponseGenerator(geometric_model, tokenizer, device)
    std_generator = ResponseGenerator(standard_model, tokenizer, device)
    
    test_prompts = [
        "User: Hello! How are you today?\nAssistant:",
        "User: Can you explain geometric attention?\nAssistant:",
        "User: What makes it different from standard transformers?\nAssistant:",
    ]
    
    for prompt in test_prompts:
        print(f"\n{'='*80}")
        print(f"Prompt: {prompt.split('Assistant:')[0]}...")
        print(f"{'='*80}")
        
        print("\n🔷 Geometric Model:")
        geo_response = geo_generator.generate_response(prompt, max_length=80, temperature=0.8)
        print(f"  {geo_response}")
        
        print("\n🔶 Standard Model:")
        std_response = std_generator.generate_response(prompt, max_length=80, temperature=0.8)
        print(f"  {std_response}")
    
    # Save results
    results = {
        'config': config,
        'geometric': {
            'best_perplexity': float(geo_best_ppl),
            'training_time': geo_time
        },
        'standard': {
            'best_perplexity': float(std_best_ppl),
            'training_time': std_time
        },
        'data_stats': {
            'train_conversations': len(train_convs),
            'val_conversations': len(val_convs),
            'test_conversations': len(test_convs)
        }
    }
    
    with open('dialogue_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("✓ Dialogue training complete!")
    print("  Results saved to dialogue_training_results.json")
    print("  Checkpoints saved in checkpoints/")
    print("="*80)
    
    print("\n💡 To chat with your trained model:")
    print("  python main.py")
    print("  → Select [2] Load Model & Chat")
    print("  → Choose best_geometric_dialogue_*.pt")


if __name__ == "__main__":
    main()
