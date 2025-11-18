#!/usr/bin/env python3
"""
Benchmark script to test various optimization strategies.
Tests torch.compile, mixed precision, cudnn settings, etc.
WITHOUT modifying working code.
"""

import torch
import torch.nn as nn
import time
from transformers import AutoTokenizer
from geometric_attention.models.language_models import GeometricCausalLM, StandardCausalLM
from geometric_attention.data import LanguageModelingDataset, load_wikitext_dataset, create_data_loader
from geometric_attention.utils import set_seed, get_device
import torch.cuda.amp as amp


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_forward_pass(model, test_batch, device, n_iterations=50):
    """Benchmark forward pass speed"""
    model.eval()
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(test_batch)
    
    torch.cuda.synchronize()
    
    # Actual benchmark
    start = time.time()
    for _ in range(n_iterations):
        with torch.no_grad():
            _ = model(test_batch)
    torch.cuda.synchronize()
    
    elapsed = time.time() - start
    return elapsed / n_iterations


def benchmark_training_step(model, loader, optimizer, criterion, device, scaler=None):
    """Benchmark one training epoch"""
    model.train()
    start = time.time()
    
    step_count = 0
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        if scaler:  # Mixed precision
            with amp.autocast():
                _, loss, _ = model(input_ids, labels=labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:  # Normal precision
            _, loss, _ = model(input_ids, labels=labels)
            loss.backward()
            optimizer.step()
        
        step_count += 1
        if step_count >= 100:  # Only do 100 steps for benchmark
            break
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    return elapsed, step_count


def main():
    print("="*80)
    print("  OPTIMIZATION BENCHMARK SUITE")
    print("  Testing: compile, AMP, cudnn, batch sizes")
    print("="*80)
    
    set_seed(42)
    device = get_device(cuda_id=1)
    
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device.index)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load small dataset for testing
    print("\nLoading WikiText-2 for benchmarking...")
    wikitext = load_wikitext_dataset("wikitext-2-raw-v1")
    
    # Use subset for quick testing
    train_dataset = LanguageModelingDataset(
        wikitext['train'].select(range(1000)),  # Just 1000 samples
        tokenizer,
        max_length=128
    )
    
    print(f"Dataset: {len(train_dataset)} samples")
    
    # Test model - medium size
    test_config = {
        'dim': 512,
        'n_layers': 4,
        'n_heads': 8
    }
    
    print(f"\nTest model: {test_config['dim']}d, {test_config['n_layers']}L, {test_config['n_heads']}H")
    
    # Prepare test batch
    test_loader = create_data_loader(train_dataset, batch_size=16, shuffle=False)
    test_batch = next(iter(test_loader))['input_ids'].to(device)
    
    print(f"Test batch shape: {test_batch.shape}")
    
    results = {}
    
    # ========================================================================
    # TEST 1: Baseline (no optimizations)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 1: BASELINE (no optimizations)")
    print("="*80)
    
    model = GeometricCausalLM(
        vocab_size=tokenizer.vocab_size,
        **test_config
    ).to(device)
    
    print(f"Parameters: {count_params(model):,}")
    
    baseline_time = benchmark_forward_pass(model, test_batch, device)
    results['baseline'] = baseline_time
    
    print(f"✓ Forward pass: {baseline_time*1000:.2f} ms/batch")
    
    del model
    torch.cuda.empty_cache()
    
    # ========================================================================
    # TEST 2: cudnn.benchmark = True
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 2: cudnn.benchmark = True")
    print("="*80)
    
    torch.backends.cudnn.benchmark = True
    
    model = GeometricCausalLM(
        vocab_size=tokenizer.vocab_size,
        **test_config
    ).to(device)
    
    cudnn_time = benchmark_forward_pass(model, test_batch, device)
    results['cudnn'] = cudnn_time
    
    speedup = baseline_time / cudnn_time
    print(f"✓ Forward pass: {cudnn_time*1000:.2f} ms/batch")
    print(f"  Speedup: {speedup:.2f}x")
    
    del model
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    
    # ========================================================================
    # TEST 3: Mixed Precision (AMP)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 3: Automatic Mixed Precision (FP16)")
    print("="*80)
    
    model = GeometricCausalLM(
        vocab_size=tokenizer.vocab_size,
        **test_config
    ).to(device)
    
    # Test with AMP
    model.eval()
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            with amp.autocast():
                _ = model(test_batch)
    
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(50):
        with torch.no_grad():
            with amp.autocast():
                _ = model(test_batch)
    torch.cuda.synchronize()
    
    amp_time = (time.time() - start) / 50
    results['amp'] = amp_time
    
    speedup = baseline_time / amp_time
    print(f"✓ Forward pass: {amp_time*1000:.2f} ms/batch")
    print(f"  Speedup: {speedup:.2f}x")
    
    del model
    torch.cuda.empty_cache()
    
    # ========================================================================
    # TEST 4: torch.compile() (PyTorch 2.0+)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 4: torch.compile() optimization")
    print("="*80)
    
    if hasattr(torch, 'compile'):
        model = GeometricCausalLM(
            vocab_size=tokenizer.vocab_size,
            **test_config
        ).to(device)
        
        # Compile model
        print("Compiling model (this may take a minute)...")
        compiled_model = torch.compile(model, mode='reduce-overhead')
        
        compile_time = benchmark_forward_pass(compiled_model, test_batch, device)
        results['compile'] = compile_time
        
        speedup = baseline_time / compile_time
        print(f"✓ Forward pass: {compile_time*1000:.2f} ms/batch")
        print(f"  Speedup: {speedup:.2f}x")
        
        del model, compiled_model
        torch.cuda.empty_cache()
    else:
        print("torch.compile() not available (need PyTorch 2.0+)")
        results['compile'] = None
    
    # ========================================================================
    # TEST 5: Combined (cudnn + AMP)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 5: cudnn.benchmark + AMP (combined)")
    print("="*80)
    
    torch.backends.cudnn.benchmark = True
    
    model = GeometricCausalLM(
        vocab_size=tokenizer.vocab_size,
        **test_config
    ).to(device)
    
    model.eval()
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            with amp.autocast():
                _ = model(test_batch)
    
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(50):
        with torch.no_grad():
            with amp.autocast():
                _ = model(test_batch)
    torch.cuda.synchronize()
    
    combined_time = (time.time() - start) / 50
    results['combined'] = combined_time
    
    speedup = baseline_time / combined_time
    print(f"✓ Forward pass: {combined_time*1000:.2f} ms/batch")
    print(f"  Speedup: {speedup:.2f}x")
    
    del model
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    
    # ========================================================================
    # TEST 6: Training Step Benchmark
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 6: Full Training Step (100 iterations)")
    print("="*80)
    
    train_loader = create_data_loader(train_dataset, batch_size=16, shuffle=True)
    
    # Baseline training
    print("\nBaseline training (FP32)...")
    model = GeometricCausalLM(vocab_size=tokenizer.vocab_size, **test_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    baseline_train_time, steps = benchmark_training_step(model, train_loader, optimizer, criterion, device)
    print(f"✓ Baseline: {baseline_train_time:.2f}s for {steps} steps ({baseline_train_time/steps*1000:.1f} ms/step)")
    
    del model, optimizer
    torch.cuda.empty_cache()
    
    # AMP training
    print("\nAMP training (FP16)...")
    model = GeometricCausalLM(vocab_size=tokenizer.vocab_size, **test_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = amp.GradScaler()
    
    amp_train_time, steps = benchmark_training_step(model, train_loader, optimizer, criterion, device, scaler)
    print(f"✓ AMP: {amp_train_time:.2f}s for {steps} steps ({amp_train_time/steps*1000:.1f} ms/step)")
    print(f"  Speedup: {baseline_train_time/amp_train_time:.2f}x")
    
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    
    # ========================================================================
    # RESULTS SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("="*80)
    
    print(f"\n{'Optimization':<30} {'Time (ms)':<12} {'Speedup':<10} {'Recommendation':<20}")
    print("-"*80)
    
    baseline_ms = baseline_time * 1000
    
    optimizations = [
        ('Baseline (no opts)', baseline_time, 1.0, 'Current'),
        ('cudnn.benchmark', results.get('cudnn'), results.get('cudnn') and baseline_time/results['cudnn'], 'Easy win'),
        ('Mixed Precision (AMP)', results.get('amp'), results.get('amp') and baseline_time/results['amp'], 'Recommended'),
        ('torch.compile()', results.get('compile'), results.get('compile') and baseline_time/results['compile'], 'Requires PyTorch 2.0+'),
        ('cudnn + AMP', results.get('combined'), results.get('combined') and baseline_time/results['combined'], 'Best combination'),
    ]
    
    for name, time_val, speedup_val, rec in optimizations:
        if time_val:
            time_str = f"{time_val*1000:.2f}"
            speedup_str = f"{speedup_val:.2f}x" if speedup_val else "-"
        else:
            time_str = "N/A"
            speedup_str = "N/A"
        
        print(f"{name:<30} {time_str:<12} {speedup_str:<10} {rec:<20}")
    
    # ========================================================================
    # RECOMMENDATIONS
    # ========================================================================
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR PRODUCTION")
    print("="*80)
    
    best_speedup = max([s for _, _, s, _ in optimizations if s and s > 1.0], default=1.0)
    
    print(f"\n🎯 Expected Speedup: {best_speedup:.2f}x")
    
    if best_speedup > 1.5:
        print("✅ Significant speedup achievable!")
    else:
        print("⚠️ Limited speedup available with standard optimizations")
    
    print("\n📝 To Apply:")
    print("  1. Add cudnn.benchmark = True at start of training")
    print("  2. Wrap model forward with torch.cuda.amp.autocast()")
    print("  3. Use GradScaler for mixed precision training")
    print("  4. Consider torch.compile() if PyTorch 2.0+")
    
    print("\n💡 Code Example:")
    print("""
    # At start of training script:
    torch.backends.cudnn.benchmark = True
    
    # Create scaler for mixed precision:
    scaler = torch.cuda.amp.GradScaler()
    
    # In training loop:
    with torch.cuda.amp.autocast():
        logits, loss, _ = model(input_ids, labels=labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    """)
    
    # ========================================================================
    # MEMORY USAGE TEST
    # ========================================================================
    print("\n" + "="*80)
    print("MEMORY USAGE COMPARISON")
    print("="*80)
    
    configs_to_test = [
        {'dim': 512, 'n_layers': 4, 'n_heads': 8, 'batch_size': 16},
        {'dim': 768, 'n_layers': 6, 'n_heads': 12, 'batch_size': 8},
        {'dim': 768, 'n_layers': 6, 'n_heads': 12, 'batch_size': 4},
    ]
    
    print(f"\n{'Config':<30} {'Batch':<8} {'FP32 (GB)':<12} {'FP16 (GB)':<12} {'Fits?':<8}")
    print("-"*80)
    
    for cfg in configs_to_test:
        config_str = f"{cfg['dim']}d, {cfg['n_layers']}L, {cfg['n_heads']}H"
        batch_size = cfg['batch_size']
        
        # Test FP32
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            model = GeometricCausalLM(
                vocab_size=tokenizer.vocab_size,
                dim=cfg['dim'],
                n_layers=cfg['n_layers'],
                n_heads=cfg['n_heads']
            ).to(device)
            
            test_loader = create_data_loader(train_dataset, batch_size=batch_size, shuffle=False)
            test_batch = next(iter(test_loader))
            input_ids = test_batch['input_ids'].to(device)
            labels = test_batch['labels'].to(device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            
            optimizer.zero_grad()
            _, loss, _ = model(input_ids, labels=labels)
            loss.backward()
            optimizer.step()
            
            torch.cuda.synchronize()
            
            fp32_memory = torch.cuda.max_memory_allocated() / 1e9
            fp32_fits = "✓"
            
            del model, optimizer
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                fp32_memory = ">24"
                fp32_fits = "✗ OOM"
            else:
                raise
        
        # Test FP16
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            model = GeometricCausalLM(
                vocab_size=tokenizer.vocab_size,
                dim=cfg['dim'],
                n_layers=cfg['n_layers'],
                n_heads=cfg['n_heads']
            ).to(device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scaler = amp.GradScaler()
            
            optimizer.zero_grad()
            with amp.autocast():
                _, loss, _ = model(input_ids, labels=labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            torch.cuda.synchronize()
            
            fp16_memory = torch.cuda.max_memory_allocated() / 1e9
            fp16_fits = "✓"
            
            del model, optimizer, scaler
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                fp16_memory = ">24"
                fp16_fits = "✗ OOM"
            else:
                raise
        
        fp32_str = f"{fp32_memory:.2f}" if isinstance(fp32_memory, float) else fp32_memory
        fp16_str = f"{fp16_memory:.2f}" if isinstance(fp16_memory, float) else fp16_memory
        
        print(f"{config_str:<30} {batch_size:<8} {fp32_str:<12} {fp16_str:<12} {fp32_fits if fp32_fits == '✓' else fp16_fits:<8}")
    
    # ========================================================================
    # FINAL RECOMMENDATIONS
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL RECOMMENDATIONS")
    print("="*80)
    
    print("\n🚀 For 768d, 6L model on 3090 (24GB):")
    print("  ✓ Use batch_size=4 with FP32")
    print("  ✓ Use batch_size=8 with FP16 (mixed precision)")
    print("  ✓ Enable cudnn.benchmark = True")
    print("  ✓ Use gradient accumulation if need larger effective batch")
    
    print("\n⚡ Best Performance Setup:")
    print("  1. torch.backends.cudnn.benchmark = True")
    print("  2. Mixed precision (AMP) for ~1.5-2x speedup")
    print("  3. Batch size 8 (with FP16)")
    print("  4. Gradient accumulation (accumulate 2-4 batches)")
    
    print("\n📊 Expected Results:")
    if best_speedup > 1:
        print(f"  Current: ~30 min for 5 epochs")
        print(f"  With opts: ~{30/best_speedup:.0f} min for 5 epochs")
        print(f"  Savings: {30 - 30/best_speedup:.0f} minutes")
    
    print("\n✓ Benchmark complete!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
