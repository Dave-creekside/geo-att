# Geometric Attention Project - Complete Overview

**Last Updated:** October 28, 2025

## What This Is

Implementation of transformer architectures where each attention head learns its optimal geometry (hyperbolic, spherical, or Euclidean) through a learnable curvature parameter. Started from a Jupyter notebook, reorganized into a production-ready Python package.

## Key Observation

In initial experiments (SST-2, MNLI, WikiText-2, NER), models consistently learned approximately 50% hyperbolic heads and 50% spherical heads, with less than 2% Euclidean. This pattern appeared across different tasks and model sizes (2 heads to 72 heads).

## Project Structure

```
geometric-attention/
├── geometric_attention/          # Core package
│   ├── models/                   # Model architectures
│   │   ├── geometric_attention.py    # Core attention mechanisms
│   │   ├── transformers.py          # Full transformer models
│   │   └── language_models.py       # LM-specific variants
│   ├── data/                     # Dataset handling
│   │   └── datasets.py
│   ├── dialogue/                 # Multi-turn conversation (NEW)
│   │   ├── conversation_dataset.py
│   │   ├── conversation_manager.py
│   │   └── response_generator.py
│   ├── training/                 # Training infrastructure
│   │   ├── trainer.py
│   │   └── evaluation.py
│   └── utils/                    # Utilities
│       ├── visualization.py
│       └── helpers.py
├── datasets/                     # User conversation data
├── checkpoints/                  # Saved models
├── comprehensive_results/        # Research experiment outputs
├── conversations/                # Saved dialogue sessions
└── docs/                        # This documentation
```

## Installation

```bash
cd geometric-attention
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Core Components

### Models

**GeometricTransformer** - Classification with learnable curvature
- Each head has learnable k parameter
- Unified distance formula works for any k
- Heads specialize to different geometries

**GeometricCausalLM** - Language modeling variant
- Causal attention for text generation
- Same geometric attention mechanism
- For conversational AI

**Standard variants** - Baseline comparisons (no geometric attention)

### Training System

**Trainer class** with:
- Automatic checkpointing (best + final)
- Timestamped filenames
- Model architecture saved in checkpoint
- Optional torch.compile() for ~2x speedup
- Sequential training (one model at a time to save VRAM)

### Dialogue Components (NEW)

**ConversationDataset** - Multi-turn training data
- Parses User:/Assistant: format
- Supports ChatML, Alpaca formats
- Masks user turns (trains only on assistant responses)

**ConversationManager** - Dialogue state tracking
- Auto-prunes old turns
- Save/load conversations
- Statistics tracking

**ResponseGenerator** - Advanced text generation
- Repetition penalty
- Nucleus sampling
- Stop sequences
- Conversation-aware

## Usage

### Interactive CLI (Main Interface)

```bash
python main.py
```

**Menu Options:**
1. Train New Model - Interactive training workflow
2. Load Model & Chat - Conversational inference
3. Compare Checkpoints - Side-by-side comparison
4. Analyze Geometry - View learned patterns
5. Run Comprehensive Experiments - Full research suite
6. Exit

### Training Scripts

**Language Modeling:**
```bash
python train_language_model.py --dim 512 --epochs 15 --compile
```

**Sentiment Classification:**
```bash
python train_sst2_full.py --dim 768 --epochs 5 --compile
```

**Dialogue Training:**
```bash
python train_dialogue.py --data datasets/my_conversations.jsonl --compile
```

**Comprehensive Research:**
```bash
python run_comprehensive_experiments.py
# Tests: 4 tasks × 4 model sizes = 16 experiments
# Output: comprehensive_results/ with JSON data and markdown report
```

### Analysis & Visualization

**Live Training Monitor:**
```bash
python plot_training_live.py --once  # Plot current training
python plot_training_live.py          # Auto-refresh monitoring
```

**Research Analysis:**
```bash
python analyze_comprehensive_results.py
# Generates publication figures in comprehensive_results/figures/
```

## Model Configurations

- **Tiny**: 128d, 1L, 2H - For quick tests
- **Small**: 256d, 2L, 4H - Fast experiments
- **Medium**: 512d, 4L, 8H - Good balance
- **Large**: 768d, 6L, 12H - Best results
- **Custom**: User-defined

## Data Formats

### For Training

**Conversations (JSONL recommended):**
```jsonl
{"text": "User: Hello!\nAssistant: Hi there!"}
{"text": "User: How are you?\nAssistant: I'm doing well!"}
```

**Classification:**
```json
{"sentence": "Great movie!", "label": 1}
```

**Language Modeling:**
```json
{"text": "Any raw text for next-token prediction"}
```

### Supported Formats
- JSONL (one JSON per line)
- JSON (array of objects)
- TXT (plain text, one sample per line)

## Performance Characteristics

### Speed
- Geometric models: ~2x slower than standard (without optimization)
- With torch.compile + TF32: ~1.4x slower
- First 2 epochs slow (JIT compilation), then fast

### Memory
- Sequential training: Only one model in VRAM at a time
- 512d model: ~8-10 GB VRAM
- 768d model: ~12-16 GB VRAM with batch_size=8

### Observed Results
- SST-2: Geometric 80.4%, Standard 81.3% (tied)
- MNLI: Geometric 52.6%, Standard 45.5% (geometric +7%)
- WikiText-2 (512d): Geometric 609 ppl, Standard 626 ppl

### Geometry Pattern
Observed consistently:
- ~50% Hyperbolic heads (k < -0.1)
- ~50% Spherical heads (k > 0.1)
- <2% Euclidean heads (|k| ≤ 0.1)

Pattern held from 2 heads to 72 heads, across all tested tasks.

## File Organization

### Training Outputs
- `checkpoints/` - Model checkpoints with timestamps
- `*.json` - Training results
- `*.png` - Training curves, geometry plots

### Research Outputs
- `comprehensive_results/` - Full experimental data
  - `comprehensive_results.json` - Raw data
  - `RESEARCH_REPORT.md` - Auto-generated analysis
  - `figures/` - Publication-quality plots

### Conversations
- `conversations/` - Saved dialogue sessions
- Load/save from chat interface

## Known Issues & Solutions

### OOM Errors
**Issue:** Large models run out of VRAM  
**Solution:** Use sequential training (now default), reduce batch size

### torch.compile() Compilation Lag
**Issue:** First 2 epochs slow (5+ min overhead)  
**Solution:** Expected behavior (LR warmup creates variations), epochs 3+ are fast

### Text Generation Repetition
**Issue:** Models repeating tokens ("is is is")  
**Solution:** Fixed with repetition penalty and nucleus sampling in generate_text()

## Dependencies

Key packages:
- PyTorch 2.0+ (for torch.compile)
- Transformers 4.30+
- Geoopt 0.5+ (for manifold operations)
- Datasets (HuggingFace)
- Matplotlib, NumPy, tqdm

See `requirements.txt` for complete list.

## Code Entry Points

**Main Interface:**
- `main.py` - Interactive CLI shell

**Training:**
- `train_language_model.py` - WikiText language modeling
- `train_sst2_full.py` - Full SST-2 dataset
- `train_dialogue.py` - Multi-turn conversations
- `run_comprehensive_experiments.py` - Systematic research

**Analysis:**
- `plot_training_live.py` - Monitor training
- `analyze_comprehensive_results.py` - Generate figures
- `benchmark_optimizations.py` - Performance testing

**Testing:**
- `quick_test.py` - Fast validation
- `train_example.py` - Configurable single experiment

## Model Architecture Details

### Learnable Curvature Attention

Each head learns a curvature parameter k:
- k < 0: Hyperbolic geometry (hierarchical structure)
- k ≈ 0: Euclidean geometry (local attention)
- k > 0: Spherical geometry (semantic similarity)

Uses unified distance formula that smoothly interpolates between geometries.

### Implementation Notes

- QKV projections standard
- Distance computation in curved space
- Softmax over negative distances
- Curvature bounded to [-2, 2] via tanh
- Fully differentiable

## Optimization Settings

**torch.compile()** - ~2x speedup
```python
Trainer(model, use_compile=True)
```

**TensorFloat32** - Auto-enabled with compile
- ~1.3x additional speedup on Ampere GPUs
- No accuracy loss

**Sequential Training** - Halves memory usage
- Train geometric first
- Free memory
- Train standard second

## Tips & Best Practices

1. **Start small** - Test with tiny/small models first
2. **Use --compile** - After warmup, significant speedup
3. **Monitor with plot_training_live.py** - Watch progress
4. **Save conversations** - Use `save <name>` command in chat
5. **Check geometry** - Option [4] in main menu to analyze patterns

## References

Original notebook: `cleaned_fresh.py` (converted from .ipynb)

Related work:
- Hyperbolic embeddings (Nickel & Kiela, 2017)
- Geometric deep learning (Bronstein et al., 2021)
- Transformer architectures (Vaswani et al., 2017)

## Notes

This is a research prototype. The 50/50 pattern observation is interesting and consistent but needs further investigation regarding:
- Scaling to larger models
- Domain-specific data
- Theoretical explanation
- Production deployment considerations

Code is organized for experimentation and research, not production deployment.
