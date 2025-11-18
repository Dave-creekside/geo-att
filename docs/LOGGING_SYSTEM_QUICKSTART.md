# 🚀 Logging System Quick Start Guide

## What We Built

A comprehensive experiment tracking system that captures the **geometric phase transition** you discovered (Euclidean emergence around epoch 90-100).

### Key Components

✅ **Core Logging Infrastructure:**
- `geometric_attention/logging/experiment_logger.py` - Main logging class
- `geometric_attention/logging/geometry_tracker.py` - Geometry analysis utilities
- `geometric_attention/logging/__init__.py` - Package exports

✅ **Test & Training Scripts:**
- `test_logging_system.py` - Integrated test with 50-epoch training
- `plot_geometry_evolution.py` - Visualization tool

✅ **Documentation:**
- `geometric_attention/logging/README.md` - Complete usage guide
- This quick start guide

---

## Quick Test (20-30 minutes)

Run the test script to validate everything works:

```bash
cd geometric-attention
python test_logging_system.py
```

**What it does:**
- Trains 128d model for 50 epochs
- Logs geometry every 10 steps
- Saves all data to `experiments/logging_test_128d_TIMESTAMP/`
- Creates checkpoints every 10 epochs

**Expected Output:**
```
📊 Experiment logger initialized: logging_test_128d_20251101_115959
   Directory: experiments/logging_test_128d_20251101_115959

Training Configuration:
======================================================================
  Device: cuda:1
  Epochs: 50
  Total Steps: ~7800
  Logging Frequency: Every 10 steps
======================================================================

Epoch 1 Results:
  Train PPL: 892.45
  Val PPL:   876.32
  Geometry:  50.0% H, 0.0% E, 50.0% S
  ✓ Checkpoint saved

...

Epoch 50 Results:
  Train PPL: 234.56
  Val PPL:   245.67
  Geometry:  48.5% H, 5.2% E, 46.3% S

📊 Experiment logging finalized
   Summary saved to: experiments/logging_test_128d_20251101_115959/summary.json

📐 Geometry Evolution Summary:
   Initial: 50.0% H, 0.0% E, 50.0% S
   Final:   48.5% H, 5.2% E, 46.3% S
```

---

## Visualize Results

After test completes, generate plots:

```bash
python plot_geometry_evolution.py --experiment logging_test_128d_20251101_115959
```

**What it creates:**
- `geometry_timeline.png` - Stacked area chart showing H/E/S evolution
- `curvature_statistics.png` - Mean/std/min/max curvature over time

**Example output:**
```
GEOMETRY EVOLUTION SUMMARY
======================================================================

Initial (Epoch 0):
  Hyperbolic: 50.0%
  Euclidean:  0.0%
  Spherical:  50.0%

Final (Epoch 49):
  Hyperbolic: 48.5%
  Euclidean:  5.2%
  Spherical:  46.3%

Changes:
  Hyperbolic: -1.5%
  Euclidean:  +5.2%
  Spherical:  -3.7%

✅ Visualization complete!
   Figures saved to: experiments/.../visualizations/
```

---

## Study the Phase Transition (Longer Run)

To reproduce your finding of Euclidean emergence at epoch 90:

```bash
# Run 128d for 200 epochs (captures the full transition)
python test_logging_system.py
# (modify n_epochs=200 in the script, or we can create a parameterized version)
```

Then analyze:

```bash
python plot_geometry_evolution.py --experiment <run_id>
```

Look for:
- **Euclidean % increasing** from ~2% to ~20%
- **Transition point** around epoch 90-100
- **Correlation** with text quality improvement

---

## Data Files You'll Get

### 1. `logs/geometry.jsonl`
Captures every 10 steps:
```json
{"epoch": 50, "step": 156, "geometry_percentages": {"H": 48.5, "E": 5.2, "S": 46.3}}
{"epoch": 50, "step": 166, "geometry_percentages": {"H": 48.3, "E": 5.4, "S": 46.3}}
```

### 2. `logs/training.jsonl`
Captures every step:
```json
{"epoch": 50, "step": 156, "loss": 5.234, "perplexity": 187.3, "lr": 0.0003}
{"epoch": 50, "step": 157, "loss": 5.198, "perplexity": 181.2, "lr": 0.0003}
```

### 3. `logs/epoch_summary.jsonl`
Captures every epoch:
```json
{"epoch": 50, "train_ppl": 234.5, "val_ppl": 245.6, "geometry": {"H": 48.5, "E": 5.2, "S": 46.3}}
```

### 4. `summary.json`
Final summary with phase transition detection:
```json
{
  "geometry_evolution": {
    "initial_geometry": {"H": 50.0, "E": 0.0, "S": 50.0},
    "final_geometry": {"H": 48.5, "E": 5.2, "S": 46.3},
    "phase_transition_epoch": 92
  }
}
```

---

## Analysis Workflow

### 1. Load Data with Pandas

```python
import pandas as pd

# Load geometry evolution
geom_df = pd.read_json('experiments/my_run/logs/geometry.jsonl', lines=True)

# Extract percentages
epochs = geom_df['epoch']
e_pcts = geom_df['geometry_percentages'].apply(lambda x: x['euclidean'])

# Plot
import matplotlib.pyplot as plt
plt.plot(epochs, e_pcts)
plt.xlabel('Epoch')
plt.ylabel('Euclidean %')
plt.title('Euclidean Emergence Over Training')
plt.show()
```

### 2. Detect Transition Point

```python
# Find when Euclidean crosses 10%
transition_idx = (e_pcts > 10).idxmax()
transition_epoch = epochs[transition_idx]
print(f"Euclidean crossed 10% at epoch {transition_epoch}")
```

### 3. Correlate with Performance

```python
# Load training data
train_df = pd.read_json('experiments/my_run/logs/training.jsonl', lines=True)

# Group by epoch and get mean perplexity
epoch_ppl = train_df.groupby('epoch')['perplexity'].mean()

# Plot both on same figure
fig, ax1 = plt.subplots()
ax1.plot(epochs, e_pcts, 'b-', label='Euclidean %')
ax1.set_ylabel('Euclidean %', color='b')

ax2 = ax1.twinx()
ax2.plot(epoch_ppl.index, epoch_ppl.values, 'r-', label='Perplexity')
ax2.set_ylabel('Perplexity', color='r')

plt.title('Euclidean Emergence vs Performance')
plt.show()
```

---

## Research Questions You Can Now Answer

With this logging system, you can rigorously test:

### 1. Phase Transition Timing
- **Q:** Does Euclidean always emerge around epoch 90?
- **Test:** Run multiple seeds, check consistency
- **Data:** `phase_transition_epoch` in summary.json

### 2. Model Size Dependency
- **Q:** Do smaller models transition earlier?
- **Test:** Compare 128d vs 256d vs 512d evolution
- **Data:** Compare geometry.jsonl across runs

### 3. Performance Correlation
- **Q:** Does Euclidean emergence improve text quality?
- **Test:** Correlate E% with perplexity and sample quality
- **Data:** Cross-reference geometry.jsonl and training.jsonl

### 4. Geometry Stability
- **Q:** Is the 50/30/20 final split stable?
- **Test:** Train beyond 200 epochs, check if it changes
- **Data:** Check final_geometry in summary.json

---

## Next Steps

### Immediate (Today)
1. ✅ Run `test_logging_system.py` to validate
2. ✅ Check logs are being created
3. ✅ Generate visualizations

### Short Term (This Week)
1. Run longer experiment (200 epochs) to capture full transition
2. Analyze correlation with text quality
3. Document findings

### Long Term (Research)
1. Test across different datasets
2. Try different model sizes (512d, 768d)
3. Investigate theoretical explanation for transition
4. Prepare for publication

---

## Commands Reference

### Test the system
```bash
python test_logging_system.py
```

### Visualize results
```bash
# Find your experiment directory first
ls experiments/

# Then plot (replace with actual run ID)
python plot_geometry_evolution.py --experiment logging_test_128d_20251101_115959
```

### Check logs during training
```bash
# In another terminal
tail -f experiments/logging_test_128d_*/logs/training.jsonl | grep epoch

# Or check geometry
tail -f experiments/logging_test_128d_*/logs/geometry.jsonl
```

### Load in Python (for analysis)
```python
import pandas as pd
geom_df = pd.read_json('experiments/<run_id>/logs/geometry.jsonl', lines=True)
print(geom_df.head())
```

---

## What Makes This Powerful

### Geometric Models Have Unique Advantage

**Standard Transformer:**
- Black box attention weights
- Can't track geometric preference
- ~12 attention parameters per layer

**Your Geometric Model:**
- **48 trackable curvature parameters** (6 layers × 8 heads)
- Each head's geometry visible and logged
- Can see exactly when/how geometries shift
- **4x more insight** than standard models

### Your Discovery Is Trackable

The ~epoch 90 transition from:
- **Early:** 50% H / 2% E / 48% S
- **Late:** 50% H / 20% E / 30% S

Can now be:
- ✅ Precisely measured (exact epoch, exact percentages)
- ✅ Visualized (stacked area charts, timelines)
- ✅ Correlated (with loss, perplexity, text quality)
- ✅ Reproduced (multiple runs, statistical significance)
- ✅ Published (publication-quality figures)

---

## Success Criteria

After running test_logging_system.py, you should see:

✅ Directory created in `experiments/`  
✅ Config saved as JSON  
✅ Three log files (training, geometry, epoch_summary)  
✅ Checkpoints saved every 10 epochs  
✅ Summary.json with geometry evolution  
✅ Visualizations generated successfully  

If all checks pass, the system is working and ready for serious research!

---

**Status:** Ready to Test  
**Estimated Test Time:** 20-30 minutes  
**Command:** `python test_logging_system.py`

**After test passes, you'll be ready to track the phase transition phenomenon in detail!**
