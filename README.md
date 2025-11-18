# Geometric Attention

A PyTorch implementation of transformers with learnable curvature attention mechanisms that can dynamically learn to operate in Hyperbolic, Spherical, or Euclidean geometries.

## 🚀 Overview

This project explores "Geometric Deep Learning" applied to Transformer attention mechanisms. Unlike standard transformers that operate solely in Euclidean space (via dot products), **Geometric Attention** allows each attention head to learn its own optimal geometry:

-   **Hyperbolic ($c < 0$):** Ideal for hierarchical data, syntax trees, and entailment.
-   **Spherical ($c > 0$):** Ideal for cyclic data and semantic similarity.
-   **Euclidean ($c \approx 0$):** Standard local attention.

### 🧬 Key Innovations

1.  **Unified Manifold Operations:** A mathematically rigorous implementation of exponential maps, logarithmic maps, and distance functions that work across all three geometries.
2.  **Learnable Curvature:** The curvature parameter $c$ is learnable per head, allowing the model to self-organize.
3.  **Taylor Expansion Stability:** Uses second-order Taylor approximations for near-zero curvature regions, ensuring smooth gradients and numerical stability across the Euclidean boundary.
4.  **Adaptive Temperature:** A learnable temperature parameter scales the attention scores, adapting to the different volume growth rates of hyperbolic vs. spherical spaces.

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/geometric-attention.git
cd geometric-attention

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🛠️ Quick Start

### Interactive CLI

 The easiest way to use the library is through the interactive CLI:

```bash
python main.py
```
This menu allows you to:
- Train new models (Language Modeling, Dialogue, Sentiment Analysis)
- Chat with trained models
- Analyze learned geometries
- Run comprehensive experiments

### Training a Language Model

To train a model from the command line:

```bash
# Train a medium-sized model on WikiText-2
python train_language_model.py --dim 512 --heads 8 --layers 4 --epochs 5 --compile
```

### Training for Dialogue

To train a conversational agent:

```bash
# Train on a custom conversation dataset
python train_dialogue.py --data datasets/my_conversations.jsonl --epochs 3
```

## 🧠 How It Works

Standard attention computes similarity as a dot product: $Attention(Q, K) = \text{softmax}(\frac{QK^T}{\sqrt{d}})$.

**Geometric Attention** replaces this with a distance-based metric on a manifold:

1.  **Projection:** Linear projections $Q, K$ are mapped from the tangent space to the manifold (Poincaré ball or Sphere) using an **Exponential Map**.
2.  **Distance:** We compute the geodesic distance $d_c(q, k)$ on the manifold with curvature $c$.
3.  **Attention:** $Attention(Q, K) = \text{softmax}(-d_c(q, k) \cdot \tau)$.

Where $\tau$ is a learnable temperature scaling factor.

### Mathematical Stability

The codebase handles the singularities of Riemannian geometry:
-   **Near Zero Curvature:** When $c \to 0$, standard formulas for `arctanh` (hyperbolic) or `arccos` (spherical) become numerically unstable. We automatically switch to a **Taylor series approximation**:
    $$ d_c(x,y) \approx ||x-y|| \cdot (1 - \frac{c}{12} ||x-y||^2) $$
-   **Boundary Conditions:** Projections ensure points stay within the Poincaré ball ($||x|| < 1/\sqrt{-c}$).

## 📂 Project Structure

```
geometric-attention/
├── geometric_attention/          # Core Package
│   ├── models/                   # Neural Architectures
│   │   ├── geometric_attention.py    # The core attention logic
│   │   ├── manifold_ops.py           # Riemannian math & projections
│   │   ├── language_models.py        # GPT-style models
│   │   └── transformers.py           # Encoder-only models
│   ├── training/                 # Training Loop & Trainers
│   ├── dialogue/                 # Chat & Inference Tools
│   └── utils/                    # Visualization & Logging
├── scripts/                      # Standalone Scripts
│   ├── quick_test.py             # Verify math stability
│   └── run_comprehensive_experiments.py
├── datasets/                     # Data storage
└── docs/                         # Documentation
```

## 📊 Key Findings (Preliminary)

Initial experiments suggest a **"50/50 Split"** phenomenon where models converge to using approximately half hyperbolic and half spherical heads, with very few remaining purely Euclidean. This suggests natural language relies heavily on both hierarchical (hyperbolic) and distributional (spherical) relationships.

## 🧪 Running Tests

To verify the mathematical stability of the manifold operations:

```bash
python scripts/quick_test.py
```

This will check:
-   Gradient flow through $c=0$
-   Numerical stability of Taylor expansions
-   Learnable parameter updates

## 📜 Citation

If you use this code in your research, please cite:

```bibtex
@misc{geometric-attention-2025,
  title={Geometric Attention: Learnable Curvature Transformers},
  author={Orion},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/yourusername/geometric-attention}}
}
```

## License

MIT License
