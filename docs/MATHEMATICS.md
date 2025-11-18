# Mathematical Principles of Geometric Attention

This document details the mathematical formulations used in the Geometric Attention project, including the manifold operations, distance metrics, and numerical stability techniques.

## 1. Manifold Geometry

We model the latent space as a product of Riemannian manifolds. Each attention head $h$ operates in a space with constant curvature $c_h$.

- **$c < 0$:** Hyperbolic Geometry (Poincaré Ball model)
- **$c > 0$:** Spherical Geometry (Hypersphere)
- **$c = 0$:** Euclidean Geometry (Flat space)

### The Poincaré Ball Model ($c < 0$)

For a curvature $c < 0$, the manifold $\mathbb{D}_c^n$ is defined as the open ball:
$$ \mathbb{D}_c^n = \{ x \in \mathbb{R}^n : ||x|| < \frac{1}{\sqrt{-c}} \} $$

The metric tensor is conformal to the Euclidean metric:
$$ g_x^{\mathbb{D}} = \lambda_x^2 g^E, \quad \text{where } \lambda_x = \frac{2}{1 + c||x||^2} $$

#### Möbius Addition
Vector addition in hyperbolic space is non-linear and non-commutative. We use **Möbius addition** $\oplus_c$:
$$ x \oplus_c y = \frac{(1 + 2c\langle x, y \rangle + c||y||^2)x + (1 - c||x||^2)y}{1 + 2c\langle x, y \rangle + c^2||x||^2 ||y||^2} $$

#### Distance
The induced geodesic distance is:
$$ d_{\mathbb{D}}(x, y) = \frac{2}{\sqrt{-c}} \text{arctanh}(\sqrt{-c} ||-x \oplus_c y||) $$

### Spherical Geometry ($c > 0$)

For $c > 0$, the manifold $\mathbb{S}_c^n$ is the sphere of radius $1/\sqrt{c}$:
$$ \mathbb{S}_c^n = \{ x \in \mathbb{R}^{n+1} : ||x|| = \frac{1}{\sqrt{c}} \} $$

*Note: In our implementation, we project points onto the unit sphere and scale distances by $1/\sqrt{c}$.*

#### Distance
The great-circle distance is:
$$ d_{\mathbb{S}}(x, y) = \frac{1}{\sqrt{c}} \arccos(\langle x, y \rangle_{\text{normalized}}) $$

## 2. Geometric Attention Mechanism

### Projections and Exponential Maps

Input embeddings $x$ live in the tangent space $T_0\mathcal{M} \cong \mathbb{R}^n$. We project them onto the manifold using the **Exponential Map**.

$$ \text{Exp}_0(v) = \tanh(\sqrt{|c|} \frac{||v||}{2}) \frac{v}{\sqrt{|c|}||v||} $$

In our optimized implementation, we perform a linear projection $W_q, W_k$ followed by a projection operator `project_to_manifold`:

1.  **Hyperbolic:** Clips norm to be $< 1/\sqrt{-c}$.
2.  **Spherical:** Normalizes norm to be $= 1$.
3.  **Euclidean:** Identity.

### Unified Distance Function

To allow the curvature $c$ to be learnable and cross zero, we implement a **Unified Distance** function.

$$
d(x, y, c) = 
\begin{cases} 
\frac{2}{\sqrt{-c}} \text{arctanh}(\sqrt{-c} ||-x \oplus_c y||) & \text{if } c < -\epsilon \\
\frac{1}{\sqrt{c}} \arccos(\langle x, y \rangle) & \text{if } c > \epsilon \\
||x - y||_2 & \text{if } |c| \le \epsilon
\end{cases}
$$

### Attention Scores

$$ \alpha_{ij} = \text{softmax}_j \left( - \frac{d(q_i, k_j)^2}{\tau} \right) $$

Where $\tau$ is a learnable temperature parameter.
*Note: We typically use distance (linear) or distance squared. Our implementation currently uses linear distance scaled by temperature.*

## 3. Numerical Stability & Optimizations

### Taylor Expansion at $c \approx 0$

The transition between geometries at $c=0$ involves singularities ($1/\sqrt{c}$). To ensure differentiability and stability, we use a second-order **Taylor approximation** when $|c| < 10^{-3}$:

$$ d_c(x, y) \approx ||x - y||_2 \left( 1 - \frac{c}{12} ||x - y||_2^2 \right) $$

This ensures that gradients can flow from hyperbolic to spherical regimes seamlessly.

### Boundary Conditions

In hyperbolic space, points near the boundary ($||x|| \to 1/\sqrt{-c}$) cause distances to explode to infinity. We enforce:
1.  **Clipping:** Norms are clipped to $(1 - \epsilon)/\sqrt{-c}$.
2.  **Safe Arctanh:** Arguments to `arctanh` are clamped to avoid NaNs.

### Learnable Temperature

In Euclidean attention, scaling by $1/\sqrt{d}$ stabilizes gradients. In curved spaces, volume growth is exponential (hyperbolic) or polynomial (spherical). A fixed scalar is insufficient.
We introduce a learnable log-temperature $\lambda$:
$$ \tau = \exp(\lambda) \cdot \sqrt{d} $$
This allows the model to learn the appropriate "sharpness" for its learned geometry.
