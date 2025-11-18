"""
Quick test script to verify geometric attention fixes and stability.
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometric_attention.models.geometric_attention import OptimizedGeometricAttention
from geometric_attention.models.manifold_ops import unified_distance

def test_taylor_expansion():
    print("\n=== Testing Taylor Expansion Stability ===")
    
    dim = 64
    x = torch.randn(10, dim)
    y = torch.randn(10, dim)
    
    # Test curvatures around 0
    curvatures = [0.0, 1e-5, -1e-5, 1e-4, -1e-4, 1e-2, -1e-2]
    
    for c in curvatures:
        c_tensor = torch.tensor(c, requires_grad=True)
        try:
            dist = unified_distance(x, y, c_tensor, dim=-1)
            grad = torch.autograd.grad(dist.sum(), c_tensor, create_graph=True)[0]
            print(f"c = {c:8.5f} | Dist mean: {dist.mean().item():8.4f} | Grad: {grad.item():8.4f}")
            
            if torch.isnan(dist).any() or torch.isnan(grad).any():
                print(f"❌ NaNs detected at c={c}")
            
        except Exception as e:
            print(f"❌ Error at c={c}: {e}")

def test_model_gradients():
    print("\n=== Testing Model Gradients ===")
    
    dim = 64
    seq_len = 16
    batch_size = 4
    
    model = OptimizedGeometricAttention(dim)
    x = torch.randn(batch_size, seq_len, dim)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print(f"Initial curvature: {model.curvature.item()}")
    print(f"Initial log_temp:  {model.log_temperature.item()}")
    
    # Single step
    output, _, k = model(x)
    loss = output.mean()
    loss.backward()
    
    print("\nGradients:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name:20s}: {param.grad.norm().item():.4f}")
        else:
            print(f"{name:20s}: None")
            
    optimizer.step()
    print("\nStep complete. New values:")
    print(f"Curvature: {model.curvature.item():.4f}")
    print(f"Log Temp:  {model.log_temperature.item():.4f}")

if __name__ == "__main__":
    test_taylor_expansion()
    test_model_gradients()
