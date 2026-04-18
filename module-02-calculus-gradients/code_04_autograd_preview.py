"""
Module 02 — Code 04: PyTorch Autograd Preview
===============================================
PyTorch computes gradients automatically using autograd.
This file shows HOW it works and verifies it produces the same results
as our manual backprop from code_03.

Key concept: PyTorch builds a computation graph during the forward pass.
Calling .backward() traverses that graph using the chain rule.

Run: python code_04_autograd_preview.py
Requires: pip install torch
"""

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def demo_requires_grad():
    """
    Show what 'requires_grad=True' does.
    When you set it on a tensor, PyTorch tracks all operations on it
    so it can compute gradients via .backward().
    """
    print("=" * 55)
    print("DEMO 1: requires_grad and the computation graph")
    print("=" * 55)

    if not TORCH_AVAILABLE:
        print("  [SKIP] Install PyTorch: pip install torch")
        return

    # A simple scalar function: L = (w - 3)^2
    w = torch.tensor(1.0, requires_grad=True)   # tell PyTorch to track this

    L = (w - 3)**2   # builds computation graph node

    print(f"\n  w = {w.item()}")
    print(f"  L = (w-3)^2 = {L.item()}")

    # Backprop: compute dL/dw analytically we know: dL/dw = 2*(w-3)
    L.backward()

    print(f"\n  After L.backward():")
    print(f"  w.grad = {w.grad.item():.4f}   (PyTorch computed)")
    print(f"  Analytical: 2*(1-3) = {2*(1-3):.4f}")
    print(f"  Match: {abs(w.grad.item() - 2*(1-3)) < 1e-6}")


def demo_multi_variable():
    """
    Gradient of a multi-variable function.
    """
    print("\n" + "=" * 55)
    print("DEMO 2: Multi-variable gradient")
    print("=" * 55)

    if not TORCH_AVAILABLE:
        print("  [SKIP] Install PyTorch: pip install torch")
        return

    # f(w1, w2) = (w1 - 3)^2 + (w2 - 5)^2
    w1 = torch.tensor(0.0, requires_grad=True)
    w2 = torch.tensor(0.0, requires_grad=True)

    L = (w1 - 3)**2 + (w2 - 5)**2
    L.backward()

    print(f"\n  w = [{w1.item()}, {w2.item()}]")
    print(f"  L = {L.item()}")
    print(f"\n  Gradients (PyTorch): dL/dw1={w1.grad.item():.2f}, dL/dw2={w2.grad.item():.2f}")
    print(f"  Analytical:          dL/dw1={2*(0-3):.2f}, dL/dw2={2*(0-5):.2f}")
    print(f"  Match: w1={abs(w1.grad.item() - 2*(0-3)) < 1e-6}, w2={abs(w2.grad.item() - 2*(0-5)) < 1e-6}")


def demo_linear_layer_gradient():
    """
    Show the gradient of a linear layer (W @ x + b), comparing:
    - PyTorch autograd
    - Our manual backprop formula from code_03

    The key formula for linear layer y = W @ x + b:
        dL/dW = dL/dy ⊗ x    (outer product)
        dL/db = dL/dy
        dL/dx = W.T @ dL/dy
    """
    print("\n" + "=" * 55)
    print("DEMO 3: Linear layer gradient — autograd vs manual")
    print("=" * 55)

    if not TORCH_AVAILABLE:
        print("  [SKIP] Install PyTorch: pip install torch")
        return

    np.random.seed(0)
    W_np = np.random.randn(3, 4)
    b_np = np.random.randn(3)
    x_np = np.random.randn(4)
    y_np = np.random.randn(3)

    # ─── PyTorch version ────────────────────────────────────────────────
    W = torch.tensor(W_np, requires_grad=True, dtype=torch.float64)
    b = torch.tensor(b_np, requires_grad=True, dtype=torch.float64)
    x = torch.tensor(x_np, dtype=torch.float64)
    y = torch.tensor(y_np, dtype=torch.float64)

    z = W @ x + b
    L = ((z - y)**2).mean()
    L.backward()

    # ─── Manual backprop ────────────────────────────────────────────────
    z_np   = W_np @ x_np + b_np
    dL_dz  = (2.0 / len(z_np)) * (z_np - y_np)   # MSE gradient
    dW_manual = np.outer(dL_dz, x_np)
    db_manual = dL_dz

    print(f"\n  W shape: {W_np.shape},  x shape: {x_np.shape}")
    print(f"  Loss: {L.item():.6f}")

    print(f"\n  dL/dW  (first row):")
    print(f"    PyTorch: {W.grad[0].numpy().round(6)}")
    print(f"    Manual:  {dW_manual[0].round(6)}")
    print(f"    Match:   {np.allclose(W.grad.numpy(), dW_manual, atol=1e-6)}")

    print(f"\n  dL/db:")
    print(f"    PyTorch: {b.grad.numpy().round(6)}")
    print(f"    Manual:  {db_manual.round(6)}")
    print(f"    Match:   {np.allclose(b.grad.numpy(), db_manual, atol=1e-6)}")


def demo_full_network():
    """
    Replicate code_03's 2-layer network in PyTorch.
    Verify the loss and gradients match our manual implementation.
    """
    print("\n" + "=" * 55)
    print("DEMO 4: Full 2-layer network — autograd vs manual backprop")
    print("=" * 55)

    if not TORCH_AVAILABLE:
        print("  [SKIP] Install PyTorch: pip install torch")
        return

    import torch.nn as nn

    np.random.seed(42)

    d_in, d_hidden, d_out = 3, 4, 2
    W1_np = np.random.randn(d_hidden, d_in)  * np.sqrt(1.0 / d_in)
    b1_np = np.zeros(d_hidden)
    W2_np = np.random.randn(d_out, d_hidden) * np.sqrt(1.0 / d_hidden)
    b2_np = np.zeros(d_out)
    x_np  = np.array([1.0, 2.0, 3.0])
    y_np  = np.array([1.0, -1.0])

    # ─── Manual backprop (from code_03) ────────────────────────────────
    from code_03_backprop_from_scratch import TwoLayerNet
    net_manual = TwoLayerNet(d_in, d_hidden, d_out)
    net_manual.W1[:] = W1_np
    net_manual.b1[:] = b1_np
    net_manual.W2[:] = W2_np
    net_manual.b2[:] = b2_np

    ypred_manual = net_manual.forward(x_np)
    loss_manual  = net_manual.loss(ypred_manual, y_np)
    grads_manual = net_manual.backward(ypred_manual, y_np)

    # ─── PyTorch autograd ────────────────────────────────────────────────
    W1 = torch.tensor(W1_np, requires_grad=True, dtype=torch.float64)
    b1 = torch.tensor(b1_np, requires_grad=True, dtype=torch.float64)
    W2 = torch.tensor(W2_np, requires_grad=True, dtype=torch.float64)
    b2 = torch.tensor(b2_np, requires_grad=True, dtype=torch.float64)
    x  = torch.tensor(x_np,  dtype=torch.float64)
    y  = torch.tensor(y_np,  dtype=torch.float64)

    z1 = W1 @ x + b1
    a1 = F.relu(z1)
    z2 = W2 @ a1 + b2
    L  = ((z2 - y)**2).mean()
    L.backward()

    print(f"\n  Loss — manual: {loss_manual:.6f},  PyTorch: {L.item():.6f}")
    print(f"  Loss match: {abs(loss_manual - L.item()) < 1e-10}")

    for name, grad_manual, grad_torch in [
        ('dW1', grads_manual['dW1'], W1.grad.numpy()),
        ('db1', grads_manual['db1'], b1.grad.numpy()),
        ('dW2', grads_manual['dW2'], W2.grad.numpy()),
        ('db2', grads_manual['db2'], b2.grad.numpy()),
    ]:
        match = np.allclose(grad_manual, grad_torch, atol=1e-8)
        print(f"  {name} match: {match}")

    print("\n  ✓ Manual backprop and PyTorch autograd produce identical gradients.")
    print("  PyTorch does nothing magical — it's the same chain rule, automated.")


def demo_training_loop_pytorch():
    """
    The same XOR training from code_03, now using PyTorch.
    Shows the clean training loop enabled by autograd.
    """
    print("\n" + "=" * 55)
    print("DEMO 5: PyTorch Training Loop (XOR)")
    print("=" * 55)

    if not TORCH_AVAILABLE:
        print("  [SKIP] Install PyTorch: pip install torch")
        return

    import torch.nn as nn
    import torch.optim as optim

    # XOR dataset
    X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float)
    Y = torch.tensor([[0],[1],[1],[0]],          dtype=torch.float)

    # Define the same 2-layer network using PyTorch nn.Module
    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
    )

    optimizer = optim.SGD(model.parameters(), lr=0.5)
    criterion = nn.MSELoss()

    for epoch in range(3001):
        optimizer.zero_grad()        # clear gradients from last step
        y_pred = model(X)
        loss   = criterion(y_pred, Y)
        loss.backward()              # backprop (PyTorch handles chain rule)
        optimizer.step()             # update parameters

        if epoch % 500 == 0:
            print(f"  Epoch {epoch:>5}: loss = {loss.item():.6f}")

    # Final predictions
    with torch.no_grad():
        preds = model(X)
    print("\n  Final predictions:")
    for i, (x_i, y_i, p_i) in enumerate(zip(X, Y, preds)):
        print(f"    {x_i.tolist()} → target={int(y_i.item())}  pred={p_i.item():.4f}  "
              f"rounded={round(p_i.item())}")

    print("\n  The PyTorch loop is shorter, but does the same thing as code_03:")
    print("    zero_grad()  → clear accumulated gradients")
    print("    forward()    → compute predictions (builds computation graph)")
    print("    backward()   → apply chain rule through the graph")
    print("    step()       → w = w - lr * w.grad")


if __name__ == '__main__':
    demo_requires_grad()
    demo_multi_variable()
    demo_linear_layer_gradient()
    demo_full_network()
    demo_training_loop_pytorch()
