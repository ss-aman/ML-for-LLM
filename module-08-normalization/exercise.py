"""
Module 08 — Exercises: Normalization & Residual Connections
============================================================
Work through these exercises to solidify your understanding of:
  1. LayerNorm from scratch (normalise, then apply learned gamma/beta)
  2. A 4-layer network with residual connections — verify gradients don't vanish
  3. (Challenge) Pre-LN vs Post-LN: compare training stability

Run with:  python exercise.py
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Exercise 1: Implement LayerNorm from scratch
# ---------------------------------------------------------------------------

def exercise_1_layernorm():
    """
    Implement LayerNorm: normalise a vector to mean=0, std=1,
    then apply learned scale γ (gamma) and shift β (beta).

    Formula for a single input vector x of length D:
        μ     = mean(x)
        σ²    = var(x)
        x_hat = (x - μ) / sqrt(σ² + ε)
        out   = γ * x_hat + β

    Here we work in batched form:
        x shape: (D, N) where D = feature dim, N = batch size
        Normalise across the D axis (feature axis) for each of the N samples.

    Tasks:
      a) Compute the per-sample mean and variance (across feature axis)
      b) Normalise: x_hat = (x - mean) / sqrt(var + eps)
      c) Apply learned parameters: out = gamma * x_hat + beta
      d) Verify: each column (sample) of `out` has mean≈0, std≈1 before
         gamma/beta are applied

    Backend analogy: Imagine each column is one API request's feature vector.
    You normalise each request independently — no shared state, no running
    averages, just a pure per-request z-score transformation.
    """
    print("=" * 55)
    print("Exercise 1: LayerNorm from scratch")
    print("=" * 55)

    np.random.seed(42)
    D = 8      # feature dimension
    N = 4      # batch size
    eps = 1e-5

    # Input with non-zero mean and varying scale (like raw activations)
    x = np.random.randn(D, N) * 5 + 3

    # Learnable parameters — start as identity (gamma=1, beta=0)
    gamma = np.ones((D, 1))
    beta = np.zeros((D, 1))

    # --- YOUR CODE HERE ---

    # Step 1: compute per-sample mean — shape should be (1, N)
    mean = None  # TODO: x.mean(axis=..., keepdims=True)

    # Step 2: compute per-sample variance — shape should be (1, N)
    var = None   # TODO: x.var(axis=..., keepdims=True)

    # Step 3: normalise
    x_hat = None  # TODO: (x - mean) / sqrt(var + eps)

    # Step 4: scale and shift
    out = None    # TODO: gamma * x_hat + beta

    raise NotImplementedError("Implement LayerNorm steps above")

    # --- END YOUR CODE ---

    print(f"Input mean per sample:  {x.mean(axis=0).round(3)}")
    print(f"x_hat mean per sample:  {x_hat.mean(axis=0).round(5)}  (should be ≈ 0)")
    print(f"x_hat std  per sample:  {x_hat.std(axis=0).round(5)}   (should be ≈ 1)")
    print(f"Output (gamma=1,beta=0) == x_hat: {np.allclose(out, x_hat)}")

    # Checks
    assert x_hat is not None, "x_hat not computed"
    assert np.allclose(x_hat.mean(axis=0), 0, atol=1e-5), "Mean should be 0 after normalisation"
    assert np.allclose(x_hat.std(axis=0), 1, atol=1e-4), "Std should be 1 after normalisation"

    # Try with non-identity gamma and beta
    gamma2 = np.arange(1, D + 1).reshape(D, 1).astype(float)
    beta2 = np.ones((D, 1)) * 0.5
    out2 = gamma2 * x_hat + beta2
    print(f"\nWith gamma=[1..{D}], beta=0.5:")
    print(f"  Output mean per sample: {out2.mean(axis=0).round(3)}")
    print("LayerNorm checks passed!\n")


# ---------------------------------------------------------------------------
# Exercise 2: 4-layer network with residual connections
# ---------------------------------------------------------------------------

def exercise_2_residual_network():
    """
    Build a 4-layer network with residual connections and verify that gradients
    flow all the way back to the first layer without vanishing.

    Architecture:
        For each of the 4 layers:
            x = x + relu(W @ x + b)    ← residual connection

        Final:
            output = W_out @ x + b_out

    Residual connection: output = F(x) + x
    Without the +x, gradients at layer 1 would be the product of all 4 local
    Jacobians, which can be < 0.01. With +x, the gradient always has an
    additive identity path.

    Tasks:
      a) Initialise 4 (W, b) pairs + output layer
      b) Forward pass: apply relu(W@x + b) + x for each layer
      c) MSE loss: loss = mean((pred - y)^2)
      d) Backward pass: manually propagate gradients back through all 4 layers
      e) Track the L2 norm of the gradient at each layer's input
      f) Print gradient norms — they should NOT collapse to near-zero
      g) Compare: add a second run WITHOUT the skip connection and show that
         gradient norms DO collapse

    Backend analogy: The skip connection is an express lane that bypasses each
    processing stage. Even if a stage's weights produce near-zero output
    (e.g., all inputs hit the "off" side of ReLU), the original signal still
    flows through unchanged — the express lane guarantees signal continuity.
    """
    print("=" * 55)
    print("Exercise 2: 4-Layer Residual Network — Gradient Check")
    print("=" * 55)

    rng = np.random.default_rng(7)
    D = 16    # feature dimension
    N = 32    # batch size
    lr = 0.01
    epochs = 200

    X = rng.standard_normal((D, N))
    y = rng.standard_normal((1, N))

    def init_params(num_layers, D, seed):
        rg = np.random.default_rng(seed)
        Ws = [rg.standard_normal((D, D)) * np.sqrt(2.0 / D) for _ in range(num_layers)]
        bs = [np.zeros((D, 1)) for _ in range(num_layers)]
        W_out = rg.standard_normal((1, D)) * 0.1
        b_out = np.zeros((1, 1))
        return Ws, bs, W_out, b_out

    def train_and_measure_gradients(use_skip: bool):
        Ws, bs, W_out, b_out = init_params(4, D, seed=42)
        label = "WITH skip" if use_skip else "WITHOUT skip"
        losses = []
        grad_norms_history = []

        for epoch in range(epochs):
            # Forward pass
            xs = [X]
            for W, b in zip(Ws, bs):
                pre_act = W @ xs[-1] + b        # linear transform
                act = np.maximum(0, pre_act)    # ReLU
                if use_skip:
                    xs.append(act + xs[-1])     # residual: F(x) + x
                else:
                    xs.append(act)              # plain: just F(x)

            pred = W_out @ xs[-1] + b_out
            loss = float(np.mean((pred - y) ** 2))
            losses.append(loss)

            # Backward pass
            d = 2 * (pred - y) / N
            dW_out = d @ xs[-1].T / N
            db_out = d.mean(axis=1, keepdims=True)
            d = W_out.T @ d

            grad_norms = []
            for i in range(3, -1, -1):
                W, b = Ws[i], bs[i]
                x_in = xs[i]
                pre_act = W @ x_in + b

                if use_skip:
                    # d is gradient w.r.t. (F(x) + x)
                    d_skip = d.copy()          # gradient through skip path (identity)
                    d_F = d.copy()             # gradient through F path
                else:
                    d_skip = 0
                    d_F = d

                d_act = d_F * (pre_act > 0)   # through ReLU
                dW = d_act @ x_in.T / N
                db = d_act.mean(axis=1, keepdims=True)
                d = W.T @ d_act
                if use_skip:
                    d = d + d_skip            # gradient from both paths

                grad_norms.append(float(np.linalg.norm(d)))
                Ws[i] -= lr * dW
                bs[i] -= lr * db

            W_out -= lr * dW_out
            b_out -= lr * db_out
            grad_norms_history.append(grad_norms[::-1])  # re-order layer 1→4

        grad_norms_last = np.array(grad_norms_history[-1])
        print(f"\n  [{label}] Final gradient norms at each layer:")
        for i, gn in enumerate(grad_norms_last):
            print(f"    Layer {i+1}: {gn:.6f}")
        print(f"  Final loss: {losses[-1]:.4f}")
        return losses, grad_norms_history

    # --- YOUR CODE: call train_and_measure_gradients for both cases ---
    # TODO: run with use_skip=True and use_skip=False
    # TODO: assert that with skip, all grad norms > 1e-4
    # TODO: print comparison
    raise NotImplementedError("Call train_and_measure_gradients for both skip=True and skip=False")

    # --- END YOUR CODE ---


# ---------------------------------------------------------------------------
# Exercise 3 (Challenge): Pre-LN vs Post-LN
# ---------------------------------------------------------------------------

def exercise_3_challenge_pre_vs_post_ln():
    """
    Implement and compare Pre-LN and Post-LN residual blocks.

    Post-LN (original transformer):
        x = LayerNorm(x + F(x))

    Pre-LN (modern transformers, e.g., GPT-2, GPT-3, LLaMA):
        x = x + F(LayerNorm(x))

    F is a simple Linear + ReLU transformation.

    The claim: Pre-LN is more stable early in training because the identity
    (skip) path is free of any normalisation — gradients flow back through
    the skip connection without going through LayerNorm, giving larger and
    more consistent gradients at early layers.

    Tasks:
      a) Implement layernorm_forward(x, gamma, beta, eps) as a pure function
      b) Implement a Post-LN block forward pass
      c) Implement a Pre-LN block forward pass
      d) Train both on the same task (regression on sin(x)) with the same
         initialisation and learning rate
      e) Track the gradient norm at layer 1 over the first 50 epochs
      f) Plot: Pre-LN should maintain larger/stabler gradients early on

    Backend analogy:
      - Post-LN: normalise AFTER merging the skip path — you normalise the
        combined (original + transformed) signal. The identity path is not
        "clean" at the output.
      - Pre-LN: normalise BEFORE the transformation — the identity path
        bypasses normalisation entirely and reaches the output unmodified.
        Like having an untouched fallback route that never goes through the
        normalisation middleware.
    """
    print("=" * 55)
    print("Exercise 3 (Challenge): Pre-LN vs Post-LN")
    print("=" * 55)

    rng = np.random.default_rng(42)
    D = 16
    N = 64
    eps = 1e-5
    lr = 0.005
    epochs = 150

    X = rng.standard_normal((D, N))
    y = np.sin(X.sum(axis=0, keepdims=True))

    def init_block_params(D, seed):
        rg = np.random.default_rng(seed)
        W = rg.standard_normal((D, D)) * np.sqrt(2.0 / D)
        b = np.zeros((D, 1))
        gamma = np.ones((D, 1))
        beta = np.zeros((D, 1))
        return W, b, gamma, beta

    def layernorm_forward(x, gamma, beta, eps=1e-5):
        """
        TODO: Implement LayerNorm forward.
        x: (D, N)
        Normalise across axis=0 (features) for each sample independently.
        Return out, x_hat, mean, var  (the last three for backprop)
        """
        raise NotImplementedError("Implement layernorm_forward")

    def post_ln_forward(x, W, b, gamma, beta):
        """
        Post-LN block: out = LayerNorm(x + relu(W @ x + b))
        Return out and any cached values needed for backprop.
        TODO: implement
        """
        raise NotImplementedError("Implement post_ln_forward")

    def pre_ln_forward(x, W, b, gamma, beta):
        """
        Pre-LN block: out = x + relu(W @ LayerNorm(x) + b)
        Return out and any cached values needed for backprop.
        TODO: implement
        """
        raise NotImplementedError("Implement pre_ln_forward")

    # --- YOUR CODE: train both variants and compare gradient norms ---
    # For each variant:
    #   1. Stack 4 blocks
    #   2. Add a final Linear(D→1) output layer
    #   3. Train for `epochs` iterations using MSE loss
    #   4. At each epoch, record the gradient norm at the INPUT to block 1
    # Plot both gradient norm curves on the same axes

    raise NotImplementedError("Implement training loop for Pre-LN vs Post-LN comparison")

    # --- END YOUR CODE ---


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    exercise_1_layernorm()
    exercise_2_residual_network()
    exercise_3_challenge_pre_vs_post_ln()


if __name__ == "__main__":
    main()
