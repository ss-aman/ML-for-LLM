"""
Module 11: Transformer Block — Code Implementation

Builds the full transformer block from scratch, then assembles a tiny GPT
and trains it on a repeating pattern to verify it actually learns.

Backend analogy: we're assembling a middleware pipeline from individual
components (LayerNorm, MHA, FFN, residuals), then wiring it to an input/output
layer to form a complete inference service.

Dependencies: numpy (core math), matplotlib (plots). No PyTorch needed until
the training loop — we use PyTorch there to get autograd for free.
"""

import numpy as np
import matplotlib.pyplot as plt

# PyTorch is used for the training demo (autograd handles gradients for us)
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. Sinusoidal Positional Encoding (NumPy)
# ---------------------------------------------------------------------------

def sinusoidal_positional_encoding(seq_len, d_model):
    """
    Compute sinusoidal positional encodings.

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    Returns: (seq_len, d_model) array

    Backend analogy: deterministic Lamport timestamps for each position —
    you can always recompute them, no storage needed.
    """
    PE = np.zeros((seq_len, d_model))
    positions = np.arange(seq_len).reshape(-1, 1)           # (seq_len, 1)
    dims = np.arange(0, d_model, 2)                         # [0, 2, 4, ...]
    # Compute the frequency denominators: 10000^(2i/d_model)
    div_term = np.power(10000.0, dims / d_model)            # (d_model/2,)

    # Even dimensions: sin; odd dimensions: cos
    PE[:, 0::2] = np.sin(positions / div_term)
    PE[:, 1::2] = np.cos(positions / div_term)

    return PE


def demo_positional_encoding():
    """Show how positional encoding looks as a heatmap."""
    seq_len, d_model = 50, 64
    PE = sinusoidal_positional_encoding(seq_len, d_model)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Heatmap of all positions × dimensions
    im = axes[0].imshow(PE, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    axes[0].set_xlabel("Dimension")
    axes[0].set_ylabel("Position")
    axes[0].set_title("Sinusoidal Positional Encoding (pos × dim)")
    plt.colorbar(im, ax=axes[0])

    # Show encoding for first 6 positions across first 20 dimensions
    for pos in range(6):
        axes[1].plot(PE[pos, :20], label=f"pos={pos}", linewidth=1.5)
    axes[1].set_xlabel("Dimension index")
    axes[1].set_ylabel("Encoding value")
    axes[1].set_title("PE values for first 6 positions, first 20 dims")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig("/home/user/ML-for-LLM/module-11-transformer-block/positional_encoding.png",
                dpi=150, bbox_inches="tight")
    print("Saved: positional_encoding.png")
    return PE


# ---------------------------------------------------------------------------
# 2. Full Transformer Block (NumPy — forward pass only)
# ---------------------------------------------------------------------------
# This section shows the math clearly without training infrastructure.
# For a trainable version see the PyTorch section below.

def layer_norm_numpy(x, gamma=None, beta=None, eps=1e-5):
    """
    LayerNorm: normalize each token's embedding to zero mean, unit variance,
    then apply learned scale (gamma) and shift (beta).

    x: (seq_len, d_model)
    Returns: (seq_len, d_model)
    """
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    x_norm = (x - mean) / (std + eps)
    if gamma is not None:
        x_norm = gamma * x_norm
    if beta is not None:
        x_norm = x_norm + beta
    return x_norm


def softmax_numpy(x):
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def scaled_dot_product_attention_numpy(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    weights = softmax_numpy(scores)
    return weights @ V, weights


def gelu_numpy(x):
    """GELU activation: smoother version of ReLU used in GPT models."""
    # Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def transformer_block_forward_numpy(x, W_Q, W_K, W_V, W_O, W1, b1, W2, b2,
                                    gamma1, beta1, gamma2, beta2, mask=None):
    """
    Full transformer block: LayerNorm → MHA → Residual → LayerNorm → FFN → Residual

    Args:
        x: input (seq_len, d_model)
        W_Q, W_K, W_V: attention projection weights (d_model, d_model)
        W_O: attention output projection (d_model, d_model)
        W1, b1: FFN layer 1 (d_model, d_ff), (d_ff,)
        W2, b2: FFN layer 2 (d_ff, d_model), (d_model,)
        gamma1, beta1: LayerNorm 1 parameters
        gamma2, beta2: LayerNorm 2 parameters
        mask: optional causal mask

    Returns:
        output (seq_len, d_model) — refined token representations
    """
    seq_len, d_model = x.shape
    num_heads = 4
    d_k = d_model // num_heads

    # --- Sub-layer 1: Multi-Head Attention + Residual ---
    x_norm1 = layer_norm_numpy(x, gamma1, beta1)    # normalize first

    # Simple single-head for clarity (multi-head in PyTorch version)
    Q = x_norm1 @ W_Q
    K = x_norm1 @ W_K
    V = x_norm1 @ W_V
    attn_out, _ = scaled_dot_product_attention_numpy(Q, K, V, mask)
    attn_out = attn_out @ W_O

    x = x + attn_out                                # residual #1

    # --- Sub-layer 2: FFN + Residual ---
    x_norm2 = layer_norm_numpy(x, gamma2, beta2)    # normalize before FFN

    # FFN: Linear → GELU → Linear
    h = gelu_numpy(x_norm2 @ W1 + b1)              # (seq_len, d_ff)
    ffn_out = h @ W2 + b2                           # (seq_len, d_model)

    x = x + ffn_out                                 # residual #2

    return x


def demo_transformer_block_numpy():
    """Show the transformer block forward pass with random weights."""
    print("\n--- NumPy Transformer Block Forward Pass ---")
    np.random.seed(42)

    seq_len, d_model, d_ff = 5, 16, 64

    # Random weights (in a trained model these are learned)
    scale = np.sqrt(2.0 / d_model)
    W_Q  = np.random.randn(d_model, d_model) * scale
    W_K  = np.random.randn(d_model, d_model) * scale
    W_V  = np.random.randn(d_model, d_model) * scale
    W_O  = np.random.randn(d_model, d_model) * scale
    W1   = np.random.randn(d_model, d_ff) * scale
    b1   = np.zeros(d_ff)
    W2   = np.random.randn(d_ff, d_model) * scale
    b2   = np.zeros(d_model)
    # LayerNorm parameters (ones and zeros = identity at init)
    gamma1 = np.ones(d_model);  beta1 = np.zeros(d_model)
    gamma2 = np.ones(d_model);  beta2 = np.zeros(d_model)

    # Causal mask
    mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)

    x = np.random.randn(seq_len, d_model)
    print(f"Input shape:  {x.shape}")

    out = transformer_block_forward_numpy(
        x, W_Q, W_K, W_V, W_O, W1, b1, W2, b2, gamma1, beta1, gamma2, beta2, mask
    )
    print(f"Output shape: {out.shape}  (same as input — block is shape-preserving)")
    print(f"Input mean/std:  {x.mean():.3f} / {x.std():.3f}")
    print(f"Output mean/std: {out.mean():.3f} / {out.std():.3f}")


# ---------------------------------------------------------------------------
# 3. Trainable Transformer (PyTorch) — Tiny GPT
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Full transformer block: Pre-LayerNorm → MHA → Residual → Pre-LayerNorm → FFN → Residual

    Backend analogy: a single middleware stage. Input comes in, gets normalized,
    routes through attention (cross-token join), residual skip, normalizes again,
    processes through FFN (per-token business logic), residual skip, output goes out.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True,
                                          dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, causal_mask=None):
        """
        x: (batch, seq_len, d_model)
        causal_mask: (seq_len, seq_len) additive mask (-inf for future positions)
        """
        # Sub-layer 1: attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=causal_mask)
        x = x + attn_out                    # residual #1

        # Sub-layer 2: FFN
        x = x + self.ffn(self.norm2(x))     # residual #2

        return x


class TinyGPT(nn.Module):
    """
    Minimal GPT-style language model.

    Architecture:
      Token embedding → Positional embedding → N transformer blocks
      → LayerNorm → Linear (d_model → vocab_size)

    Backend analogy: a complete inference service that:
      1. Converts token IDs to internal representations (embedding = DB lookup)
      2. Adds sequence position info (positional encoding = timestamp injection)
      3. Passes through N middleware stages (transformer blocks)
      4. Final layer converts internal state to output distribution (serializer)
    """

    def __init__(self, vocab_size, d_model=64, num_heads=4, d_ff=128,
                 num_layers=2, max_seq_len=64):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embedding: vocab_size integers → d_model-dimensional vectors
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Learned positional embedding (simpler than sinusoidal for training)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        # Final normalization before output projection
        self.norm_final = nn.LayerNorm(d_model)

        # Output projection: d_model → vocab_size logits
        # These logits become next-token probabilities after softmax
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Small initialization for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, token_ids):
        """
        token_ids: (batch, seq_len) integer tensor

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch, seq_len = token_ids.shape
        assert seq_len <= self.max_seq_len

        # 1. Token embeddings: integer IDs → vectors
        tok_emb = self.token_embedding(token_ids)      # (batch, seq_len, d_model)

        # 2. Positional embeddings: inject position information
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)         # (1, seq_len, d_model)

        x = tok_emb + pos_emb                           # (batch, seq_len, d_model)

        # 3. Build causal mask: upper triangle = -inf (block future tokens)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=token_ids.device),
            diagonal=1
        )

        # 4. Pass through N transformer blocks
        for block in self.blocks:
            x = block(x, causal_mask=causal_mask)

        # 5. Final LayerNorm
        x = self.norm_final(x)

        # 6. Project to vocab logits
        logits = self.output_proj(x)                    # (batch, seq_len, vocab_size)

        return logits

    def generate(self, prompt_ids, max_new_tokens=20, temperature=1.0):
        """
        Autoregressive generation: generate max_new_tokens tokens one at a time.

        Backend analogy: streaming response — each token depends on all previous ones.
        """
        self.eval()
        with torch.no_grad():
            ids = prompt_ids.clone()
            for _ in range(max_new_tokens):
                # Trim to max_seq_len if needed
                context = ids[:, -self.max_seq_len:]

                # Forward pass: get logits for the last position
                logits = self(context)                  # (batch, seq_len, vocab_size)
                next_logits = logits[:, -1, :]          # (batch, vocab_size)

                # Apply temperature: lower T = more confident/deterministic
                next_logits = next_logits / temperature

                # Sample next token
                probs = torch.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)  # (batch, 1)

                ids = torch.cat([ids, next_id], dim=1)

        return ids


# ---------------------------------------------------------------------------
# 4. Training: Tiny GPT on "ABCABC..." pattern
# ---------------------------------------------------------------------------

def train_tiny_gpt():
    """
    Train a 2-layer GPT on the repeating pattern "1 2 3 4 5 1 2 3 4 5..."

    If the model is working correctly, it should quickly learn to predict
    the next number in the cycle.

    Backend analogy: we're training a model to predict the next value in
    a round-robin load balancer sequence — a simple, learnable pattern.
    """
    print("\n--- Training Tiny GPT on '1 2 3 4 5' repeating pattern ---")

    torch.manual_seed(42)

    # Vocabulary: tokens 0–5 (using 0 as a dummy, 1-5 as sequence values)
    vocab_size = 6      # tokens: 0, 1, 2, 3, 4, 5
    seq_len = 20        # train on windows of 20 tokens

    # Build training data: the repeating sequence 1 2 3 4 5 1 2 3 4 5 ...
    full_sequence = [((i % 5) + 1) for i in range(200)]   # 1,2,3,4,5,1,2,3,4,5,...
    full_tensor = torch.tensor(full_sequence, dtype=torch.long)

    # Create training windows: (input, target) pairs
    # input  = tokens[i : i+seq_len]
    # target = tokens[i+1 : i+seq_len+1]  (next-token prediction)
    inputs, targets = [], []
    for i in range(0, len(full_sequence) - seq_len - 1, 1):
        inputs.append(full_tensor[i:i + seq_len])
        targets.append(full_tensor[i + 1:i + seq_len + 1])

    inputs = torch.stack(inputs)    # (N, seq_len)
    targets = torch.stack(targets)  # (N, seq_len)
    print(f"Training samples: {len(inputs)}, seq_len: {seq_len}")
    print(f"First input:  {inputs[0].tolist()}")
    print(f"First target: {targets[0].tolist()}")

    # Build model
    model = TinyGPT(
        vocab_size=vocab_size,
        d_model=64,
        num_heads=4,
        d_ff=128,
        num_layers=2,
        max_seq_len=seq_len + 5,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    batch_size = 32
    num_epochs = 50
    losses = []

    print(f"\nTraining for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        # Mini-batch training
        perm = torch.randperm(len(inputs))
        for i in range(0, len(inputs), batch_size):
            batch_idx = perm[i:i + batch_size]
            x_batch = inputs[batch_idx]     # (batch, seq_len)
            y_batch = targets[batch_idx]    # (batch, seq_len)

            optimizer.zero_grad()
            logits = model(x_batch)         # (batch, seq_len, vocab_size)

            # Cross-entropy loss: predict each next token
            # Reshape: (batch*seq_len, vocab_size) vs (batch*seq_len,)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                y_batch.reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1:3d}/{num_epochs}  loss={avg_loss:.4f}")

    # Plot training curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Tiny GPT: Training Loss on '1 2 3 4 5' Pattern")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig("/home/user/ML-for-LLM/module-11-transformer-block/training_loss.png",
                dpi=150, bbox_inches="tight")
    print("Saved: training_loss.png")

    # Test generation
    print("\n--- Generation after training ---")
    model.eval()
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)  # start with "1 2 3"
    generated = model.generate(prompt, max_new_tokens=12, temperature=0.1)
    print(f"Prompt:    {prompt[0].tolist()}")
    print(f"Generated: {generated[0].tolist()}")
    print(f"Expected:  1 2 3 4 5 1 2 3 4 5 1 2 3 4 5")

    final_loss = losses[-1]
    print(f"\nFinal training loss: {final_loss:.4f}")
    print(f"(Perfect prediction would approach 0; random guessing ≈ {np.log(vocab_size):.2f})")

    return model, losses


# ---------------------------------------------------------------------------
# 5. Two-block stacking demo
# ---------------------------------------------------------------------------

def demo_stacked_blocks():
    """Show that stacking blocks refines representations progressively."""
    print("\n--- Stacking 2 Transformer Blocks ---")

    d_model, num_heads, d_ff = 32, 4, 64
    seq_len = 5

    block1 = TransformerBlock(d_model, num_heads, d_ff)
    block2 = TransformerBlock(d_model, num_heads, d_ff)

    torch.manual_seed(7)
    x = torch.randn(1, seq_len, d_model)
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)

    print(f"Input shape:    {tuple(x.shape)}")

    out1 = block1(x, causal_mask=mask)
    print(f"After block 1:  {tuple(out1.shape)}  (same shape, refined content)")

    out2 = block2(out1, causal_mask=mask)
    print(f"After block 2:  {tuple(out2.shape)}  (same shape, further refined)")

    # Residual check: output should not be wildly different from input
    diff1 = (out1 - x).abs().mean().item()
    diff2 = (out2 - out1).abs().mean().item()
    print(f"\nMean |block1_out - input|:  {diff1:.4f}  (residuals keep it close)")
    print(f"Mean |block2_out - block1_out|: {diff2:.4f}  (each block refines gently)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Module 11: Transformer Block")
    print("=" * 60)

    # 1. Positional encoding visualization
    PE = demo_positional_encoding()
    print(f"PE shape: {PE.shape}")
    print(f"PE[0, :4] (position 0): {PE[0, :4].round(3)}")
    print(f"PE[1, :4] (position 1): {PE[1, :4].round(3)}")

    # 2. NumPy forward pass demo
    demo_transformer_block_numpy()

    # 3. Stack 2 PyTorch blocks
    demo_stacked_blocks()

    # 4. Train tiny GPT
    model, losses = train_tiny_gpt()

    print("\n--- Architecture Summary ---")
    print("Transformer block = middleware stack:")
    print("  Input x")
    print("  → LayerNorm  (normalize before attention)")
    print("  → MHA        (soft database lookup — who attends to whom)")
    print("  → + x        (residual #1 — preserve original signal)")
    print("  → LayerNorm  (normalize before FFN)")
    print("  → FFN        (per-token processing — factual knowledge storage)")
    print("  → + x        (residual #2 — preserve again)")
    print("  Output (same shape as input)")
    print()
    print("Full GPT = Token Embedding + Positional Encoding + N blocks + LM head")
