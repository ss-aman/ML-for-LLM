"""
Module 11: Transformer Block — Exercises

Target audience: backend developer (Python/APIs/databases), no ML background.

You will:
  1. Implement sinusoidal positional encoding
  2. Build a complete transformer block from scratch
  3. (Challenge) Build a 2-layer GPT and train it on a simple pattern

Run with:  python exercise.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Exercise 1: Sinusoidal Positional Encoding
# ---------------------------------------------------------------------------

def exercise1_positional_encoding():
    """
    Implement sinusoidal positional encoding.

    The formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    Where:
        pos   = position in the sequence (0, 1, 2, ...)
        i     = dimension index (0, 1, 2, ..., d_model/2 - 1)
        2i    = even dimension index
        2i+1  = odd dimension index

    Backend analogy: a deterministic Lamport timestamp injected into each
    token's embedding before the first transformer block. Without this,
    the model sees an unordered bag of tokens — position is invisible.

    Returns:
        PE matrix of shape (seq_len, d_model)

    Verification:
        - Shape should be (seq_len, d_model)
        - All values should be in [-1, 1]
        - Different positions should produce different encodings
        - PE[pos, 0] = sin(pos / 1.0) for the zeroth dimension
    """
    # --- YOUR IMPLEMENTATION BELOW ---

    def positional_encoding(seq_len, d_model):
        PE = np.zeros((seq_len, d_model))
        positions = np.arange(seq_len).reshape(-1, 1)    # (seq_len, 1)
        dims = np.arange(0, d_model, 2)                  # [0, 2, 4, ...]
        div_term = np.power(10000.0, dims / d_model)     # frequency denominators

        PE[:, 0::2] = np.sin(positions / div_term)       # even dims: sin
        PE[:, 1::2] = np.cos(positions / div_term)       # odd dims: cos
        return PE

    # --- TESTS ---
    seq_len, d_model = 10, 16
    PE = positional_encoding(seq_len, d_model)

    print("=== Exercise 1: Sinusoidal Positional Encoding ===")
    print(f"PE shape: {PE.shape}  (should be ({seq_len}, {d_model}))")
    assert PE.shape == (seq_len, d_model), f"Wrong shape: {PE.shape}"

    # All values in [-1, 1] (sin/cos range)
    assert PE.min() >= -1.0 - 1e-9 and PE.max() <= 1.0 + 1e-9, \
        "Values should be in [-1, 1]"
    print(f"Value range: [{PE.min():.3f}, {PE.max():.3f}]  (should be in [-1, 1])")

    # Different positions → different encodings
    for i in range(seq_len - 1):
        assert not np.allclose(PE[i], PE[i + 1]), \
            f"Positions {i} and {i+1} have identical encodings!"
    print("All positions produce distinct encodings  OK")

    # Check PE[pos, 0] = sin(pos / 10000^0) = sin(pos)
    expected_col0 = np.sin(np.arange(seq_len).astype(float))
    assert np.allclose(PE[:, 0], expected_col0, atol=1e-6), \
        "Column 0 should be sin(pos)"
    print(f"Column 0 (sin(pos)): {PE[:5, 0].round(3)}  OK")

    # Show first 3 positions
    print(f"\nFirst 3 positions, first 8 dims:")
    print(PE[:3, :8].round(3))
    print("PASSED\n")

    return positional_encoding


# ---------------------------------------------------------------------------
# Exercise 2: Complete Transformer Block
# ---------------------------------------------------------------------------

def exercise2_transformer_block():
    """
    Build a complete transformer block with the correct data flow:

        x  →  LayerNorm  →  MHA  →  + x  →  LayerNorm  →  FFN  →  + x  →  output

    The two residual additions (+x) are the key structural element:
    they allow gradients to flow backwards without vanishing, and ensure
    each block can only *refine* representations, never destroy them.

    Backend analogy: a two-stage middleware:
      Stage 1: normalize → attention (routing/join across tokens) → merge with original
      Stage 2: normalize → FFN (per-token transform/business logic) → merge with original

    Use PyTorch nn.Module for simplicity (handles weight initialization + autograd).

    Verify:
        - Output shape equals input shape
        - Block works with and without a causal mask
        - Residuals prevent explosion: output should be similar magnitude to input
    """
    # --- YOUR IMPLEMENTATION BELOW ---

    class TransformerBlock(nn.Module):
        def __init__(self, d_model, num_heads, d_ff):
            super().__init__()
            # Two LayerNorm layers (one before each sub-layer)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

            # Multi-Head Attention: PyTorch's built-in handles the H heads
            # batch_first=True means input shape is (batch, seq_len, d_model)
            self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

            # Feed-Forward Network: Linear → GELU → Linear
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model),
            )

        def forward(self, x, causal_mask=None):
            # Sub-layer 1: Pre-Norm → MHA → Residual
            x_norm1 = self.norm1(x)
            attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1,
                                    attn_mask=causal_mask)
            x = x + attn_out                    # residual connection #1

            # Sub-layer 2: Pre-Norm → FFN → Residual
            x = x + self.ffn(self.norm2(x))     # residual connection #2

            return x

    # --- TESTS ---
    torch.manual_seed(0)
    batch, seq_len, d_model, num_heads, d_ff = 2, 6, 32, 4, 128

    block = TransformerBlock(d_model, num_heads, d_ff)

    x = torch.randn(batch, seq_len, d_model)

    print("=== Exercise 2: Complete Transformer Block ===")
    print(f"Input shape: {tuple(x.shape)}")

    # Test without mask (bidirectional, like BERT)
    out_bidir = block(x)
    print(f"Output (no mask): {tuple(out_bidir.shape)}  (should be {(batch, seq_len, d_model)})")
    assert out_bidir.shape == x.shape, "Output shape must match input shape!"

    # Test with causal mask (autoregressive, like GPT)
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf")), diagonal=1
    )
    out_causal = block(x, causal_mask=causal_mask)
    print(f"Output (causal):  {tuple(out_causal.shape)}  (should be {(batch, seq_len, d_model)})")
    assert out_causal.shape == x.shape

    # Residual check: output magnitude should be similar to input (not exploding)
    input_norm  = x.norm().item()
    output_norm = out_causal.norm().item()
    ratio = output_norm / input_norm
    print(f"\nInput  L2 norm: {input_norm:.3f}")
    print(f"Output L2 norm: {output_norm:.3f}")
    print(f"Ratio (should be near 1.0, residuals stabilize magnitude): {ratio:.3f}")

    # Bidirectional and causal outputs should differ (mask actually does something)
    diff = (out_bidir - out_causal).abs().mean().item()
    print(f"\nMean |bidir_out - causal_out|: {diff:.4f}  (>0 means mask is working)")
    assert diff > 0, "Causal and bidirectional outputs should differ!"

    print("PASSED\n")
    return TransformerBlock


# ---------------------------------------------------------------------------
# Exercise 3 (Challenge): 2-Layer GPT on "1 2 3 4 5" Pattern
# ---------------------------------------------------------------------------

def exercise3_mini_gpt():
    """
    CHALLENGE: Build a 2-layer GPT and train it to predict the next token
    in the repeating sequence "1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 ..."

    Architecture:
        token_ids → Embedding → + PositionalEmbedding
                  → TransformerBlock × 2
                  → LayerNorm
                  → Linear(d_model, vocab_size)
                  → logits

    Training objective (next-token prediction):
        input  = [1, 2, 3, 4, 5, 1, 2, 3]
        target = [2, 3, 4, 5, 1, 2, 3, 4]

    Success criterion:
        After training, the model should correctly predict the next token
        with high probability (e.g., given [1, 2, 3] it should predict 4).

    Backend analogy: you're training a service to predict the next value
    in a round-robin schedule. The model must learn the period-5 cycle
    purely from data — no one hardcodes the rule.
    """
    # --- YOUR IMPLEMENTATION BELOW ---

    class MiniGPT(nn.Module):
        def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len):
            super().__init__()
            self.max_seq_len = max_seq_len

            # Step 1: token IDs → embeddings (like a DB lookup table)
            self.token_emb = nn.Embedding(vocab_size, d_model)
            # Step 2: learned positional embeddings (add order information)
            self.pos_emb   = nn.Embedding(max_seq_len, d_model)

            # Step 3: N transformer blocks (the middleware pipeline)
            self.blocks = nn.ModuleList([
                TransformerBlockLocal(d_model, num_heads, d_ff)
                for _ in range(num_layers)
            ])

            # Step 4: final norm + output projection (serialize to vocab distribution)
            self.norm_out = nn.LayerNorm(d_model)
            self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)

            # Small init for stable training
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.normal_(p, std=0.02)

        def forward(self, ids):
            B, T = ids.shape
            tok = self.token_emb(ids)
            pos = self.pos_emb(torch.arange(T, device=ids.device).unsqueeze(0))
            x = tok + pos

            mask = torch.triu(torch.full((T, T), float("-inf"), device=ids.device),
                              diagonal=1)
            for block in self.blocks:
                x = block(x, mask)

            return self.lm_head(self.norm_out(x))

    class TransformerBlockLocal(nn.Module):
        def __init__(self, d_model, num_heads, d_ff):
            super().__init__()
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.attn  = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            self.ffn   = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
            )

        def forward(self, x, mask=None):
            a, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                             attn_mask=mask)
            x = x + a
            x = x + self.ffn(self.norm2(x))
            return x

    # --- BUILD DATASET ---
    torch.manual_seed(42)
    vocab_size = 6      # tokens 0..5; we use 1..5 in the pattern
    seq_len    = 15

    pattern = [((i % 5) + 1) for i in range(500)]
    t = torch.tensor(pattern, dtype=torch.long)

    inputs, targets = [], []
    for i in range(len(pattern) - seq_len - 1):
        inputs.append(t[i:i + seq_len])
        targets.append(t[i + 1:i + seq_len + 1])
    inputs  = torch.stack(inputs)
    targets = torch.stack(targets)

    # --- BUILD & TRAIN MODEL ---
    model = MiniGPT(vocab_size=vocab_size, d_model=32, num_heads=4,
                    d_ff=64, num_layers=2, max_seq_len=seq_len + 5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    batch_size = 64
    epochs     = 80
    losses     = []

    print("=== Exercise 3 (Challenge): 2-Layer GPT on '1 2 3 4 5' Pattern ===")
    print(f"Dataset: {len(inputs)} training windows of length {seq_len}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\nTraining for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(inputs))
        total_loss = 0.0
        n_batches  = 0
        for i in range(0, len(inputs), batch_size):
            idx = perm[i:i + batch_size]
            x_b, y_b = inputs[idx], targets[idx]
            logits = model(x_b)                          # (B, T, V)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y_b.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
        losses.append(total_loss / n_batches)

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1:3d}/{epochs}  loss={losses[-1]:.4f}")

    # --- EVALUATION ---
    model.eval()
    with torch.no_grad():
        # Test: prompt "1 2 3" → should predict 4, 5, 1, 2, 3, ...
        prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
        generated = [1, 2, 3]
        for _ in range(9):
            logits = model(prompt)[:, -1, :]           # last position logits
            next_tok = logits.argmax(dim=-1).item()    # greedy decode
            generated.append(next_tok)
            prompt = torch.cat([prompt, torch.tensor([[next_tok]])], dim=1)

    print(f"\nGenerated: {generated}")
    print(f"Expected:  [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2]")
    print(f"Final loss: {losses[-1]:.4f}  (random baseline ≈ {np.log(vocab_size):.2f})")

    # Check that at least the first few predictions are correct
    expected = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2]
    matches = sum(g == e for g, e in zip(generated, expected))
    print(f"Correct predictions: {matches}/{len(expected)}")
    if matches >= 10:
        print("Model learned the pattern!  PASSED")
    else:
        print(f"Partial learning (may need more epochs). Got {matches}/12 correct.")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Module 11: Transformer Block — Exercises\n")

    exercise1_positional_encoding()
    exercise2_transformer_block()
    exercise3_mini_gpt()

    print("All exercises complete.")
    print()
    print("Key takeaways:")
    print("  - Positional encoding = Lamport timestamps for token order")
    print("  - Transformer block = normalize → attend → residual → normalize → FFN → residual")
    print("  - Residual connections prevent gradient vanishing in deep networks")
    print("  - FFN stores factual knowledge; attention routes information between tokens")
    print("  - Stack N blocks to increase model capacity and reasoning depth")


if __name__ == "__main__":
    main()
