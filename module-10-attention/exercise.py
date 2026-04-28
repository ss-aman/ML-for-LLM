"""
Module 10: Attention Mechanism — Exercises

Target audience: backend developer (Python/APIs/databases), no ML background.

Backend frame: you're implementing the "soft database lookup" engine from scratch.
Each exercise builds on the last — by the end you'll have multi-head attention running.

Run with:  python exercise.py
"""

import numpy as np


# ---------------------------------------------------------------------------
# Exercise 1: Scaled Dot-Product Attention
# ---------------------------------------------------------------------------

def exercise1_scaled_dot_product_attention():
    """
    Implement scaled dot-product attention from scratch.

    Given Q, K, V matrices, compute the attention output using:
        Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V

    Think of it as a soft database lookup:
      - Q = what you're querying for
      - K = the index keys in the database
      - V = the values stored at each key
      - Output = a weighted blend of all values, weighted by query-key similarity

    Steps:
      1. Compute raw scores: scores = Q @ K.T
      2. Scale:              scores = scores / sqrt(d_k)
      3. Softmax:            weights = softmax(scores)   (each row sums to 1)
      4. Weighted sum:       output  = weights @ V

    Returns:
        output  shape (seq_len, d_v)   — context-enriched token representations
        weights shape (seq_len, seq_len) — attention distribution (for inspection)
    """
    # --- YOUR IMPLEMENTATION BELOW ---

    def softmax(x):
        # Numerically stable: subtract max before exp
        # (prevents overflow; doesn't change output because softmax is shift-invariant)
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def scaled_dot_product_attention(Q, K, V):
        d_k = Q.shape[-1]                      # dimension of key vectors
        scores = Q @ K.T                        # (seq_len, seq_len)
        scores = scores / np.sqrt(d_k)          # scale to prevent saturation
        weights = softmax(scores)               # (seq_len, seq_len), rows sum to 1
        output = weights @ V                    # (seq_len, d_v)
        return output, weights

    # --- TESTS ---
    np.random.seed(0)
    seq_len, d_k, d_v = 4, 8, 8
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_v)

    output, weights = scaled_dot_product_attention(Q, K, V)

    print("=== Exercise 1: Scaled Dot-Product Attention ===")
    print(f"Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")
    print(f"Output shape: {output.shape}   (should be ({seq_len}, {d_v}))")
    print(f"Weights shape: {weights.shape} (should be ({seq_len}, {seq_len}))")

    # Each row of weights should sum to 1 (it's a probability distribution)
    row_sums = weights.sum(axis=-1)
    print(f"\nRow sums of weights (each should be 1.0):")
    print(f"  {row_sums.round(6)}")
    assert np.allclose(row_sums, 1.0), "Rows must sum to 1!"

    # All weights should be non-negative
    assert (weights >= 0).all(), "Weights must be non-negative!"

    print("\nAttention weight matrix (row i = how token i distributes attention):")
    print(weights.round(3))
    print("PASSED\n")

    return scaled_dot_product_attention


# ---------------------------------------------------------------------------
# Exercise 2: Causal Mask
# ---------------------------------------------------------------------------

def exercise2_causal_mask(scaled_dot_product_attention_fn):
    """
    Apply a causal (autoregressive) mask so each token can only attend
    to itself and previous tokens — never future ones.

    Backend analogy: an append-only event log.
    When processing event at position i, events i+1, i+2, ... don't exist yet.
    The causal mask enforces this: future positions get weight 0.

    How it works:
      - Build a mask of shape (seq_len, seq_len)
      - mask[i, j] = 0    if j <= i  (j is visible to i)
      - mask[i, j] = -inf if j > i   (j is in the future — blocked)
      - Add the mask to the scores BEFORE softmax
      - -inf + anything = -inf, so softmax(−inf) = 0 → zero attention weight

    Verify:
      - Token 0 attends only to position 0
      - Token 1 attends only to positions [0, 1]
      - Token i attends only to positions [0, ..., i]
    """
    # --- YOUR IMPLEMENTATION BELOW ---

    def make_causal_mask(seq_len):
        # np.triu with k=1 gives the upper triangle ABOVE the diagonal
        # Those positions = future tokens → set to -inf
        return np.triu(np.full((seq_len, seq_len), -np.inf), k=1)

    def masked_attention(Q, K, V, mask):
        d_k = Q.shape[-1]
        scores = Q @ K.T / np.sqrt(d_k)
        scores = scores + mask          # -inf entries survive addition
        # Softmax: exp(-inf) = 0, so those positions get zero weight
        x_shifted = scores - np.max(scores, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        weights = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        output = weights @ V
        return output, weights

    # --- TESTS ---
    np.random.seed(1)
    seq_len, d_k = 5, 8
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_k)
    mask = make_causal_mask(seq_len)

    print("=== Exercise 2: Causal Mask ===")
    print(f"Causal mask for seq_len={seq_len}:")
    print(mask)

    output, weights = masked_attention(Q, K, V, mask)

    print("\nAttention weights with causal mask:")
    print(weights.round(3))

    print("\nVerification — future tokens must have zero weight:")
    all_passed = True
    for i in range(seq_len):
        future_weight = np.sum(weights[i, i + 1:])
        status = "OK" if np.isclose(future_weight, 0.0, atol=1e-9) else "FAIL"
        if status == "FAIL":
            all_passed = False
        visible_positions = list(range(i + 1))
        print(f"  Token {i}: visible={visible_positions}, "
              f"future_weight={future_weight:.2e}  [{status}]")

    # Also verify that row sums are still 1
    row_sums = weights.sum(axis=-1)
    assert np.allclose(row_sums, 1.0), "Masked rows must still sum to 1!"
    assert all_passed, "Some future tokens had non-zero weight!"
    print("PASSED\n")


# ---------------------------------------------------------------------------
# Exercise 3 (Challenge): Multi-Head Attention
# ---------------------------------------------------------------------------

def exercise3_multi_head_attention():
    """
    CHALLENGE: Implement multi-head attention with 4 heads.

    Why multiple heads?
    Backend analogy: imagine running PARALLEL database queries with DIFFERENT indexes.
    - Query 1: "find related entities by foreign key"
    - Query 2: "find nearby records by timestamp"
    - Query 3: "find similar records by content hash"
    Each query returns a result set; you concatenate all result sets, then a final
    projection layer combines them into one unified response.

    Multi-head attention:
      For each head h in range(H):
          Q_h = X @ W_Q[h]              project to smaller d_k = d_model / H
          K_h = X @ W_K[h]
          V_h = X @ W_V[h]
          head_h = Attention(Q_h, K_h, V_h)    (seq_len, d_k)

      output = concat(head_0, ..., head_{H-1}) @ W_O
               (seq_len, H*d_k) @ (H*d_k, d_model) = (seq_len, d_model)

    Dimensions:
      d_model = 32    full model embedding dimension
      H       = 4     number of heads
      d_k     = 8     per-head dimension (= d_model / H)
      seq_len = 8     tokens in the sequence

    Verify:
      - Output shape is (seq_len, d_model)
      - Each head's attention weights are a valid probability distribution (rows sum to 1)
      - Different heads produce different weight patterns (they've specialized)
    """
    # --- YOUR IMPLEMENTATION BELOW ---

    def softmax(x):
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def single_head_attention(Q, K, V, mask=None):
        d_k = Q.shape[-1]
        scores = Q @ K.T / np.sqrt(d_k)
        if mask is not None:
            scores = scores + mask
        weights = softmax(scores)
        return weights @ V, weights

    def multi_head_attention(X, W_Q, W_K, W_V, W_O, mask=None):
        """
        X:   (seq_len, d_model)
        W_Q: (H, d_model, d_k)   — one Q projection per head
        W_K: (H, d_model, d_k)
        W_V: (H, d_model, d_k)
        W_O: (H*d_k, d_model)    — output projection
        """
        H = W_Q.shape[0]
        head_outputs = []
        head_weights = []

        for h in range(H):
            Q_h = X @ W_Q[h]           # (seq_len, d_k)
            K_h = X @ W_K[h]
            V_h = X @ W_V[h]
            out_h, w_h = single_head_attention(Q_h, K_h, V_h, mask)
            head_outputs.append(out_h)  # (seq_len, d_k)
            head_weights.append(w_h)    # (seq_len, seq_len)

        # Concatenate all head outputs: (seq_len, H*d_k) = (seq_len, d_model)
        concat = np.concatenate(head_outputs, axis=-1)
        output = concat @ W_O           # (seq_len, d_model)
        return output, head_weights

    # --- SETUP ---
    np.random.seed(42)
    seq_len = 8
    d_model = 32
    H = 4
    d_k = d_model // H    # = 8

    X = np.random.randn(seq_len, d_model)

    # Initialize projection weights (in a trained model these are learned)
    scale = np.sqrt(2.0 / d_model)
    W_Q = np.random.randn(H, d_model, d_k) * scale
    W_K = np.random.randn(H, d_model, d_k) * scale
    W_V = np.random.randn(H, d_model, d_k) * scale
    W_O = np.random.randn(H * d_k, d_model) * scale   # = (d_model, d_model)

    # Build causal mask for this sequence length
    mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)

    # --- RUN ---
    output, head_weights = multi_head_attention(X, W_Q, W_K, W_V, W_O, mask=mask)

    print("=== Exercise 3 (Challenge): Multi-Head Attention ===")
    print(f"Input shape:  {X.shape}")
    print(f"Output shape: {output.shape}  (should be ({seq_len}, {d_model}))")
    print(f"Number of heads: {H}, d_k per head: {d_k}")

    # Verify output shape
    assert output.shape == (seq_len, d_model), \
        f"Expected ({seq_len}, {d_model}), got {output.shape}"

    # Verify each head's weights are a valid probability distribution
    print(f"\nPer-head weight row sums (each should be 1.0):")
    for h, w in enumerate(head_weights):
        row_sums = w.sum(axis=-1)
        assert np.allclose(row_sums, 1.0), f"Head {h} rows don't sum to 1!"
        print(f"  Head {h + 1}: {row_sums.round(4)}  OK")

    # Show that heads differ (they'd be identical if there were no specialization)
    print(f"\nDo different heads produce different patterns?")
    for h in range(H - 1):
        diff = np.mean(np.abs(head_weights[h] - head_weights[h + 1]))
        print(f"  Mean |head_{h+1} - head_{h+2}| = {diff:.4f}  (>0 means different)")

    # Causal mask check — no future tokens attended to
    print(f"\nCausal mask verification (future weight should be ~0):")
    for i in range(seq_len):
        future_weight = sum(head_weights[0][i, i + 1:])
        assert np.isclose(future_weight, 0.0, atol=1e-9)
    print("  All heads: future tokens have zero weight  OK")
    print("PASSED\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Module 10: Attention Mechanism — Exercises\n")

    # Exercise 1: implement basic attention, get back the function for reuse
    attn_fn = exercise1_scaled_dot_product_attention()

    # Exercise 2: apply causal mask, verify temporal constraint
    exercise2_causal_mask(attn_fn)

    # Exercise 3 (challenge): full multi-head attention
    exercise3_multi_head_attention()

    print("All exercises complete.")
    print()
    print("Key takeaways:")
    print("  - Attention = soft database lookup: Q queries, K indexes, V values")
    print("  - Scaling by sqrt(d_k) prevents vanishing gradients as model grows")
    print("  - Causal mask = append-only log constraint baked into attention math")
    print("  - Multi-head = parallel queries with different indexes, results merged")


if __name__ == "__main__":
    main()
