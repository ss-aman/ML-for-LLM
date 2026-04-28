"""
Module 10: Attention Mechanism — Code Implementation

Implements single-head and multi-head scaled dot-product attention from scratch
using only NumPy. Demonstrates causal masking and visualizes attention weights.

Backend analogy: We're building the core "soft database lookup" engine from scratch.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ---------------------------------------------------------------------------
# 1. Scaled Dot-Product Attention (single head)
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Core attention operation: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    Args:
        Q: Query matrix  shape (seq_len, d_k)
        K: Key matrix    shape (seq_len, d_k)
        V: Value matrix  shape (seq_len, d_v)
        mask: Optional boolean/float mask shape (seq_len, seq_len)
              True/-inf positions are masked out (set to -inf before softmax)

    Returns:
        output: shape (seq_len, d_v)  — context-aware blended values
        weights: shape (seq_len, seq_len) — attention weights (for visualization)
    """
    d_k = Q.shape[-1]

    # Step 1: Compute raw similarity scores — every query vs every key
    # Q @ K.T shape: (seq_len, seq_len)
    # Entry [i,j] = "how much should token i attend to token j?"
    scores = Q @ K.T                    # (seq_len, seq_len)

    # Step 2: Scale to prevent dot products from getting too large
    # Without this, softmax saturates → gradients vanish → model stops learning
    scores = scores / np.sqrt(d_k)      # still (seq_len, seq_len)

    # Step 3: Apply mask (for causal attention — block future tokens)
    if mask is not None:
        scores = scores + mask          # -inf entries become -inf after addition

    # Step 4: Softmax — convert scores to probability weights (each row sums to 1)
    # Backend analogy: like normalizing load-balancing weights so they sum to 100%
    weights = softmax(scores)           # (seq_len, seq_len)

    # Step 5: Weighted sum of Values — the actual "retrieval"
    # Backend analogy: each token gets a weighted blend of all values,
    # where weights came from how relevant each key was to the query
    output = weights @ V                # (seq_len, d_v)

    return output, weights


def softmax(x):
    """
    Numerically stable softmax along the last axis.
    Subtracts max before exp to avoid overflow (standard trick).
    """
    # Subtract max for numerical stability (doesn't change output, prevents overflow)
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# 2. Causal Mask
# ---------------------------------------------------------------------------

def make_causal_mask(seq_len):
    """
    Create a causal (lower-triangular) mask for autoregressive attention.

    Returns a matrix where upper-triangular positions are -inf.
    After softmax, these positions get weight 0 — effectively invisible.

    Backend analogy: like an append-only event log constraint —
    when processing event at position i, only events 0..i are visible.

    Shape: (seq_len, seq_len)
    mask[i, j] = 0      if j <= i  (position j is visible to position i)
    mask[i, j] = -inf   if j > i   (position j is in the future — blocked)
    """
    # np.triu gets upper triangle (above diagonal)
    # k=1 means start one above the main diagonal (so diagonal itself is 0)
    mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
    return mask


def verify_causal_mask(seq_len=4):
    """
    Verify that causal masking works correctly:
    - Token at position 0 attends only to position 0
    - Token at position 1 attends to positions 0 and 1
    - Token at position i attends to positions 0..i
    """
    print("\n--- Causal Mask Verification ---")
    mask = make_causal_mask(seq_len)
    print(f"Causal mask (0=visible, -inf=blocked) for seq_len={seq_len}:")
    print(mask)

    # Use random Q, K, V and apply the mask
    np.random.seed(42)
    d_k = 8
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_k)

    _, weights = scaled_dot_product_attention(Q, K, V, mask=mask)

    print("\nAttention weights after causal masking (rows = queries, cols = keys):")
    np.set_printoptions(precision=3, suppress=True)
    print(weights)

    print("\nVerification:")
    for i in range(seq_len):
        future_weight = np.sum(weights[i, i+1:])
        print(f"  Token {i}: weight on future tokens = {future_weight:.6f} (should be ~0)")

    return weights


# ---------------------------------------------------------------------------
# 3. Multi-Head Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention:
    """
    Multi-Head Attention: run attention H times with different learned projections.

    Each head learns to attend to different relationship types:
    - Head 1 might focus on syntactic relationships
    - Head 2 might focus on semantic similarity
    - Head 3 might focus on positional proximity
    etc.

    Backend analogy: like parallel database queries with different indexes —
    each "replica" specializes in a different lookup strategy, then results are merged.
    """

    def __init__(self, d_model, num_heads, seed=42):
        """
        Args:
            d_model: Full model dimension (e.g., 64)
            num_heads: Number of parallel attention heads (e.g., 4)
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads   # per-head dimension

        rng = np.random.RandomState(seed)

        # Projection matrices for Q, K, V — one per head
        # Shape: (num_heads, d_model, d_k)
        scale = np.sqrt(2.0 / d_model)
        self.W_Q = rng.randn(num_heads, d_model, self.d_k) * scale
        self.W_K = rng.randn(num_heads, d_model, self.d_k) * scale
        self.W_V = rng.randn(num_heads, d_model, self.d_k) * scale

        # Output projection: maps concatenated heads back to d_model
        # Input: num_heads * d_k = d_model, Output: d_model
        self.W_O = rng.randn(d_model, d_model) * scale

    def forward(self, X, mask=None):
        """
        Args:
            X: Input shape (seq_len, d_model)
            mask: Optional causal mask shape (seq_len, seq_len)

        Returns:
            output: shape (seq_len, d_model)
            all_weights: list of (seq_len, seq_len) attention weight matrices, one per head
        """
        seq_len = X.shape[0]
        head_outputs = []
        all_weights = []

        for h in range(self.num_heads):
            # Project input to Q, K, V for this head
            # X @ W_Q[h]: (seq_len, d_model) @ (d_model, d_k) = (seq_len, d_k)
            Q_h = X @ self.W_Q[h]
            K_h = X @ self.W_K[h]
            V_h = X @ self.W_V[h]

            # Run attention for this head
            head_out, weights = scaled_dot_product_attention(Q_h, K_h, V_h, mask=mask)
            head_outputs.append(head_out)    # (seq_len, d_k)
            all_weights.append(weights)       # (seq_len, seq_len)

        # Concatenate all heads: (seq_len, num_heads * d_k) = (seq_len, d_model)
        # This preserves all H perspectives — averaging would lose information
        concat = np.concatenate(head_outputs, axis=-1)

        # Final linear projection: blend the multi-head perspectives
        output = concat @ self.W_O           # (seq_len, d_model)

        return output, all_weights


# ---------------------------------------------------------------------------
# 4. Visualization: Attention Heatmaps
# ---------------------------------------------------------------------------

def visualize_attention_weights(weights, tokens, title="Attention Weights"):
    """
    Plot attention weights as a heatmap.

    Rows = queries (which token is "looking")
    Cols = keys    (which token is "being looked at")
    Color = weight (bright = high attention)
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(weights, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Attention weight")

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_yticklabels(tokens)
    ax.set_xlabel("Keys (token being attended to)")
    ax.set_ylabel("Queries (token doing the attending)")
    ax.set_title(title)

    # Annotate cells with weight values
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            color = "white" if weights[i, j] > 0.6 else "black"
            ax.text(j, i, f"{weights[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color=color)

    plt.tight_layout()
    return fig


def visualize_multi_head(all_weights, tokens, num_heads):
    """Plot attention weights for each head in a grid."""
    cols = min(4, num_heads)
    rows = (num_heads + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    if num_heads == 1:
        axes = [axes]
    axes = np.array(axes).flatten()

    for h, (weights, ax) in enumerate(zip(all_weights, axes)):
        im = ax.imshow(weights, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(tokens, fontsize=8)
        ax.set_title(f"Head {h+1}")
        plt.colorbar(im, ax=ax)

    # Hide any unused subplots
    for ax in axes[num_heads:]:
        ax.set_visible(False)

    fig.suptitle("Multi-Head Attention Weights", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Demo: "The cat sat" — who attends to whom?
# ---------------------------------------------------------------------------

def demo_sentence_attention():
    """
    Demo: Given ["The", "cat", "sat"], show which tokens attend to which.

    We create simple hand-crafted embeddings where:
    - "The" has a high-dimension signal for articles
    - "cat" has a signal for nouns
    - "sat" has a signal for verbs

    Then we watch how attention distributes across tokens.
    """
    print("\n--- Demo: Attention on ['The', 'cat', 'sat'] ---")

    tokens = ["The", "cat", "sat"]
    seq_len = len(tokens)
    d_k = 8

    # Seed for reproducibility — in a real model, Q/K/V are learned projections
    np.random.seed(123)

    # Simulate Q, K, V as if they were learned projections of token embeddings
    # (In a real transformer, these come from: token_embedding @ W_Q etc.)
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_k)

    # --- Without causal mask (bidirectional — like BERT) ---
    output_bidir, weights_bidir = scaled_dot_product_attention(Q, K, V)
    print("\nBidirectional attention weights (each token can see all others):")
    print(weights_bidir.round(3))

    fig1 = visualize_attention_weights(
        weights_bidir, tokens, "Bidirectional Attention — 'The cat sat'"
    )
    fig1.savefig("/home/user/ML-for-LLM/module-10-attention/attention_bidirectional.png",
                 dpi=150, bbox_inches="tight")
    print("  Saved: attention_bidirectional.png")

    # --- With causal mask (autoregressive — like GPT) ---
    mask = make_causal_mask(seq_len)
    output_causal, weights_causal = scaled_dot_product_attention(Q, K, V, mask=mask)
    print("\nCausal attention weights (each token sees only itself + past):")
    print(weights_causal.round(3))

    fig2 = visualize_attention_weights(
        weights_causal, tokens, "Causal Attention — 'The cat sat'"
    )
    fig2.savefig("/home/user/ML-for-LLM/module-10-attention/attention_causal.png",
                 dpi=150, bbox_inches="tight")
    print("  Saved: attention_causal.png")

    return weights_bidir, weights_causal


def demo_multi_head_attention():
    """
    Demo: Multi-head attention on ["The", "cat", "sat"].
    Shows how different heads can develop different attention patterns.
    """
    print("\n--- Demo: Multi-Head Attention ---")

    tokens = ["The", "cat", "sat"]
    seq_len = len(tokens)
    d_model = 16
    num_heads = 4

    # Create a simple "embedding" for each token (random, for demo purposes)
    np.random.seed(42)
    X = np.random.randn(seq_len, d_model)

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, seed=7)

    # Without causal mask (bidirectional)
    output, all_weights = mha.forward(X)

    print(f"Input shape:  {X.shape}       (seq_len={seq_len}, d_model={d_model})")
    print(f"Output shape: {output.shape}  (same shape — attention is a transformation)")
    print(f"Number of heads: {num_heads}, d_k per head: {d_model // num_heads}")
    print("\nAttention weights per head:")
    for h, w in enumerate(all_weights):
        print(f"  Head {h+1}: {w.round(3)}")

    fig = visualize_multi_head(all_weights, tokens, num_heads)
    fig.savefig("/home/user/ML-for-LLM/module-10-attention/attention_multihead.png",
                dpi=150, bbox_inches="tight")
    print("  Saved: attention_multihead.png")

    # Also run with causal mask
    mask = make_causal_mask(seq_len)
    output_causal, weights_causal = mha.forward(X, mask=mask)
    print(f"\nWith causal mask — head 1 weights:\n{weights_causal[0].round(3)}")

    return all_weights


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Module 10: Attention Mechanism")
    print("=" * 60)

    # 1. Verify causal masking
    causal_weights = verify_causal_mask(seq_len=4)

    # 2. Demo: sentence attention
    w_bidir, w_causal = demo_sentence_attention()

    # 3. Demo: multi-head attention
    mha_weights = demo_multi_head_attention()

    print("\n--- Attention Formula Recap ---")
    print("Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V")
    print()
    print("Steps:")
    print("  1. scores   = Q @ K.T          — similarity of every query to every key")
    print("  2. scaled   = scores / sqrt(dk) — prevent gradient vanishing")
    print("  3. masked   = scaled + mask     — block future positions (causal LMs)")
    print("  4. weights  = softmax(masked)   — convert scores to probability weights")
    print("  5. output   = weights @ V       — weighted blend of values")
    print()
    print("Multi-Head: run H times with different Q/K/V projections, concatenate results.")
    print("Each head specializes in different relationship types.")
    print()
    print("Saved plots to module-10-attention/")
