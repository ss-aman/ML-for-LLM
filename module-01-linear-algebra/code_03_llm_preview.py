"""
Module 01 — LLM Forward Pass Preview
======================================
This ties everything together: a minimal but complete forward pass through
one transformer "block" using only linear algebra.

No PyTorch. No magic. Just numpy.

This is a PREVIEW — you'll implement each piece properly in later modules.
Goal here: see that the whole thing is just matrices.

Run: python code_03_llm_preview.py
"""

import numpy as np

np.random.seed(42)


# =============================================================================
# BUILDING BLOCKS
# =============================================================================

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Convert raw scores to probabilities.
    Each row sums to 1. Higher score → higher probability.
    Subtract max per row for numerical stability.
    """
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation: max(0, x). Sets negatives to zero."""
    return np.maximum(0, x)


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Normalize x to have mean=0, std=1 across the feature dimension.
    Keeps training stable by preventing activations from exploding/vanishing.
    """
    mean = x.mean(axis=-1, keepdims=True)
    std  = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


# =============================================================================
# TRANSFORMER COMPONENTS
# =============================================================================

class TinyEmbedding:
    """
    A lookup table mapping token IDs to dense vectors.
    Shape: (vocab_size, d_model)
    """
    def __init__(self, vocab_size: int, d_model: int):
        self.table = np.random.randn(vocab_size, d_model) * 0.1
        self.shape = (vocab_size, d_model)

    def forward(self, token_ids: list) -> np.ndarray:
        """Look up embeddings for a list of token IDs. Returns (seq_len, d_model)."""
        return self.table[token_ids]   # just row indexing!


class TinySelfAttention:
    """
    Single-head self-attention.
    The core mechanism: each token attends to all other tokens.

    Full formula: Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
    """
    def __init__(self, d_model: int, d_k: int):
        self.d_k = d_k
        # Three weight matrices (learned during training)
        self.W_q = np.random.randn(d_k, d_model) * 0.1
        self.W_k = np.random.randn(d_k, d_model) * 0.1
        self.W_v = np.random.randn(d_k, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_k) * 0.1

    def forward(self, X: np.ndarray, causal_mask: bool = True) -> np.ndarray:
        """
        X: (seq_len, d_model)
        Returns: (seq_len, d_model)
        """
        seq_len, d_model = X.shape

        # Project to Q, K, V — three separate linear transformations
        Q = X @ self.W_q.T    # (seq_len, d_k)
        K = X @ self.W_k.T    # (seq_len, d_k)
        V = X @ self.W_v.T    # (seq_len, d_k)

        # Compute attention scores: every query vs every key
        scores = Q @ K.T / np.sqrt(self.d_k)   # (seq_len, seq_len)

        if causal_mask:
            # Mask future tokens (each position can only see previous positions)
            # This is what makes GPT auto-regressive: position i can't see i+1, i+2, ...
            mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
            scores[mask] = -1e9   # -infinity before softmax → 0 after softmax

        # Softmax: convert scores to weights that sum to 1
        attn_weights = softmax(scores)   # (seq_len, seq_len)

        # Weighted sum of values
        attn_output = attn_weights @ V   # (seq_len, d_k)

        # Project back to d_model
        output = attn_output @ self.W_o.T   # (seq_len, d_model)

        return output, attn_weights


class TinyFFN:
    """
    Feed-Forward Network: two linear layers with ReLU in between.
    Processes each token independently (no cross-token interaction).
    Stores "factual knowledge" learned during training.
    """
    def __init__(self, d_model: int, d_ff: int):
        # d_ff is typically 4 × d_model
        self.W1 = np.random.randn(d_ff, d_model) * 0.1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_model, d_ff) * 0.1
        self.b2 = np.zeros(d_model)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        X: (seq_len, d_model)
        Returns: (seq_len, d_model)
        """
        # Expand dimension, apply ReLU, compress back
        h = relu(X @ self.W1.T + self.b1)   # (seq_len, d_ff)
        return h @ self.W2.T + self.b2       # (seq_len, d_model)


class TinyTransformerBlock:
    """
    One full transformer block:
      x → LayerNorm → Attention → +x (residual)
        → LayerNorm → FFN       → +x (residual)
    """
    def __init__(self, d_model: int, d_k: int, d_ff: int):
        self.attention = TinySelfAttention(d_model, d_k)
        self.ffn       = TinyFFN(d_model, d_ff)

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Attention sub-layer with residual connection
        attn_out, weights = self.attention.forward(layer_norm(X))
        X = X + attn_out   # residual: add input back

        # FFN sub-layer with residual connection
        ffn_out = self.ffn.forward(layer_norm(X))
        X = X + ffn_out    # residual: add input back

        return X, weights


class TinyLM:
    """
    A miniature language model with:
    - Token embedding
    - Positional encoding
    - 1 transformer block
    - Output projection to vocabulary logits

    Architecture matches GPT's design at small scale.
    """
    def __init__(self, vocab_size: int, d_model: int, d_k: int, d_ff: int, seq_len: int):
        self.vocab_size = vocab_size
        self.d_model    = d_model
        self.seq_len    = seq_len

        self.embedding   = TinyEmbedding(vocab_size, d_model)
        self.pos_enc     = self._build_positional_encoding(seq_len, d_model)
        self.transformer = TinyTransformerBlock(d_model, d_k, d_ff)
        self.W_out       = np.random.randn(vocab_size, d_model) * 0.1

    def _build_positional_encoding(self, seq_len: int, d_model: int) -> np.ndarray:
        """
        Sinusoidal positional encoding. Each position gets a unique vector.
        Formula: PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
                 PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))
        """
        PE = np.zeros((seq_len, d_model))
        positions = np.arange(seq_len)[:, None]    # (seq_len, 1)
        dims      = np.arange(0, d_model, 2)       # [0, 2, 4, ...]
        freqs     = 1.0 / (10000 ** (dims / d_model))

        PE[:, 0::2] = np.sin(positions * freqs)
        PE[:, 1::2] = np.cos(positions * freqs)
        return PE

    def forward(self, token_ids: list):
        """
        Full forward pass: token IDs → probability distribution over next token.

        token_ids: list of integers (token sequence)
        Returns: logits (vocab_size,) for the last position
        """
        seq = len(token_ids)

        # 1. Token embeddings (matrix row lookup)
        X = self.embedding.forward(token_ids)      # (seq, d_model)

        # 2. Add positional encoding (vector addition)
        X = X + self.pos_enc[:seq]                  # (seq, d_model)

        # 3. Transformer block
        X, attn_weights = self.transformer.forward(X)  # (seq, d_model)

        # 4. Output projection: map last token's vector to vocab logits
        logits = X[-1] @ self.W_out.T               # (vocab_size,)

        return logits, attn_weights


# =============================================================================
# DEMO
# =============================================================================

def run_demo():
    print("=" * 60)
    print("Tiny LLM Forward Pass — Pure NumPy")
    print("=" * 60)

    # Vocabulary (10 words for simplicity)
    vocab = ["<pad>", "the", "cat", "sat", "on", "mat", "dog", "ran", "fast", "<eos>"]
    word2id = {w: i for i, w in enumerate(vocab)}
    id2word = {i: w for i, w in enumerate(vocab)}

    # Model parameters
    VOCAB_SIZE = len(vocab)   # 10
    D_MODEL    = 16           # embedding dimension (GPT-2 uses 768)
    D_K        = 8            # attention head dimension
    D_FF       = 32           # feed-forward hidden dimension
    SEQ_LEN    = 8            # max sequence length

    model = TinyLM(VOCAB_SIZE, D_MODEL, D_K, D_FF, SEQ_LEN)

    print(f"\nModel parameters:")
    print(f"  Vocab size:          {VOCAB_SIZE}")
    print(f"  Embedding dimension: {D_MODEL}")
    print(f"  Attention d_k:       {D_K}")
    print(f"  FFN hidden dim:      {D_FF}")
    print(f"\nApprox parameter count:")
    emb_params  = VOCAB_SIZE * D_MODEL
    attn_params = 3 * D_K * D_MODEL + D_MODEL * D_K    # W_q, W_k, W_v, W_o
    ffn_params  = D_FF * D_MODEL + D_MODEL * D_FF      # W1, W2
    out_params  = VOCAB_SIZE * D_MODEL
    total       = emb_params + attn_params + ffn_params + out_params
    print(f"  Embedding:  {emb_params:,}")
    print(f"  Attention:  {attn_params:,}")
    print(f"  FFN:        {ffn_params:,}")
    print(f"  Output:     {out_params:,}")
    print(f"  Total:      {total:,}  (GPT-3 has 175 billion)")

    # Forward pass
    sentence = ["the", "cat", "sat"]
    token_ids = [word2id[w] for w in sentence]

    print(f"\nInput: {sentence}")
    print(f"Token IDs: {token_ids}")

    logits, attn_weights = model.forward(token_ids)

    probs = softmax(logits)

    print(f"\nOutput logits (one per vocab token): {logits.round(3)}")
    print(f"\nProbabilities over next token:")
    for word, prob in sorted(zip(vocab, probs), key=lambda x: -x[1]):
        bar = "█" * int(prob * 40)
        print(f"  {word:8s}  {prob:.4f}  {bar}")

    predicted_id   = int(np.argmax(probs))
    predicted_word = id2word[predicted_id]
    print(f"\nPredicted next token: '{predicted_word}' (id={predicted_id})")
    print(f"(Weights are random — prediction is random at this stage.)")
    print(f"After training on data, the model would learn to predict 'on' or 'mat'.")

    # Show attention weights
    print(f"\nAttention weights for this sequence:")
    print(f"  shape: {attn_weights.shape}  (seq_len × seq_len)")
    print(f"  attn_weights[i][j] = how much token i attends to token j")
    print()
    header = "         " + " ".join(f"{w:6s}" for w in sentence)
    print(header)
    for i, word in enumerate(sentence):
        row = " ".join(f"{attn_weights[i][j]:.3f} " for j in range(len(sentence)))
        print(f"  {word:6s}   {row}")

    print("\n" + "=" * 60)
    print("Summary: every operation above was just matrix multiplications,")
    print("vector additions, and softmax. That IS an LLM forward pass.")
    print("=" * 60)


if __name__ == '__main__':
    run_demo()
