# 06 — How Linear Algebra Powers LLMs

## The Payoff

You've learned vectors, dot products, matrices, and matrix multiplication.
Now let's see exactly where each concept appears in a real LLM.

We'll trace what happens when GPT processes the sentence:

```
"the cat sat"
```

---

## Step 1: Tokenization → Integer IDs (not linalg yet)

First, each word gets converted to an integer token ID. This is just a
dictionary lookup (covered in Module 12). For now, assume:

```python
tokens = [the=464, cat=3797, sat=3332]
```

---

## Step 2: Token Embedding → Vectors

**Linear algebra concept: Matrix row lookup**

The embedding table is a matrix `E` of shape `(vocab_size, d_model)`.
Each row is the embedding vector for one token.

```python
vocab_size = 50257   # GPT-2
d_model    = 768

E = np.random.randn(vocab_size, d_model)   # shape (50257, 768)

# Look up embeddings for our 3 tokens:
token_ids = [464, 3797, 3332]
X = E[token_ids]   # shape (3, 768) — 3 vectors, each 768-dim
```

At this point `X` is a matrix of shape `(seq_len=3, d_model=768)`.
Each row is the initial "meaning vector" for one token.

---

## Step 3: Positional Encoding → Adding Position to Vectors

**Linear algebra concept: Vector addition**

Transformers have no built-in sense of order. To tell the model that "cat" is
token #2 (not token #1 or #3), we add a position-encoding vector to each
embedding:

```python
PE = compute_positional_encoding(seq_len=3, d_model=768)  # shape (3, 768)
X = X + PE   # element-wise vector addition, shape still (3, 768)
```

Each token's embedding now encodes BOTH its meaning AND its position.

---

## Step 4: Self-Attention — The Core of a Transformer

This is where it all comes together. The attention mechanism computes:

```
Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
```

Every symbol here is a linear algebra operation. Let's unpack it.

### 4a. Project to Q, K, V

Three weight matrices project the input `X` into Query, Key, and Value spaces:

```python
d_k = 64   # dimension per attention head

W_q = np.random.randn(d_k, d_model)   # shape (64, 768)
W_k = np.random.randn(d_k, d_model)   # shape (64, 768)
W_v = np.random.randn(d_k, d_model)   # shape (64, 768)

# Project each token's embedding into Q, K, V spaces:
Q = X @ W_q.T   # shape (3, 64) — 3 query vectors
K = X @ W_k.T   # shape (3, 64) — 3 key vectors
V = X @ W_v.T   # shape (3, 64) — 3 value vectors
```

**Intuition:**
- `Q` — "what am I looking for?" (one query vector per token)
- `K` — "what do I offer?" (one key vector per token)
- `V` — "what I'll actually share if attended to" (one value vector per token)

This is literally a database metaphor: Query → Key → Value lookup.

### 4b. Compute Attention Scores

```python
scores = Q @ K.T   # shape (3, 3)
# scores[i][j] = dot product of query_i with key_j
# = "how much should token i attend to token j?"
```

This produces a `(seq_len × seq_len)` matrix. For our 3-token example:

```
          "the"  "cat"  "sat"
"the"   [ s00,   s01,   s02 ]
"cat"   [ s10,   s11,   s12 ]
"sat"   [ s20,   s21,   s22 ]
```

Each row tells one token how much it wants to look at every other token.

### 4c. Scale and Normalize

```python
import numpy as np

scores = scores / np.sqrt(d_k)    # scale down — vector scalar multiply
weights = softmax(scores)          # convert to probabilities (row-wise)
# weights[i] is a probability distribution over which tokens token i attends to
```

The softmax converts raw scores into a probability distribution that sums to 1.
High-score tokens get most of the attention weight.

### 4d. Weighted Sum of Values

```python
output = weights @ V   # shape (3, 64)
# output[i] = weighted average of value vectors,
#             weighted by how much token i attended to each token
```

This is the attention output: each token's new representation is a **blend**
of all value vectors, weighted by attention scores.

The entire attention operation in one line:
```python
output = softmax(X @ W_q.T @ (X @ W_k.T).T / np.sqrt(d_k)) @ (X @ W_v.T)
```

**100% linear algebra.** Dot products. Matrix multiplications. Vector addition.

---

## Step 5: Feed-Forward Network (FFN)

After attention, each token's vector passes through two linear layers:

```python
W1 = np.random.randn(d_ff, d_model)   # shape (3072, 768)  — expand
b1 = np.zeros(d_ff)                    # shape (3072,)
W2 = np.random.randn(d_model, d_ff)   # shape (768, 3072)  — compress
b2 = np.zeros(d_model)                 # shape (768,)

# For one token's vector x of shape (768,):
h = relu(W1 @ x + b1)   # (3072,) — expand to 4× width
y = W2 @ h + b2          # (768,)  — compress back to original size
```

This FFN stores "factual knowledge" learned during training.
The expansion to 4× width (`d_ff = 4 × d_model`) gives it capacity to store
patterns that attention alone can't represent.

---

## Step 6: Residual Connections

After both attention and the FFN, the original vector `x` is added back:

```python
x = x + attention_output   # vector addition
x = x + ffn_output         # vector addition
```

This is the "residual connection" or "skip connection." It passes the original
signal through unchanged, and the layer only needs to learn the **difference**
(residual) from the input. This makes training much more stable (Module 08).

---

## Step 7: Repeat × N Layers

The same transformer block (attention + FFN + residuals) runs `N` times.
GPT-2 small: 12 layers. GPT-3: 96 layers.

Each layer refines the token representations. Early layers capture syntax,
middle layers capture semantics, later layers compute task-specific outputs.

---

## Step 8: Output Projection → Logits

After all transformer blocks, a final linear layer maps each token's
`d_model`-dimensional vector to a `vocab_size`-dimensional vector:

```python
W_out = np.random.randn(vocab_size, d_model)   # shape (50257, 768)
logits = x @ W_out.T   # shape (50257,) — one score per vocabulary token
```

The highest score is the model's prediction for the next token.
Apply softmax to get probabilities. Sample from them. That's generation.

---

## The Full Picture

```
tokens  → [embedding matrix lookup]  → (seq_len, d_model)
        → [+ positional encoding]    → (seq_len, d_model)
        → [× N transformer blocks]:
            ├─ [Q, K, V projections] → matrix multiplications
            ├─ [attention scores]    → Q @ K.T
            ├─ [softmax]             → normalize rows
            ├─ [weighted sum]        → weights @ V
            ├─ [FFN]                 → two matrix multiplications
            └─ [residuals]           → vector additions
        → [output projection]        → matrix multiplication
        → [softmax]                  → probability distribution
        → next token prediction
```

**Every single box in this diagram is linear algebra.**

---

## Why This Module Is the Foundation

| Concept | Where It Appears |
|---|---|
| Vector | Every token representation, every embedding |
| Vector addition | Residual connections, positional encoding |
| Dot product | Attention scoring, neuron computation |
| Cosine similarity | Semantic search, measuring embedding similarity |
| Matrix | Weight matrices, embedding table, attention score grid |
| Matrix multiplication | Linear layers, Q/K/V projections, attention output |
| Transpose | `Q @ K.T` in attention |
| Eigenvectors | PCA of embeddings, weight analysis (advanced) |

---

## Key Takeaway

> An LLM's forward pass is entirely composed of the operations you just
> learned. There is no magic — only vectors, matrix multiplications, and
> a few nonlinearities. Understanding linear algebra means you can read
> and implement transformer code without treating it as a black box.
