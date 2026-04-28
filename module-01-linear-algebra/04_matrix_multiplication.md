# 04 — Matrix Multiplication: The Heart of Deep Learning

## Why This File Matters

If there is ONE thing to understand deeply before studying transformers, it's
matrix multiplication. Every forward pass through a neural network is a
sequence of matrix multiplications. The attention mechanism is matrix
multiplication. Training is computing gradients through matrix multiplications.

This file explains it from multiple angles until it clicks.

---

## The Setup

To multiply matrix `A` by matrix `B`, written `A @ B` or `AB`:

1. `A` has shape `(m, k)` — m rows, k columns
2. `B` has shape `(k, n)` — k rows, n columns
3. The result `C` has shape `(m, n)` — m rows, n columns

**The inner dimensions (k) must match.**

```
A: (m × k)
B:    (k × n)
C: (m    × n)
      ^--- k cancels
```

---

## How to Compute It (Step by Step)

Each element `C[i][j]` is the **dot product** of row `i` of A with column `j` of B.

### Small example:

```
A = [[1, 2],      B = [[5, 6],
     [3, 4]]           [7, 8]]
```

Step through each output element:

```
C[0][0] = row 0 of A · col 0 of B = [1,2] · [5,7] = 1*5 + 2*7 = 19
C[0][1] = row 0 of A · col 1 of B = [1,2] · [6,8] = 1*6 + 2*8 = 22
C[1][0] = row 1 of A · col 0 of B = [3,4] · [5,7] = 3*5 + 4*7 = 43
C[1][1] = row 1 of A · col 1 of B = [3,4] · [6,8] = 3*6 + 4*8 = 50

         C = [[19, 22],
              [43, 50]]
```

```python
import numpy as np
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A @ B
# array([[19, 22],
#        [43, 50]])
```

---

## What Is It Actually Computing?

### View 1: Many dot products at once

Computing `A @ B` is the same as computing the dot product of **every row of A**
with **every column of B**, and putting the results in a grid.

This is very useful when you have:
- Many query vectors (rows of A)
- Many key vectors (columns of B)
- And want ALL query-key similarities in one shot

This is literally what attention does. Q and K are matrices, `Q @ K.T` computes
ALL query-key dot products simultaneously. Efficient.

### View 2: A function applied to every column

`A @ B` applies the transformation `A` to each column of `B`.

If `B` has n columns (n input vectors), the result `A @ B` has n columns
(n transformed vectors). You transform all n vectors with one operation.

**Backend analogy:** Like applying a map() transformation to a list of items
in parallel. `A` is the function, each column of `B` is an item.

### View 3: Composing two functions (transformations)

If matrix `A` represents the function f, and matrix `B` represents function g,
then `A @ B` represents the composed function `f(g(x))` — applying g first,
then f.

```python
# Instead of:
result = A @ (B @ x)

# You can precompute:
AB = A @ B
result = AB @ x   # same result, but AB is reused for multiple x's
```

This is why neural network layers can be combined — each layer is a matrix, and
a sequence of layers is a product of matrices.

---

## Critical Property: NOT Commutative

```python
A @ B  !=  B @ A   # in general
```

Order matters. Swapping the order of transformations gives a different result —
just like the order of middleware in a web server matters.

```python
A = np.array([[1, 2], [0, 1]])
B = np.array([[1, 0], [3, 1]])

print(A @ B)   # [[7, 2], [3, 1]]
print(B @ A)   # [[1, 2], [3, 7]]  ← different!
```

Always check your shape order.

---

## Matrix × Vector: The Most Common Case

The most common usage in neural networks is multiplying a matrix by a single
vector:

```
W: (d_out, d_in)   — weight matrix
x: (d_in,)         — input vector
y = W @ x: (d_out,) — output vector
```

This is a **linear layer** (also called a fully-connected or dense layer). It:
1. Takes input of dimension `d_in`
2. Outputs a vector of dimension `d_out`
3. The matrix `W` contains all the learned weights

In a transformer, the feed-forward network does:
```python
y = W2 @ relu(W1 @ x + b1) + b2
```

Two matrix-vector multiplications, with a nonlinearity in between.

---

## Batch Matrix Multiplication

In practice, you never process one input at a time — you process a **batch**
of inputs. NumPy and PyTorch handle this with batched operations.

```python
# Single input:
W = np.random.randn(4, 3)  # weight matrix (4, 3)
x = np.random.randn(3)     # one input (3,)
y = W @ x                  # output (4,)

# Batch of 32 inputs:
X = np.random.randn(32, 3)  # 32 inputs (32, 3)
Y = X @ W.T                 # 32 outputs (32, 4)
# Note: W.T is (3, 4) so X @ W.T gives (32, 4)
```

In transformer code you'll often see shapes like `(batch_size, seq_len, d_model)` —
a 3D tensor. The matrix multiplications operate on the last two dimensions,
processing all batch items in parallel.

---

## Real LLM Example: One Linear Layer

In GPT-2, each attention head projects the input like this:

```python
d_model = 768   # input dimension
d_head   = 64   # per-head dimension
seq_len  = 512  # number of tokens

# Weight matrix for query projection:
W_q = np.random.randn(d_head, d_model)  # shape (64, 768)

# One token's embedding:
x = np.random.randn(d_model)  # shape (768,)

# Project to query vector:
q = W_q @ x   # shape (64,) — a 64-dimensional query vector
```

This one multiplication maps a 768D embedding to a 64D query. The values in
`W_q` were learned during training to extract "query-relevant features" from
the input.

---

## Matrix Multiplication: Computational Cost

For `(m, k) @ (k, n)`, the number of multiplications is `m × k × n`.

For a single GPT-2 attention head's Q projection: `64 × 768 = 49,152` ops.
GPT-2 has 12 heads × 12 layers × 2 (forward + key) × 49,152 ≈ **14 million** ops
just for Q projections, per token.

This is why GPUs matter: they run thousands of multiplications in parallel.
A matrix multiply is embarrassingly parallel — each output element can be
computed independently.

---

## Summary

| Property | Value |
|---|---|
| Shape rule | `(m, k) @ (k, n) → (m, n)` |
| Commutative? | No: `A @ B ≠ B @ A` |
| Associative? | Yes: `(A @ B) @ C = A @ (B @ C)` |
| Each output element | Dot product of one row with one column |
| In neural networks | The core computation of every layer |
| In attention | `Q @ K.T` computes all query-key similarities |

---

## Key Takeaway

> Matrix multiplication is function application at scale: it transforms
> vectors, composes transformations, and runs efficiently on hardware. Every
> forward pass through an LLM is hundreds of matrix multiplications chained
> together.

---

## What's Next

`05_special_matrices.md`: The identity matrix, inverse, and why the
transpose shows up constantly in transformer formulas.
