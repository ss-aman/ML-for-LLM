# 03 — Matrices: Organizing Vectors into Tables

## What Is a Matrix?

A matrix is a 2D grid of numbers — rows and columns.

```python
W = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
```

We describe its size as `rows × columns`. This matrix is `3 × 3`.

```python
import numpy as np
W = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
W.shape  # → (3, 3)
```

---

## Three Ways to Think About a Matrix

### 1. As a table of data

The most familiar view. A database table, a spreadsheet, a CSV file.
Each row is one record, each column is one feature.

```
         feature_1  feature_2  feature_3
user_1  [  0.45,     127,        85.3   ]
user_2  [  0.72,      43,        210.0  ]
user_3  [  0.12,     891,        12.5   ]
```

Shape: `(3 users, 3 features)` = `(3, 3)` matrix.

### 2. As a collection of row vectors (or column vectors)

Every row is a vector. Stack N vectors of dimension D → matrix of shape `(N, D)`.

```python
# 3 word embeddings, each 4-dimensional
embedding_table = np.array([
    [0.2, -0.5,  1.3,  0.8],   # "cat"
    [0.9,  0.1,  0.7, -0.2],   # "dog"
    [0.1,  0.8, -0.4,  0.6],   # "fish"
])
# shape: (3, 4) — 3 words, each is a 4D vector
```

This is EXACTLY what an LLM's embedding table is. Shape `(vocab_size, d_model)`.
To look up the embedding for a token, you grab the corresponding row.

### 3. As a function (linear transformation)

This is the most powerful view. A matrix defines a **function** that transforms
input vectors into output vectors.

If `W` is a `(4, 3)` matrix and `x` is a `(3,)` vector, then `W @ x` is a
`(4,)` vector — a transformation from 3D space to 4D space.

```python
W = np.random.randn(4, 3)  # a 4×3 weight matrix
x = np.array([1.0, 2.0, 3.0])  # a 3D input vector
y = W @ x   # y is a 4D output vector
```

**Backend analogy:** Think of `W` as a function definition and `W @ x` as
calling that function with argument `x`. The matrix encodes a learned
transformation. During training, the matrix's values are adjusted so the
transformation does something useful.

---

## The Embedding Table: Your First Real-World Matrix

In an LLM:
- The vocabulary has `V` tokens (e.g., V = 50,257 in GPT-2)
- Each token is represented by a `d_model`-dimensional vector
- All these vectors are stored as one big matrix: shape `(V, d_model)`

```python
V = 50257       # vocab size
d_model = 768   # embedding dimension

embedding_table = np.random.randn(V, d_model)
# shape: (50257, 768) — roughly 38.6 million numbers

# To look up token 423 (e.g., the token for "cat"):
token_id = 423
embedding = embedding_table[token_id]  # shape: (768,)
```

This is just a row lookup. The "intelligence" comes from the values stored in
those rows — which are learned during training over billions of examples.

---

## Indexing and Slicing

```python
M = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

M[0]        # first row:    [1, 2, 3]
M[:, 0]     # first column: [1, 4, 7]
M[1, 2]     # row 1, col 2:  6
M[0:2, :]   # first 2 rows: [[1,2,3],[4,5,6]]
```

---

## Shapes — The Most Important Thing to Track

In ML code, shape errors are the #1 cause of bugs. Get comfortable reasoning
about shapes before writing a single line of model code.

```python
A = np.zeros((3, 4))   # 3 rows, 4 columns — shape (3, 4)
B = np.zeros((4, 5))   # 4 rows, 5 columns — shape (4, 5)
C = A @ B              # valid! → shape (3, 5)

# Rule: (m, k) @ (k, n) → (m, n)
# The inner dimensions (k) must match.
```

A useful mental trick: read shapes like fractions that cancel:
```
(3, 4) @ (4, 5)
      ^---^ these cancel
→     (3,    5)
```

**Common shape patterns in LLMs:**

| Operation | Shape |
|---|---|
| Token embedding lookup | `(vocab_size, d_model)` |
| Input to an attention layer | `(seq_len, d_model)` |
| Weight matrix of a linear layer | `(d_out, d_in)` |
| Attention score matrix | `(seq_len, seq_len)` |
| Batch of inputs | `(batch_size, seq_len, d_model)` |

---

## Creating Matrices in NumPy

```python
# All zeros
np.zeros((3, 4))

# All ones
np.ones((3, 4))

# Identity matrix (1s on diagonal)
np.eye(3)          # 3×3 identity

# Random (Gaussian)
np.random.randn(3, 4)

# Random (uniform 0 to 1)
np.random.rand(3, 4)

# From a list
np.array([[1, 2], [3, 4]])
```

---

## The Transpose

Transpose flips rows and columns. Element `[i, j]` moves to `[j, i]`.

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])
# shape: (2, 3)

A.T
# [[1, 4],
#  [2, 5],
#  [3, 6]]
# shape: (3, 2)
```

Shape rule: `(m, n).T → (n, m)`

**When you need it:**
- Aligning dimensions for matrix multiplication
- Computing `Q @ K.T` in attention (both Q and K have the same shape — you
  transpose K so the shapes can multiply)
- Converting row vectors to column vectors

---

## Key Takeaway

> A matrix is a 2D array that can represent: a dataset (rows = samples),
> a collection of vectors (rows = embeddings), or a function (a learned
> transformation). In LLMs, every layer is defined by weight matrices that
> transform vectors from one representation to another.

---

## What's Next

In `04_matrix_multiplication.md`: how matrices multiply — the single most
important computation in all of deep learning.
