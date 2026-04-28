# 05 — Special Matrices and Eigenvectors

## The Identity Matrix

The identity matrix `I` is the matrix equivalent of the number 1.

Multiplying anything by it leaves it unchanged:

```
A @ I = A
I @ A = A
```

It has 1s on the diagonal and 0s everywhere else:

```python
np.eye(3)
# [[1., 0., 0.],
#  [0., 1., 0.],
#  [0., 0., 1.]]
```

**Where it appears in LLMs:** Residual connections effectively add the identity
function. When you do `output = x + F(x)`, the network can learn to make
`F(x)` approach zero, leaving only the identity path. This makes deep networks
easier to train — at initialization, the whole network approximates identity,
and learning happens incrementally.

---

## The Transpose (Revisited for Multiplication)

You saw the transpose in `03_matrices.md` (swaps rows and columns). Its most
important use is **aligning shapes** for multiplication.

Key identity: `(A @ B).T = B.T @ A.T`

The order reverses when you transpose a product.

**The attention formula:**

```
scores = Q @ K.T    # (seq_len, d_k) @ (d_k, seq_len) → (seq_len, seq_len)
```

Both `Q` and `K` have shape `(seq_len, d_k)`. To get a `(seq_len, seq_len)`
score matrix (every query vs every key), you transpose `K` so the shapes work:

```
Q:   (seq_len, d_k)
K.T: (d_k, seq_len)
     (d_k) cancels → result: (seq_len, seq_len)
```

Every entry `scores[i, j]` = dot product of query i with key j = "how much
should token i attend to token j?"

---

## The Inverse

For a square matrix `A`, its inverse `A⁻¹` satisfies:

```
A @ A⁻¹ = I
A⁻¹ @ A = I
```

```python
A = np.array([[4., 7.], [2., 6.]])
A_inv = np.linalg.inv(A)
A @ A_inv  # → identity matrix (up to floating point noise)
```

**Intuition:** If `A` encodes a transformation (rotate + stretch), `A⁻¹`
encodes the exact reverse (un-rotate + un-stretch).

**Backend analogy:** Encryption / decryption. `encrypt(x) = A @ x`,
`decrypt(y) = A⁻¹ @ y`.

**Important for ML:** You almost never compute inverses explicitly in practice
because:
1. It's numerically unstable
2. It's expensive: O(n³) for an n×n matrix
3. Usually you want to solve `A @ x = b` for `x` — use `np.linalg.solve(A, b)`
   which is faster and more stable

The inverse mostly appears in mathematical derivations, not in running code.

---

## Eigenvectors and Eigenvalues

This is a slightly more advanced concept. You can come back to it after Module 06.

### The Definition

An **eigenvector** of matrix `A` is a special vector `v` where:

```
A @ v = λ * v
```

Applying the matrix transformation to `v` only **scales** it — it doesn't
change direction. `λ` (lambda) is the **eigenvalue**: the scaling factor.

### The Intuition

Most vectors, when you apply a matrix transformation, get rotated AND scaled.
Eigenvectors are special: they only get scaled, never rotated.

```python
A = np.array([[3., 0.],
              [0., 2.]])

eigenvalues, eigenvectors = np.linalg.eig(A)
# eigenvalues:  [3., 2.]
# eigenvectors: [[1., 0.],   ← [1,0] scaled by 3
#                [0., 1.]]   ← [0,1] scaled by 2
```

For this diagonal matrix, the eigenvectors are just the coordinate axes — makes
sense, since the matrix only stretches each axis independently.

### Why This Matters for ML

**1. PCA (Principal Component Analysis)**

Given a dataset (a matrix of shape `(n_samples, n_features)`), the eigenvectors
of the covariance matrix point in the directions of maximum variance.

Think of it as finding the "natural axes" of your data. If your data forms an
elongated ellipse in 2D, the eigenvectors point along the long axis and the
short axis. This is used for:
- Dimensionality reduction (keep only the top-k eigenvectors)
- Understanding what structure exists in the data
- Visualizing high-dimensional embeddings (Module 06 uses PCA to plot 2D views
  of word embeddings)

**2. Understanding what a model learned**

The eigenvectors of attention weight matrices (or embedding matrices) reveal
the "principal concepts" the model captured. Researchers use this to interpret
what a layer "encodes."

**3. Numerical stability**

The largest eigenvalue of a weight matrix relates to how quickly gradients
grow or shrink during backpropagation. Very large eigenvalues → exploding
gradients. Very small → vanishing gradients. This is why weight initialization
and normalization matter (Module 08).

### Quick Code Demo

```python
# A matrix that stretches x by 3×, y by 1×
A = np.array([[3., 0.],
              [0., 1.]])

v = np.array([1., 0.])   # x-axis direction
A @ v   # → [3., 0.] — same direction, scaled by 3 ← eigenvector!

u = np.array([1., 1.])   # diagonal direction (NOT an eigenvector)
A @ u   # → [3., 1.] — DIFFERENT direction! not an eigenvector
```

---

## Summary

| Concept | What It Is | Where in LLMs |
|---|---|---|
| Identity matrix `I` | The "do nothing" transformation | Residual connections (`x + F(x)`) |
| Transpose `A.T` | Swap rows and columns | `Q @ K.T` in attention |
| Inverse `A⁻¹` | Undo the transformation | Derivations, not runtime code |
| Eigenvector | Direction unchanged by transformation | PCA, interpreting learned weights |
| Eigenvalue | Scale factor for the eigenvector | Stability analysis, weight init |

---

## What's Next

`06_how_linalg_powers_llms.md`: We put everything together and walk through
a transformer's forward pass entirely in linear algebra terms.
