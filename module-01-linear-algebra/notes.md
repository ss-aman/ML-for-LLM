# Module 01: Linear Algebra for LLMs

## Why This Matters

Every LLM operation — storing a word's meaning, transforming embeddings, computing attention — is linear algebra. If you've ever worked with arrays and matrix operations in NumPy, you've already been doing linear algebra. This module gives you the mental model behind those operations.

---

## 1. Vectors — Arrays with Direction

A **vector** is just an ordered list of numbers. You already know this as a Python list or a 1D NumPy array.

```python
word_embedding = [0.2, -0.5, 1.3, 0.8]  # a 4-dimensional vector
```

The difference from a plain list: vectors have **geometric meaning**. Each number is a coordinate in some N-dimensional space.

**Backend analogy:** Think of a vector as a row in a database table. A user profile might have columns `[age, purchase_count, avg_order_value, churn_risk]`. That row *is* a vector — it locates the user in a 4D "feature space." Two users with similar rows are "close" to each other in that space.

### Key vector properties

- **Dimension**: how many elements. `[1, 2, 3]` is 3-dimensional.
- **Magnitude (norm)**: the "length" of the vector. `||v|| = sqrt(v[0]^2 + v[1]^2 + ...)`
- **Direction**: which way the vector points, independent of its magnitude.
- **Unit vector**: a vector rescaled to have magnitude 1. Useful when you care only about direction, not scale.

### Vector operations

- **Addition**: element-wise. `[1,2] + [3,4] = [4,6]`. Like merging two feature sets.
- **Scalar multiplication**: multiply every element by a constant. Scales the vector without changing direction.
- **Dot product**: multiply element-wise, then sum everything. See Section 3.

---

## 2. Matrices — 2D Arrays / Lookup Tables

A **matrix** is a 2D grid of numbers — a list of vectors stacked as rows (or columns).

```python
weights = [
    [0.1, 0.4, 0.2],
    [0.3, 0.1, 0.5],
    [0.6, 0.2, 0.3],
]
# Shape: 3 rows × 3 columns → a (3, 3) matrix
```

**Backend analogy:** A matrix is like a database join table, or a 2D configuration grid. In web analytics, you might have a matrix where rows are users and columns are features — each row is one user's feature vector.

In ML, matrices most often appear as:
- **Weight matrices**: the learned parameters in a neural network layer
- **Embedding tables**: each row is the embedding for one token in the vocabulary
- **Attention scores**: a square matrix showing how much each token attends to every other token

---

## 3. Dot Product — Element-Wise Multiply, Then Sum

The dot product of two vectors `a` and `b`:

```
a · b = a[0]*b[0] + a[1]*b[1] + ... + a[n]*b[n]
```

The result is a **single number** (a scalar).

**Backend analogy:** Imagine scoring a job applicant. You have a weight vector `w = [0.5, 0.3, 0.2]` for `[experience, education, interview_score]`. You have the applicant's feature vector `a = [8, 4, 9]`. The dot product `w · a = 0.5*8 + 0.3*4 + 0.2*9 = 4 + 1.2 + 1.8 = 7.0` gives you a single score.

**Geometric meaning:** `a · b = ||a|| * ||b|| * cos(θ)`, where θ is the angle between the vectors.

- If `a · b > 0`: vectors point in roughly the same direction (similar)
- If `a · b = 0`: vectors are perpendicular (orthogonal / unrelated)
- If `a · b < 0`: vectors point in opposite directions (dissimilar)

**In LLMs:** Query-key dot products in attention measure how relevant each key is to the query. High dot product = high attention weight.

### Cosine Similarity

To compare vectors independent of their magnitude (length), use **cosine similarity**:

```
cosine_sim(a, b) = (a · b) / (||a|| * ||b||)
```

Range: -1 (opposite) to +1 (identical direction). This is how search engines and LLMs compare the "meaning" of text chunks.

---

## 4. Matrix Multiplication — Composing Transformations

Matrix multiplication `C = A @ B` applies the transformation encoded in `B` and then the one in `A` — or equivalently, maps each column of B through A.

For `A` of shape `(m, k)` and `B` of shape `(k, n)`, the result `C` has shape `(m, n)`. The inner dimensions must match.

Each element `C[i][j]` is the dot product of row `i` of A with column `j` of B.

**Backend analogy:** Think of matrix multiplication as **function composition**. If you have a pipeline `raw_data → normalize → embed → project`, each step is a matrix. Composing two matrices is like chaining two middleware functions. The input passes through each transformation in sequence.

In a neural network layer: `output = W @ input + bias`. The weight matrix `W` transforms the input vector into the output vector — it's a learned linear function.

**Key properties:**
- NOT commutative: `A @ B ≠ B @ A` in general
- IS associative: `(A @ B) @ C = A @ (B @ C)`
- Shapes must be compatible: `(m,k) @ (k,n) → (m,n)`

---

## 5. Transpose — Flipping Rows and Columns

The transpose `A.T` swaps rows and columns. Element at `[i][j]` moves to `[j][i]`.

```
A = [[1, 2, 3],        A.T = [[1, 4],
     [4, 5, 6]]               [2, 5],
                               [3, 6]]
```

Shape changes: `(m, n) → (n, m)`

**Backend analogy:** Transposing is like pivoting a SQL result — switching what's a row vs. a column. Useful when you need to align dimensions for matrix multiplication.

**Common use in ML:**
- Computing `W.T @ gradient` during backpropagation
- Converting row vectors to column vectors
- The attention formula uses `Q @ K.T` to compute scores between queries and keys

---

## 6. Inverse — The "Undo" Matrix

The inverse `A⁻¹` of a matrix `A` satisfies: `A @ A⁻¹ = I` (the identity matrix).

The **identity matrix** is the matrix version of the number 1 — it leaves any vector unchanged when multiplied.

```
I = [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]
```

**Backend analogy:** The inverse is like an undo operation. If matrix `A` encodes an encryption function, `A⁻¹` is the decryption function. Applying encrypt then decrypt returns the original data.

**Important limitations:**
- Only square matrices can have inverses
- Not all square matrices are invertible (singular matrices — like dividing by zero)
- In practice, ML rarely explicitly computes inverses (it's numerically unstable and slow). Instead, systems solve `Ax = b` directly.

---

## 7. Eigenvectors and Eigenvalues — Invariant Directions

An **eigenvector** `v` of matrix `A` is a special vector that, when you apply the matrix transformation, only gets *scaled* — it doesn't rotate or change direction:

```
A @ v = λ * v
```

`λ` (lambda) is the corresponding **eigenvalue** — the scaling factor.

**Backend analogy:** Imagine you have a system that transforms request feature vectors (latency, error_rate, throughput). Most vectors get rotated and stretched in complicated ways. But eigenvectors are the "natural axes" of that transformation — they only get amplified or shrunk, not redirected. They represent the directions of maximum variance in your data.

**Why this matters for ML:**

1. **PCA (Principal Component Analysis):** The eigenvectors of a data's covariance matrix are the "principal components" — the directions that explain the most variance in the dataset. Used for dimensionality reduction.

2. **Understanding weight matrices:** The largest eigenvectors of attention weight matrices represent the most important "concepts" the model learned.

3. **Stable patterns:** Eigenvectors are directions that a transformation "preserves." In a recommendation system, the top eigenvectors of a user-item interaction matrix represent underlying taste profiles.

**Intuition check:** If you rotate a square, every vector in it changes direction — no eigenvectors (except for the zero vector). But if you *stretch* a square along the x-axis, the x-axis and y-axis vectors stay pointing the same way — they're eigenvectors with eigenvalues equal to the stretch factors.

---

## 8. Summary Table

| Concept | Math | Backend Analogy |
|---|---|---|
| Vector | 1D array of numbers | A database row / feature array |
| Matrix | 2D array of numbers | A DB table / weight grid |
| Dot product | Σ aᵢbᵢ | A weighted score calculation |
| Matrix multiply | Row-col dot products | Chaining two functions / middleware |
| Transpose | Swap rows↔cols | SQL PIVOT |
| Inverse | A⁻¹ such that AA⁻¹=I | Undo / decrypt operation |
| Eigenvector | Av = λv | Invariant data pattern / principal axis |

---

## 9. How This Appears in LLMs

- **Token embeddings**: Each word/token is a vector of ~768–4096 dimensions
- **Attention**: `Attention = softmax(Q @ K.T / sqrt(d_k)) @ V` — pure matrix operations
- **Feed-forward layers**: `output = W₂ @ relu(W₁ @ input + b₁) + b₂` — matrix multiplications
- **Weight matrices**: Every learned parameter is a matrix; training adjusts these matrices

Everything an LLM does at inference time is vectors being multiplied by matrices, producing new vectors. Understanding this is the foundation for understanding transformer architecture.
