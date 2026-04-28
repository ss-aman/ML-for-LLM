"""
Module 01 — Linear Algebra with NumPy
=======================================
Same concepts as code_01_from_scratch.py, but using NumPy.
NumPy is what you'll actually use — it's 100–1000× faster than pure Python
because it calls optimized BLAS/LAPACK routines under the hood.

Run: python code_02_numpy.py
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# SECTION 1: Vectors with NumPy
# =============================================================================

def section_vectors():
    print("=" * 55)
    print("SECTION 1: Vectors")
    print("=" * 55)

    # Creating vectors
    v = np.array([1.0, 2.0, 3.0])
    w = np.array([4.0, 5.0, 6.0])

    print(f"\nv = {v},  shape = {v.shape},  dtype = {v.dtype}")
    print(f"w = {w}")

    # Operations (all element-wise)
    print(f"\nv + w       = {v + w}")
    print(f"v - w       = {v - w}")
    print(f"v * 2       = {v * 2}")
    print(f"v / 2       = {v / 2}")

    # Dot product — the important one
    dot = np.dot(v, w)             # method 1
    dot_alt = v @ w                # method 2: @ operator (preferred in ML code)
    dot_manual = (v * w).sum()     # method 3: element-wise then sum

    print(f"\ndot(v, w) = {dot}")
    print(f"  v @ w   = {dot_alt}  (same result)")
    print(f"  manual  = {dot_manual}  (same result)")
    print(f"  by hand: {1*4}+{2*5}+{3*6} = {1*4 + 2*5 + 3*6}")

    # Norm
    norm_v = np.linalg.norm(v)
    print(f"\n||v|| = {norm_v:.4f}")

    # Unit vector
    unit_v = v / np.linalg.norm(v)
    print(f"unit(v) = {unit_v.round(4)}")
    print(f"||unit(v)|| = {np.linalg.norm(unit_v):.6f}  ← should be 1.0")

    # Cosine similarity
    cos_sim = np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w))
    print(f"\ncosine_similarity(v, w) = {cos_sim:.4f}")


# =============================================================================
# SECTION 2: Matrices with NumPy
# =============================================================================

def section_matrices():
    print("\n" + "=" * 55)
    print("SECTION 2: Matrices")
    print("=" * 55)

    A = np.array([[1, 2, 3],
                  [4, 5, 6]])     # shape (2, 3)

    B = np.array([[7,  8],
                  [9,  10],
                  [11, 12]])      # shape (3, 2)

    print(f"\nA shape: {A.shape}")
    print(A)
    print(f"\nB shape: {B.shape}")
    print(B)

    # Matrix multiplication
    C = A @ B    # (2,3) @ (3,2) → (2,2)
    print(f"\nA @ B  shape: {C.shape}")
    print(C)
    print(f"\nC[0,0] = row 0 of A · col 0 of B = {A[0]} · {B[:,0]} = {A[0] @ B[:,0]}")

    # Transpose
    print(f"\nA.T  shape: {A.T.shape}")
    print(A.T)

    # Identity
    I = np.eye(3)
    print(f"\nIdentity (3×3):\n{I}")
    print(f"A @ I[:3,:3] (should equal first 3 cols of A):\n{(A @ I).round(4)}")


# =============================================================================
# SECTION 3: Embedding Table Lookup
# =============================================================================

def section_embedding_lookup():
    print("\n" + "=" * 55)
    print("SECTION 3: Embedding Table (how LLMs represent tokens)")
    print("=" * 55)

    vocab_size = 10      # tiny vocab: 10 words
    d_model    = 6       # tiny embedding: 6 dimensions

    # In real GPT-2: vocab_size=50257, d_model=768
    np.random.seed(42)
    embedding_table = np.random.randn(vocab_size, d_model)

    print(f"\nEmbedding table shape: {embedding_table.shape}")
    print(f"  → {vocab_size} tokens, each with a {d_model}-dimensional vector")
    print(f"\nFirst 3 rows (embeddings for tokens 0, 1, 2):")
    print(embedding_table[:3].round(3))

    # Lookup: which token are we embedding?
    token_id = 3
    embedding = embedding_table[token_id]   # just a row lookup!

    print(f"\nLooking up token_id={token_id}:")
    print(f"  embedding = {embedding.round(3)},  shape = {embedding.shape}")

    # Batch lookup: embed multiple tokens at once
    token_sequence = [2, 7, 1, 5]    # a sequence of 4 token IDs
    batch_embeddings = embedding_table[token_sequence]

    print(f"\nLooking up sequence {token_sequence}:")
    print(f"  batch_embeddings shape: {batch_embeddings.shape}")
    print(f"  (one 6D vector per token)")
    print(batch_embeddings.round(3))


# =============================================================================
# SECTION 4: Attention Scores (Q @ K.T)
# =============================================================================

def section_attention_scores():
    print("\n" + "=" * 55)
    print("SECTION 4: Attention Scores — Q @ K.T")
    print("=" * 55)

    seq_len = 4    # 4 tokens in our sequence
    d_k     = 8   # dimension of Q and K

    np.random.seed(0)
    Q = np.random.randn(seq_len, d_k)   # (4, 8): one query per token
    K = np.random.randn(seq_len, d_k)   # (4, 8): one key per token

    # Raw scores: every query vs every key
    scores_raw = Q @ K.T                     # (4, 4)
    print(f"\nQ shape: {Q.shape}")
    print(f"K shape: {K.shape}")
    print(f"Q @ K.T shape: {scores_raw.shape}  ← one score per (token_i, token_j) pair")

    # Scaled scores (divide by sqrt(d_k) for stability)
    scores_scaled = scores_raw / np.sqrt(d_k)

    print(f"\nScaled attention scores (Q @ K.T / sqrt({d_k})):")
    print(scores_scaled.round(3))

    # Softmax: convert scores to probabilities (rows sum to 1)
    def softmax_rows(x):
        e = np.exp(x - x.max(axis=1, keepdims=True))  # subtract max for stability
        return e / e.sum(axis=1, keepdims=True)

    weights = softmax_rows(scores_scaled)

    print(f"\nAttention weights (after softmax) — each row sums to 1:")
    print(weights.round(3))
    print(f"\nRow sums (should all be 1.0): {weights.sum(axis=1).round(4)}")

    # Weighted sum of values
    V = np.random.randn(seq_len, d_k)   # (4, 8): one value per token
    output = weights @ V                 # (4, 8): one output vector per token

    print(f"\nV shape: {V.shape}")
    print(f"Attention output (weights @ V) shape: {output.shape}")
    print(f"\nThis is self-attention. The full transformer uses it in Module 10.")


# =============================================================================
# SECTION 5: Visualizing Eigenvectors
# =============================================================================

def section_eigenvectors():
    print("\n" + "=" * 55)
    print("SECTION 5: Eigenvectors")
    print("=" * 55)

    # A matrix that stretches x by 3×, leaves y unchanged
    A = np.array([[3.0, 0.0],
                  [0.0, 1.0]])

    eigenvalues, eigenvectors = np.linalg.eig(A)

    print(f"\nMatrix A:\n{A}")
    print(f"\nEigenvalues:  {eigenvalues}")
    print(f"Eigenvectors (columns):\n{eigenvectors}")

    # Verify: A @ v = λ * v
    for i in range(2):
        v   = eigenvectors[:, i]
        lam = eigenvalues[i]
        print(f"\nEigenvector {i}: {v}")
        print(f"  A @ v     = {(A @ v).round(4)}")
        print(f"  λ * v     = {(lam * v).round(4)}")
        print(f"  Same?       {np.allclose(A @ v, lam * v)}")

    # A non-trivial case: symmetric matrix with off-diagonal entries
    B = np.array([[2.0, 1.0],
                  [1.0, 3.0]])
    evals_B, evecs_B = np.linalg.eig(B)

    print(f"\nMatrix B (cross-terms):\n{B}")
    print(f"Eigenvalues:  {evals_B.round(4)}")
    print(f"Eigenvectors:\n{evecs_B.round(4)}")
    print("\nThese eigenvectors are at ~26° and ~116° — the natural axes of B's structure.")

    # Visualize (save to PNG)
    _plot_eigenvectors(B, evals_B, evecs_B)


def _plot_eigenvectors(M, eigenvalues, eigenvectors):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Left: show how matrix transforms a circle of points
    ax = axes[0]
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = np.vstack([np.cos(theta), np.sin(theta)])   # 2×100

    transformed = M @ circle   # apply transformation

    ax.plot(circle[0], circle[1], 'b--', alpha=0.5, label='Unit circle')
    ax.plot(transformed[0], transformed[1], 'r-', alpha=0.8, label='Transformed')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.legend()
    ax.set_title('Matrix M transforms a circle into an ellipse')

    # Right: show eigenvectors
    ax = axes[1]
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)

    colors = ['blue', 'red']
    for i in range(2):
        v = eigenvectors[:, i]
        lam = eigenvalues[i]
        # original eigenvector
        ax.annotate('', xy=v, xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=colors[i], lw=2))
        # transformed eigenvector (should point same direction)
        ax.annotate('', xy=lam * v, xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=colors[i], lw=2, alpha=0.4))
        ax.text(v[0] * 1.1, v[1] * 1.1, f'v{i} (λ={lam:.1f})', color=colors[i])

    ax.set_title('Eigenvectors: same direction after transformation')

    plt.tight_layout()
    plt.savefig('eigenvectors_demo.png', dpi=100)
    plt.close()
    print("\nSaved: eigenvectors_demo.png")


# =============================================================================
# SECTION 6: Speed Comparison (NumPy vs pure Python)
# =============================================================================

def section_speed_comparison():
    import time

    print("\n" + "=" * 55)
    print("SECTION 6: Why NumPy (not pure Python)")
    print("=" * 55)

    n = 1000
    A_np = np.random.randn(n, n)
    B_np = np.random.randn(n, n)

    # NumPy
    t0 = time.perf_counter()
    _ = A_np @ B_np
    t_numpy = time.perf_counter() - t0

    print(f"\n{n}×{n} matrix multiplication:")
    print(f"  NumPy:       {t_numpy*1000:.2f} ms")
    print(f"\nNumPy calls optimized BLAS (LAPACK) routines.")
    print("Pure Python loops would take ~1000× longer.")
    print("Real LLMs use GPU (cuBLAS) which is 100×+ faster than CPU NumPy.")


if __name__ == '__main__':
    section_vectors()
    section_matrices()
    section_embedding_lookup()
    section_attention_scores()
    section_eigenvectors()
    section_speed_comparison()
