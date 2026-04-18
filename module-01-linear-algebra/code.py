"""
Module 01: Linear Algebra — NumPy Implementations
==================================================
Run this file directly:  python code.py

Each section demonstrates a core linear algebra concept with
intuitive comments aimed at Python/backend developers.
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# SECTION 1: Vector Operations
# Think of vectors as feature arrays for a data record.
# =============================================================================

def vector_add(a, b):
    """
    Element-wise addition of two vectors.
    Like merging two feature sets by summing each feature.
    """
    return np.array(a) + np.array(b)


def vector_subtract(a, b):
    """
    Element-wise subtraction.
    The result vector points from b toward a — the 'difference' in feature space.
    """
    return np.array(a) - np.array(b)


def dot_product(a, b):
    """
    Multiply element-wise, then sum everything.
    Measures how much two vectors 'agree' or point in the same direction.

    Equivalent to: sum(a[i] * b[i] for i in range(len(a)))
    """
    return np.dot(np.array(a), np.array(b))


def vector_norm(v):
    """
    The 'length' of a vector — like Euclidean distance from the origin.
    Formula: sqrt(v[0]^2 + v[1]^2 + ... + v[n]^2)

    Backend analogy: normalizing a feature vector so its scale doesn't
    dominate other features (like converting raw bytes to a 0–1 range).
    """
    return np.linalg.norm(np.array(v))


def cosine_similarity(a, b):
    """
    Compares the *direction* of two vectors, ignoring their magnitudes.
    Range: -1 (opposite) to +1 (identical direction).

    This is how semantic search works: compare the direction of
    query embedding vs. document embeddings.
    """
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# =============================================================================
# SECTION 2: Matrix Operations
# Think of matrices as 2D configuration tables or transformation functions.
# =============================================================================

def matrix_multiply(A, B):
    """
    Matrix multiplication: apply transformation B, then A.
    Each output element is a dot product of one row of A with one column of B.

    Shape rule: (m, k) @ (k, n) → (m, n)
    The inner dimensions must match — like a type-safe function composition.
    """
    A, B = np.array(A), np.array(B)
    assert A.shape[1] == B.shape[0], (
        f"Shape mismatch: A is {A.shape}, B is {B.shape}. "
        f"A's columns ({A.shape[1]}) must equal B's rows ({B.shape[0]})"
    )
    return A @ B


def matrix_transpose(A):
    """
    Swap rows and columns.
    Element [i][j] moves to [j][i].

    Backend analogy: like a SQL PIVOT — what was a row becomes a column.
    Used in attention: Q @ K.T computes similarity between every query-key pair.
    """
    return np.array(A).T


def matrix_inverse(A):
    """
    Compute A⁻¹ such that A @ A⁻¹ = I (identity matrix).
    The 'undo' operation for a linear transformation.

    Will raise LinAlgError if the matrix is singular (non-invertible),
    like a function with no inverse (e.g., many-to-one mappings).
    """
    return np.linalg.inv(np.array(A, dtype=float))


def identity_matrix(n):
    """
    The identity matrix: the '1' of matrix multiplication.
    Any matrix multiplied by the identity is unchanged.
    Diagonal of 1s, everywhere else 0s.
    """
    return np.eye(n)


# =============================================================================
# SECTION 3: Practical Example — Rotating a 2D Point
# A rotation matrix is a matrix that preserves lengths but changes direction.
# Backend analogy: like a coordinate transform between two reference frames
# (e.g., converting geographic coords to screen coords).
# =============================================================================

def rotation_matrix_2d(angle_degrees):
    """
    Build a 2D rotation matrix for the given angle.
    Applying this matrix to a 2D point rotates it by that angle around the origin.

    The matrix encodes the transformation:
      new_x =  x*cos(θ) - y*sin(θ)
      new_y =  x*sin(θ) + y*cos(θ)
    """
    theta = np.radians(angle_degrees)
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
    ])


def demo_rotation():
    """Rotate a point [1, 0] (pointing right) by 90 degrees → should become [0, 1]."""
    point = np.array([1.0, 0.0])
    R = rotation_matrix_2d(90)
    rotated = R @ point

    print("=== 2D Rotation Demo ===")
    print(f"Original point:  {point}")
    print(f"Rotation matrix (90°):\n{R.round(4)}")
    print(f"Rotated point:   {rotated.round(4)}")
    print(f"Expected:        [0. 1.]")

    # Visualize the rotation
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.grid(True, alpha=0.3)

    ax.annotate('', xy=point, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.annotate('', xy=rotated, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax.text(point[0] + 0.1, point[1], 'original [1,0]', color='blue')
    ax.text(rotated[0] + 0.1, rotated[1], 'rotated [0,1]', color='red')
    ax.set_title('2D Vector Rotation by 90°')
    plt.tight_layout()
    plt.savefig('rotation_demo.png', dpi=100)
    plt.close()
    print("Saved: rotation_demo.png")


# =============================================================================
# SECTION 4: Eigenvector Demo
# Eigenvectors are directions a transformation preserves (only scales, no rotation).
# Backend analogy: like finding the "natural partition key" of your data —
# the axis along which variance is greatest and the structure is most stable.
# =============================================================================

def demo_eigenvectors():
    """
    Show what eigenvectors mean geometrically.

    We create a transformation matrix that stretches along one axis more than
    another, then find the eigenvectors — they point along those stretch axes.
    """
    # A matrix that stretches x by 3x and y by 1x (no stretching in y)
    A = np.array([
        [3.0, 0.0],
        [0.0, 1.0],
    ])

    eigenvalues, eigenvectors = np.linalg.eig(A)

    print("\n=== Eigenvector Demo ===")
    print(f"Transformation matrix A:\n{A}")
    print(f"\nEigenvalues:  {eigenvalues}")
    print(f"Eigenvectors (columns):\n{eigenvectors}")

    # Verify: A @ v = λ * v for each eigenvector
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]   # i-th eigenvector is the i-th column
        lam = eigenvalues[i]
        result_transform = A @ v
        result_scale = lam * v
        print(f"\nEigenvector {i}: {v.round(4)}")
        print(f"  A @ v     = {result_transform.round(4)}")
        print(f"  λ * v     = {result_scale.round(4)}")
        print(f"  Equal?      {np.allclose(result_transform, result_scale)}")

    print("\nInterpretation:")
    print("  Eigenvector [1,0] (x-axis) gets stretched by λ=3 — no rotation")
    print("  Eigenvector [0,1] (y-axis) gets stretched by λ=1 — no change at all")
    print("  These are the 'invariant directions' of the transformation.")

    # Now try with a matrix that has off-diagonal (mixing) terms
    B = np.array([
        [2.0, 1.0],
        [1.0, 2.0],
    ])
    eigenvalues_B, eigenvectors_B = np.linalg.eig(B)
    print(f"\nMatrix B (with cross-terms):\n{B}")
    print(f"Eigenvalues of B: {eigenvalues_B}")
    print(f"Eigenvectors of B (columns):\n{eigenvectors_B.round(4)}")
    print("Notice: eigenvectors are at 45° — the natural 'axes' of B's structure.")


# =============================================================================
# SECTION 5: Putting It Together — Word Vector Similarity
# Demonstrate how dot products / cosine similarity compare "meaning"
# =============================================================================

def demo_word_vectors():
    """
    Simplified word embeddings: compare how similar words are
    using cosine similarity — the same approach used in semantic search.
    """
    # Pretend these are learned embeddings for three words
    # (in real LLMs these are 768–4096 dimensional)
    king   = np.array([0.9, 0.1, 0.8, 0.0])
    queen  = np.array([0.8, 0.9, 0.7, 0.1])
    dog    = np.array([0.1, 0.0, 0.2, 0.9])

    print("\n=== Word Vector Similarity Demo ===")
    print(f"king  vector: {king}")
    print(f"queen vector: {queen}")
    print(f"dog   vector: {dog}")

    sim_king_queen = cosine_similarity(king, queen)
    sim_king_dog   = cosine_similarity(king, dog)

    print(f"\ncosine_similarity(king, queen) = {sim_king_queen:.4f}  ← high (semantically related)")
    print(f"cosine_similarity(king, dog)   = {sim_king_dog:.4f}  ← low (semantically different)")
    print("\nIn semantic search: query vector is compared to all document vectors,")
    print("top-k highest cosine similarities are returned as results.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("MODULE 01: Linear Algebra Demos")
    print("=" * 60)

    # --- Vector ops ---
    print("\n--- Vector Operations ---")
    a = [1, 2, 3]
    b = [4, 5, 6]
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b          = {vector_add(a, b)}")
    print(f"a - b          = {vector_subtract(a, b)}")
    print(f"dot(a, b)      = {dot_product(a, b)}")   # 1*4 + 2*5 + 3*6 = 32
    print(f"norm(a)        = {vector_norm(a):.4f}")
    print(f"cosine_sim     = {cosine_similarity(a, b):.4f}")

    # --- Matrix ops ---
    print("\n--- Matrix Operations ---")
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    print(f"A =\n{np.array(A)}")
    print(f"B =\n{np.array(B)}")
    print(f"A @ B =\n{matrix_multiply(A, B)}")
    print(f"A.T =\n{matrix_transpose(A)}")

    A_inv = matrix_inverse(A)
    print(f"A_inv =\n{A_inv.round(4)}")
    print(f"A @ A_inv (should be Identity) =\n{(np.array(A, dtype=float) @ A_inv).round(4)}")

    # --- Rotation demo ---
    print()
    demo_rotation()

    # --- Eigenvector demo ---
    demo_eigenvectors()

    # --- Word vectors ---
    demo_word_vectors()
