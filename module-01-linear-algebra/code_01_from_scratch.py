"""
Module 01 — Linear Algebra from Scratch
========================================
No numpy, no imports except math.
Goal: understand WHAT each operation computes before using the library shortcut.

Run: python code_01_from_scratch.py
"""

import math


# =============================================================================
# VECTORS
# =============================================================================

def dot_product(a: list, b: list) -> float:
    """
    The most important operation in ML.
    Multiply element-wise, then sum. Result: a single number.

    Intuition: how much do the two vectors "agree"?
    """
    assert len(a) == len(b), f"Vectors must match in length: {len(a)} vs {len(b)}"
    return sum(a[i] * b[i] for i in range(len(a)))


def vector_add(a: list, b: list) -> list:
    """Element-wise addition."""
    assert len(a) == len(b)
    return [a[i] + b[i] for i in range(len(a))]


def vector_scale(v: list, scalar: float) -> list:
    """Multiply every element by scalar."""
    return [x * scalar for x in v]


def vector_norm(v: list) -> float:
    """
    Euclidean length: sqrt(v[0]^2 + v[1]^2 + ...)
    Distance from origin to the point the vector represents.
    """
    return math.sqrt(sum(x ** 2 for x in v))


def unit_vector(v: list) -> list:
    """
    Scale vector to have length 1 (divide by its norm).
    Captures direction only — removes the effect of magnitude.
    """
    norm = vector_norm(v)
    assert norm > 0, "Cannot normalize the zero vector"
    return [x / norm for x in v]


def cosine_similarity(a: list, b: list) -> float:
    """
    Measure how similar two vectors' DIRECTIONS are.
    Range: -1 (opposite) to +1 (identical direction).

    Formula: (a · b) / (||a|| * ||b||)

    In semantic search: query vector vs document vector.
    In attention: related to how query matches key.
    """
    return dot_product(a, b) / (vector_norm(a) * vector_norm(b))


# =============================================================================
# MATRICES
# =============================================================================
# Represent a matrix as a list of lists (rows).
# matrix[row][col] — row first, then column.

def matrix_shape(M: list) -> tuple:
    """Return (num_rows, num_cols)."""
    rows = len(M)
    cols = len(M[0]) if rows > 0 else 0
    return (rows, cols)


def matrix_transpose(M: list) -> list:
    """
    Swap rows and columns.
    Element [i][j] moves to [j][i].
    Shape (m, n) becomes (n, m).
    """
    rows, cols = matrix_shape(M)
    return [[M[r][c] for r in range(rows)] for c in range(cols)]


def matrix_row(M: list, i: int) -> list:
    """Return row i of the matrix."""
    return M[i]


def matrix_col(M: list, j: int) -> list:
    """Return column j of the matrix."""
    return [M[i][j] for i in range(len(M))]


def matmul(A: list, B: list) -> list:
    """
    Matrix multiplication: C = A @ B
    C[i][j] = dot product of row i of A with column j of B.

    Shape rule: (m, k) @ (k, n) → (m, n)
    The inner dimension k must match.

    This is the core computation of every neural network layer.
    """
    m, k_a = matrix_shape(A)
    k_b, n = matrix_shape(B)
    assert k_a == k_b, f"Shape mismatch: A has {k_a} cols, B has {k_b} rows"

    C = [[0.0] * n for _ in range(m)]
    for i in range(m):          # each row of A
        for j in range(n):      # each col of B
            C[i][j] = dot_product(matrix_row(A, i), matrix_col(B, j))
    return C


def matrix_vector_multiply(M: list, v: list) -> list:
    """
    Apply matrix M to vector v: result[i] = dot(row_i_of_M, v).
    This is what a neural network layer does: output = W @ input.
    """
    rows, cols = matrix_shape(M)
    assert cols == len(v), f"Shape mismatch: matrix has {cols} cols, vector has {len(v)} elements"
    return [dot_product(M[i], v) for i in range(rows)]


def identity_matrix(n: int) -> list:
    """
    n×n identity matrix: 1s on diagonal, 0s elsewhere.
    The 'do nothing' transformation — any matrix times identity is itself.
    """
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


# =============================================================================
# DEMOS
# =============================================================================

def demo_vectors():
    print("=" * 50)
    print("VECTORS")
    print("=" * 50)

    a = [1, 2, 3]
    b = [4, 5, 6]

    print(f"\na = {a}")
    print(f"b = {b}")
    print(f"\ndot(a, b) = {dot_product(a, b)}")
    # Manually: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    print(f"  → manually: 1*4 + 2*5 + 3*6 = {1*4 + 2*5 + 3*6}")

    print(f"\na + b = {vector_add(a, b)}")
    print(f"2 * a = {vector_scale(a, 2)}")
    print(f"norm(a) = {vector_norm(a):.4f}")
    print(f"unit(a) = {[round(x, 4) for x in unit_vector(a)]}")

    # Cosine similarity: related words should be more similar
    king  = [0.9, 0.1, 0.8, 0.0]
    queen = [0.8, 0.9, 0.7, 0.1]
    dog   = [0.1, 0.0, 0.2, 0.9]

    print(f"\nWord vector similarity:")
    print(f"  cosine(king, queen) = {cosine_similarity(king, queen):.4f}  ← high, semantically related")
    print(f"  cosine(king, dog)   = {cosine_similarity(king, dog):.4f}  ← lower, less related")


def demo_matmul():
    print("\n" + "=" * 50)
    print("MATRIX MULTIPLICATION")
    print("=" * 50)

    A = [[1, 2],
         [3, 4]]

    B = [[5, 6],
         [7, 8]]

    print(f"\nA = {A}")
    print(f"B = {B}")

    C = matmul(A, B)
    print(f"\nA @ B = {C}")
    print(f"  C[0][0] = row[0] of A · col[0] of B = {matrix_row(A,0)} · {matrix_col(B,0)} = {dot_product(matrix_row(A,0), matrix_col(B,0))}")
    print(f"  C[0][1] = row[0] of A · col[1] of B = {matrix_row(A,0)} · {matrix_col(B,1)} = {dot_product(matrix_row(A,0), matrix_col(B,1))}")
    print(f"  C[1][0] = row[1] of A · col[0] of B = {matrix_row(A,1)} · {matrix_col(B,0)} = {dot_product(matrix_row(A,1), matrix_col(B,0))}")
    print(f"  C[1][1] = row[1] of A · col[1] of B = {matrix_row(A,1)} · {matrix_col(B,1)} = {dot_product(matrix_row(A,1), matrix_col(B,1))}")


def demo_linear_layer():
    print("\n" + "=" * 50)
    print("A NEURAL NETWORK LINEAR LAYER (from scratch)")
    print("=" * 50)

    # This is what a single neuron layer does:
    # output = W @ input + bias

    # Weight matrix: maps 3D input → 2D output
    W = [[0.5, -0.2,  0.8],   # row 0: weights for output neuron 0
         [0.1,  0.9, -0.3]]   # row 1: weights for output neuron 1

    bias = [0.1, -0.1]

    input_vec = [1.0, 2.0, 3.0]

    # Matrix-vector multiply
    output = matrix_vector_multiply(W, input_vec)

    # Add bias
    output = vector_add(output, bias)

    print(f"\nWeight matrix W ({len(W)}×{len(W[0])}):")
    for row in W:
        print(f"  {row}")
    print(f"\nInput vector: {input_vec}")
    print(f"Bias vector:  {bias}")
    print(f"\nOutput = W @ input + bias = {[round(x, 4) for x in output]}")
    print(f"\nBreaking it down:")
    print(f"  output[0] = {W[0]} · {input_vec} + {bias[0]}")
    print(f"            = {dot_product(W[0], input_vec)} + {bias[0]}")
    print(f"            = {dot_product(W[0], input_vec) + bias[0]:.4f}")


def demo_mini_attention():
    print("\n" + "=" * 50)
    print("MINI ATTENTION: dot products between Q and K")
    print("=" * 50)

    # 3 tokens, each represented as a 4D query/key vector
    # (In reality: 768D, but same concept)
    queries = [
        [1.0, 0.0, 1.0, 0.0],   # query for token 0 "the"
        [0.0, 1.0, 0.0, 1.0],   # query for token 1 "cat"
        [1.0, 1.0, 0.0, 0.0],   # query for token 2 "sat"
    ]

    keys = [
        [1.0, 0.1, 0.9, 0.1],   # key for token 0
        [0.1, 0.9, 0.1, 0.9],   # key for token 1
        [0.8, 0.8, 0.1, 0.2],   # key for token 2
    ]

    print("\nAttention scores (Q @ K.T):")
    print("score[i][j] = how much token i attends to token j")
    print()
    print(f"{'':8}", end="")
    tokens = ["the", "cat", "sat"]
    for t in tokens:
        print(f"  {t:6}", end="")
    print()

    for i, q in enumerate(queries):
        print(f"{tokens[i]:8}", end="")
        for j, k in enumerate(keys):
            score = dot_product(q, k)
            print(f"  {score:6.2f}", end="")
        print()

    print("\nNotice: each score is just a dot product between query and key vectors.")
    print("The transformer learns W_q and W_k so these scores capture meaningful relationships.")


if __name__ == '__main__':
    demo_vectors()
    demo_matmul()
    demo_linear_layer()
    demo_mini_attention()
