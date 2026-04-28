"""
Module 01 — Exercises
======================
Work through these exercises in order — they build on each other.
All exercises are intentionally BLANK. Fill in the TODO sections.

Check your work: python exercises.py
Compare with: solutions.py

Do NOT look at solutions.py until you've tried each exercise.
"""


# =============================================================================
# EXERCISE 1: Dot Product (Pure Python)
# Difficulty: Easy
# =============================================================================

def dot_product(a: list, b: list) -> float:
    """
    Compute the dot product of two vectors without using numpy.

    Formula: dot(a, b) = a[0]*b[0] + a[1]*b[1] + ... + a[n]*b[n]

    This is the single most important operation in deep learning.
    A neuron's output = dot(weights, inputs) + bias.

    Args:
        a: list of numbers, e.g. [1, 2, 3]
        b: list of numbers, same length as a

    Returns:
        A single float: the dot product.

    Example:
        dot_product([1, 2, 3], [4, 5, 6]) → 32
        # because 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    """
    assert len(a) == len(b), f"Vectors must have the same length"
    # TODO: implement using a loop or sum() + list comprehension
    raise NotImplementedError


# =============================================================================
# EXERCISE 2: Vector Norm (Pure Python)
# Difficulty: Easy
# =============================================================================

def vector_norm(v: list) -> float:
    """
    Compute the Euclidean norm (length) of a vector.

    Formula: ||v|| = sqrt(v[0]^2 + v[1]^2 + ... + v[n]^2)

    Hint: use Python's built-in math.sqrt

    Example:
        vector_norm([3, 4]) → 5.0    (3-4-5 right triangle)
        vector_norm([1, 0]) → 1.0
    """
    import math
    # TODO: implement
    raise NotImplementedError


# =============================================================================
# EXERCISE 3: Cosine Similarity (Pure Python)
# Difficulty: Medium
# =============================================================================

def cosine_similarity(a: list, b: list) -> float:
    """
    Compute cosine similarity between two vectors.

    Formula: cos_sim(a, b) = dot(a, b) / (||a|| * ||b||)

    Use your dot_product() and vector_norm() functions from above.

    Range: -1.0 (opposite directions) to +1.0 (same direction).

    This is how semantic search engines compare query to document embeddings.
    It's also used in clustering LLM embeddings.

    Example:
        cosine_similarity([1, 0], [1, 0]) → 1.0   (same direction)
        cosine_similarity([1, 0], [0, 1]) → 0.0   (perpendicular)
        cosine_similarity([1, 0], [-1, 0]) → -1.0 (opposite)
    """
    # TODO: use dot_product() and vector_norm() from above
    raise NotImplementedError


# =============================================================================
# EXERCISE 4: Matrix Multiplication (Pure Python)
# Difficulty: Medium
# =============================================================================

def matmul(A: list, B: list) -> list:
    """
    Multiply matrix A by matrix B. Return the result as a 2D list.

    Rule: C[i][j] = dot product of row i of A with column j of B
    Shape: (m, k) @ (k, n) → (m, n)

    The inner dimension k must match.

    Hint:
        - Use 3 nested loops: i (rows of A), j (cols of B), p (shared dim)
        - OR: reuse your dot_product() function

    Example:
        A = [[1, 2],   B = [[5, 6],
             [3, 4]]        [7, 8]]

        matmul(A, B) → [[19, 22],
                         [43, 50]]

        Verify: C[0][0] = 1*5 + 2*7 = 19 ✓
    """
    m  = len(A)
    k  = len(A[0])
    assert len(B) == k, f"Shape mismatch: A has {k} cols, B has {len(B)} rows"
    n  = len(B[0])

    # Initialize result with zeros
    C = [[0.0] * n for _ in range(m)]

    # TODO: fill in C[i][j] for all i, j
    raise NotImplementedError


# =============================================================================
# EXERCISE 5: Embedding Lookup (NumPy)
# Difficulty: Easy
# =============================================================================

def embedding_lookup(embedding_table: "np.ndarray", token_ids: list) -> "np.ndarray":
    """
    Retrieve embeddings for a list of token IDs from an embedding table.

    An embedding table has shape (vocab_size, d_model).
    Retrieving an embedding = selecting a row from this matrix.

    Args:
        embedding_table: numpy array of shape (vocab_size, d_model)
        token_ids: list of integer token IDs

    Returns:
        numpy array of shape (len(token_ids), d_model)

    This is the very first operation in every LLM forward pass.

    Example:
        table = np.random.randn(100, 8)   # 100 tokens, 8 dims each
        embedding_lookup(table, [3, 7, 1]) → array of shape (3, 8)
    """
    # TODO: one line — numpy indexing makes this trivial
    raise NotImplementedError


# =============================================================================
# EXERCISE 6: Attention Scores (NumPy)
# Difficulty: Hard — this is the real thing
# =============================================================================

def compute_attention_scores(Q: "np.ndarray", K: "np.ndarray", d_k: int) -> "np.ndarray":
    """
    Compute scaled attention scores from Query and Key matrices.

    Formula: scores = Q @ K.T / sqrt(d_k)

    This is the first step of self-attention in every transformer.

    Args:
        Q: query matrix of shape (seq_len, d_k)
        K: key matrix  of shape (seq_len, d_k)
        d_k: the key/query dimension (for scaling)

    Returns:
        scores matrix of shape (seq_len, seq_len)
        scores[i][j] = how much token i should attend to token j

    Hint:
        - Use @ for matrix multiplication
        - Use numpy.sqrt for square root
        - K.T transposes K from (seq_len, d_k) to (d_k, seq_len)
        - Then Q (seq_len, d_k) @ K.T (d_k, seq_len) → (seq_len, seq_len) ✓
    """
    import numpy as np
    # TODO: implement in one line
    raise NotImplementedError


# =============================================================================
# RUN ALL EXERCISES
# =============================================================================

def check():
    import math
    import numpy as np

    print("=" * 55)
    print("Module 01 — Exercise Results")
    print("=" * 55)

    passed  = 0
    total   = 0
    skipped = 0

    def test(name, got, expected, tol=1e-6):
        nonlocal passed, total, skipped
        total += 1
        try:
            if isinstance(expected, float):
                ok = abs(got - expected) < tol
            elif isinstance(expected, list):
                ok = all(abs(got[i][j] - expected[i][j]) < tol
                         for i in range(len(expected))
                         for j in range(len(expected[i])))
            else:
                ok = np.allclose(got, expected, atol=tol)
        except Exception:
            ok = False
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if ok:
            passed += 1

    # --- Exercise 1 ---
    print("\nExercise 1: dot_product")
    try:
        test("dot([1,2,3], [4,5,6]) = 32",  dot_product([1,2,3],[4,5,6]), 32.0)
        test("dot([1,0], [0,1]) = 0",         dot_product([1,0],[0,1]), 0.0)
        test("dot([2], [3]) = 6",             dot_product([2],[3]), 6.0)
    except NotImplementedError:
        skipped += 1
        print("  [SKIP] Not implemented yet")

    # --- Exercise 2 ---
    print("\nExercise 2: vector_norm")
    try:
        test("norm([3,4]) = 5.0",   vector_norm([3,4]), 5.0)
        test("norm([1,0]) = 1.0",   vector_norm([1,0]), 1.0)
        test("norm([0,0]) = 0.0",   vector_norm([0,0]), 0.0)
    except NotImplementedError:
        skipped += 1
        print("  [SKIP] Not implemented yet")

    # --- Exercise 3 ---
    print("\nExercise 3: cosine_similarity")
    try:
        test("cos([1,0],[1,0]) = 1.0",    cosine_similarity([1,0],[1,0]), 1.0)
        test("cos([1,0],[0,1]) = 0.0",    cosine_similarity([1,0],[0,1]), 0.0)
        test("cos([1,0],[-1,0]) = -1.0",  cosine_similarity([1,0],[-1,0]), -1.0)
        # Word vectors
        king  = [0.9, 0.1, 0.8, 0.0]
        queen = [0.8, 0.9, 0.7, 0.1]
        dog   = [0.1, 0.0, 0.2, 0.9]
        sim_kq = cosine_similarity(king, queen)
        sim_kd = cosine_similarity(king, dog)
        ok = sim_kq > sim_kd
        total += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] cos(king,queen)={sim_kq:.3f} > cos(king,dog)={sim_kd:.3f}")
        if ok:
            passed += 1
    except NotImplementedError:
        skipped += 1
        print("  [SKIP] Not implemented yet")

    # --- Exercise 4 ---
    print("\nExercise 4: matmul")
    try:
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        C = matmul(A, B)
        test("matmul([[1,2],[3,4]], [[5,6],[7,8]])", C, [[19.0,22.0],[43.0,50.0]])

        # Identity check
        I = [[1,0],[0,1]]
        AI = matmul(A, I)
        test("matmul(A, Identity) = A", AI, [[1.0,2.0],[3.0,4.0]])
    except NotImplementedError:
        skipped += 1
        print("  [SKIP] Not implemented yet")

    # --- Exercise 5 ---
    print("\nExercise 5: embedding_lookup")
    try:
        table = np.arange(20).reshape(5, 4).astype(float)
        # table[2] should be [8, 9, 10, 11]
        result = embedding_lookup(table, [2, 0])
        expected = np.array([[8., 9., 10., 11.], [0., 1., 2., 3.]])
        test("lookup [2, 0] from 5×4 table", result, expected)
    except NotImplementedError:
        skipped += 1
        print("  [SKIP] Not implemented yet")

    # --- Exercise 6 ---
    print("\nExercise 6: compute_attention_scores")
    try:
        np.random.seed(1)
        Q = np.random.randn(3, 4)
        K = np.random.randn(3, 4)
        d_k = 4
        scores = compute_attention_scores(Q, K, d_k)

        expected_shape = (3, 3)
        shape_ok = scores.shape == expected_shape
        total += 1
        print(f"  [{'PASS' if shape_ok else 'FAIL'}] scores shape = {scores.shape} (expected (3,3))")
        if shape_ok:
            passed += 1

        expected_00 = float(np.dot(Q[0], K[0]) / np.sqrt(d_k))
        test("scores[0,0] = Q[0]·K[0]/sqrt(d_k)", float(scores[0,0]), expected_00)
    except NotImplementedError:
        skipped += 1
        print("  [SKIP] Not implemented yet")

    print(f"\n{'='*55}")
    if skipped > 0:
        print(f"Score: {passed}/{total} passed,  {skipped} exercise(s) not yet implemented.")
        print("Implement the TODO sections, then re-run.")
    elif passed == total and total > 0:
        print(f"Score: {passed}/{total} — All exercises complete! Move on to Module 02.")
    else:
        print(f"Score: {passed}/{total} — {total - passed} failing.")
    print("=" * 55)


if __name__ == '__main__':
    check()
