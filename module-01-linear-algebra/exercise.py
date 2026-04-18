"""
Module 01: Linear Algebra — Exercises
======================================
Run this file directly:  python exercise.py

These exercises build hands-on intuition for the concepts in notes.md.
Each exercise is a function with a clear docstring.
The check() function at the bottom runs all exercises and prints results.

Target: someone who knows Python well but has never studied ML math.
"""


# =============================================================================
# EXERCISE 1: Dot Product Without NumPy
# Build the operation from scratch so you understand what numpy is doing.
# =============================================================================

def exercise_1_dot_product(a: list, b: list) -> float:
    """
    Implement the dot product using only plain Python (no numpy, no imports).

    The dot product: multiply element-wise, then sum everything.
        dot(a, b) = a[0]*b[0] + a[1]*b[1] + ... + a[n]*b[n]

    Backend analogy: scoring a record against a weight vector.
    Example: applicant scores [8, 4, 9], weights [0.5, 0.3, 0.2]
             dot = 0.5*8 + 0.3*4 + 0.2*9 = 7.0

    Args:
        a: first vector as a Python list of numbers
        b: second vector as a Python list of numbers

    Returns:
        The dot product as a float.

    Raises:
        ValueError if the vectors have different lengths.

    TODO: Implement this function body.
    """
    if len(a) != len(b):
        raise ValueError(f"Vectors must be the same length: got {len(a)} and {len(b)}")

    # YOUR CODE HERE
    # Hint: use a loop (or a list comprehension + sum) to multiply element-wise
    # then add them all up.
    result = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result


# =============================================================================
# EXERCISE 2: Matrix Multiplication By Hand
# Understand what numpy's @ operator is actually computing.
# =============================================================================

def exercise_2_matmul(A: list, B: list) -> list:
    """
    Compute matrix multiplication A @ B using nested loops (no numpy).

    Rule: C[i][j] = dot product of row i of A with column j of B.
    Shape: if A is (m x k) and B is (k x n), result C is (m x n).

    Backend analogy: each output element combines one "query" (row of A)
    with one "key" (column of B) — exactly like attention in transformers.

    Args:
        A: 2D list of shape (m, k)
        B: 2D list of shape (k, n)

    Returns:
        C: 2D list of shape (m, n)

    Raises:
        ValueError if the inner dimensions don't match.

    TODO: Implement this function body.
    """
    m = len(A)
    k = len(A[0])
    if len(B) != k:
        raise ValueError(f"Inner dimensions must match: A has {k} cols, B has {len(B)} rows")
    n = len(B[0])

    # Initialize result matrix with zeros
    C = [[0.0] * n for _ in range(m)]

    # YOUR CODE HERE
    # Hint: three nested loops — i over rows of A, j over cols of B, p over the shared dimension
    for i in range(m):
        for j in range(n):
            for p in range(k):
                C[i][j] += A[i][p] * B[p][j]

    return C


# =============================================================================
# EXERCISE 3: Verify A @ A_inverse = Identity
# Demonstrates that the inverse "undoes" the transformation.
# =============================================================================

def exercise_3_verify_inverse() -> bool:
    """
    Show that multiplying a matrix by its inverse gives the identity matrix.

    The identity matrix I has 1s on the diagonal and 0s elsewhere.
    A @ A⁻¹ = I (up to floating point rounding).

    Backend analogy: like calling encrypt(decrypt(data)) == data.
    The two operations are perfect inverses of each other.

    Returns:
        True if A @ A_inverse is approximately the identity matrix.

    TODO: Fill in the matrix A, compute its inverse using numpy,
    multiply A @ A_inv, and verify it equals the identity.
    """
    import numpy as np

    # A sample invertible matrix
    A = np.array([
        [4.0, 7.0],
        [2.0, 6.0]
    ])

    # YOUR CODE HERE
    # Step 1: compute the inverse
    A_inv = np.linalg.inv(A)

    # Step 2: multiply A @ A_inv
    product = A @ A_inv

    # Step 3: build the expected identity matrix
    I = np.eye(A.shape[0])

    # Step 4: check they're approximately equal (floating point tolerance)
    is_identity = np.allclose(product, I)

    print(f"\n[Exercise 3]")
    print(f"A =\n{A}")
    print(f"A_inv =\n{A_inv.round(4)}")
    print(f"A @ A_inv =\n{product.round(6)}")
    print(f"Expected (Identity) =\n{I}")
    print(f"A @ A_inv ≈ Identity? {is_identity}")

    return is_identity


# =============================================================================
# EXERCISE 4 (Challenge): Cosine Similarity for Word Vectors
# This is how search engines and LLMs compare the "meaning" of text chunks.
# =============================================================================

def exercise_4_cosine_similarity(v1: list, v2: list) -> float:
    """
    Compute cosine similarity between two vectors WITHOUT using numpy's
    built-in dot or norm functions. Use only basic math operations.

    Formula:
        cosine_similarity(a, b) = (a · b) / (||a|| * ||b||)

    Where:
        a · b = sum of element-wise products (dot product)
        ||a|| = sqrt(sum of squares) = magnitude/norm of a

    Range: -1 (pointing opposite directions) to +1 (pointing same direction).
    Two vectors with cosine_sim near 1.0 are semantically similar.

    Backend analogy: imagine two API endpoint usage patterns as vectors
    [GET_rate, POST_rate, error_rate]. Cosine similarity tells you how
    similar the usage *patterns* are, regardless of total traffic volume.

    Args:
        v1: first vector as a Python list of floats
        v2: second vector as a Python list of floats

    Returns:
        cosine similarity as a float in [-1, 1]

    Example vectors from the challenge:
        word_A = [2, 3, 1]
        word_B = [1, 4, 2]

    TODO: Implement using only Python's built-in math.sqrt and basic arithmetic.
    """
    import math

    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same length")

    # YOUR CODE HERE
    # Step 1: compute the dot product (element-wise multiply, then sum)
    dot = sum(v1[i] * v2[i] for i in range(len(v1)))

    # Step 2: compute the norm (magnitude) of each vector
    norm_v1 = math.sqrt(sum(x ** 2 for x in v1))
    norm_v2 = math.sqrt(sum(x ** 2 for x in v2))

    # Step 3: divide
    if norm_v1 == 0 or norm_v2 == 0:
        raise ValueError("Cannot compute cosine similarity for zero vector")

    return dot / (norm_v1 * norm_v2)


# =============================================================================
# CHECK FUNCTION — runs all exercises and prints results
# =============================================================================

def check():
    import math
    import numpy as np

    print("=" * 60)
    print("MODULE 01 EXERCISES — Results")
    print("=" * 60)

    # ---- Exercise 1 ----
    print("\n[Exercise 1] Dot product without numpy")
    test_cases = [
        ([1, 2, 3], [4, 5, 6], 32),       # 1*4 + 2*5 + 3*6 = 32
        ([0.5, 0.3, 0.2], [8, 4, 9], 7.0), # weighted score = 7.0
        ([1, 0], [0, 1], 0),               # perpendicular vectors → 0
    ]
    all_pass = True
    for a, b, expected in test_cases:
        result = exercise_1_dot_product(a, b)
        ok = math.isclose(result, expected, rel_tol=1e-9)
        print(f"  dot({a}, {b}) = {result}  (expected {expected})  {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False
    print(f"  Exercise 1: {'ALL PASS' if all_pass else 'SOME FAILURES'}")

    # ---- Exercise 2 ----
    print("\n[Exercise 2] Matrix multiplication by hand")
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    # Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    expected_C = [[19.0, 22.0], [43.0, 50.0]]
    result_C = exercise_2_matmul(A, B)
    numpy_C = (np.array(A) @ np.array(B)).tolist()
    match = np.allclose(result_C, expected_C)
    matches_numpy = np.allclose(result_C, numpy_C)
    print(f"  A = {A}")
    print(f"  B = {B}")
    print(f"  A @ B (your result) = {result_C}")
    print(f"  A @ B (numpy)       = {[[int(x) for x in row] for row in numpy_C]}")
    print(f"  Matches expected?   {match}  {'PASS' if match else 'FAIL'}")
    print(f"  Matches numpy?      {matches_numpy}  {'PASS' if matches_numpy else 'FAIL'}")

    # ---- Exercise 3 ----
    print()  # exercise_3 prints its own output
    result_3 = exercise_3_verify_inverse()
    print(f"  Exercise 3: {'PASS' if result_3 else 'FAIL'}")

    # ---- Exercise 4 ----
    print("\n[Exercise 4] Cosine similarity for word vectors")
    word_A = [2.0, 3.0, 1.0]
    word_B = [1.0, 4.0, 2.0]
    result_sim = exercise_4_cosine_similarity(word_A, word_B)
    # Verify against numpy
    a_np = np.array(word_A)
    b_np = np.array(word_B)
    expected_sim = float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))
    ok = math.isclose(result_sim, expected_sim, rel_tol=1e-6)
    print(f"  word_A = {word_A}")
    print(f"  word_B = {word_B}")
    print(f"  cosine_similarity = {result_sim:.6f}  (numpy reference: {expected_sim:.6f})")
    print(f"  Interpretation: {result_sim:.3f} is {'high — similar direction' if result_sim > 0.8 else 'moderate' if result_sim > 0.5 else 'low'}")
    print(f"  Exercise 4: {'PASS' if ok else 'FAIL'}")

    print("\n" + "=" * 60)
    print("Tip: Read the docstrings above each exercise for the full context.")
    print("=" * 60)


if __name__ == '__main__':
    check()
