"""
Module 01 — Exercise Solutions
================================
Reference solutions for exercises.py.
Only look here after you've genuinely attempted each exercise.

Run: python solutions.py   (runs all checks and shows expected output)
"""

import math
import numpy as np


def dot_product(a: list, b: list) -> float:
    assert len(a) == len(b)
    return sum(a[i] * b[i] for i in range(len(a)))


def vector_norm(v: list) -> float:
    return math.sqrt(sum(x ** 2 for x in v))


def cosine_similarity(a: list, b: list) -> float:
    return dot_product(a, b) / (vector_norm(a) * vector_norm(b))


def matmul(A: list, B: list) -> list:
    m, k = len(A), len(A[0])
    assert len(B) == k
    n = len(B[0])
    # Build column cache so we don't recompute each time
    B_cols = [[B[r][c] for r in range(k)] for c in range(n)]
    return [[dot_product(A[i], B_cols[j]) for j in range(n)] for i in range(m)]


def embedding_lookup(embedding_table: np.ndarray, token_ids: list) -> np.ndarray:
    return embedding_table[token_ids]


def compute_attention_scores(Q: np.ndarray, K: np.ndarray, d_k: int) -> np.ndarray:
    return Q @ K.T / np.sqrt(d_k)


# =============================================================================
# CHECK (same as exercises.py — verifies solutions above)
# =============================================================================

def check():
    print("=" * 55)
    print("Module 01 — Solutions Check")
    print("=" * 55)

    passed = 0
    total  = 0

    def test(name, got, expected, tol=1e-6):
        nonlocal passed, total
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

    print("\nExercise 1: dot_product")
    test("dot([1,2,3], [4,5,6]) = 32",  dot_product([1,2,3],[4,5,6]), 32.0)
    test("dot([1,0], [0,1]) = 0",         dot_product([1,0],[0,1]), 0.0)
    test("dot([2], [3]) = 6",             dot_product([2],[3]), 6.0)

    print("\nExercise 2: vector_norm")
    test("norm([3,4]) = 5.0",   vector_norm([3,4]), 5.0)
    test("norm([1,0]) = 1.0",   vector_norm([1,0]), 1.0)
    test("norm([0,0]) = 0.0",   vector_norm([0,0]), 0.0)

    print("\nExercise 3: cosine_similarity")
    test("cos([1,0],[1,0]) = 1.0",    cosine_similarity([1,0],[1,0]), 1.0)
    test("cos([1,0],[0,1]) = 0.0",    cosine_similarity([1,0],[0,1]), 0.0)
    test("cos([1,0],[-1,0]) = -1.0",  cosine_similarity([1,0],[-1,0]), -1.0)

    print("\nExercise 4: matmul")
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    C = matmul(A, B)
    test("matmul(A, B)", C, [[19.0,22.0],[43.0,50.0]])
    I = [[1,0],[0,1]]
    test("matmul(A, I) = A", matmul(A, I), [[1.0,2.0],[3.0,4.0]])

    print("\nExercise 5: embedding_lookup")
    table = np.arange(20).reshape(5, 4).astype(float)
    result = embedding_lookup(table, [2, 0])
    expected = np.array([[8., 9., 10., 11.], [0., 1., 2., 3.]])
    test("lookup [2, 0] from 5×4 table", result, expected)

    print("\nExercise 6: compute_attention_scores")
    np.random.seed(1)
    Q = np.random.randn(3, 4)
    K = np.random.randn(3, 4)
    d_k = 4
    scores = compute_attention_scores(Q, K, d_k)
    shape_ok = scores.shape == (3, 3)
    total += 1
    print(f"  [{'PASS' if shape_ok else 'FAIL'}] scores shape = {scores.shape}")
    if shape_ok:
        passed += 1
    expected_00 = float(np.dot(Q[0], K[0]) / np.sqrt(d_k))
    test("scores[0,0] = Q[0]·K[0]/sqrt(d_k)", float(scores[0,0]), expected_00)

    print(f"\n{'='*55}")
    print(f"Score: {passed}/{total}")
    print("=" * 55)


if __name__ == '__main__':
    check()
