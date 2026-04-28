"""
Module 03 — Exercises: Probability & Information Theory
========================================================
Implement each function. Do NOT look at solutions.py until you've tried.

Run: python exercises.py
"""

import numpy as np


# =============================================================================
# EXERCISE 1: Entropy
# Difficulty: Easy
# =============================================================================

def entropy(p):
    """
    Compute the Shannon entropy (in nats, using natural log) of distribution p.

        H(p) = -Σ p(x) · ln(p(x))

    Convention: 0 · log(0) = 0  (terms where p[i] == 0 contribute nothing)

    Args:
        p: numpy array of probabilities (must sum to 1, all ≥ 0)

    Returns:
        scalar entropy value ≥ 0

    Examples:
        entropy([1.0, 0.0])          → 0.0       (certain outcome)
        entropy([0.5, 0.5])          → 0.6931    (fair coin, = ln(2))
        entropy([0.25]*4)            → 1.3863    (uniform over 4, = ln(4))
    """
    # TODO: one or two lines — sum only where p > 0
    raise NotImplementedError


# =============================================================================
# EXERCISE 2: KL Divergence
# Difficulty: Easy-Medium
# =============================================================================

def kl_divergence(p, q):
    """
    Compute KL divergence KL(p || q) = Σ p(x) · ln(p(x) / q(x))

    p is the reference (true) distribution.
    q is the approximation.

    Properties:
      - Always ≥ 0
      - = 0 iff p == q
      - Returns float('inf') if q[i] == 0 where p[i] > 0

    Args:
        p: reference distribution, numpy array summing to 1
        q: approximate distribution, numpy array summing to 1

    Returns:
        scalar KL divergence ≥ 0

    Examples:
        kl_divergence([0.5, 0.5], [0.5, 0.5])  → 0.0
        kl_divergence([0.9, 0.1], [0.5, 0.5])  → ~0.368
        kl_divergence([1.0, 0.0], [0.0, 1.0])  → inf
    """
    # TODO: iterate only where p > 0; return inf if q[i]=0 there
    raise NotImplementedError


# =============================================================================
# EXERCISE 3: Softmax
# Difficulty: Easy
# =============================================================================

def softmax(z):
    """
    Compute the numerically stable softmax of vector z.

        softmax(z)_i = exp(z_i) / Σ_j exp(z_j)

    Numerical stability trick: subtract max(z) before exponentiating.
    This doesn't change the result: softmax(z) = softmax(z - c) for any c.

    Args:
        z: numpy array of real-valued logits (any values)

    Returns:
        numpy array of same shape, values in (0, 1), summing to 1

    Examples:
        softmax([1.0, 1.0, 1.0])   → [1/3, 1/3, 1/3]   (uniform)
        softmax([10.0, 0.0, 0.0])  → [~1.0, ~0.0, ~0.0] (peaked)
        softmax([0.0])             → [1.0]
    """
    # TODO: subtract max, exponentiate, normalize
    raise NotImplementedError


# =============================================================================
# EXERCISE 4: Cross-Entropy Loss
# Difficulty: Easy-Medium
# =============================================================================

def cross_entropy_loss(logits, label):
    """
    Compute cross-entropy loss for a single classification example.

        loss = -log(softmax(logits)[label])

    Internally:
        1. Apply softmax to get probabilities
        2. Return -log of the probability at the correct label

    Args:
        logits: numpy array of raw model outputs (shape: vocab_size,)
        label:  integer index of the correct class

    Returns:
        scalar loss ≥ 0 (= 0 when model perfectly predicts label)

    Examples:
        logits = [0, 0, 10, 0]   label = 2
        → softmax ≈ [0, 0, 1, 0]  → loss ≈ 0.0  (model is certain and correct)

        logits = [10, 0, 0, 0]   label = 2
        → softmax ≈ [1, 0, 0, 0]  → loss ≈ 10.0  (model is certain and wrong)
    """
    # TODO: use your softmax function, then -log(prob at label)
    raise NotImplementedError


# =============================================================================
# EXERCISE 5: Top-k Sampling
# Difficulty: Medium
# =============================================================================

def top_k_sample(probs, k):
    """
    Sample a token index using top-k sampling.

    Algorithm:
        1. Find the k tokens with the highest probability
        2. Zero out all other tokens
        3. Renormalize the remaining k probabilities to sum to 1
        4. Sample from the renormalized distribution

    Args:
        probs: numpy array of probabilities (summing to 1)
        k:     number of top tokens to keep

    Returns:
        integer index of sampled token

    Examples:
        probs = [0.5, 0.3, 0.15, 0.05],  k=2
        → only keep indices [0, 1] (top 2)
        → renormalize: [0.5/0.8, 0.3/0.8] = [0.625, 0.375]
        → sample from {0: 62.5%, 1: 37.5%}
        → returns 0 or 1 (randomly)
    """
    # TODO: find top-k indices, renormalize, sample
    raise NotImplementedError


# =============================================================================
# EXERCISE 6: Nucleus (Top-p) Sampling
# Difficulty: Hard
# =============================================================================

def nucleus_sample(probs, p):
    """
    Sample using nucleus (top-p) sampling.

    Algorithm:
        1. Sort tokens by probability, descending
        2. Compute cumulative sum of sorted probabilities
        3. Keep the smallest set of tokens whose cumulative prob ≥ p
           (this is the "nucleus")
        4. Renormalize nucleus probabilities to sum to 1
        5. Sample from the nucleus

    Args:
        probs: numpy array of probabilities (summing to 1)
        p:     nucleus threshold in (0, 1] — fraction of probability mass to keep

    Returns:
        integer index of sampled token

    Examples:
        probs = [0.7, 0.2, 0.07, 0.03],  p=0.9
        Sorted: [0.7, 0.2, 0.07, 0.03]  cumsum: [0.7, 0.9, 0.97, 1.0]
        Nucleus at p=0.9: indices with cumsum ≤ 0.9 → first 2 tokens
        Renormalize: [0.7/0.9, 0.2/0.9] = [0.778, 0.222]
        Sample from {orig_index_0: 77.8%, orig_index_1: 22.2%}
    """
    # TODO: sort descending, cumsum, find cutoff, renormalize, sample
    raise NotImplementedError


# =============================================================================
# RUN ALL EXERCISES
# =============================================================================

def check():
    print("=" * 55)
    print("Module 03 — Exercise Results")
    print("=" * 55)

    passed  = 0
    total   = 0
    skipped = 0

    def test(name, got, expected, tol=1e-4):
        nonlocal passed, total
        total += 1
        if isinstance(expected, np.ndarray):
            ok = np.allclose(got, expected, atol=tol)
        elif expected == float('inf') or expected == float('-inf'):
            ok = got == expected
        elif isinstance(expected, (float, int)):
            ok = abs(float(got) - float(expected)) < tol
        else:
            ok = np.allclose(got, expected, atol=tol)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if ok:
            passed += 1

    # ── Exercise 1: entropy ───────────────────────────────────────────────────
    print("\nExercise 1: entropy")
    try:
        test("H([1,0]) = 0",          entropy([1.0, 0.0]),    0.0)
        test("H([0.5,0.5]) = ln(2)",   entropy([0.5, 0.5]),   np.log(2))
        test("H([0.25]*4) = ln(4)",    entropy([0.25]*4),     np.log(4))
        test("H([0.9,0.1]) < H([0.5,0.5])",
             entropy([0.9, 0.1]) < entropy([0.5, 0.5]), True)
    except NotImplementedError:
        skipped += 1
        print("  [SKIP] Not implemented yet")

    # ── Exercise 2: kl_divergence ─────────────────────────────────────────────
    print("\nExercise 2: kl_divergence")
    try:
        test("KL(p, p) = 0",     kl_divergence([0.5, 0.5], [0.5, 0.5]), 0.0)
        kl_val = kl_divergence([0.9, 0.1], [0.5, 0.5])
        test("KL([0.9,0.1]||[0.5,0.5]) ≈ 0.368", kl_val, 0.3681, tol=1e-3)
        test("KL(p, q) ≥ 0",     kl_val >= 0, True)
        test("KL when q has zero = inf",
             kl_divergence([1.0, 0.0], [0.0, 1.0]), float('inf'))
    except NotImplementedError:
        skipped += 1
        print("  [SKIP] Not implemented yet")

    # ── Exercise 3: softmax ───────────────────────────────────────────────────
    print("\nExercise 3: softmax")
    try:
        result_uniform = softmax([1.0, 1.0, 1.0])
        test("softmax([1,1,1]) is uniform", result_uniform,
             np.array([1/3, 1/3, 1/3]))
        test("softmax sums to 1", softmax([2.0, -1.0, 0.5]).sum(), 1.0)
        test("softmax stable (large values)", softmax([1000.0, 1001.0])[1] > 0.5, True)
        test("softmax all positive", all(x > 0 for x in softmax([5.0, -5.0, 0.0])), True)
    except NotImplementedError:
        skipped += 1
        print("  [SKIP] Not implemented yet")

    # ── Exercise 4: cross_entropy_loss ────────────────────────────────────────
    print("\nExercise 4: cross_entropy_loss")
    try:
        # Perfect prediction: logit 10 on correct class
        perfect_logits = np.array([0.0, 0.0, 10.0, 0.0])
        test("loss ≈ 0 when model is certain and correct",
             cross_entropy_loss(perfect_logits, 2), 0.0, tol=0.1)
        # Wrong: high logit on wrong class
        wrong_logits = np.array([10.0, 0.0, 0.0, 0.0])
        test("loss is high when model is certain and wrong",
             cross_entropy_loss(wrong_logits, 2) > 5, True)
        # Uniform logits: loss = log(4) for 4-class problem
        uniform_logits = np.array([0.0, 0.0, 0.0, 0.0])
        test("uniform logits: loss = log(n)",
             cross_entropy_loss(uniform_logits, 0), np.log(4), tol=1e-4)
    except NotImplementedError:
        skipped += 1
        print("  [SKIP] Not implemented yet")

    # ── Exercise 5: top_k_sample ──────────────────────────────────────────────
    print("\nExercise 5: top_k_sample")
    try:
        np.random.seed(42)
        probs = np.array([0.5, 0.3, 0.15, 0.05])
        samples_k2 = [top_k_sample(probs.copy(), k=2) for _ in range(200)]
        # With k=2, only indices 0 and 1 should appear
        test("top-k=2 only returns top 2 indices",
             all(s in (0, 1) for s in samples_k2), True)
        # k=1 should always return index 0 (highest prob)
        samples_k1 = [top_k_sample(probs.copy(), k=1) for _ in range(10)]
        test("top-k=1 always returns argmax",
             all(s == 0 for s in samples_k1), True)
        # k=len(probs) should sample from all
        samples_full = [top_k_sample(probs.copy(), k=len(probs)) for _ in range(500)]
        test("top-k=n can return any index",
             3 in samples_full, True)
    except NotImplementedError:
        skipped += 1
        print("  [SKIP] Not implemented yet")

    # ── Exercise 6: nucleus_sample ────────────────────────────────────────────
    print("\nExercise 6: nucleus_sample")
    try:
        np.random.seed(42)
        # Peaked: first token covers 95%, p=0.9 → nucleus = 1 token → always index 0
        peaked = np.array([0.95, 0.03, 0.01, 0.01])
        samples_peaked = [nucleus_sample(peaked, p=0.9) for _ in range(50)]
        test("nucleus peaked p=0.9 → always index 0",
             all(s == 0 for s in samples_peaked), True)

        # Flat: uniform over 4, p=0.9 → nucleus covers ≥3 tokens
        flat = np.array([0.25, 0.25, 0.25, 0.25])
        samples_flat = [nucleus_sample(flat, p=0.9) for _ in range(200)]
        unique_sampled = len(set(samples_flat))
        test("nucleus flat p=0.9 → samples from multiple tokens",
             unique_sampled >= 3, True)

        # p=1.0 → include all tokens
        result = nucleus_sample(np.array([0.7, 0.2, 0.07, 0.03]), p=1.0)
        test("nucleus p=1.0 returns valid index", result in (0, 1, 2, 3), True)
    except NotImplementedError:
        skipped += 1
        print("  [SKIP] Not implemented yet")

    print(f"\n{'='*55}")
    if skipped > 0:
        print(f"Score: {passed}/{total} passed,  {skipped} exercise(s) not yet implemented.")
        print("Implement the TODO sections, then re-run.")
    elif passed == total and total > 0:
        print(f"Score: {passed}/{total} — All exercises complete! Move on to Module 04.")
    else:
        print(f"Score: {passed}/{total} — {total - passed} failing.")
    print("=" * 55)


if __name__ == '__main__':
    check()
