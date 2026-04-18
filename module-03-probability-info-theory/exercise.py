"""
Module 03: Probability and Information Theory — Exercises
==========================================================
Run this file directly:  python exercise.py

These exercises build hands-on intuition for entropy, softmax,
and cross-entropy loss. Each exercise is a function with a clear docstring.
The check() function at the bottom runs all exercises and prints results.

Target: someone who knows Python well but has never studied ML math.
"""

import math
import numpy as np


# =============================================================================
# EXERCISE 1: Entropy of a Fair Coin vs a Biased Coin
#
# Entropy measures uncertainty. The more predictable the outcome,
# the lower the entropy.
# =============================================================================

def exercise_1_entropy(probs: list) -> float:
    """
    Compute the Shannon entropy of a discrete probability distribution.

    Formula:
        H(p) = -Σ p(x) * log₂(p(x))

    Convention: if p(x) = 0, that term contributes 0 to the sum
    (because lim(p→0) p*log(p) = 0).

    Backend analogy: entropy of your server request distribution.
    - All requests go to server A: H = 0 (perfectly predictable)
    - Requests split 50/50: H = 1 bit (one binary question answers it)
    - Requests split uniformly across 4 servers: H = 2 bits

    Args:
        probs: list of probabilities that sum to 1.
               Example: [0.5, 0.5] for fair coin, [0.9, 0.1] for biased.

    Returns:
        Entropy in bits (using log base 2).

    TODO: Implement using Python's math.log2. Skip any p=0 terms.
    """
    total = 0.0
    for p in probs:
        if p > 0:
            total += p * math.log2(p)
    return -total


# =============================================================================
# EXERCISE 2: Softmax from Scratch
#
# Convert raw logit scores (unbounded real numbers) to probabilities
# (non-negative, sum to 1).
# =============================================================================

def exercise_2_softmax(logits: list) -> list:
    """
    Implement the softmax function from scratch (no numpy allowed).

    Formula:
        softmax(z)_i = exp(z_i) / Σ_j exp(z_j)

    IMPORTANT: Use the numerically stable version — subtract max(logits)
    before taking exp(). This prevents overflow without changing the result:
        softmax(z) = softmax(z - max(z))   (provable by algebra)

    Backend analogy: you have raw "capacity scores" for N servers.
    Softmax converts them into traffic allocation percentages.
    A server with score 5 gets much more traffic than one with score 1 —
    but all percentages sum to exactly 100%.

    Args:
        logits: list of raw scores (any real numbers, positive or negative)

    Returns:
        list of probabilities (each in [0,1], all sum to 1.0)

    TODO: Implement using only Python's built-in math.exp, max, sum.
    """
    # Step 1: find the max for numerical stability
    max_val = max(logits)

    # Step 2: compute exp(z_i - max) for each logit
    exp_vals = [math.exp(z - max_val) for z in logits]

    # Step 3: sum all the exp values
    total = sum(exp_vals)

    # Step 4: divide each by the total
    return [e / total for e in exp_vals]


# =============================================================================
# EXERCISE 3: Cross-Entropy Loss for a Classification Prediction
#
# The core loss function for language models and classifiers.
# Measures how "surprised" the model is by the correct answer.
# =============================================================================

def exercise_3_cross_entropy(predicted_probs: list, true_class_index: int) -> float:
    """
    Compute cross-entropy loss for a single prediction.

    For one-hot true labels, the formula simplifies to:
        Loss = -log(predicted_probs[true_class_index])

    This is because the one-hot vector is all zeros except at true_class_index,
    so only the term for the correct class survives the sum:
        H(p, q) = -Σ p(x) * log(q(x)) = -1 * log(q[correct]) + 0 + 0 + ...

    The result equals: "negative log probability of the correct answer."
    - Model says 90% on correct class → loss = -log(0.9) ≈ 0.105   (low)
    - Model says 10% on correct class → loss = -log(0.1) ≈ 2.303   (high)
    - Model says 1% on correct class  → loss = -log(0.01) ≈ 4.605  (very high)

    Backend analogy: you predicted 90% of traffic would hit the US east region,
    but 90% actually hit EU. Cross-entropy loss measures how wrong your traffic
    model was — and it grows sharply when you were confidently wrong.

    Args:
        predicted_probs: model's output probabilities (list, sums to 1)
        true_class_index: the index of the correct class

    Returns:
        cross-entropy loss as a float (always ≥ 0, lower is better)

    Example from the exercise prompt:
        predicted_probs = [0.7, 0.2, 0.1]
        true_class_index = 0   (class 0 is correct)
        loss = -log(0.7) ≈ 0.3567

    TODO: Implement using math.log. Add a tiny epsilon (1e-10) to the
          probability before taking log to avoid log(0).
    """
    epsilon = 1e-10
    p_correct = predicted_probs[true_class_index] + epsilon
    return -math.log(p_correct)


# =============================================================================
# EXERCISE 4 (Challenge): Cross-Entropy ≥ Entropy
#
# Prove experimentally that cross-entropy H(p, q) ≥ true entropy H(p).
# The gap is the KL divergence KL(p || q) — always non-negative.
#
# This means no model can achieve a loss lower than the true entropy
# of the data — it's the information-theoretic lower bound on compression.
# =============================================================================

def exercise_4_cross_entropy_lower_bound(p: list, q: list) -> dict:
    """
    Compute and verify: H(p, q) = H(p) + KL(p || q) ≥ H(p).

    Given a true distribution p and a predicted distribution q, compute:
        1. True entropy:    H(p)    = -Σ p(x) * log(p(x))
        2. Cross-entropy:   H(p,q)  = -Σ p(x) * log(q(x))
        3. KL divergence:   KL(p||q) = Σ p(x) * log(p(x)/q(x))
        4. Verify:          H(p,q) ≈ H(p) + KL(p||q)  (up to floating point)

    Use natural logarithm (math.log) for all calculations — this is what
    ML frameworks use internally (the "nats" unit instead of "bits").

    Backend analogy: p is the actual traffic distribution.
    q is your model's prediction of traffic.
    H(p) is the fundamental unpredictability of traffic (you can't do better).
    KL(p||q) is the extra cost of using the wrong model.
    H(p,q) is your total modeling cost — always at least H(p).

    Args:
        p: true probability distribution (list, sums to 1)
        q: predicted probability distribution (list, sums to 1)

    Returns:
        dict with keys: 'entropy_p', 'cross_entropy_pq', 'kl_divergence',
                        'sum_check' (H(p) + KL), 'lower_bound_holds' (bool)

    TODO: Implement using math.log (natural log). Add epsilon to avoid log(0).
    """
    epsilon = 1e-10

    # YOUR CODE HERE
    # Compute H(p) = -Σ p(x) * log(p(x))
    entropy_p = -sum(
        px * math.log(px + epsilon)
        for px in p if px > 0
    )

    # Compute H(p, q) = -Σ p(x) * log(q(x))
    cross_entropy_pq = -sum(
        px * math.log(qx + epsilon)
        for px, qx in zip(p, q)
        if px > 0
    )

    # Compute KL(p||q) = Σ p(x) * log(p(x)/q(x))
    kl_divergence = sum(
        px * math.log((px + epsilon) / (qx + epsilon))
        for px, qx in zip(p, q)
        if px > 0
    )

    # Verify the relationship: H(p,q) = H(p) + KL(p||q)
    sum_check = entropy_p + kl_divergence
    lower_bound_holds = cross_entropy_pq >= entropy_p - 1e-9

    return {
        'entropy_p': entropy_p,
        'cross_entropy_pq': cross_entropy_pq,
        'kl_divergence': kl_divergence,
        'sum_check': sum_check,
        'lower_bound_holds': lower_bound_holds,
    }


# =============================================================================
# CHECK FUNCTION — runs all exercises and prints results
# =============================================================================

def check():
    print("=" * 65)
    print("MODULE 03 EXERCISES — Results")
    print("=" * 65)

    # ---- Exercise 1 ----
    print("\n[Exercise 1] Entropy: fair coin vs biased coin")

    cases = [
        ([0.5, 0.5],   1.0,    "Fair coin (should be 1.0 bit)"),
        ([0.9, 0.1],   None,   "Biased coin (should be < 1 bit)"),
        ([1.0, 0.0],   0.0,    "Certain (always heads, should be 0)"),
        ([0.25]*4,     2.0,    "4-sided fair die (should be 2 bits)"),
    ]

    for probs, expected, label in cases:
        result = exercise_1_entropy(probs)
        ok_str = ""
        if expected is not None:
            ok = math.isclose(result, expected, abs_tol=1e-6)
            ok_str = f"  {'PASS' if ok else 'FAIL'}"
        print(f"  {label}")
        print(f"    probs={probs}  →  H = {result:.6f} bits{ok_str}")

    # ---- Exercise 2 ----
    print("\n[Exercise 2] Softmax from scratch")
    test_inputs = [
        ([2.0, 1.0, 0.1],   "Normal logits"),
        ([0.0, 0.0, 0.0],   "All equal → uniform"),
        ([10.0, 1.0, 0.1],  "Large spread → peaked"),
        ([-1000, 0, 1000],  "Extreme values → numerically stable?"),
    ]
    all_pass = True
    for logits, label in test_inputs:
        result = exercise_2_softmax(logits)
        total = sum(result)
        sums_to_one = math.isclose(total, 1.0, abs_tol=1e-9)
        all_non_neg = all(p >= 0 for p in result)
        # Compare to numpy reference
        import numpy as np
        np_ref = (lambda z: np.exp(z - np.max(z)) / np.sum(np.exp(z - np.max(z))))(np.array(logits))
        matches_numpy = all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(result, np_ref))
        ok = sums_to_one and all_non_neg and matches_numpy
        print(f"  {label}")
        print(f"    input: {logits}")
        print(f"    output: {[round(p, 4) for p in result]}")
        print(f"    sum={total:.8f}  sums_to_1={'PASS' if sums_to_one else 'FAIL'}"
              f"  matches_numpy={'PASS' if matches_numpy else 'FAIL'}")
        if not ok:
            all_pass = False
    print(f"  Exercise 2: {'ALL PASS' if all_pass else 'SOME FAILURES'}")

    # ---- Exercise 3 ----
    print("\n[Exercise 3] Cross-entropy loss")
    print("  Model prediction: [0.7, 0.2, 0.1], true class: 0")
    predicted = [0.7, 0.2, 0.1]
    true_class = 0
    loss = exercise_3_cross_entropy(predicted, true_class)
    expected_loss = -math.log(0.7)
    ok = math.isclose(loss, expected_loss, rel_tol=1e-6)
    print(f"  Cross-entropy loss = {loss:.6f}")
    print(f"  Expected: -log(0.7) = {expected_loss:.6f}")
    print(f"  Exercise 3: {'PASS' if ok else 'FAIL'}")

    # Additional intuition cases
    print("\n  Intuition: how loss varies with model confidence")
    for pred, true_idx, desc in [
        ([0.99, 0.009, 0.001], 0, "Very confident, correct"),
        ([0.5,  0.3,   0.2  ], 0, "Moderate confidence, correct"),
        ([0.1,  0.8,   0.1  ], 0, "Confident, WRONG"),
        ([0.01, 0.01,  0.98 ], 0, "Very confident, VERY WRONG"),
    ]:
        l = exercise_3_cross_entropy(pred, true_idx)
        print(f"    {desc:>35}: loss = {l:.4f}")

    # ---- Exercise 4 ----
    print("\n[Exercise 4] Cross-entropy lower bound: H(p,q) >= H(p)")
    test_pairs = [
        ([0.5, 0.3, 0.15, 0.05], [0.5, 0.3, 0.15, 0.05], "Perfect model: q = p"),
        ([0.5, 0.3, 0.15, 0.05], [0.4, 0.3, 0.2,  0.1 ], "Good model"),
        ([0.5, 0.3, 0.15, 0.05], [0.25,0.25,0.25, 0.25], "Uniform (worst-ish)"),
        ([0.5, 0.3, 0.15, 0.05], [0.1, 0.1, 0.1,  0.7 ], "Terrible model"),
    ]

    print(f"\n  {'Scenario':>25}  {'H(p)':>8}  {'H(p,q)':>8}  {'KL(p||q)':>10}  "
          f"{'H(p)+KL≈H(p,q)?':>18}  {'≥H(p)?':>8}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*18}  {'-'*8}")

    all_pass = True
    for p, q, label in test_pairs:
        result = exercise_4_cross_entropy_lower_bound(p, q)
        sum_matches = math.isclose(result['sum_check'], result['cross_entropy_pq'], abs_tol=1e-6)
        lb_holds = result['lower_bound_holds']
        print(f"  {label:>25}  "
              f"{result['entropy_p']:>8.4f}  "
              f"{result['cross_entropy_pq']:>8.4f}  "
              f"{result['kl_divergence']:>10.4f}  "
              f"{'PASS' if sum_matches else 'FAIL':>18}  "
              f"{'PASS' if lb_holds else 'FAIL':>8}")
        if not (sum_matches and lb_holds):
            all_pass = False

    print(f"\n  Exercise 4: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    print("\n  Key insight: H(p,q) = H(p) + KL(p||q)")
    print("  The minimum achievable loss is H(p) — the true entropy of the data.")
    print("  No model can do better than the information content of the true distribution.")

    print("\n" + "=" * 65)
    print("Tip: Experiment with different distributions in Exercise 1 to build")
    print("intuition for when entropy is high vs. low.")
    print("=" * 65)


if __name__ == '__main__':
    check()
