"""
Module 03 — Solutions
======================
Reference solutions. Only look here after genuinely attempting exercises.

Run: python solutions.py
"""

import numpy as np


def entropy(p):
    p = np.array(p, dtype=float)
    mask = p > 0
    return -np.sum(p[mask] * np.log(p[mask]))


def kl_divergence(p, q):
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    mask = p > 0
    if np.any(q[mask] == 0):
        return float('inf')
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))


def softmax(z):
    z = np.array(z, dtype=float)
    z = z - z.max()          # numerical stability
    exp_z = np.exp(z)
    return exp_z / exp_z.sum()


def cross_entropy_loss(logits, label):
    probs = softmax(logits)
    return -np.log(probs[label] + 1e-10)


def top_k_sample(probs, k):
    probs       = np.array(probs, dtype=float)
    k           = min(k, len(probs))
    top_indices = np.argsort(probs)[-k:]
    top_probs   = probs[top_indices]
    top_probs   = top_probs / top_probs.sum()
    return int(np.random.choice(top_indices, p=top_probs))


def nucleus_sample(probs, p):
    probs          = np.array(probs, dtype=float)
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs   = probs[sorted_indices]
    cumulative     = np.cumsum(sorted_probs)
    cutoff         = int(np.searchsorted(cumulative, p))
    nucleus_idx    = sorted_indices[:cutoff + 1]
    nucleus_probs  = probs[nucleus_idx]
    nucleus_probs  = nucleus_probs / nucleus_probs.sum()
    return int(np.random.choice(nucleus_idx, p=nucleus_probs))


# =============================================================================
# CHECK (same tests as exercises.py)
# =============================================================================

def check():
    print("=" * 55)
    print("Module 03 — Solutions Check")
    print("=" * 55)

    passed = 0
    total  = 0

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

    print("\nExercise 1: entropy")
    test("H([1,0]) = 0",          entropy([1.0, 0.0]),    0.0)
    test("H([0.5,0.5]) = ln(2)",   entropy([0.5, 0.5]),   np.log(2))
    test("H([0.25]*4) = ln(4)",    entropy([0.25]*4),     np.log(4))
    test("H([0.9,0.1]) < H([0.5,0.5])",
         entropy([0.9, 0.1]) < entropy([0.5, 0.5]), True)

    print("\nExercise 2: kl_divergence")
    test("KL(p, p) = 0",     kl_divergence([0.5, 0.5], [0.5, 0.5]), 0.0)
    kl_val = kl_divergence([0.9, 0.1], [0.5, 0.5])
    test("KL([0.9,0.1]||[0.5,0.5]) ≈ 0.368", kl_val, 0.3681, tol=1e-3)
    test("KL(p, q) ≥ 0",     kl_val >= 0, True)
    test("KL when q has zero = inf",
         kl_divergence([1.0, 0.0], [0.0, 1.0]), float('inf'))

    print("\nExercise 3: softmax")
    result_uniform = softmax([1.0, 1.0, 1.0])
    test("softmax([1,1,1]) is uniform", result_uniform, np.array([1/3, 1/3, 1/3]))
    test("softmax sums to 1", softmax([2.0, -1.0, 0.5]).sum(), 1.0)
    test("softmax stable (large values)", softmax([1000.0, 1001.0])[1] > 0.5, True)
    test("softmax all positive", all(x > 0 for x in softmax([5.0, -5.0, 0.0])), True)

    print("\nExercise 4: cross_entropy_loss")
    perfect_logits = np.array([0.0, 0.0, 10.0, 0.0])
    test("loss ≈ 0 when model is certain and correct",
         cross_entropy_loss(perfect_logits, 2), 0.0, tol=0.1)
    wrong_logits = np.array([10.0, 0.0, 0.0, 0.0])
    test("loss is high when model is certain and wrong",
         cross_entropy_loss(wrong_logits, 2) > 5, True)
    uniform_logits = np.array([0.0, 0.0, 0.0, 0.0])
    test("uniform logits: loss = log(n)",
         cross_entropy_loss(uniform_logits, 0), np.log(4), tol=1e-4)

    print("\nExercise 5: top_k_sample")
    np.random.seed(42)
    probs = np.array([0.5, 0.3, 0.15, 0.05])
    samples_k2 = [top_k_sample(probs.copy(), k=2) for _ in range(200)]
    test("top-k=2 only returns top 2 indices",
         all(s in (0, 1) for s in samples_k2), True)
    samples_k1 = [top_k_sample(probs.copy(), k=1) for _ in range(10)]
    test("top-k=1 always returns argmax", all(s == 0 for s in samples_k1), True)
    samples_full = [top_k_sample(probs.copy(), k=len(probs)) for _ in range(500)]
    test("top-k=n can return any index", 3 in samples_full, True)

    print("\nExercise 6: nucleus_sample")
    np.random.seed(42)
    peaked = np.array([0.95, 0.03, 0.01, 0.01])
    samples_peaked = [nucleus_sample(peaked, p=0.9) for _ in range(50)]
    test("nucleus peaked p=0.9 → always index 0",
         all(s == 0 for s in samples_peaked), True)
    flat = np.array([0.25, 0.25, 0.25, 0.25])
    samples_flat = [nucleus_sample(flat, p=0.9) for _ in range(200)]
    unique_sampled = len(set(samples_flat))
    test("nucleus flat p=0.9 → samples from multiple tokens",
         unique_sampled >= 3, True)
    result = nucleus_sample(np.array([0.7, 0.2, 0.07, 0.03]), p=1.0)
    test("nucleus p=1.0 returns valid index", result in (0, 1, 2, 3), True)

    print(f"\n{'='*55}")
    print(f"Score: {passed}/{total}")
    print("=" * 55)


if __name__ == '__main__':
    check()
