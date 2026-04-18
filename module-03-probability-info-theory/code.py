"""
Module 03: Probability and Information Theory — Python Implementations
=======================================================================
Run this file directly:  python code.py

Each section demonstrates a core probability/info-theory concept with
intuitive comments aimed at Python/backend developers.

Key theme: these tools let us measure uncertainty and quantify how
"surprised" a model is — which is exactly what cross-entropy loss does.
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# SECTION 1: Gaussian (Normal) Distribution
#
# The bell curve — parameterized by mean (μ) and std deviation (σ).
# Appears everywhere in ML: weight initialization, latency distributions,
# noise modeling, VAE latent spaces.
#
# Backend analogy: your API response time distribution under steady load.
# =============================================================================

def gaussian_pdf(x, mu=0.0, sigma=1.0):
    """
    Probability density function (PDF) of a Gaussian distribution.

    f(x) = (1 / σ√2π) * exp(-(x-μ)²/2σ²)

    Note: this gives a *density*, not a probability. Integrate over a range
    to get the actual probability that X falls in that range.
    """
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def demo_gaussian():
    """Plot Gaussian distributions with different parameters."""
    print("=== Gaussian Distribution Demo ===")

    x = np.linspace(-5, 5, 300)

    configs = [
        (0, 1, 'blue',   'N(μ=0, σ=1) — standard normal'),
        (1, 0.5, 'green', 'N(μ=1, σ=0.5) — narrower, shifted'),
        (-1, 2, 'red',   'N(μ=-1, σ=2) — wider, shifted left'),
    ]

    plt.figure(figsize=(8, 4))
    for mu, sigma, color, label in configs:
        y = gaussian_pdf(x, mu, sigma)
        plt.plot(x, y, color=color, linewidth=2, label=label)
        print(f"  {label}: peak at x={mu}, height={gaussian_pdf(mu, mu, sigma):.4f}")

    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.title('Gaussian Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('gaussian.png', dpi=100)
    plt.close()
    print("Saved: gaussian.png")


# =============================================================================
# SECTION 2: Entropy — Measuring Uncertainty
#
# H(p) = -Σ p(x) * log₂(p(x))
#
# Zero entropy = fully predictable.
# Maximum entropy = maximum uncertainty (uniform distribution).
#
# Backend analogy: entropy of your request routing distribution.
# Low entropy = deterministic routing (one server gets everything).
# High entropy = uniform load balancing.
# =============================================================================

def entropy(probs, base=2):
    """
    Compute Shannon entropy of a discrete probability distribution.

    H(p) = -Σ p(x) * log_base(p(x))

    Args:
        probs: array of probabilities (must sum to ~1)
        base: logarithm base. base=2 → bits; base=e → nats (used in ML)

    Returns:
        entropy as a float
    """
    probs = np.array(probs, dtype=float)
    # Guard against log(0) — 0 * log(0) is defined as 0 in information theory
    # because lim(p→0) p*log(p) = 0
    nonzero = probs[probs > 0]
    return -np.sum(nonzero * np.log(nonzero) / np.log(base))


def demo_entropy():
    """Show how entropy varies across different distributions."""
    print("\n=== Entropy Demo ===")

    distributions = [
        ([1.0, 0.0, 0.0, 0.0], "Certain (always server A)"),
        ([0.9, 0.1, 0.0, 0.0], "Biased (mostly server A)"),
        ([0.5, 0.5, 0.0, 0.0], "Fair coin (2 options)"),
        ([0.25, 0.25, 0.25, 0.25], "Uniform (4 servers — max entropy)"),
        ([0.97, 0.01, 0.01, 0.01], "Very biased"),
    ]

    print(f"\n  {'Distribution':>35}  {'Entropy (bits)':>14}")
    print(f"  {'-'*35}  {'-'*14}")
    for probs, label in distributions:
        h = entropy(probs)
        bar = '█' * int(h * 10)
        print(f"  {label:>35}  {h:>8.4f} bits  {bar}")

    print("\n  Observation: uniform distribution has the highest entropy.")
    print("  Certain outcome has 0 entropy — no information needed to encode it.")

    # Plot entropy vs p for a binary distribution
    p_values = np.linspace(0.001, 0.999, 200)
    h_values = [entropy([p, 1 - p]) for p in p_values]

    plt.figure(figsize=(7, 4))
    plt.plot(p_values, h_values, 'b-', linewidth=2)
    plt.xlabel('p (probability of outcome 1)')
    plt.ylabel('Entropy H(p) in bits')
    plt.title('Binary Entropy: H(p, 1-p)\nMaximum at p=0.5 (fair coin = 1 bit)')
    plt.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Maximum at p=0.5')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('entropy.png', dpi=100)
    plt.close()
    print("Saved: entropy.png")


# =============================================================================
# SECTION 3: KL Divergence — How Different Are Two Distributions?
#
# KL(p || q) = Σ p(x) * log(p(x) / q(x))
#
# Measures how much extra information you need if you use q to encode
# data that actually follows p. Always ≥ 0; equals 0 iff p = q.
#
# Backend analogy: you designed your infrastructure assuming traffic
# distribution q (your model), but actual traffic follows p.
# KL divergence measures the cost of being wrong about your traffic model.
# =============================================================================

def kl_divergence(p, q, epsilon=1e-10):
    """
    Compute KL divergence KL(p || q).

    KL(p || q) = Σ p(x) * log(p(x) / q(x))

    Args:
        p: true distribution (array, sums to 1)
        q: approximate distribution (array, sums to 1)
        epsilon: small value to avoid log(0)

    Returns:
        KL divergence as a float (always ≥ 0)
    """
    p = np.array(p, dtype=float) + epsilon
    q = np.array(q, dtype=float) + epsilon
    # Normalize to ensure they sum to 1 (after epsilon addition)
    p = p / p.sum()
    q = q / q.sum()
    return np.sum(p * np.log(p / q))


def demo_kl_divergence():
    """Show KL divergence between different distributions."""
    print("\n=== KL Divergence Demo ===")

    # True distribution p (actual traffic across 4 servers)
    p = np.array([0.4, 0.3, 0.2, 0.1])

    # Different approximations q
    scenarios = [
        (p.copy(),                    "q = p (perfect model)"),
        ([0.25, 0.25, 0.25, 0.25],    "q = uniform (round-robin)"),
        ([0.7, 0.1, 0.1, 0.1],        "q = over-estimates server 0"),
        ([0.1, 0.1, 0.1, 0.7],        "q = wrong server dominates"),
        ([0.39, 0.31, 0.19, 0.11],    "q ≈ p (close but not exact)"),
    ]

    print(f"  True distribution p = {p.tolist()}")
    print(f"\n  {'Scenario':>35}  {'KL(p||q)':>10}")
    print(f"  {'-'*35}  {'-'*10}")
    for q_vals, label in scenarios:
        kl = kl_divergence(p, q_vals)
        print(f"  {label:>35}  {kl:>10.6f}")

    print("\n  KL = 0 means perfect match. Higher = more different.")
    print("  Note: KL(p||q) ≠ KL(q||p) — it's asymmetric.")

    # Demonstrate asymmetry
    q_test = [0.7, 0.1, 0.1, 0.1]
    kl_pq = kl_divergence(p, q_test)
    kl_qp = kl_divergence(q_test, p)
    print(f"\n  KL(p||q) = {kl_pq:.4f}  vs  KL(q||p) = {kl_qp:.4f}  ← not equal!")


# =============================================================================
# SECTION 4: Softmax — Raw Scores → Probabilities
#
# Neural networks output "logits" — raw scores with no constraints.
# Softmax converts them to a valid probability distribution.
#
# softmax(z)_i = exp(z_i) / Σ_j exp(z_j)
#
# Backend analogy: converting raw server capacity scores into
# percentage-of-traffic allocations for a load balancer.
# =============================================================================

def softmax(logits):
    """
    Convert raw logit scores to a probability distribution.

    Uses the numerically stable version: subtract max(logits) first
    to prevent overflow when exp() is applied to large numbers.

    softmax(z)_i = exp(z_i - max(z)) / Σ_j exp(z_j - max(z))

    Args:
        logits: array of raw scores (any real numbers)

    Returns:
        array of probabilities (non-negative, sums to 1.0)
    """
    logits = np.array(logits, dtype=float)
    # Numerical stability: subtract max before exponentiating
    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)


def demo_softmax():
    """Show how softmax converts logits to probabilities."""
    print("\n=== Softmax Demo ===")

    examples = [
        ([2.0, 1.0, 0.1],        "Slightly biased toward class 0"),
        ([10.0, 1.0, 0.1],       "Very confident class 0"),
        ([0.0, 0.0, 0.0],        "Uniform logits → uniform probs"),
        ([-1.0, 0.0, 1.0, 2.0],  "Four classes, increasing logits"),
    ]

    for logits, label in examples:
        probs = softmax(logits)
        print(f"\n  {label}")
        print(f"    Logits: {[round(l, 2) for l in logits]}")
        print(f"    Probs:  {[round(p, 4) for p in probs.tolist()]}  (sum = {probs.sum():.6f})")

    print("\n  Key properties:")
    print("  - Probabilities always sum to 1")
    print("  - Higher logit → higher probability (order preserved)")
    print("  - Exponentiation amplifies differences between logits")

    # Effect of temperature scaling
    print("\n  --- Temperature Scaling ---")
    logits = np.array([2.0, 1.0, 0.5])
    for temp in [0.1, 0.5, 1.0, 2.0, 10.0]:
        probs = softmax(logits / temp)
        print(f"  T={temp:>4.1f}: probs = {probs.round(4).tolist()}  "
              f"({'peaked/confident' if temp < 1 else 'flat/uncertain' if temp > 1 else 'default'})")
    print("  Low temperature → more peaked (greedy). High temperature → more uniform (creative).")


# =============================================================================
# SECTION 5: Cross-Entropy Loss
#
# H(p, q) = -Σ p(x) * log(q(x))
#
# For one-hot true labels (standard classification):
#   Loss = -log(q[correct_class])
#
# This is the main training loss for LLMs.
# It equals the negative log probability of the correct answer.
#
# Backend analogy: how "surprised" is your model by the correct answer?
# Low loss = model was confident and right.
# High loss = model was either uncertain or confidently wrong.
# =============================================================================

def cross_entropy_loss(true_probs, predicted_probs, epsilon=1e-10):
    """
    Compute cross-entropy loss between true distribution p and predicted q.

    H(p, q) = -Σ p(x) * log(q(x))

    For one-hot true_probs (standard classification):
        = -log(predicted_probs[correct_class])

    Args:
        true_probs: the true distribution p (often one-hot)
        predicted_probs: model's predicted distribution q
        epsilon: small value to avoid log(0)

    Returns:
        cross-entropy loss as a float (lower = better)
    """
    p = np.array(true_probs, dtype=float)
    q = np.array(predicted_probs, dtype=float) + epsilon
    return -np.sum(p * np.log(q))


def demo_cross_entropy():
    """
    Show how cross-entropy loss behaves in different prediction scenarios.
    Demonstrates: confident+correct → low loss, wrong → high loss.
    """
    print("\n=== Cross-Entropy Loss Demo ===")
    print("Scenario: 3-class classification (classes: 0, 1, 2)")
    print("Correct answer: class 0 (one-hot: [1, 0, 0])\n")

    true_label = [1.0, 0.0, 0.0]   # one-hot: correct class is 0

    scenarios = [
        ([0.9,  0.05, 0.05], "Very confident, correct"),
        ([0.7,  0.2,  0.1 ], "Confident, correct"),
        ([0.33, 0.33, 0.34], "Uniform — no confidence"),
        ([0.1,  0.5,  0.4 ], "Confident, WRONG (thinks class 1)"),
        ([0.01, 0.01, 0.98], "Very confident, VERY WRONG"),
    ]

    print(f"  {'Scenario':>35}  {'Prediction':>22}  {'Loss':>8}")
    print(f"  {'-'*35}  {'-'*22}  {'-'*8}")
    for probs, label in scenarios:
        loss = cross_entropy_loss(true_label, probs)
        # Loss should equal -log(probs[0]) for one-hot true label
        expected = -np.log(probs[0] + 1e-10)
        print(f"  {label:>35}  {str(probs):>22}  {loss:>8.4f}")

    print("\n  Key insight: loss = -log(probability_of_correct_class)")
    print("  Being confidently WRONG is penalized much more than being uncertain.")

    # Visualize: loss vs probability of correct class
    p_correct = np.linspace(0.01, 0.99, 200)
    loss_vals = -np.log(p_correct)

    plt.figure(figsize=(7, 4))
    plt.plot(p_correct, loss_vals, 'b-', linewidth=2)
    plt.xlabel('Model\'s probability assigned to correct class')
    plt.ylabel('Cross-Entropy Loss = -log(p_correct)')
    plt.title('Cross-Entropy Loss vs Model Confidence\n'
              'Confident & correct → near 0. Wrong → very high.')
    plt.axvline(0.9, color='green', linestyle='--', alpha=0.7, label='90% correct → loss≈0.1')
    plt.axvline(0.1, color='red',   linestyle='--', alpha=0.7, label='10% correct → loss≈2.3')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 5)
    plt.tight_layout()
    plt.savefig('cross_entropy.png', dpi=100)
    plt.close()
    print("Saved: cross_entropy.png")


# =============================================================================
# SECTION 6: Cross-Entropy ≥ Entropy
#
# Mathematical proof via KL divergence:
#   H(p, q) = H(p) + KL(p || q)
#   Since KL ≥ 0, therefore H(p, q) ≥ H(p)
#
# This means no model can achieve a cross-entropy loss lower than the
# true entropy of the data — an information-theoretic lower bound.
# =============================================================================

def demo_cross_entropy_lower_bound():
    """
    Show that cross-entropy loss is always ≥ true entropy.
    The gap is the KL divergence — how far the model is from perfect.
    """
    print("\n=== Cross-Entropy ≥ Entropy Demo ===")
    print("H(p, q) = H(p) + KL(p || q)  →  cross-entropy ≥ true entropy\n")

    # True distribution (e.g., true distribution of next tokens in a corpus)
    p = np.array([0.5, 0.3, 0.15, 0.05])

    true_entropy = entropy(p, base=np.e)  # nats (natural log, as used in ML)

    model_predictions = [
        (p.copy(),                         "Perfect model q = p"),
        ([0.4, 0.3, 0.2, 0.1],            "Good model (close to p)"),
        ([0.25, 0.25, 0.25, 0.25],         "Uniform model (knows nothing)"),
        ([0.1, 0.1, 0.1, 0.7],            "Terrible model (wrong)"),
    ]

    print(f"  True distribution p = {p.tolist()}")
    print(f"  True entropy H(p)   = {true_entropy:.4f} nats")
    print()
    print(f"  {'Model':>35}  {'H(p,q)':>8}  {'H(p)':>8}  {'KL(p||q)':>10}  {'≥ H(p)?':>9}")
    print(f"  {'-'*35}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*9}")

    for q_vals, label in model_predictions:
        q = np.array(q_vals, dtype=float)
        h_pq = cross_entropy_loss(p, q)          # H(p, q) in nats
        h_p = true_entropy
        kl = kl_divergence(p, q)
        ge = h_pq >= h_p - 1e-9  # allow tiny floating-point tolerance
        print(f"  {label:>35}  {h_pq:>8.4f}  {h_p:>8.4f}  {kl:>10.4f}  {'YES ✓' if ge else 'NO ✗':>9}")

    print("\n  Cross-entropy is always ≥ entropy. The gap = KL divergence.")
    print("  Training minimizes this gap — pushing the model distribution toward truth.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 65)
    print("MODULE 03: Probability and Information Theory Demos")
    print("=" * 65)

    demo_gaussian()
    demo_entropy()
    demo_kl_divergence()
    demo_softmax()
    demo_cross_entropy()
    demo_cross_entropy_lower_bound()

    print("\n" + "=" * 65)
    print("All demos complete. Check the generated PNG files:")
    print("  gaussian.png       — Gaussian distributions with different μ, σ")
    print("  entropy.png        — Binary entropy curve")
    print("  cross_entropy.png  — Loss vs model confidence")
    print("=" * 65)
