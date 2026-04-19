"""
Module 03 — Code 01: Probability Distributions
================================================
Build probability distributions from scratch and connect them
to how LLMs work internally.

Run: python code_01_distributions.py
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# SECTION 1: Discrete Distributions
# =============================================================================

def section_discrete():
    print("=" * 55)
    print("SECTION 1: Discrete Distributions")
    print("=" * 55)

    # A simple discrete distribution over 4 servers
    probs = np.array([0.5, 0.3, 0.15, 0.05])
    outcomes = ["server_A", "server_B", "server_C", "server_D"]

    print("\nServer traffic distribution:")
    for name, p in zip(outcomes, probs):
        bar = "█" * int(p * 40)
        print(f"  {name}: {p:.2f}  {bar}")
    print(f"  Sum = {probs.sum():.4f}  (must be 1.0)")

    # Simulate 1000 requests
    samples = np.random.choice(outcomes, size=1000, p=probs)
    unique, counts = np.unique(samples, return_counts=True)
    print("\nSimulated 1000 requests — empirical frequencies:")
    for name, count in zip(unique, counts):
        print(f"  {name}: {count/1000:.3f}  (theoretical: {probs[outcomes.index(name)]:.3f})")

    # LLM vocabulary distribution — tiny 6-token vocabulary example
    print("\n--- LLM token distribution example ---")
    vocab = ["the", "a", "Paris", "London", "cat", "dog"]
    # Context: "The capital of France is"
    token_probs = np.array([0.03, 0.01, 0.82, 0.10, 0.02, 0.02])

    print("Context: 'The capital of France is ___'")
    print("Token probabilities:")
    for tok, p in zip(vocab, token_probs):
        bar = "█" * int(p * 50)
        print(f"  {tok:8s}: {p:.3f}  {bar}")
    print(f"\nMost likely next token: '{vocab[np.argmax(token_probs)]}' "
          f"({token_probs.max():.1%})")


# =============================================================================
# SECTION 2: Gaussian Distribution
# =============================================================================

def gaussian_pdf(x, mu, sigma):
    """Probability density function of N(mu, sigma^2)."""
    coeff = 1.0 / (sigma * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * np.exp(exponent)


def section_gaussian():
    print("\n" + "=" * 55)
    print("SECTION 2: Gaussian Distribution")
    print("=" * 55)

    x = np.linspace(-5, 5, 1000)

    configs = [
        (0.0, 1.0, "Standard N(0,1)"),
        (2.0, 0.5, "Narrow N(2, 0.25)"),
        (0.0, 2.0, "Wide N(0, 4)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for mu, sigma, label in configs:
        y = gaussian_pdf(x, mu, sigma)
        axes[0].plot(x, y, label=label, linewidth=2)

    axes[0].set_xlabel("x")
    axes[0].set_ylabel("Probability density")
    axes[0].set_title("Gaussian PDFs")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    print("\nGaussian (normal) distribution N(μ, σ²):")
    for mu, sigma, label in configs:
        peak = gaussian_pdf(mu, mu, sigma)
        print(f"  {label}: peak at x={mu} with density {peak:.3f}")

    # Why Gaussian matters for LLMs: weight initialization
    print("\n--- Weight initialization in LLMs ---")
    d_model = 768   # GPT-2 hidden size
    sigma_xavier = np.sqrt(1.0 / d_model)

    weights = np.random.normal(0, sigma_xavier, size=(d_model, d_model))
    print(f"  Weight matrix shape: {weights.shape}")
    print(f"  Initialization: N(0, {sigma_xavier:.5f}²)")
    print(f"  Actual mean:    {weights.mean():.6f}  (should be ≈ 0)")
    print(f"  Actual std:     {weights.std():.5f}  (should be ≈ {sigma_xavier:.5f})")
    print("  Small values prevent gradient explosion in deep networks")

    # Visualize weight distribution
    axes[1].hist(weights.flatten()[:5000], bins=60, density=True,
                 alpha=0.7, color='steelblue', label='Sampled weights')
    x_range = np.linspace(-0.15, 0.15, 300)
    axes[1].plot(x_range, gaussian_pdf(x_range, 0, sigma_xavier),
                 'r-', linewidth=2, label=f'N(0, {sigma_xavier:.4f}²)')
    axes[1].set_xlabel("Weight value")
    axes[1].set_ylabel("Density")
    axes[1].set_title(f"LLM Weight Initialization (d_model={d_model})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("distributions.png", dpi=100)
    plt.close()
    print("\nSaved: distributions.png")


# =============================================================================
# SECTION 3: Expectation and Variance
# =============================================================================

def section_expectation():
    print("\n" + "=" * 55)
    print("SECTION 3: Expectation and Variance")
    print("=" * 55)

    # Dice roll
    outcomes = np.array([1, 2, 3, 4, 5, 6])
    probs    = np.array([1/6] * 6)

    E_x  = np.sum(outcomes * probs)
    E_x2 = np.sum(outcomes**2 * probs)
    Var  = E_x2 - E_x**2
    std  = np.sqrt(Var)

    print(f"\nFair die:")
    print(f"  E[X]    = {E_x:.4f}  (expected roll)")
    print(f"  Var[X]  = {Var:.4f}  (variance)")
    print(f"  Std[X]  = {std:.4f}  (standard deviation)")

    # API latency comparison
    print("\nAPI latency comparison (same mean, different variance):")
    np.random.seed(42)
    service_A = np.random.normal(loc=50, scale=5,  size=10000)  # tight distribution
    service_B = np.random.normal(loc=50, scale=30, size=10000)  # wide distribution

    for name, data in [("Service A (σ=5)",  service_A),
                       ("Service B (σ=30)", service_B)]:
        p99 = np.percentile(data, 99)
        print(f"  {name}: mean={data.mean():.1f}ms, std={data.std():.1f}ms, "
              f"p99={p99:.1f}ms")

    print("\n  Same average latency, but Service B has terrible tail latency (p99)")
    print("  Variance reveals what averages hide — exactly why p99 monitoring matters")

    # Expectation in LLMs: expected token
    print("\nExpected token in LLM output:")
    vocab = ["the", "a", "Paris", "London", "cat"]
    probs_llm = np.array([0.03, 0.01, 0.82, 0.10, 0.04])
    # For discrete tokens, "expected token" doesn't make direct sense,
    # but expected loss does:
    true_probs = np.array([0.0, 0.0, 1.0, 0.0, 0.0])  # "Paris" is correct

    cross_entropy = -np.sum(true_probs * np.log(probs_llm + 1e-10))
    print(f"  Model assigns P('Paris') = {probs_llm[2]:.2f}")
    print(f"  Cross-entropy loss = -log({probs_llm[2]:.2f}) = {cross_entropy:.4f}")
    print(f"  This is the expected surprise under the true distribution")


# =============================================================================
# SECTION 4: Conditional Probability in LLMs
# =============================================================================

def section_conditional():
    print("\n" + "=" * 55)
    print("SECTION 4: Conditional Probability in LLMs")
    print("=" * 55)

    print("""
The entire LLM task: P(next_token | all_previous_tokens)

Context 1: "The capital of France is"
  → P("Paris") = 0.82   P("London") = 0.03   P("city") = 0.08   ...

Context 2: "The best programming language is"
  → P("Python") = 0.25  P("JavaScript") = 0.20  P("C") = 0.15   ...

Context 3: "I am feeling very"
  → P("happy") = 0.18   P("tired") = 0.15   P("good") = 0.14    ...

Each context gives a DIFFERENT probability distribution over the vocabulary.
The transformer learns all these conditional distributions simultaneously,
from billions of context-token pairs in the training data.
""")

    # Simulate how context changes the distribution
    print("Demonstration: context changes the token distribution")
    contexts = [
        ("The weather is",     {"sunny": 0.20, "cold": 0.15, "changing": 0.10, "nice": 0.18}),
        ("The capital is",     {"Paris": 0.35, "London": 0.20, "Berlin": 0.15, "Rome": 0.12}),
        ("The answer is",      {"42": 0.30, "yes": 0.20, "no": 0.15, "correct": 0.10}),
    ]

    for context, top_preds in contexts:
        print(f"\n  Context: '{context}'")
        total = sum(top_preds.values())
        for token, p in sorted(top_preds.items(), key=lambda x: -x[1]):
            bar = "█" * int(p / total * 30)
            print(f"    {token:12s}: {p/total:.2f} {bar}")
        print(f"    (+ many more tokens with small probabilities)")


# =============================================================================
# RUN ALL SECTIONS
# =============================================================================

if __name__ == '__main__':
    section_discrete()
    section_gaussian()
    section_expectation()
    section_conditional()

    print("\n" + "=" * 55)
    print("Key takeaways:")
    print("  1. A probability distribution sums to 1 over all outcomes")
    print("  2. Gaussian used for weight init; discrete for token probs")
    print("  3. Variance reveals tail behavior that means hide")
    print("  4. LLMs learn P(token | context) — conditional distributions")
    print("=" * 55)
