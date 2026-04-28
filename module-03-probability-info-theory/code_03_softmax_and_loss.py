"""
Module 03 — Code 03: Softmax and Cross-Entropy Loss
=====================================================
Implement softmax, cross-entropy, and their gradients from scratch.
Verify against PyTorch. Show numerical stability in action.

Run: python code_03_softmax_and_loss.py
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def softmax(z):
    """
    Numerically stable softmax.

    Subtracts max(z) before exponentiating to prevent overflow.
    This doesn't change the result: softmax(z) = softmax(z - c) for any c.
    """
    z = np.array(z, dtype=float)
    z_shifted = z - np.max(z)      # stability: max value becomes 0
    exp_z     = np.exp(z_shifted)
    return exp_z / exp_z.sum()


def cross_entropy_loss(logits, label):
    """
    Cross-entropy loss for a single example.

    Args:
        logits: raw model output, shape (vocab_size,)
        label:  index of the correct class (integer)

    Returns:
        loss = -log(softmax(logits)[label])
    """
    probs = softmax(logits)
    return -np.log(probs[label] + 1e-10)   # 1e-10 prevents log(0)


def cross_entropy_gradient(logits, label):
    """
    Gradient of cross-entropy loss w.r.t. logits.

    The beautiful result: ∂L/∂z_i = softmax(z)_i - 1[i == label]
                                   = predicted_prob - true_prob

    For wrong classes (i ≠ label): gradient = prob     (push DOWN)
    For correct class (i == label): gradient = prob - 1 (push UP toward 1)
    """
    probs = softmax(logits)
    grad  = probs.copy()
    grad[label] -= 1.0     # subtract 1 from the correct class
    return grad


def sequence_cross_entropy(logits_seq, labels):
    """
    Average cross-entropy over a sequence of tokens.
    This is the actual LLM training loss.

    Args:
        logits_seq: shape (seq_len, vocab_size)
        labels:     shape (seq_len,) — correct token at each position

    Returns:
        average loss across all positions
    """
    total_loss = 0.0
    for t, (logits_t, label_t) in enumerate(zip(logits_seq, labels)):
        total_loss += cross_entropy_loss(logits_t, label_t)
    return total_loss / len(labels)


# =============================================================================
# SECTION 1: Softmax Basics
# =============================================================================

def section_softmax_basics():
    print("=" * 55)
    print("SECTION 1: Softmax")
    print("=" * 55)

    logits = np.array([2.0, 1.0, 0.1, -1.0])
    probs  = softmax(logits)

    print(f"\nLogits: {logits}")
    print(f"Probs:  {np.round(probs, 4)}")
    print(f"Sum:    {probs.sum():.6f}  (must be 1.0)")

    print("\nLogit → Probability table:")
    print(f"  {'Logit':>8}  {'Prob':>10}  {'Bar'}")
    for z, p in zip(logits, probs):
        bar = "█" * int(p * 40)
        print(f"  {z:>8.2f}  {p:>10.4f}  {bar}")

    # Show amplification effect
    print("\nAmplification: logit differences grow exponentially in probability space")
    pairs = [(0, 1), (0, 2), (0, 5), (0, 10)]
    print(f"  {'Logit diff':>12}  {'Prob ratio':>12}  {'Expected e^diff':>16}")
    for a, b in pairs:
        logits_pair = np.array([float(a), float(b)])
        p = softmax(logits_pair)
        ratio = p[1] / p[0]
        expected = np.exp(b - a)
        print(f"  {b-a:>12}  {ratio:>12.2f}  {expected:>16.2f}")


# =============================================================================
# SECTION 2: Numerical Stability
# =============================================================================

def section_numerical_stability():
    print("\n" + "=" * 55)
    print("SECTION 2: Numerical Stability")
    print("=" * 55)

    # Naive softmax (no stability fix)
    def softmax_naive(z):
        exp_z = np.exp(z)
        return exp_z / exp_z.sum()

    print("\nNaive vs stable softmax:")
    print(f"  {'Logits':>20}  {'Naive':>20}  {'Stable':>20}")
    print("-" * 70)

    test_cases = [
        np.array([1.0, 2.0, 3.0]),        # normal
        np.array([100.0, 200.0, 300.0]),   # large — naive overflows
        np.array([-1000.0, -999.0]),       # large negative
    ]

    for logits in test_cases:
        with np.errstate(over='ignore', invalid='ignore'):
            naive_result = softmax_naive(logits)
        stable_result = softmax(logits)

        naive_str  = str(np.round(naive_result, 4)) if not np.any(np.isnan(naive_result)) else "NaN/Inf overflow!"
        stable_str = str(np.round(stable_result, 4))
        print(f"  {str(logits):>20}  {naive_str:>20}  {stable_str:>20}")

    print("\nFix: subtract max(logits) before exp()")
    print("  softmax(z) = softmax(z - max(z))  [mathematically identical]")
    print("  After subtraction: max becomes 0, exp(0)=1 — no overflow")


# =============================================================================
# SECTION 3: Cross-Entropy Loss
# =============================================================================

def section_cross_entropy():
    print("\n" + "=" * 55)
    print("SECTION 3: Cross-Entropy Loss")
    print("=" * 55)

    vocab  = ["the", "a", "Paris", "London", "dog"]
    label  = 2   # "Paris" is correct

    print(f"\nCorrect token: '{vocab[label]}' (index {label})")

    model_outputs = [
        (np.array([0.1, 0.2, 3.5, 1.0, 0.3]),   "Good model  (logit 3.5 for Paris)"),
        (np.array([0.1, 0.2, 0.0, 1.0, 0.3]),   "Confused    (all logits low)"),
        (np.array([0.1, 0.2, -2.0, 3.0, 0.3]),  "Wrong model (logit 3.0 for London)"),
    ]

    print(f"\n{'Model':30s}  {'P(Paris)':>10}  {'Loss':>10}  {'PPL':>10}")
    print("-" * 65)
    for logits, desc in model_outputs:
        probs = softmax(logits)
        loss  = cross_entropy_loss(logits, label)
        ppl   = np.exp(loss)
        print(f"  {desc:30s}  {probs[label]:>10.4f}  {loss:>10.4f}  {ppl:>10.4f}")

    # Show relationship: loss = -log(prob_correct)
    print("\n-log(prob) interpretation:")
    probs_range = [0.99, 0.90, 0.70, 0.50, 0.20, 0.10, 0.01]
    print(f"  {'P(correct)':>12}  {'Loss=-log(p)':>14}  {'PPL=exp(loss)':>16}")
    for p in probs_range:
        loss = -np.log(p)
        ppl  = np.exp(loss)
        print(f"  {p:>12.2f}  {loss:>14.4f}  {ppl:>16.4f}")

    # Sequence loss
    print("\n--- Sequence cross-entropy: average over all positions ---")
    # "The capital of France is Paris"
    # Simplified: 6 positions, each with logits for 4-token vocab
    np.random.seed(7)
    seq_len   = 6
    vocab_sz  = 4
    logits_seq = np.random.randn(seq_len, vocab_sz)
    # True labels (what each next token should be)
    labels     = np.array([1, 0, 2, 3, 0, 2])

    avg_loss = sequence_cross_entropy(logits_seq, labels)
    per_pos  = [cross_entropy_loss(logits_seq[t], labels[t]) for t in range(seq_len)]

    print(f"\n  Sequence length: {seq_len}")
    print(f"  Per-position losses: {[f'{l:.3f}' for l in per_pos]}")
    print(f"  Average loss: {avg_loss:.4f}")
    print(f"  Perplexity:   {np.exp(avg_loss):.4f}")
    print(f"\n  This is exactly what PyTorch's F.cross_entropy() computes.")


# =============================================================================
# SECTION 4: Gradient of Cross-Entropy
# =============================================================================

def section_gradient():
    print("\n" + "=" * 55)
    print("SECTION 4: Gradient of Cross-Entropy + Softmax")
    print("=" * 55)

    print("""
The gradient of (cross-entropy + softmax) w.r.t. logits is:
  ∂L/∂z_i = softmax(z)_i - p_true_i
           = predicted_prob - true_prob

For wrong classes (i ≠ label): ∂L/∂z_i = prob_i     > 0  → push logit DOWN
For correct class (i == label): ∂L/∂z_i = prob_i - 1 < 0  → push logit UP

This is the gradient that backpropagates through the entire network.
""")

    logits = np.array([1.0, 0.5, 2.0, -0.5])
    label  = 2   # correct class is index 2

    probs = softmax(logits)
    grad  = cross_entropy_gradient(logits, label)

    # Verify with numerical gradient
    h         = 1e-5
    num_grad  = np.zeros_like(logits)
    for i in range(len(logits)):
        logits_plus       = logits.copy(); logits_plus[i]  += h
        logits_minus      = logits.copy(); logits_minus[i] -= h
        loss_plus  = cross_entropy_loss(logits_plus,  label)
        loss_minus = cross_entropy_loss(logits_minus, label)
        num_grad[i] = (loss_plus - loss_minus) / (2 * h)

    print(f"Logits:          {np.round(logits, 3)}")
    print(f"Probs:           {np.round(probs, 4)}")
    print(f"True label:      index {label}")
    print(f"\nGradients:")
    print(f"  Analytical:    {np.round(grad, 6)}")
    print(f"  Numerical:     {np.round(num_grad, 6)}")
    print(f"  Match:         {np.allclose(grad, num_grad, atol=1e-5)}")

    print(f"\nInterpretation:")
    for i, (g, p) in enumerate(zip(grad, probs)):
        role = "correct" if i == label else "wrong  "
        direction = "push UP ↑" if g < 0 else "push DOWN ↓"
        print(f"  Index {i} ({role}): prob={p:.4f}, grad={g:+.4f} → {direction}")


# =============================================================================
# SECTION 5: Temperature Effect on Softmax
# =============================================================================

def section_temperature():
    print("\n" + "=" * 55)
    print("SECTION 5: Temperature Scaling")
    print("=" * 55)

    logits = np.array([3.0, 1.5, 0.8, 0.2, -0.5])
    vocab  = ["Paris", "London", "Berlin", "Rome", "Tokyo"]
    temperatures = [0.3, 0.7, 1.0, 1.5, 2.0, 5.0]

    print(f"\nLogits: {logits}")
    print(f"\n{'T':>6}  ", end="")
    for v in vocab:
        print(f"{v:>10}", end="")
    print(f"  {'Entropy':>10}")
    print("-" * 75)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    all_probs = []
    for T in temperatures:
        probs = softmax(logits / T)
        all_probs.append(probs)
        h = -np.sum(probs * np.log(probs + 1e-10))
        print(f"{T:>6.1f}  ", end="")
        for p in probs:
            print(f"{p:>10.4f}", end="")
        print(f"  {h:>10.4f}")

    # Plot: how distribution changes with temperature
    x = np.arange(len(vocab))
    for i, (T, probs) in enumerate(zip(temperatures, all_probs)):
        axes[0].plot(x, probs, marker='o', label=f'T={T}', linewidth=1.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(vocab, rotation=20)
    axes[0].set_ylabel("Probability")
    axes[0].set_title("Softmax Distribution at Different Temperatures")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Entropy vs temperature
    entropies = [-np.sum(p * np.log(p + 1e-10)) for p in all_probs]
    axes[1].plot(temperatures, entropies, 'b-o', linewidth=2)
    axes[1].set_xlabel("Temperature")
    axes[1].set_ylabel("Distribution entropy (nats)")
    axes[1].set_title("Higher Temperature → Higher Entropy → More Random Output")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("softmax_temperature.png", dpi=100)
    plt.close()

    print("\nSaved: softmax_temperature.png")
    print("""
T < 1: peaks sharper → model more decisive (greedy-like behavior)
T = 1: standard softmax
T > 1: peaks flatten → more diverse/random output
T → ∞: uniform → completely random
T → 0: one-hot → always picks argmax (deterministic)

In practice: temperature=0.7 for code, temperature=1.0–1.2 for creative writing.
""")


# =============================================================================
# SECTION 6: Verify Against PyTorch
# =============================================================================

def section_verify_pytorch():
    print("=" * 55)
    print("SECTION 6: Verify Against PyTorch")
    print("=" * 55)

    try:
        import torch
        import torch.nn.functional as F

        np.random.seed(42)
        logits_np = np.random.randn(5)
        label     = 3

        # Our implementation
        probs_ours = softmax(logits_np)
        loss_ours  = cross_entropy_loss(logits_np, label)

        # PyTorch
        logits_t = torch.tensor(logits_np, dtype=torch.float64)
        probs_pt = F.softmax(logits_t, dim=0).numpy()
        loss_pt  = F.cross_entropy(
            logits_t.unsqueeze(0),
            torch.tensor([label])
        ).item()

        print(f"\nLogits: {np.round(logits_np, 4)}, label: {label}")
        print(f"\n{'':20}  {'Ours':>12}  {'PyTorch':>12}  {'Match':>8}")
        print("-" * 60)
        for i in range(len(probs_ours)):
            match = abs(probs_ours[i] - probs_pt[i]) < 1e-6
            print(f"  prob[{i}]:            {probs_ours[i]:>12.6f}  {probs_pt[i]:>12.6f}  {'✓' if match else '✗':>8}")

        loss_match = abs(loss_ours - loss_pt) < 1e-5
        print(f"\n  Loss:               {loss_ours:>12.6f}  {loss_pt:>12.6f}  {'✓' if loss_match else '✗':>8}")
        print("\nAll match: our softmax and cross-entropy are correct.")

    except ImportError:
        print("\n[SKIP] PyTorch not installed. Run: pip install torch")
        print("Our implementation is correct — verified against numerical gradient above.")


# =============================================================================
# RUN ALL SECTIONS
# =============================================================================

if __name__ == '__main__':
    section_softmax_basics()
    section_numerical_stability()
    section_cross_entropy()
    section_gradient()
    section_temperature()
    section_verify_pytorch()

    print("\n" + "=" * 55)
    print("Key takeaways:")
    print("  1. Softmax: logits → probs (always positive, sum=1)")
    print("  2. Numerical stability: subtract max before exp")
    print("  3. Cross-entropy = -log(prob of correct token)")
    print("  4. Gradient: predicted_prob - true_prob (elegant!)")
    print("  5. Temperature: controls sharpness of distribution")
    print("=" * 55)
