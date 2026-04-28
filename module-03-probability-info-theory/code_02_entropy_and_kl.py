"""
Module 03 — Code 02: Entropy and KL Divergence
================================================
Implement entropy and KL divergence, visualize their properties,
and connect them to LLM training and RLHF.

Run: python code_02_entropy_and_kl.py
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def entropy(p, base=2):
    """
    Shannon entropy H(p) = -Σ p(x) · log_base(p(x))

    Args:
        p: probability distribution (must sum to 1)
        base: logarithm base (2 → bits, e → nats)

    Returns:
        scalar entropy value
    """
    p = np.array(p, dtype=float)
    assert np.isclose(p.sum(), 1.0), f"Probabilities must sum to 1, got {p.sum()}"
    # Exclude zeros: 0 * log(0) = 0 by convention (limit)
    mask = p > 0
    if base == 2:
        return -np.sum(p[mask] * np.log2(p[mask]))
    else:
        return -np.sum(p[mask] * np.log(p[mask]))


def kl_divergence(p, q, base='e'):
    """
    KL divergence KL(p || q) = Σ p(x) · log(p(x) / q(x))

    p is the reference (true) distribution.
    q is the approximation (model prediction).
    Result ≥ 0; = 0 iff p == q.

    WARNING: returns inf if q(x)=0 anywhere that p(x)>0.
    """
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)

    mask = p > 0
    if np.any(q[mask] == 0):
        return float('inf')   # q misses mass that p has → undefined/infinite

    if base == 2:
        return np.sum(p[mask] * np.log2(p[mask] / q[mask]))
    else:
        return np.sum(p[mask] * np.log(p[mask] / q[mask]))


def cross_entropy(p, q):
    """
    Cross-entropy H(p, q) = -Σ p(x) · log(q(x))

    Relationship: H(p, q) = H(p) + KL(p || q)
    """
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    mask = p > 0
    return -np.sum(p[mask] * np.log(q[mask]))


# =============================================================================
# SECTION 1: Entropy — Understanding the Formula
# =============================================================================

def section_entropy_basics():
    print("=" * 55)
    print("SECTION 1: Shannon Entropy")
    print("=" * 55)

    examples = [
        ([1.0, 0.0],              "Certain (always heads)"),
        ([0.9, 0.1],              "Biased coin (90/10)"),
        ([0.7, 0.3],              "Biased coin (70/30)"),
        ([0.5, 0.5],              "Fair coin"),
        ([0.25, 0.25, 0.25, 0.25],"Fair 4-sided die"),
        ([1/6]*6,                 "Fair 6-sided die"),
    ]

    print(f"\n{'Distribution':35s}  {'Entropy (bits)':>16}  {'Max possible':>14}")
    print("-" * 72)
    for p, label in examples:
        h    = entropy(p)
        n    = len(p)
        hmax = np.log2(n)
        print(f"  {label:33s}: {h:>14.4f}  /  {hmax:>10.4f}")

    print("\nKey observation: maximum entropy = log₂(n) for n outcomes")
    print("Achieved by uniform distribution — maximum uncertainty")


# =============================================================================
# SECTION 2: Entropy as a Function of Probability (Binary Case)
# =============================================================================

def section_entropy_curve():
    print("\n" + "=" * 55)
    print("SECTION 2: Binary Entropy Curve")
    print("=" * 55)

    p_vals = np.linspace(0.001, 0.999, 1000)
    h_vals = [entropy([p, 1-p]) for p in p_vals]

    plt.figure(figsize=(8, 4))
    plt.plot(p_vals, h_vals, 'b-', linewidth=2)
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Max entropy (1 bit)')
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.scatter([0.5], [1.0], color='red', s=100, zorder=5, label='Fair coin: H=1 bit')
    plt.xlabel("P(heads)")
    plt.ylabel("Entropy (bits)")
    plt.title("Binary Entropy H(p, 1-p)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("entropy_curve.png", dpi=100)
    plt.close()
    print("Saved: entropy_curve.png")
    print("Entropy peaks at p=0.5 (maximum uncertainty), hits 0 at extremes")


# =============================================================================
# SECTION 3: KL Divergence
# =============================================================================

def section_kl_basics():
    print("\n" + "=" * 55)
    print("SECTION 3: KL Divergence")
    print("=" * 55)

    # Reference distribution p (true data)
    p = np.array([0.5, 0.3, 0.2])

    approximations = [
        (np.array([0.5, 0.3, 0.2]), "q = p (identical)"),
        (np.array([0.4, 0.4, 0.2]), "q slightly different"),
        (np.array([0.9, 0.05, 0.05]), "q very different"),
        (np.array([0.33, 0.33, 0.34]), "q = uniform"),
    ]

    print(f"\nTrue distribution p = {p}")
    print(f"\n{'Approximation q':35s}  {'KL(p||q)':>10}  {'KL(q||p)':>10}  {'Symmetric?':>12}")
    print("-" * 75)
    for q, label in approximations:
        kl_fwd = kl_divergence(p, q)
        kl_rev = kl_divergence(q, p)
        sym = "YES" if abs(kl_fwd - kl_rev) < 0.001 else "NO"
        print(f"  {label:33s}: {kl_fwd:>10.4f}  {kl_rev:>10.4f}  {sym:>12}")

    print("\nKey: KL(p||q) = 0 only when p == q")
    print("     KL is NOT symmetric: forward ≠ reverse")

    # Infinite KL
    print("\n--- Infinite KL: model assigns zero to something that happens ---")
    p_danger = np.array([0.5, 0.3, 0.2])
    q_danger = np.array([0.7, 0.3, 0.0])   # zero at index 2, but p[2] = 0.2

    kl = kl_divergence(p_danger, q_danger)
    print(f"  p = {p_danger},  q = {q_danger}")
    print(f"  KL(p || q) = {kl}")
    print("  If model assigns q=0 to something that actually occurs → infinite surprise")
    print("  This is why models never output exactly 0 probability (softmax ensures > 0)")


# =============================================================================
# SECTION 4: Cross-entropy = Entropy + KL
# =============================================================================

def section_crossentropy_decomposition():
    print("\n" + "=" * 55)
    print("SECTION 4: H(p,q) = H(p) + KL(p||q)")
    print("=" * 55)

    np.random.seed(42)
    p = np.array([0.4, 0.3, 0.2, 0.1])   # true distribution (one-hot in LLM training)
    q = np.array([0.35, 0.25, 0.25, 0.15])  # model prediction

    h_p  = entropy(p, base='e')
    kl   = kl_divergence(p, q, base='e')
    h_pq = cross_entropy(p, q)

    print(f"\np (true)   = {p}")
    print(f"q (model)  = {q}")
    print(f"\nH(p)       = {h_p:.6f} nats   (true entropy — irreducible)")
    print(f"KL(p||q)   = {kl:.6f} nats   (model's error)")
    print(f"H(p,q)     = {h_pq:.6f} nats   (cross-entropy loss)")
    print(f"H(p)+KL    = {h_p + kl:.6f} nats   (should equal H(p,q))")
    print(f"\nMatch: {np.isclose(h_pq, h_p + kl)}")

    print("""
Interpretation for LLM training:
  Cross-entropy loss = irreducible noise + how wrong the model is
  Minimizing loss = making the model less wrong (can't reduce noise)
  Perfect model achieves: loss = H(true data distribution)
""")


# =============================================================================
# SECTION 5: KL in RLHF
# =============================================================================

def section_kl_rlhf():
    print("\n" + "=" * 55)
    print("SECTION 5: KL Divergence in RLHF")
    print("=" * 55)

    print("""
RLHF training objective:
  maximize: E[reward] - β · KL(π_RL || π_SFT)

The KL penalty prevents the model from drifting too far
from the supervised fine-tuned (SFT) baseline.
""")

    # Simulate: how does a model change during RLHF fine-tuning?
    vocab_size = 8
    np.random.seed(1)

    # SFT baseline distribution (pretrained, sensible)
    sft_logits = np.array([2.1, 0.5, -0.3, 1.2, 0.8, -1.0, 0.3, -0.5])
    sft_probs  = np.exp(sft_logits) / np.sum(np.exp(sft_logits))

    # RL model after fine-tuning with different betas
    # Simulate: RL model learned to concentrate on token 0 for high reward
    rl_logits_strong = np.array([5.0, 0.1, -2.0, 0.1, 0.1, -2.0, 0.1, -2.0])
    rl_probs_strong  = np.exp(rl_logits_strong) / np.sum(np.exp(rl_logits_strong))

    rl_logits_mild   = np.array([3.0, 0.4, -0.5, 0.9, 0.6, -1.2, 0.2, -0.8])
    rl_probs_mild    = np.exp(rl_logits_mild) / np.sum(np.exp(rl_logits_mild))

    kl_strong = kl_divergence(rl_probs_strong, sft_probs)
    kl_mild   = kl_divergence(rl_probs_mild,   sft_probs)

    print(f"{'':25s}  {'SFT baseline':>14}  {'RL (mild)':>12}  {'RL (strong)':>12}")
    print("-" * 70)
    for i in range(vocab_size):
        print(f"  Token {i}:               {sft_probs[i]:>14.4f}  "
              f"{rl_probs_mild[i]:>12.4f}  {rl_probs_strong[i]:>12.4f}")
    print(f"\n  KL(RL || SFT):          {'N/A':>14}  {kl_mild:>12.4f}  {kl_strong:>12.4f}")

    print(f"""
  Mild fine-tuning (KL={kl_mild:.3f}): small drift from SFT — safe
  Strong fine-tuning (KL={kl_strong:.3f}): large drift — reward hacking risk

  The β coefficient in RLHF controls this trade-off:
    Small β (0.01): more fine-tuning allowed, higher reward potential, more risk
    Large β (0.1) : stays close to SFT baseline, lower reward, safer behavior

  All major aligned LLMs (GPT-4, Claude, Gemini) use this exact mechanism.
""")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    x = np.arange(vocab_size)
    width = 0.25
    axes[0].bar(x - width, sft_probs,      width, label='SFT baseline', alpha=0.8)
    axes[0].bar(x,          rl_probs_mild,  width, label='RL mild',      alpha=0.8)
    axes[0].bar(x + width,  rl_probs_strong,width, label='RL strong',    alpha=0.8)
    axes[0].set_xlabel("Token index")
    axes[0].set_ylabel("Probability")
    axes[0].set_title("RLHF: Distribution Drift from SFT Baseline")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # KL as function of beta
    betas = np.linspace(0.01, 0.5, 100)
    # Simulate: higher beta → model stays closer to SFT → less KL
    kl_vs_beta = kl_strong * np.exp(-betas * 10) + kl_mild * 0.2
    axes[1].plot(betas, kl_vs_beta, 'b-', linewidth=2)
    axes[1].axhline(y=kl_mild,   color='green', linestyle='--', alpha=0.7,
                    label=f'Mild KL={kl_mild:.2f}')
    axes[1].axhline(y=kl_strong, color='red',   linestyle='--', alpha=0.7,
                    label=f'Strong KL={kl_strong:.2f}')
    axes[1].set_xlabel("β (KL penalty coefficient)")
    axes[1].set_ylabel("KL(RL || SFT)")
    axes[1].set_title("KL Penalty Controls Distribution Drift")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("kl_divergence.png", dpi=100)
    plt.close()
    print("Saved: kl_divergence.png")


# =============================================================================
# RUN ALL SECTIONS
# =============================================================================

if __name__ == '__main__':
    section_entropy_basics()
    section_entropy_curve()
    section_kl_basics()
    section_crossentropy_decomposition()
    section_kl_rlhf()

    print("\n" + "=" * 55)
    print("Key takeaways:")
    print("  1. Entropy = average surprise = uncertainty in distribution")
    print("  2. KL(p||q) = extra bits wasted by using q instead of p")
    print("  3. Cross-entropy = entropy + KL = total coding cost")
    print("  4. Minimizing cross-entropy loss = minimizing KL from true distribution")
    print("  5. RLHF uses explicit KL penalty to prevent reward hacking")
    print("=" * 55)
