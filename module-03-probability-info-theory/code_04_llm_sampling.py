"""
Module 03 — Code 04: LLM Sampling Strategies
=============================================
Implement greedy, top-k, nucleus (top-p), and beam search.
Show how temperature + sampling strategy shapes generation.

Run: python code_04_llm_sampling.py
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# CORE UTILITIES
# =============================================================================

def softmax(z, temperature=1.0):
    z = np.array(z, dtype=float) / temperature
    z -= z.max()
    exp_z = np.exp(z)
    return exp_z / exp_z.sum()


# =============================================================================
# SAMPLING STRATEGIES
# =============================================================================

def greedy(probs):
    """Always pick the highest probability token. Deterministic."""
    return int(np.argmax(probs))


def random_sample(probs):
    """Sample from the full distribution."""
    return int(np.random.choice(len(probs), p=probs))


def top_k_sample(probs, k=50):
    """
    Restrict sampling to the top-k most probable tokens.
    Renormalize those k probabilities, then sample.
    """
    k = min(k, len(probs))
    top_indices = np.argsort(probs)[-k:]         # k highest-prob indices
    top_probs   = probs[top_indices]
    top_probs   = top_probs / top_probs.sum()    # renormalize
    chosen_idx  = np.random.choice(top_indices, p=top_probs)
    return int(chosen_idx)


def nucleus_sample(probs, p=0.9):
    """
    Top-p (nucleus) sampling.
    Take the smallest set of tokens whose cumulative probability ≥ p.
    Renormalize and sample from that nucleus.

    Adapts automatically: when model is confident (peaked distribution),
    nucleus is small; when uncertain (flat), nucleus is larger.
    """
    sorted_indices = np.argsort(probs)[::-1]      # descending order
    sorted_probs   = probs[sorted_indices]
    cumulative     = np.cumsum(sorted_probs)

    # Find smallest nucleus whose cumulative prob ≥ p
    cutoff         = int(np.searchsorted(cumulative, p))
    nucleus_idx    = sorted_indices[:cutoff + 1]
    nucleus_probs  = probs[nucleus_idx]
    nucleus_probs  = nucleus_probs / nucleus_probs.sum()

    return int(np.random.choice(nucleus_idx, p=nucleus_probs))


# =============================================================================
# SECTION 1: Sampling Strategy Comparison
# =============================================================================

def section_sampling_comparison():
    print("=" * 55)
    print("SECTION 1: Sampling Strategy Comparison")
    print("=" * 55)

    # A realistic 10-token distribution
    np.random.seed(42)
    vocab = [f"token_{i}" for i in range(10)]
    logits = np.array([3.5, 1.2, 0.8, 0.5, 0.3, 0.1, -0.2, -0.5, -1.0, -2.0])
    probs  = softmax(logits)

    print("\nToken distribution:")
    for tok, p in zip(vocab, probs):
        bar = "█" * int(p * 60)
        print(f"  {tok}: {p:.4f}  {bar}")

    print(f"\nGreedy always picks: '{vocab[greedy(probs)]}' (prob={probs[greedy(probs)]:.4f})")

    # Run each strategy 10 times to show variability
    print("\n10 samples from each strategy:")
    strategies = [
        ("Greedy",        lambda: greedy(probs)),
        ("Random",        lambda: random_sample(probs)),
        ("Top-k (k=3)",   lambda: top_k_sample(probs, k=3)),
        ("Nucleus (p=0.9)", lambda: nucleus_sample(probs, p=0.9)),
    ]

    for name, fn in strategies:
        np.random.seed(0)
        samples = [vocab[fn()] for _ in range(10)]
        print(f"  {name:20s}: {samples}")


# =============================================================================
# SECTION 2: Top-k vs Nucleus — When Each Adapts Better
# =============================================================================

def section_topk_vs_nucleus():
    print("\n" + "=" * 55)
    print("SECTION 2: Top-k vs Nucleus — Adaptive Behavior")
    print("=" * 55)

    print("""
Top-k uses a fixed window regardless of distribution shape.
Nucleus adapts the window to the shape.
""")

    # Case A: Model is very confident (peaked)
    logits_confident = np.array([8.0, 1.0, 0.5, 0.2, 0.1])
    probs_confident  = softmax(logits_confident)

    # Case B: Model is very uncertain (flat)
    logits_uncertain = np.array([1.2, 1.0, 0.9, 0.8, 0.7])
    probs_uncertain  = softmax(logits_uncertain)

    for case_name, probs in [("Confident model", probs_confident),
                              ("Uncertain model", probs_uncertain)]:
        print(f"\n{case_name}: probs = {np.round(probs, 3)}")

        # Top-k=3: always picks from top 3
        top3_indices = np.argsort(probs)[-3:]
        top3_cumprob = probs[top3_indices].sum()

        # Nucleus p=0.9: picks smallest set covering 90%
        sorted_p = np.sort(probs)[::-1]
        cumsum   = np.cumsum(sorted_p)
        nucleus_size = int(np.searchsorted(cumsum, 0.9)) + 1

        print(f"  Top-k (k=3):    samples from {3} tokens, covering {top3_cumprob:.1%} of probability")
        print(f"  Nucleus (p=0.9): samples from {nucleus_size} tokens, covering ≥90% of probability")

    print("""
Confident model (prob=[0.96, 0.02, ...]):
  Top-k (k=3): still samples from 3 tokens even though 2 are near-zero
  Nucleus:     nucleus = 1 token (that one token covers 96% > 90%)

Uncertain model (prob=[0.25, 0.23, 0.22, ...]):
  Top-k (k=3): cuts off at position 3, missing valid tokens 4, 5
  Nucleus:     includes all tokens needed to cover 90% (maybe 5+ tokens)

This is why nucleus is the default for modern LLMs.
""")


# =============================================================================
# SECTION 3: Temperature + Sampling — The Standard Combo
# =============================================================================

def section_temperature_sampling():
    print("=" * 55)
    print("SECTION 3: Temperature + Nucleus — Production Setup")
    print("=" * 55)

    # Realistic logits for "The capital of France is ___"
    vocab  = ["Paris", "London", "Berlin", "Rome", "a", "the", "city", "known"]
    logits = np.array([5.2, 1.8, 1.2, 1.0, 0.5, 0.3, 0.8, 0.6])

    configs = [
        (0.3, 0.9, "Conservative (code generation)"),
        (1.0, 0.9, "Standard (balanced)"),
        (1.5, 0.95,"Creative (brainstorming)"),
    ]

    print(f"\nContext: 'The capital of France is ___'")
    print(f"\n{'Config':35s}  {'Top tokens sampled (20 trials)':>30}")
    print("-" * 70)

    np.random.seed(42)
    for temp, p_nucleus, label in configs:
        probs   = softmax(logits, temperature=temp)
        samples = [vocab[nucleus_sample(probs, p=p_nucleus)] for _ in range(20)]
        counts  = {v: samples.count(v) for v in set(samples)}
        top3    = sorted(counts.items(), key=lambda x: -x[1])[:3]
        summary = ", ".join(f"{tok}×{cnt}" for tok, cnt in top3)
        print(f"  {label:35s}: {summary}")

    # Show the actual distributions at each temperature
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    for ax, (temp, p_nucleus, label) in zip(axes, configs):
        probs = softmax(logits, temperature=temp)
        bars  = ax.bar(range(len(vocab)), probs, color='steelblue', alpha=0.8)
        ax.set_xticks(range(len(vocab)))
        ax.set_xticklabels(vocab, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel("Probability")
        ax.set_title(f"{label}\n(T={temp}, p={p_nucleus})")
        ax.grid(True, alpha=0.3)

        # Shade nucleus (top-p tokens)
        sorted_idx = np.argsort(probs)[::-1]
        cumprobs   = np.cumsum(probs[sorted_idx])
        nucleus_n  = int(np.searchsorted(cumprobs, p_nucleus)) + 1
        nucleus_tokens = sorted_idx[:nucleus_n]
        for i, bar in enumerate(bars):
            if i in nucleus_tokens:
                bar.set_color('orange')
        ax.text(0.95, 0.95, f"nucleus={nucleus_n} tokens",
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle("Temperature + Nucleus Sampling (orange = nucleus)", fontsize=12)
    plt.tight_layout()
    plt.savefig("sampling_strategies.png", dpi=100)
    plt.close()
    print("\nSaved: sampling_strategies.png")


# =============================================================================
# SECTION 4: Beam Search
# =============================================================================

def section_beam_search():
    print("\n" + "=" * 55)
    print("SECTION 4: Beam Search (Structured Generation)")
    print("=" * 55)

    print("""
Beam search keeps the B best sequences at each step (beam width B).
Unlike sampling, it's deterministic and finds high-probability sequences.

Used for: translation, summarization, structured output.
NOT used for: open-ended chat (too repetitive).
""")

    # Tiny vocabulary for demonstration
    vocab  = ["A", "B", "C", "D", "<EOS>"]
    n_vocab = len(vocab)

    # Fake log-probabilities for each step
    # step t, given token i at t-1, log-prob of each token at step t
    np.random.seed(5)
    log_probs_table = [
        np.log(softmax(np.array([2.5, 1.0, 0.5, 0.3, 0.1]))),  # step 0
        np.log(softmax(np.array([0.2, 2.0, 1.5, 0.8, 0.5]))),  # step 1 after A
        np.log(softmax(np.array([0.5, 0.3, 3.0, 0.2, 0.4]))),  # step 2 after A,B
    ]

    def beam_search(beam_width=3, max_steps=3):
        # Each beam: (cumulative_log_prob, sequence)
        beams = [(0.0, [])]

        for step in range(max_steps):
            new_beams = []
            log_probs = log_probs_table[min(step, len(log_probs_table)-1)]

            for cum_logprob, seq in beams:
                for tok_id in range(n_vocab):
                    new_logprob = cum_logprob + log_probs[tok_id]
                    new_beams.append((new_logprob, seq + [vocab[tok_id]]))

            # Keep top beam_width sequences
            new_beams.sort(key=lambda x: -x[0])
            beams = new_beams[:beam_width]

            print(f"\n  After step {step + 1} (top {beam_width} beams):")
            for logprob, seq in beams:
                prob = np.exp(logprob)
                print(f"    {' '.join(seq):20s}  logprob={logprob:.3f}  prob={prob:.4f}")

        print(f"\n  Best sequence: {' '.join(beams[0][1])}")
        return beams[0]

    print("Beam search with width=3, max 3 steps:")
    beam_search(beam_width=3, max_steps=3)

    print("""
Beam width trade-off:
  Width=1:   equivalent to greedy (fastest, often suboptimal)
  Width=5:   standard for translation (good balance)
  Width=50+: expensive, diminishing returns
""")


# =============================================================================
# SECTION 5: Perplexity in Action
# =============================================================================

def section_perplexity():
    print("=" * 55)
    print("SECTION 5: Perplexity — Evaluating LLM Quality")
    print("=" * 55)

    print("""
Perplexity = exp(average cross-entropy loss)
           = geometric mean of 1/P(correct_token) at each position

Lower perplexity = better model.
""")

    # Compare three hypothetical models on the same sequence
    sequence = ["The", "capital", "of", "France", "is", "Paris"]
    n_tokens  = len(sequence)

    model_scenarios = [
        ("Strong LLM",   [0.72, 0.45, 0.68, 0.85, 0.62, 0.91]),  # high probs
        ("Baseline LM",  [0.30, 0.18, 0.22, 0.41, 0.25, 0.55]),  # medium probs
        ("Weak model",   [0.05, 0.04, 0.07, 0.12, 0.08, 0.15]),  # low probs
    ]

    print(f"\n{'Model':15s}  ", end="")
    for tok in sequence:
        print(f"{tok:>10}", end="")
    print(f"  {'Avg loss':>10}  {'PPL':>8}")
    print("-" * 100)

    for name, token_probs in model_scenarios:
        losses  = [-np.log(p) for p in token_probs]
        avg_loss = np.mean(losses)
        ppl      = np.exp(avg_loss)

        print(f"{name:15s}  ", end="")
        for p in token_probs:
            print(f"{p:>10.2f}", end="")
        print(f"  {avg_loss:>10.4f}  {ppl:>8.2f}")

    print("""
Strong LLM  → low PPL (~3-8 on common benchmarks for GPT-4-class models)
Baseline LM → medium PPL (~15-30, like GPT-2 on common text)
Weak model  → high PPL (~50-200, poor language model)
Random guess → PPL ≈ vocab_size (~50000 for GPT-style vocab)

PPL is reported on test sets that the model hasn't seen during training.
""")


# =============================================================================
# RUN ALL SECTIONS
# =============================================================================

if __name__ == '__main__':
    section_sampling_comparison()
    section_topk_vs_nucleus()
    section_temperature_sampling()
    section_beam_search()
    section_perplexity()

    print("\n" + "=" * 55)
    print("Key takeaways:")
    print("  1. Greedy: deterministic, fast, can loop/repeat")
    print("  2. Top-k: fixed window, misses tail, simpler to implement")
    print("  3. Nucleus (top-p): adaptive window, better in practice")
    print("  4. Temperature: controls entropy/creativity of output")
    print("  5. Beam search: best for structured tasks (translation/code)")
    print("  6. Perplexity: standard benchmark = exp(cross-entropy loss)")
    print("=" * 55)
