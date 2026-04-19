# Module 03 — Probability & Information Theory

## Why this module matters for LLMs

When an LLM generates a word, it produces a **probability distribution** over every word in the vocabulary. Training teaches it to make those distributions better — and we measure "better" using **cross-entropy**, a concept from information theory.

Without this module, the LLM training loop is a black box. After this module, you'll understand exactly what the loss number means and why minimizing it makes the model smarter.

---

## Reading Order

| File | Topic | Core idea |
|------|-------|-----------|
| `01_probability_basics.md` | Probability fundamentals | Events, distributions, expectation |
| `02_entropy.md` | Information & entropy | How much "surprise" is in a distribution |
| `03_cross_entropy_loss.md` | Cross-entropy loss | The loss function used to train every LLM |
| `04_kl_divergence.md` | KL divergence | How different are two distributions? |
| `05_softmax_and_sampling.md` | Softmax + temperature | How LLMs convert numbers to probabilities |
| `06_how_probability_powers_llms.md` | Full picture | Every concept mapped to transformer code |

---

## Code Files

| File | What it demonstrates |
|------|---------------------|
| `code_01_distributions.py` | Probability distributions from scratch |
| `code_02_entropy_and_kl.py` | Entropy, KL divergence with visualizations |
| `code_03_softmax_and_loss.py` | Softmax, cross-entropy loss, temperature |
| `code_04_llm_sampling.py` | Greedy, top-k, nucleus sampling strategies |

---

## Exercises

`exercises.py` — Implement from scratch, verify with the checker  
`solutions.py` — Reference solutions (try first!)

---

## The one-sentence summary

> **Training an LLM = repeatedly telling it "your probability for the correct next word was X, but it should have been higher" — cross-entropy quantifies exactly how much higher, and gradients fix it.**

---

## Module connections

```
Module 01 (Linear Algebra)
  └─ Embedding lookup → gives vectors
  └─ Q @ K.T attention → gives raw scores (logits)

Module 02 (Calculus/Gradients)
  └─ Gradient descent → the update mechanism

Module 03 (Probability) ← YOU ARE HERE
  └─ Softmax    → converts logits to probabilities
  └─ Cross-entropy → measures how wrong the probabilities are
  └─ Gradients flow back through cross-entropy → training signal

All three together = the complete LLM training loop
```
