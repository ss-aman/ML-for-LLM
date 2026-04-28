# 06 — How Probability Powers LLMs

## The full picture

Every concept from this module appears in the LLM training loop. Here it is all at once, annotated:

```
Training data: "The capital of France is Paris."
Tokenized:     [464, 3139, 286, 4881, 318, 6342, 13]
               "The" "capital" "of" "France" "is" "Paris" "."

For each position t, the model learns:
  P(token_t | token_0, token_1, ..., token_{t-1})
```

---

## Step-by-step: one training step

```python
import numpy as np

# Vocabulary size
vocab_size = 50257   # GPT-2 vocabulary

# Input tokens (a sequence)
tokens = [464, 3139, 286, 4881, 318, 6342]   # "The capital of France is Paris"
labels = [3139, 286, 4881, 318, 6342, 13]    # shifted right: each token predicts the next

# ── FORWARD PASS ─────────────────────────────────────────────────────────────

# 1. Embedding lookup (Module 01 linear algebra)
#    Each token index → a dense vector (d_model dimensions)
#    embedding_table: shape (vocab_size, d_model)
#    output: shape (seq_len, d_model)
x = embedding_table[tokens]                   # lookup rows

# 2. Transformer layers (attention + FFN, Module 01 and 02)
#    Many matrix multiplications and nonlinearities
#    Each layer refines the representation
for layer in transformer_layers:
    x = layer(x)                              # shape stays (seq_len, d_model)

# 3. Output projection: d_model → vocab_size (Module 01 linear algebra)
#    Produces logits: one raw score per token in vocabulary
logits = x @ W_output.T + b_output           # shape: (seq_len, vocab_size)

# ── PROBABILITY (this module) ─────────────────────────────────────────────────

# 4. Softmax: logits → probabilities
#    Each row sums to 1; positive values; higher logit = higher probability
probs = softmax(logits)                       # shape: (seq_len, vocab_size)

# 5. Cross-entropy loss: measure how wrong the probabilities are
#    For each position, loss = -log(prob[correct_token])
loss = cross_entropy(logits, labels)          # scalar

# ── BACKWARD PASS (Module 02 gradients) ──────────────────────────────────────

# 6. Compute gradients
loss.backward()                               # chain rule through all layers

# 7. Update parameters (Adam optimizer, Module 02)
optimizer.step()
```

---

## The probability distribution at each step

At position t (predicting the token after "The capital of France is"):

```
Logits (50257 numbers): [ 0.3, -1.2, ..., 4.7, ..., -0.5, ... ]
                                          ↑
                                   index for "Paris"

Softmax → probs:         [0.001, 0.0003, ..., 0.73, ..., 0.002, ...]
                                               ↑
                                        73% for "Paris"

Cross-entropy loss = -log(0.73) = 0.315   ← model is pretty good here

After training on millions of examples, this 73% rises toward 99%.
```

---

## What the loss number means

When you see training output like:

```
Epoch 1: loss = 4.23
Epoch 2: loss = 3.87
Epoch 5: loss = 2.94
Epoch 10: loss = 2.11
```

Each number means:

```
loss = 4.23 → avg probability of correct token = exp(-4.23) = 1.5%
loss = 3.87 → avg probability of correct token = exp(-3.87) = 2.1%
loss = 2.94 → avg probability of correct token = exp(-2.94) = 5.3%
loss = 2.11 → avg probability of correct token = exp(-2.11) = 12.1%
```

And in perplexity:

```
loss = 4.23 → PPL = exp(4.23) = 68.7  (as confused as uniform over 68 tokens)
loss = 2.11 → PPL = exp(2.11) = 8.2   (as confused as uniform over 8 tokens)
```

Lower loss = model assigns higher probability to the correct next token on average = better language model.

---

## The complete training objective

LLM pretraining maximizes the likelihood of the training data:

```
maximize:   Π_{t} P(token_t | token_0...token_{t-1})    # product over all positions

equivalent: maximize Σ_{t} log P(token_t | context_t)   # sum in log space

equivalent: minimize -Σ_{t} log P(token_t | context_t)  # cross-entropy loss
```

All three formulations are the same objective. We minimize cross-entropy because:
1. Log space avoids underflow
2. Negative sign turns maximization into minimization (gradient descent)
3. Cross-entropy has clean gradients (from Module 02: `∂L/∂logit = prob - label`)

---

## Entropy and the training floor

For GPT-2 on WebText:
- Achieved: cross-entropy ≈ 2.92 nats/token → PPL ≈ 18.3
- True entropy of English text: ≈ 1.3 nats/character ≈ 8–12 nats/token (depending on vocabulary)

The model is still far above the theoretical floor — room to improve. Modern GPT-4-class models:
- PPL ≈ 3–8 on common benchmarks
- Approaching but not reaching the true entropy of language

---

## KL divergence in the full training picture

```
Cross-entropy loss = H(p_data) + KL(p_data || p_model)
                     ↑               ↑
               constant term    what we actually minimize
               (can't change)   (model getting closer to data)

Training = reducing KL between model and reality.
```

Every gradient step pushes `p_model` closer to `p_data` in KL divergence sense.

---

## Text generation: inference pipeline with probabilities

After training, generating text:

```python
def generate(model, prompt_tokens, max_new_tokens=100,
             temperature=1.0, top_p=0.9):
    tokens = list(prompt_tokens)

    for _ in range(max_new_tokens):
        # Forward pass: get logits for the last position
        logits = model.forward(tokens)[-1]   # shape: (vocab_size,)

        # Temperature scaling
        logits = logits / temperature

        # Softmax → probabilities
        probs = softmax(logits)

        # Nucleus sampling (top-p)
        next_token = nucleus_sample(probs, p=top_p)

        tokens.append(next_token)

        if next_token == EOS_TOKEN:
            break

    return tokens
```

Every generated token goes through the full probability pipeline.

---

## Temperature's effect on creativity

```
Low temperature (T=0.3):
  The capital of France is Paris, and it is known for the Eiffel Tower.
  → Predictable, factual, safe

Standard temperature (T=1.0):
  The capital of France is Paris, a city celebrated for its art, cuisine...
  → Natural, varied

High temperature (T=1.5):
  The capital of France is Paris, though some argue the real cultural heart...
  → Creative, sometimes surprising, occasionally wrong
```

Temperature controls the entropy of the output distribution at each step:
- Low T → low entropy → conservative
- High T → high entropy → creative/risky

---

## RLHF: probability connects training to alignment

After pretraining, RLHF fine-tuning uses all three probability concepts together:

```python
# Reward: how good is this response? (learned from human feedback)
reward = reward_model(prompt, response)

# KL penalty: don't drift too far from pretrained model
kl = kl_divergence(policy_logprobs, reference_logprobs)

# RLHF objective: maximize reward, minimize KL drift
loss = -reward + beta * kl

# beta controls the trade-off:
#   beta=0   → pure reward maximization (dangerous: reward hacking)
#   beta=∞   → no fine-tuning (stays at SFT baseline)
#   beta=0.02 → typical value in InstructGPT / GPT-4 training
```

The KL term is the mathematical guardrail that keeps alignment training from breaking the base model.

---

## Annotated transformer forward pass with all shapes

```python
# GPT-2 small: d_model=768, n_heads=12, vocab_size=50257

# INPUT
tokens: (1024,)                    # sequence of token indices

# EMBEDDING
x: (1024, 768)                     # each token → 768-dim vector

# TRANSFORMER BLOCKS (12 of them)
# Each block: attention + FFN
#   attention: Q,K,V projections (768→768), scores (1024×1024), output (1024×768)
#   FFN: 768→3072→768
x: (1024, 768)                     # shape unchanged through all blocks

# OUTPUT PROJECTION
logits: (1024, 50257)              # (seq_len, vocab_size)
# Each of 1024 positions has a score for each of 50257 vocab tokens

# SOFTMAX (during inference)
probs: (1024, 50257)               # each row sums to 1

# CROSS-ENTROPY (during training)
loss: scalar                       # -mean(log(probs[t, labels[t]]))
```

---

## The module 03 → LLM connection table

| Concept | Where it appears in LLMs |
|---------|--------------------------|
| Discrete distribution | Probability over vocab at each step |
| Gaussian | Weight initialization; noise in training |
| Conditional probability | `P(token_t \| context)` — the core task |
| Entropy | Lower bound on achievable loss; perplexity |
| Cross-entropy | Training loss for all pretraining |
| KL divergence | RLHF penalty; comparing model versions |
| Softmax | Logits → probabilities at output layer |
| Temperature | Controls creativity/randomness at inference |
| Top-p sampling | Nucleus sampling; default inference strategy |
| Perplexity | Standard benchmark metric = exp(cross-entropy) |

---

## What you now understand

After Modules 01, 02, and 03, you can read this and understand every line:

```python
# LLM pretraining step — fully annotated
logits = transformer(tokens)               # Module 01: matrix ops, attention
loss   = F.cross_entropy(logits, labels)  # Module 03: -log P(correct token)
loss.backward()                            # Module 02: chain rule, backprop
optimizer.step()                           # Module 02: Adam update, w -= lr*grad
```

Four lines. Three modules. That's the entire LLM training loop.

---

## What's next

The remaining modules build on this foundation:

- **Module 04** (Neural Networks) — the architecture that produces the logits
- **Module 05** (Optimization) — advanced optimizers, regularization
- **Module 06** (Transformers) — attention in full detail, positional encoding
- **Module 07** (LLM Architecture) — GPT, BERT, T5 architectures
- **Module 08+** (Training, Evaluation, Deployment) — scaling, fine-tuning, inference
