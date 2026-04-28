# Module 09: From RNNs to Transformers — The Sequential Problem

## Why This Matters

Before transformers, language models used Recurrent Neural Networks (RNNs)
and LSTMs. Understanding *why those approaches were limited* is the best way
to understand *why transformers are designed the way they are*. This module
is the "why" before the "how" of the transformer architecture.

---

## 1. The Sequential Problem

Natural language is a sequence: "The cat sat on the mat." To model language,
a network must:
1. Process tokens in order
2. Remember context from earlier in the sequence
3. Use that context to predict or understand later tokens

The naive approach: process one token at a time, carry state forward.

> **Backend analogy:** Imagine a database cursor that can only read one row
> at a time, in order. You cannot skip ahead, you cannot look back without
> re-scanning, and you cannot parallelise the scan. Every row depends on
> having processed every previous row. This is the fundamental bottleneck
> of sequential models.

---

## 2. Recurrent Neural Networks (RNNs)

An RNN maintains a **hidden state** that carries memory across time steps:

```
h_t = tanh(W_h @ h_{t-1} + W_x @ x_t + b)
```

Where:
- `x_t` is the input at time step t (e.g., a token embedding)
- `h_{t-1}` is the hidden state from the previous step (memory)
- `h_t` is the new hidden state (updated memory)
- `W_h`, `W_x`, `b` are learned parameters

At each step, the RNN reads the current input and updates its hidden state.
The hidden state is the network's "working memory" — a fixed-size vector
that must compress everything relevant from past tokens.

> **Backend analogy:** An RNN is like a stateful API server where each request
> carries a session ID. The server loads the session state, processes the current
> request, updates the session state, and stores it back. Each request has access
> to the accumulated history of all previous requests in that session. But the
> session object has a fixed memory size — and if the session has thousands of
> requests, important early information may get overwritten.

### What RNNs can do well:
- Short sequences: remembering context from the last few steps works fine.
- Simple patterns: detecting the last word, counting, basic grammar.

### What RNNs struggle with:
- Long sequences: the fixed-size hidden state must compress ALL history.
- Long-range dependencies: "The **keys** that the locksmith dropped on the
  counter of the shop were **rusty**." — connecting "keys" (word 2) to
  "rusty" (word 14) requires remembering across 12 intermediate tokens.

---

## 3. The Vanishing Gradient Problem in RNNs

During backpropagation through time (BPTT), gradients must flow backwards
through every time step. The gradient of the loss w.r.t. h_1 (the very first
hidden state) passes through the weight matrix W_h once per step:

```
∂Loss/∂h_1 = (∂Loss/∂h_T) × W_h^T × W_h^T × ... × W_h^T
                                    ↑
                           (T-1) multiplications
```

If the largest singular value of W_h is < 1, the gradient shrinks
exponentially with sequence length T. By step 50, it may be < 10^(-15).

> **Backend analogy:** Think of passing a message through a queue of 50
> services, each of which scales the message amplitude by 0.9x. By the time
> it reaches service 1, the amplitude is 0.9^50 ≈ 0.005 — barely detectable.
> The first service in the chain learns almost nothing from the error signal.
> This is vanishing gradient: early tokens receive near-zero gradient updates
> and barely contribute to what the network learns.

**Exploding gradients** are the opposite: if the singular value is > 1,
gradients grow exponentially and training diverges. Gradient clipping
(capping gradient norms) is the standard fix for exploding gradients, but
there is no easy fix for vanishing gradients in plain RNNs.

---

## 4. LSTMs — Adding Gates to Control Memory

Long Short-Term Memory networks (Hochreiter & Schmidhuber, 1997) solve the
vanishing gradient problem by introducing **gating mechanisms** and a
dedicated **cell state**.

An LSTM has two memory vectors at each step:
- `h_t` — the hidden state (short-term working memory, what gets passed to the next layer)
- `c_t` — the cell state (long-term memory, protected by gates)

Four components computed at each step:

```python
# All inputs: h_{t-1} and x_t concatenated
combined = concat(h_{t-1}, x_t)

forget_gate = sigmoid(W_f @ combined + b_f)   # 0=forget everything, 1=keep all
input_gate  = sigmoid(W_i @ combined + b_i)   # how much new info to store
cell_gate   = tanh(W_c @ combined + b_c)      # candidate new values (-1 to +1)
output_gate = sigmoid(W_o @ combined + b_o)   # how much cell state to expose

c_t = forget_gate * c_{t-1} + input_gate * cell_gate   # update cell state
h_t = output_gate * tanh(c_t)                          # update hidden state
```

> **Backend analogy:** An LSTM is like a cache with explicit eviction and
> admission policies:
> - **Forget gate** = eviction policy: "how much of the current cache entry
>   should I keep?" (0 = delete, 1 = keep as-is). Like an LRU entry's
>   relevance score.
> - **Input gate** = admission filter: "how much of this new information
>   should I write to cache?" Like a cache write-through policy with
>   selective updates.
> - **Cell gate** = the candidate value to write (the content of the new
>   cache entry).
> - **Output gate** = read filter: "how much of the cached value should I
>   expose as output right now?"
>
> Unlike a plain RNN where memory is completely overwritten each step, an
> LSTM can choose to **hold a piece of information indefinitely** (keep
> forget_gate ≈ 1, input_gate ≈ 0 for that dimension) until it becomes
> relevant.

### Why LSTMs Mitigate Vanishing Gradients

The gradient path through the cell state is:
```
∂c_t/∂c_{t-1} = forget_gate
```

If the forget gate stays ≈ 1 (the network learns to keep the memory), the
gradient flows back through time without being multiplied by a weight matrix
— it's multiplied by a scalar close to 1. This is the **constant error
carousel**: the cell state provides a near-linear gradient path.

---

## 5. The Core Problem with RNNs (and LSTMs)

Even with LSTMs solving the vanishing gradient issue, a fundamental
architectural problem remains: **sequential computation**.

To compute h_50 you must first compute h_1, h_2, ..., h_49. In order. There
is no way to compute h_10 and h_20 in parallel — h_20 depends on h_10.

For a sequence of length N:
- **Forward pass**: O(N) sequential steps, each taking O(d²) time
- Cannot be parallelised across sequence positions
- Training on long sequences (e.g., N=2048 tokens) takes N×longer than short
  sequences (N=10 tokens)

In 2017, GPUs with thousands of cores existed. But RNNs could only use one
core at a time for the sequential dimension. The hardware sat mostly idle.

> **Backend analogy:** Imagine a database query that must process 10,000 rows
> but can only process one row at a time because each row's computation depends
> on the previous row's result. You have 8,000 CPU cores idle. The bottleneck
> is not computation — it's the sequential dependency chain. You need a
> fundamentally different algorithm to exploit parallelism.

### Summary of RNN/LSTM Limitations

| Problem | Details |
|---|---|
| Sequential computation | O(N) unavoidable serial steps per forward/backward pass |
| Fixed-size hidden state | Must compress all prior context into a fixed-dim vector |
| Vanishing gradients (RNN) | Gradient ∝ W^N → exponential decay or explosion |
| Long-range dependencies | Early tokens barely influence gradients at step N |
| GPU utilisation | Terrible — cores sit idle during sequential steps |

---

## 6. Transformers — The Solution

The Transformer (Vaswani et al., 2017 — "Attention is All You Need") solves
the sequential problem by replacing recurrence with **attention**.

**Key insight:** instead of reading the sequence one token at a time and
compressing context into a hidden state, the transformer reads **all tokens
simultaneously** and allows each token to directly attend to every other
token.

```
# Attention lets token i look at token j directly — O(1) "distance"
attention_output[i] = weighted_sum(all tokens j, weighted by relevance to i)
```

- **O(1) path length** between any two tokens: no chain of multiplications to
  degrade the gradient.
- **Fully parallelisable**: compute all attention outputs at once — every token
  attends to every other token simultaneously. 100% GPU utilisation.
- **No fixed-size bottleneck**: context isn't compressed into a single vector;
  the full sequence representation is available at every layer.

> **Backend analogy:** Replacing an RNN with a transformer is like replacing a
> sequential cursor scan with a hash-map lookup. Instead of iterating through
> all previous rows to find the relevant context (O(N) sequential), you jump
> directly to the relevant position in O(1). The entire sequence becomes a
> random-access structure rather than a read-only tape.
>
> Or think of it as replacing a message queue (FIFO, one-at-a-time) with a
> broadcast channel where every subscriber can simultaneously see every message
> ever sent and pick exactly what's relevant to them.

### The Trade-off

Transformers are not strictly better in every way:

| Property | RNN/LSTM | Transformer |
|---|---|---|
| Sequence parallelism (training) | O(N) serial | O(1) parallel |
| Long-range dependencies | Hard | Easy (direct attention) |
| Memory (inference) | O(1) — just hidden state | O(N²) — KV cache grows |
| Inference speed (token by token) | O(1) per token | O(N) per token (must re-attend) |
| Fixed-length context window | No (in theory) | Yes (in practice, bounded by O(N²) cost) |

The O(N²) cost of attention (every token attending to every other token) is
why modern LLMs have context window limits. Active research areas like
linear attention, sparse attention, and state-space models (Mamba) try to
recover the O(N) inference properties of RNNs while keeping the parallelism
advantages of transformers.

---

## 7. The Progression

```
Plain RNN
  → vanishing gradients on long sequences
  → LSTMs add gates to control memory flow
     → mitigates vanishing gradients
     → but still sequential: O(N) steps, poor GPU utilisation
     → fixed-size hidden state still bottlenecks long-range context
  → Transformers replace recurrence with attention
     → fully parallel: all tokens processed simultaneously
     → O(1) path between any two tokens
     → every token can directly access every other token's representation
     → enables training on very long contexts with large GPU clusters
```

---

## 8. How This Connects to LLMs

GPT, BERT, LLaMA, and every other modern LLM is a **transformer**, not an
RNN or LSTM. The architecture choices you'll study in the next modules
(attention, positional encoding, feed-forward layers) are all motivated by
the problems described above:

- **Multi-head attention** gives each token direct access to all other tokens.
- **Positional encoding** compensates for the fact that transformers are
  order-agnostic (unlike RNNs, they see all tokens at once with no built-in
  notion of position).
- **KV cache** restores O(1) per-token inference by caching the attended
  key/value pairs so they don't have to be recomputed at each generation step.

The next module dives into the attention mechanism itself.
