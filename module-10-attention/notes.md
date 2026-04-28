# Module 10: Attention Mechanism

## The Core Insight: Attention IS a Soft Database Lookup

This is not a metaphor or analogy. Attention literally IS a soft database lookup.

In a normal key-value store (Redis, a Python dict, a database index), you:
1. Have a **query** (what you're looking for)
2. Match it against **keys** (the index)
3. Retrieve the associated **values**

The match is **hard** — either the key matches or it doesn't.

**Attention does the same thing, but softly** — instead of returning one value for an exact match, it returns a **weighted blend of all values**, where the weights come from how well your query matches each key.

```
Hard lookup:  GET "city"  →  "Paris"        (one exact match)
Soft lookup:  GET "city"  →  0.9*"Paris" + 0.07*"London" + 0.03*"Berlin"
```

That's it. That's attention.

---

## Query, Key, Value (Q, K, V)

Every token in a sequence produces three vectors:

- **Q (Query)**: "What information am I looking for?"
- **K (Key)**: "What information do I advertise that I contain?"
- **V (Value)**: "What information will I actually contribute if selected?"

> **Backend analogy:** Think of a microservice ecosystem.
> - Q = the API request payload (what the caller wants)
> - K = the service's OpenAPI spec / endpoint description (what it claims to offer)
> - V = the actual response body (what it delivers when called)
>
> A token asks: "Who has what I need?" It checks all tokens' Keys, gets a relevance score, and pulls a weighted blend of their Values.

### Why separate K and V?

Because what a token *advertises* and what it *delivers* can be different projections of the same information — optimized for matching vs. for downstream use. The model **learns** the best Q, K, V projections during training.

---

## The Attention Formula

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Let's break this down step by step.

### Step 1: Compute Similarity Scores — `Q @ K.T`

The dot product `Q · K` measures how similar two vectors are. If Q and K point in the same direction, the dot product is large (high relevance). If they're orthogonal, it's zero.

For a sequence of N tokens, we compute **all N×N pairs** at once using matrix multiplication:

```
Q shape: (N, d_k)   — N tokens, each with a d_k-dimensional query vector
K shape: (N, d_k)   — N tokens, each with a d_k-dimensional key vector
Q @ K.T shape: (N, N)  — every token's query vs. every token's key
```

Entry `[i, j]` in the result = "how much should token `i` attend to token `j`?"

### Step 2: Scale — `/ sqrt(d_k)`

Why divide by √d_k?

- Dot products grow with dimension: for d_k-dimensional vectors, the expected dot product magnitude is ~√d_k
- Without scaling, scores get large → softmax saturates → gradients vanish → model stops learning
- Dividing by √d_k keeps scores in a reasonable range regardless of model size

> **Backend analogy:** This is normalization, like normalizing scores to a 0–100 scale before ranking. Without it, different-sized systems aren't comparable.

### Step 3: Softmax — convert scores to weights

```python
weights = softmax(scores / sqrt(d_k))
```

Softmax takes a vector of arbitrary real numbers and converts them to a probability distribution (all positive, sum to 1).

- High score → high weight (token pays a lot of attention here)
- Low score → low weight (token mostly ignores this)
- Each **row** sums to 1 — each token distributes its "attention budget" across all positions

> **Backend analogy:** Like normalizing votes: if three services score 10, 3, 1, softmax gives ~0.99, 0.009, 0.001 — the clear winner gets almost all the weight.

### Step 4: Weighted Sum of Values

```python
output = weights @ V
```

The final output for token `i` is the weighted average of all Value vectors, where the weights came from step 3.

```
output[i] = sum over j of (weight[i,j] * V[j])
```

This is a **context-aware representation** — token `i`'s output embedding now contains information blended from all relevant tokens in the sequence.

> **Backend analogy:** Like a consensus query across a database cluster — you ask all nodes, weight their responses by reliability, and return the blended answer.

---

## Causal / Masked Attention

For autoregressive language models (GPT-style), when generating token N, the model must **not peek at tokens N+1, N+2, ...**

This is enforced with a **causal mask**: set all scores where `j > i` to `-infinity` before softmax.

```
Mask for sequence length 4:

Position:  0    1    2    3
Token 0: [ OK  -inf -inf -inf ]  ← can only see itself
Token 1: [ OK   OK  -inf -inf ]  ← can see tokens 0 and 1
Token 2: [ OK   OK   OK  -inf ]  ← can see tokens 0, 1, 2
Token 3: [ OK   OK   OK   OK  ]  ← can see all tokens
```

After softmax, `-inf` becomes `0` weight — those positions are effectively invisible.

> **Backend analogy:** This is exactly like an **append-only log** or an **event stream**. When processing event at position N, you can read all previous events (0..N-1) but future events don't exist yet. The causal mask enforces this constraint at the attention level.

---

## Multi-Head Attention

Instead of running attention once, run it **H times in parallel** with **different learned projections**.

```python
for head in range(H):
    Q_h = Q @ W_Q[head]   # project to smaller dimension: d_model → d_k
    K_h = K @ W_K[head]   # d_k = d_model / H
    V_h = V @ W_V[head]
    head_output[head] = Attention(Q_h, K_h, V_h)

output = concat(head_outputs) @ W_O  # project back to d_model
```

Each head can specialize in **different relationship types**:
- Head 1: syntactic dependencies (subject-verb agreement)
- Head 2: coreference (pronouns → their antecedents)
- Head 3: positional proximity
- Head 4: semantic similarity
- etc.

> **Backend analogy:** Like running **parallel database queries** with different indexes — one query for geographic relationships, another for temporal relationships, another for categorical. Each replica specializes, you concatenate the results, and a final layer combines them.

### Why concatenate, not average?

Averaging loses information. Concatenation preserves all H perspectives, and the output projection W_O learns how to blend them optimally.

### Dimensions with H heads:
- `d_model`: full model dimension (e.g., 512)
- `d_k = d_v = d_model / H`: per-head dimension (e.g., 512/8 = 64)
- Total computation ≈ same as single-head attention (parallel across heads)

---

## Full Attention Formula (Again)

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_H) \cdot W^O$$

$$\text{where } \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

---

## Summary Table

| Component | What it does | Backend equivalent |
|-----------|-------------|-------------------|
| Q (Query) | What token i is looking for | API request payload |
| K (Key) | What token j advertises | Service endpoint spec |
| V (Value) | What token j delivers | Service response body |
| `Q @ K.T` | Compute all pairwise similarities | Full-text search scoring |
| `/ sqrt(d_k)` | Prevent score explosion | Score normalization |
| Softmax | Convert scores → probability weights | Weighted load balancing |
| `weights @ V` | Weighted blend of values | Consensus-weighted response |
| Causal mask | Block future tokens | Append-only log constraint |
| Multi-head | H parallel attention runs | Parallel query with different indexes |

---

## Key Takeaways

1. **Attention = soft database lookup.** Not an analogy — the math is identical, just with soft matching instead of hard matching.
2. **Q, K, V are learned projections** — the model learns what to query for, what to advertise, and what to deliver.
3. **Scaling by √d_k** prevents gradient vanishing as model size grows.
4. **Causal masking** enforces temporal causality — models can't cheat by looking at the future.
5. **Multi-head attention** lets the model attend to multiple relationship types simultaneously.
6. **The output is a context-aware blend** — each token's representation after attention incorporates information from all relevant positions.
