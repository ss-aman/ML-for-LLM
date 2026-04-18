# Module 06: Embeddings

> **Goal:** Understand how language models turn words (and subword tokens) into numbers they can do math on. Embeddings are the bridge between discrete symbolic data (words, IDs) and continuous vector spaces where neural networks operate.

---

## 1. Why Embeddings? The Representation Problem

Neural networks work with floating-point numbers. But language is made of discrete tokens: `["the", "cat", "sat"]` — or after tokenization, subword pieces like `["the", "cat", "s", "at"]`.

How do you feed a word into a network?

### Naive Approach: Integer IDs

Assign each word an integer: `cat=42, dog=43, fish=199`. Problem: **the integer values mean nothing**. Why is `cat` closer to `dog` than to `fish`? (42 is not closer to 43 than to 199 in any meaningful sense.) And arithmetic on them is nonsensical: `cat + dog = 85` means nothing.

### One-Hot Encoding

Represent each word as a sparse vector of length `vocab_size`, with a 1 in one position and 0s everywhere else.

```
cat  → [0, 0, ..., 1, 0, 0, ...]   (position 42)
dog  → [0, 0, ..., 0, 1, 0, ...]   (position 43)
fish → [0, 0, ..., 0, 0, ..., 1, ...]  (position 199)
```

Problems:
- Every word is equally dissimilar to every other word (all vectors are orthogonal)
- Vocabulary of 50,000 words = 50,000-dimensional vectors — very high-dimensional and sparse
- No information about meaning or usage is encoded

### Embeddings: Dense Learned Representations

Instead of a sparse 50k-dim vector, give each token a **dense, low-dimensional vector** (e.g., 128 or 512 floats). These are *learned* during training — the model adjusts them to make its task easier.

```
cat  → [0.32, -0.14,  0.87, ...,  0.05]   (128 floats)
dog  → [0.29, -0.11,  0.84, ...,  0.08]   (128 floats — similar to cat!)
fish → [-0.12, 0.63, -0.21, ..., -0.44]   (128 floats — different)
```

> **Backend analogy:** Think of embeddings as a **learned hash function** where semantically similar items hash to nearby memory locations. A regular hash function (`SHA256(cat)`) maps similar inputs to completely unrelated buckets. An embedding function maps "cat" and "dog" to nearby points in a 128-dimensional space because they're used in similar contexts. It's the difference between a hash index (equality lookup only) and a vector index (nearest-neighbour lookup).

---

## 2. The Embedding Table

Concretely, an embedding table is a matrix of shape `[vocab_size × embedding_dim]`.

```
         dim_0   dim_1   dim_2   ...  dim_127
token_0: [0.12,  -0.33,   0.05,  ...,  0.18 ]
token_1: [0.87,   0.21,  -0.44,  ..., -0.09 ]
...
token_N: [0.03,  -0.71,   0.99,  ...,  0.52 ]
```

**Lookup operation:** Given token ID `i`, return row `i` of the matrix. That's it — it's a pure table lookup.

```python
embedding_table = np.random.randn(vocab_size, embedding_dim) * 0.01
token_id = 42
vector = embedding_table[token_id]   # shape: (embedding_dim,)
```

> **Backend analogy:** This is exactly a database table with a primary key lookup: `SELECT embedding FROM token_embeddings WHERE token_id = 42`. The difference is that this "database" gets updated during training — the rows are learned parameters, not static data.

**Initialization:** Typically small random values (e.g., `~N(0, 0.01)`). The network learns the values through backpropagation during training.

---

## 3. Word2Vec: How Embeddings Get Their Meaning

Word2Vec (Mikolov et al., 2013) showed that embeddings trained to **predict context** develop rich semantic structure as a side effect.

### The Skip-Gram Idea

**Hypothesis:** Words that appear in similar contexts have similar meanings.

**Training task:** Given a center word, predict which words appear nearby (the context window).

```
Sentence: "the quick brown fox jumps over the lazy dog"
Center: "fox"   →   Predict: ["quick", "brown", "jumps", "over"]
Center: "cat"   →   Predict: ["the", "quick", "chased", "the", "mouse"]
```

Because `fox` and `cat` appear near similar context words, their embeddings end up similar — not because we told the model "fox and cat are both animals" but because the *distributional statistics* of the language force them together.

> **Backend analogy:** Two API endpoints that appear in the same request traces, used by the same clients, in response to the same upstream calls, will naturally "cluster" in any embedding trained on those traces. You don't label them as similar — the access patterns reveal their similarity.

### The Famous Result: Arithmetic on Meanings

```
king - man + woman ≈ queen
Paris - France + Italy ≈ Rome
```

Why does this work? The embedding space encodes semantic relationships as *directions*. The "gender direction" is approximately the vector from `man` to `woman`. The "capital city direction" is approximately the vector from `France` to `Paris`. Because these directions are consistent across the vocabulary, arithmetic transfers.

---

## 4. Cosine Similarity

To measure how similar two vectors are, we use **cosine similarity** — the cosine of the angle between them.

```
cosine_similarity(a, b) = (a · b) / (|a| * |b|)
```

| Value | Meaning |
|---|---|
| 1.0 | Same direction — identical meaning |
| 0.0 | Orthogonal — unrelated |
| -1.0 | Opposite directions — opposite meaning |

**Why cosine, not Euclidean distance?**

Euclidean distance is sensitive to vector magnitude. Two vectors pointing in the same direction might be far apart in Euclidean space if one is longer. Cosine normalizes for magnitude and only measures the *angle*.

> **Backend analogy:** Cosine similarity is like comparing the *shape* of two time-series (normalized by amplitude) rather than their absolute values. A service that spikes at 9am and 5pm has the same shape as another 9am/5pm-spiking service even if one handles 10x more traffic. Cosine captures the structural similarity; Euclidean distance doesn't.

---

## 5. Dimensionality

LLMs use embedding dimensions in the range of 512–4096:

| Model | Embedding Dim |
|---|---|
| GPT-2 (small) | 768 |
| GPT-3 (175B) | 12,288 |
| LLaMA-7B | 4,096 |
| BERT-base | 768 |

**Why higher dimensions?**
- More dimensions = more capacity to encode fine-grained semantic distinctions
- A 4-dim embedding can only represent 4 "independent axes of meaning"
- 4096 dims can represent thousands of nuanced relationships

**Why not infinite dimensions?**
- Memory: a vocab of 50,000 tokens at 4096 dims = 800M parameters just for the embedding table
- Training: more parameters = more data needed to train them well
- Diminishing returns: after some point, adding dimensions stops helping

> **Backend analogy:** Dimensionality is like the number of metrics you collect per request (latency, CPU, memory, error_code, user_tier, region...). More metrics → better anomaly detection. But at some point you're tracking so many metrics that your observability system becomes expensive and noisy. There's an optimal "embedding dimension" for your monitoring setup too.

---

## 6. Embedding Arithmetic: Math on Meanings

Once embeddings are trained, you can do arithmetic to manipulate meaning:

```python
# Analogy: "king is to man as queen is to woman"
result = embeddings["king"] - embeddings["man"] + embeddings["woman"]
# Find the word whose embedding is closest to result:
nearest = find_nearest_neighbour(result, all_embeddings)
# nearest ≈ "queen"
```

This works because the vector offset from `man` to `king` encodes "royalty added" and the vector offset from `woman` to `queen` encodes the same concept — the "royalty" direction is consistent across gender.

---

## 7. Positional Embeddings (Preview)

In a transformer, the same token at different positions in a sentence should have different representations: "bank" in position 3 vs position 15 of a sentence has different context.

The solution: add a **positional embedding** to each token embedding — a learned or fixed vector that encodes position.

```
final_embedding[i] = token_embedding[token_id[i]] + positional_embedding[i]
```

We'll cover this in depth in the Transformer module. For now: just know that the embedding table is only half the story — position matters too.

---

## Key Formulas Summary

| Concept | Formula |
|---|---|
| Embedding lookup | `v = E[token_id]`  where `E ∈ ℝ^{vocab × dim}` |
| Cosine similarity | `cos(a,b) = (a·b) / (‖a‖ · ‖b‖)` |
| Skip-gram objective | Maximise `P(context_word \| center_word)` |
| Analogy arithmetic | `king - man + woman ≈ queen` |

---

## What's Next

Module 07 covers **neural networks** — how to stack layers of transformations on top of embeddings to build a complete forward pass. Embeddings are the input layer; the neural network is everything that follows.
