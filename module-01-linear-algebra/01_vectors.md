# 01 — Vectors: The Basic Unit of Everything in ML

## Why Start Here?

Every piece of data in an LLM — every word, every sentence, every internal
representation — is a **vector**. Not metaphorically. Literally. When GPT-4
processes the word "cat", it immediately converts it to a vector of 12,288
numbers. All the computation that follows is math on those vectors.

So understanding vectors is not optional background. It IS the foundation.

---

## What Is a Vector?

A vector is an **ordered list of numbers**.

```python
[0.2, -0.5, 1.3, 0.8]
```

That's it. A Python list is already a vector. The difference is we also give
it geometric meaning: each number is a coordinate in space.

- A 2D vector lives in a plane: `[x, y]`
- A 3D vector lives in 3D space: `[x, y, z]`
- A 768D vector lives in a 768-dimensional space (GPT-2 does this)

The number of elements is the **dimension** of the vector.

---

## The Backend Analogy: A Row in a Feature Table

Imagine you're building a recommendation system. For each user you store:

```
user_features = [age_normalized, purchase_count, avg_order_value, churn_risk]
               = [0.45,          127,            85.3,             0.12     ]
```

That row IS a vector. It "locates" the user in a 4-dimensional feature space.
Two users with similar rows are "close" to each other in that space.

This is exactly how LLMs work — except instead of users, they have tokens
(words/subwords), and instead of 4 features, they use 768 to 12,288 features.

---

## Notation

Vectors are usually written in **bold lowercase**: **v**, **x**, **w**

Or with an arrow: v⃗

Individual elements use subscripts: v₁, v₂, v₃ (sometimes v[0], v[1], v[2] in code)

Shape: when we say a vector has shape `(4,)` or `(4, 1)`, we mean it has 4 elements.

---

## Two Ways to Think About a Vector

### 1. As a point in space

`[3, 2]` is the point located at x=3, y=2. Its "address" in 2D space.

### 2. As an arrow from the origin

`[3, 2]` is an arrow pointing from (0,0) to (3,2). It has a **direction** and a
**magnitude** (length).

Both views are correct and useful. ML mostly uses the "point in space" view for
data representation, and the "arrow" view when computing similarities.

---

## Magnitude (Norm) — How Long Is the Vector?

The **norm** (or magnitude, or length) of a vector is the distance from the
origin to the point it represents.

For a 2D vector `[a, b]`, this is just the Pythagorean theorem:
```
||v|| = sqrt(a² + b²)
```

For an n-dimensional vector:
```
||v|| = sqrt(v₁² + v₂² + ... + vₙ²)
```

In code:
```python
import numpy as np
v = np.array([3, 4])
norm = np.linalg.norm(v)  # → 5.0  (3-4-5 right triangle)
```

### Why norms matter

- **Normalization**: dividing a vector by its norm gives a **unit vector** (length = 1)
- Unit vectors let you compare directions without being fooled by scale
- LLMs normalize certain vectors (e.g., in some attention variants, query/key norms are controlled)

```python
unit_v = v / np.linalg.norm(v)  # direction only, length = 1
```

---

## Why LLMs Use High-Dimensional Vectors

You might wonder: why 768 dimensions? Why not 10?

More dimensions = more capacity to encode subtle distinctions.

Think of it like a richer API response. A 4-field user object can represent
basic demographics. A 768-field object can encode: grammar role, semantic
category, sentiment, part-of-speech, whether it's a name, what century it
was commonly used in, its relationship to hundreds of concepts...

In practice:
| Model | Embedding Dimension |
|---|---|
| Word2Vec (2013) | 300 |
| BERT-base (2018) | 768 |
| GPT-2 (2019) | 768 |
| GPT-3 (2020) | 12,288 |
| GPT-4 (est.) | ~12,288–25,600 |

---

## Key Takeaway

> A vector is just a list of numbers. In ML, it's the way we represent
> ANY object (a word, a sentence, an image patch) as something math can
> operate on. Every LLM operation starts and ends with vectors.

---

## What's Next

In `02_vector_operations.md` we cover the math you do WITH vectors —
especially the **dot product**, which is the most important operation in
all of deep learning.
