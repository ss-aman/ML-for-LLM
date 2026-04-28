# 02 — Vector Operations: The Math You Actually Need

## Overview

There are four vector operations used constantly in ML:

1. Addition / Subtraction
2. Scalar multiplication
3. **Dot product** ← most important by far
4. Norm (already covered in `01_vectors.md`)

And one derived measure:
5. Cosine similarity (built from dot product + norms)

---

## 1. Vector Addition

Add element-by-element. Vectors must be the same dimension.

```
[1, 2, 3]
+ [4, 5, 6]
-----------
= [5, 7, 9]
```

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = a + b   # → [5, 7, 9]
```

**Geometric meaning:** The result vector goes from the origin to the point you
reach by first following vector a, then following vector b.

**In LLMs:** When a transformer adds a residual connection — `output = x + F(x)` —
it's literally vector addition. The original signal `x` is added to the
transformed version `F(x)`. This is called a **skip connection** and it's
critical for training deep networks (you'll see this in Module 11).

---

## 2. Scalar Multiplication

Multiply every element by a single number (the "scalar").

```python
v = np.array([1, 2, 3])
v * 3    # → [3, 6, 9]
v * 0.5  # → [0.5, 1.0, 1.5]
v * -1   # → [-1, -2, -3]  (reverses direction)
```

**Effect:** scales the magnitude without changing direction (unless scalar is
negative, which flips direction).

**In LLMs:** The attention formula divides by `sqrt(d_k)` — that's scalar
multiplication of a vector. It keeps the dot products from growing too large
as dimension increases.

---

## 3. The Dot Product — The Most Important Operation

The dot product takes two vectors and produces a **single number**.

```
a · b = a[0]*b[0] + a[1]*b[1] + ... + a[n]*b[n]
```

Example:
```
a = [1, 2, 3]
b = [4, 5, 6]

a · b = 1*4 + 2*5 + 3*6
      = 4   + 10  + 18
      = 32
```

```python
np.dot(a, b)  # → 32
# or equivalently:
(a * b).sum()
```

### Why the dot product is everywhere

The dot product answers the question: **"How much do these two vectors agree?"**

- High positive dot product → vectors point in similar directions (agree)
- Dot product near zero → vectors are perpendicular (unrelated)
- Negative dot product → vectors point in opposite directions (disagree)

### Backend analogy: weighted scoring

You have a request feature vector `r = [5, 2, 8]` representing
`[cpu_usage, memory_pressure, request_rate]`.

You have a weight vector `w = [0.6, 0.1, 0.3]` representing how much each
feature contributes to your "danger score".

```
danger_score = r · w
             = 5*0.6 + 2*0.1 + 8*0.3
             = 3.0  + 0.2   + 2.4
             = 5.6
```

The dot product "summarizes" a feature vector into a single score using weights.

**Every neuron in a neural network does exactly this.** A neuron computes
`w · x + bias` and that's its output. The entire forward pass of a neural
network is just stacked dot products.

### The dot product in attention (preview)

In a transformer, when deciding how much token A should "attend to" token B:

```
score(A, B) = query_A · key_B
```

High score → A attends heavily to B. Low score → A mostly ignores B.

This is the ENTIRE mechanism behind "attention" in GPT, BERT, etc.
(You'll implement this fully in Module 10.)

---

## 4. The Geometric Meaning of the Dot Product

There's a beautiful geometric identity:

```
a · b = ||a|| × ||b|| × cos(θ)
```

Where θ is the angle between the two vectors.

This means:
- θ = 0° (same direction): cos(0) = 1, maximum dot product
- θ = 90° (perpendicular): cos(90) = 0, dot product = 0
- θ = 180° (opposite): cos(180) = -1, most negative dot product

```
   b
   ↑
   |   /
   |  / a
   | /
   |/___
      θ small → high dot product
      θ = 90° → dot product = 0
```

This is why the dot product measures "similarity of direction".

---

## 5. Cosine Similarity — Direction Without Scale

Problem with raw dot products: if vector a is 10× longer than vector b, the
dot product is 10× larger even if they point the same direction. We want to
compare directions regardless of length.

**Solution:** normalize by both norms.

```
cosine_similarity(a, b) = (a · b) / (||a|| × ||b||)
```

This gives a value between -1 and +1, regardless of how long the vectors are.

- +1.0 → identical direction (very similar)
- 0.0 → perpendicular (unrelated)
- -1.0 → opposite direction (contradictory)

```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

### Where you see this

**Semantic search:** Your query is embedded into a vector. Every document in
your database is also embedded. You compute cosine similarity between the
query vector and all document vectors. Return the top-k most similar documents.

**LLM internals:** Transformer key-query matching is essentially cosine
similarity (the scaling by `sqrt(d_k)` is there to control magnitude, similar
to normalization).

---

## Quick Reference

| Operation | Formula | Result shape | Use in LLMs |
|---|---|---|---|
| Addition | a + b (element-wise) | same as inputs | Residual connections |
| Scalar multiply | c * v | same as v | Attention scaling (÷√d_k) |
| Dot product | Σ aᵢbᵢ | scalar (1 number) | Attention scores, neuron output |
| Norm | √(Σ aᵢ²) | scalar | Normalization, cosine similarity |
| Cosine similarity | (a·b)/(‖a‖‖b‖) | scalar in [-1,1] | Semantic search, similarity |

---

## Key Takeaway

> The dot product is the core operation of deep learning. A single neuron
> computes a dot product. An attention head computes many dot products.
> Training adjusts weight vectors so their dot products with input vectors
> produce the right outputs.

---

## What's Next

In `03_matrices.md`: a matrix is just a organized collection of vectors —
the natural way to represent many dot products at once.
