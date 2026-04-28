# Module 01 — Linear Algebra for ML

## Who This Is For

Backend developers who know Python well but have never studied ML math.
Goal: understand the linear algebra that powers every LLM forward pass.

---

## Reading Order

Work through these files in order. Each file is short and builds on the previous.

### Theory (read first)

| File | What You'll Learn |
|---|---|
| `01_vectors.md` | What vectors are, why LLMs use 768–12k dimensional ones |
| `02_vector_operations.md` | Addition, dot product, cosine similarity — the core ops |
| `03_matrices.md` | Matrices as tables, transformations, embedding lookups |
| `04_matrix_multiplication.md` | The most important computation in deep learning |
| `05_special_matrices.md` | Transpose, inverse, eigenvectors |
| `06_how_linalg_powers_llms.md` | Full transformer forward pass, annotated in linear algebra |

### Code (run after each theory section)

| File | What It Does |
|---|---|
| `code_01_from_scratch.py` | Implement everything in pure Python (no numpy) |
| `code_02_numpy.py` | Same operations with numpy; includes attention score demo |
| `code_03_llm_preview.py` | Complete mini-LLM forward pass: embedding → attention → output |

### Exercises

| File | What It Is |
|---|---|
| `exercises.py` | 6 blank exercises for you to implement |
| `solutions.py` | Reference solutions (look only after you try) |

---

## The One-Paragraph Summary

An LLM converts each token (word piece) into a vector of numbers, then
repeatedly transforms those vectors through matrix multiplications and dot
products. The "attention" mechanism computes dot products between query and
key vectors to decide which tokens should influence which. The "weights" of
the model are matrices whose values were tuned over billions of training steps.
Understanding linear algebra means you can read transformer source code
without treating it as magic.

---

## Setup

```bash
pip install numpy matplotlib
```

No GPU needed for this module.

---

## Key Concepts at a Glance

| Concept | Formula | Role in LLMs |
|---|---|---|
| Vector | `[v₁, v₂, ..., vₙ]` | Token embedding, any intermediate representation |
| Dot product | `Σ aᵢbᵢ` | Attention scoring, neuron output |
| Cosine similarity | `(a·b)/(‖a‖‖b‖)` | Semantic search, embedding comparison |
| Matrix | `(m×n)` array | Weight matrix, embedding table, attention scores |
| Matrix multiply | `(m,k)@(k,n)→(m,n)` | Linear layers, Q/K/V projection, attention output |
| Transpose | `A.T` | `Q@K.T` in attention formula |
| Softmax | `exp(xᵢ)/Σexp(xⱼ)` | Convert scores to probability distribution |
