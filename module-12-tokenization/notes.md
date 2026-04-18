# Module 12: Tokenization

## Overview

Before any text reaches a transformer, it must be converted to integers.
Transformers don't work with strings — they work with sequences of token IDs,
exactly like how a database doesn't store "New York" in every row — it stores
a foreign key ID that maps to the canonical string.

Tokenization is that conversion layer.

---

## Why Tokenization?

> **Backend analogy:** Tokenization is like normalizing a free-text field into a
> foreign key before storing it. Instead of passing arbitrary strings through
> your API, you map them to integer IDs from a fixed vocabulary. The model works
> entirely in integer-ID space; the tokenizer handles the conversion at the boundary.

The model needs:
1. A **finite vocabulary** so the embedding table has a fixed size
2. **Integer IDs** so it can index into that table
3. **Consistent encoding** so the same text always becomes the same IDs

---

## Approach 1: Character-Level Tokenization

The simplest possible approach: each character is one token.

```python
vocab = {c: i for i, c in enumerate(sorted(set(corpus)))}
encode("hello") → [7, 4, 11, 11, 14]   # 'h'→7, 'e'→4, etc.
```

**Pros:**
- Tiny vocabulary (≈100 chars for ASCII)
- No out-of-vocabulary (OOV) problem — any text can be encoded
- Perfect reconstruction on decode

**Cons:**
- Very long sequences — "hello world" = 11 tokens
- Characters alone carry little meaning, so the model must learn more context
- Slow: longer sequences = more attention operations = quadratic cost

---

## Approach 2: Word-Level Tokenization

Split on whitespace/punctuation, one token per word.

```python
vocab = {"the": 0, "cat": 1, "sat": 2, ...}
encode("the cat sat") → [0, 1, 2]
```

**Pros:**
- Short sequences
- Each token carries semantic meaning

**Cons:**
- Vocabulary explodes: English has 170,000+ words
- **OOV problem**: "tokenization" might not be in vocab
- Can't handle inflections: "run", "runs", "running", "ran" are 4 separate tokens
- Different for every language

---

## Approach 3: Byte Pair Encoding (BPE) — The Dominant Approach

BPE is the tokenization algorithm used by GPT-2, GPT-3, GPT-4, LLaMA, and most
modern LLMs. It's a **data-compression algorithm** repurposed for tokenization.

> **Backend analogy:** BPE is like building a **compression dictionary** at encoding
> time. You start with individual characters, then repeatedly find the most frequent
> adjacent pair and merge it into a new symbol. The result is a vocabulary that
> efficiently encodes common patterns without fixing the vocab upfront.

### The BPE Algorithm

```
STEP 0: Start with character-level tokenization
  Corpus: "the cat sat on the mat"
  Initial tokens: t h e _ c a t _ s a t _ o n _ t h e _ m a t

STEP 1: Count all adjacent pairs
  (t,h): 3   (h,e): 2   (e,_): 2   ...

STEP 2: Merge the most frequent pair → new token
  Merge (t,h) → "th"
  Corpus: th e _ c a t _ s a t _ o n _ th e _ m a t

STEP 3: Repeat until vocab_size reached
  Merge (th,e) → "the"   (still the most frequent)
  Merge (a,t) → "at"
  Merge (s,at) → "sat"
  ...
```

Each merge step adds one entry to the vocabulary. After 50,000+ merges, common
English words are single tokens, rare words are split into meaningful subwords.

### Example: GPT-4 Tokenization

```
"tokenization" → ["token", "ization"]       (2 tokens)
"unhelpfulness" → ["un", "help", "ful", "ness"]  (4 tokens)
"the" → ["the"]                              (1 token — very common)
"antidisestablishmentarianism" → [...many subwords...]
```

### Why BPE Works

- Common words → single tokens (efficient)
- Rare words → broken into meaningful subword pieces (handles OOV)
- Any byte sequence can be encoded (no true OOV)
- Vocabulary size is a tunable hyperparameter

---

## Vocabulary Size

| Model | Vocab Size |
|-------|-----------|
| GPT-2 | 50,257 |
| GPT-3 | 50,257 |
| GPT-4 / GPT-4o | ~100,277 |
| LLaMA 3 | 128,256 |
| BERT | 30,522 |

Larger vocabulary = shorter sequences but larger embedding table.

> **Backend analogy:** vocab size is like the cardinality of an enum — bigger enums
> need more storage per entry but can represent more distinct values with one lookup.

---

## Special Tokens

Every tokenizer reserves some IDs for structural signals:

| Token | Typical ID | Meaning |
|-------|-----------|---------|
| `[BOS]` / `<s>` | 1 | Begin of sequence |
| `[EOS]` / `</s>` | 2 | End of sequence |
| `[PAD]` | 0 | Padding (used to batch sequences of different lengths) |
| `[UNK]` | 3 | Unknown token (rare in BPE — almost everything can be encoded) |
| `<\|im_start\|>` | custom | Start of a chat message (ChatML format) |
| `<\|im_end\|>` | custom | End of a chat message |

> **Backend analogy:** Special tokens are exactly like **HTTP status codes** or
> **sentinel values** in a protocol — reserved constants that carry structural
> meaning rather than content. Just as `\r\n\r\n` signals the end of HTTP headers,
> `[EOS]` signals the end of generation.

---

## Tokenization Gotchas for Backend Developers

These are real issues that cause bugs in production LLM systems:

### 1. One token ≠ one word
```
"hello"       → 1 token
"Hello"       → 1 token (different token than "hello"!)
"hello world" → 2 tokens
"Hello World" → 2 tokens  (possibly different pair)
```

### 2. Whitespace is part of the token
GPT tokenizers encode leading spaces as part of the token:
```
"cat"  →  token_id=9246
" cat" →  token_id=3797   ← space+cat is a DIFFERENT token
```

This is why you see `Ġcat` in GPT tokenizer vocabulary — Ġ represents a leading space.

### 3. Numbers tokenize unpredictably
```
"2024" → ["2024"]            (1 token — common year)
"2027" → ["202", "7"]        (2 tokens — less common)
"1234567" → ["123", "456", "7"]  (3 tokens — split arbitrarily)
```
This is why LLMs are bad at arithmetic — numbers aren't atomic units.

### 4. Code tokenizes differently than prose
Python reserved words are often single tokens; variable names split.

### 5. Token count drives cost and context limits
- OpenAI API charges per token, not per character
- Context window limit (e.g., 128k tokens) is in **token count**, not words
- Rule of thumb: 1 token ≈ 0.75 words for English prose
- Code, JSON, and non-English languages use more tokens per character

### 6. The same text may tokenize differently with different models
GPT-4 and LLaMA 3 use different tokenizers — never assume IDs are portable.

---

## The Tokenization Pipeline

```
Raw text
    │
    ▼
Pre-tokenization   (split on whitespace, punctuation — model-specific rules)
    │
    ▼
BPE encoding       (apply learned merge rules to get subword tokens)
    │
    ▼
Add special tokens ([BOS] at start, [EOS] at end, [PAD] to reach target length)
    │
    ▼
Integer IDs        → fed into embedding table
```

> **Backend analogy:** This is your **input validation and normalization middleware**.
> Raw user text comes in, gets sanitized (pre-tokenization), mapped to canonical IDs
> (BPE encoding), and wrapped with protocol headers/footers (special tokens) before
> being dispatched to the model service.

---

## Key Takeaways

1. **Tokenization = text → integer IDs.** Not a metaphor — it's the literal boundary
   between string-land and model-land.
2. **BPE** is the dominant algorithm: builds a vocabulary by iteratively merging the
   most frequent adjacent pairs. Efficient, handles OOV, vocabulary size is tunable.
3. **1 token ≠ 1 word.** This matters for billing, context limits, and understanding
   model behavior on numbers/code.
4. **Special tokens** are protocol sentinels — they tell the model where sequences
   start, end, and how messages are structured in chat models.
5. **Vocab size is a tradeoff**: larger = shorter sequences but more memory for
   the embedding table.
6. **Tokenizers are model-specific** — never reuse token IDs across models.
