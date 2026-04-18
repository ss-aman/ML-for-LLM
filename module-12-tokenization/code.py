"""
Module 12: Tokenization — Code Implementation

Implements character-level tokenization and Byte Pair Encoding (BPE) from scratch.

Backend analogy: we're building the input-validation and normalization middleware
that sits at the boundary between raw user text and the model's integer-ID world.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import re


# ---------------------------------------------------------------------------
# 1. Character-Level Tokenizer
# ---------------------------------------------------------------------------

class CharTokenizer:
    """
    Simplest possible tokenizer: one character = one token.

    Backend analogy: treating each character as an enum member.
    The vocabulary is all unique characters in the training corpus.
    """

    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        # Reserve ID 0 for unknown characters
        self._add_special("<UNK>", 0)

    def _add_special(self, token, idx):
        self.char_to_id[token] = idx
        self.id_to_char[idx] = token

    def fit(self, corpus: str):
        """Build vocabulary from a corpus string."""
        chars = sorted(set(corpus))
        for i, c in enumerate(chars, start=len(self.char_to_id)):
            self.char_to_id[c] = i
            self.id_to_char[i] = c
        print(f"CharTokenizer: vocab_size={self.vocab_size}")

    @property
    def vocab_size(self):
        return len(self.char_to_id)

    def encode(self, text: str) -> list:
        """Convert a string to a list of integer IDs."""
        return [self.char_to_id.get(c, 0) for c in text]  # 0 = <UNK>

    def decode(self, ids: list) -> str:
        """Convert a list of integer IDs back to a string."""
        return "".join(self.id_to_char.get(i, "?") for i in ids)


def demo_char_tokenizer():
    """Show character-level tokenization."""
    print("--- Character-Level Tokenizer ---")
    corpus = "the quick brown fox jumps over the lazy dog"

    tokenizer = CharTokenizer()
    tokenizer.fit(corpus)

    test_texts = ["hello", "the cat sat", "xyz123"]
    for text in test_texts:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        print(f"  '{text}' → {ids} → '{decoded}'")

    # Show token count comparison
    long_text = "the quick brown fox"
    ids = tokenizer.encode(long_text)
    print(f"\n  '{long_text}'")
    print(f"  Characters: {len(long_text)}, Tokens: {len(ids)}")
    print(f"  (char tokenizer: always 1 token per character)")
    print()
    return tokenizer


# ---------------------------------------------------------------------------
# 2. Byte Pair Encoding (BPE) from Scratch
# ---------------------------------------------------------------------------

class BPETokenizer:
    """
    Byte Pair Encoding tokenizer.

    Algorithm:
      1. Start with character-level tokenization (each char = 1 token)
      2. Count all adjacent pairs of tokens across the corpus
      3. Merge the most frequent pair into a new single token
      4. Repeat steps 2-3 until vocab_size is reached

    Backend analogy: building a compression dictionary.
    You start with atomic units (bytes/chars) and progressively merge
    frequent patterns into shorthand symbols — exactly how gzip works.
    """

    def __init__(self, vocab_size: int = 100):
        self.target_vocab_size = vocab_size
        self.merges = []           # list of (pair_tuple, new_token_str) in order learned
        self.merge_lookup = {}     # pair → new_token (for fast encode)
        self.vocab = {}            # token_str → int_id
        self.vocab_inv = {}        # int_id → token_str
        self._fitted = False

    # ------------------------------------------------------------------
    # Training (Fitting)
    # ------------------------------------------------------------------

    def _text_to_words(self, corpus: str) -> list:
        """
        Split corpus into words, represent each word as a list of characters
        with a space marker Ġ prepended to non-first tokens to track word boundaries.

        This is how GPT-2's tokenizer works: leading space is baked into the token.
        """
        # Simplified: split on whitespace, wrap each word as tuple of chars
        words = corpus.lower().split()
        # Represent each word as a list of characters (space-separated for BPE)
        return [list(word) for word in words]

    def _get_pair_counts(self, words: list) -> Counter:
        """Count all adjacent pairs across all words."""
        counts = Counter()
        for word in words:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                counts[pair] += 1
        return counts

    def _merge_pair(self, words: list, pair: tuple) -> list:
        """Replace all occurrences of `pair` in every word with a merged token."""
        a, b = pair
        merged = a + b
        new_words = []
        for word in words:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_words.append(new_word)
        return new_words

    def fit(self, corpus: str):
        """
        Learn BPE merges from corpus until vocab_size is reached.

        Backend analogy: this is the "build compression dictionary" phase.
        You run this once offline; the result is a table of merge rules
        you apply at inference time to encode new text.
        """
        # Step 0: start with character vocabulary
        words = self._text_to_words(corpus)

        # Build initial character vocabulary (all unique chars in corpus)
        all_chars = sorted(set(c for word in words for c in word))
        self.vocab = {c: i for i, c in enumerate(all_chars)}
        self.vocab_inv = {i: c for c, i in self.vocab.items()}
        current_vocab_size = len(self.vocab)

        print(f"BPE: initial char vocab = {current_vocab_size}")
        print(f"BPE: target vocab size  = {self.target_vocab_size}")

        num_merges = self.target_vocab_size - current_vocab_size
        merge_history = []  # for visualization

        # Iteratively merge most frequent pair
        for step in range(num_merges):
            pair_counts = self._get_pair_counts(words)
            if not pair_counts:
                break

            # Find the most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            best_count = pair_counts[best_pair]

            if best_count < 2:
                # No pair appears more than once — stop early
                break

            # Merge it
            new_token = best_pair[0] + best_pair[1]
            words = self._merge_pair(words, best_pair)

            # Update vocabulary
            new_id = len(self.vocab)
            self.vocab[new_token] = new_id
            self.vocab_inv[new_id] = new_token

            # Record merge
            self.merges.append((best_pair, new_token))
            self.merge_lookup[best_pair] = new_token
            merge_history.append((step + 1, best_pair, new_token, best_count))

            if (step + 1) % 10 == 0 or step < 5:
                print(f"  Merge {step + 1:3d}: '{best_pair[0]}' + '{best_pair[1]}'"
                      f" → '{new_token}'  (count={best_count})")

        self._fitted = True
        print(f"BPE: learned {len(self.merges)} merges, "
              f"final vocab size = {len(self.vocab)}")
        return merge_history

    # ------------------------------------------------------------------
    # Encoding / Decoding
    # ------------------------------------------------------------------

    def _tokenize_word(self, word: str) -> list:
        """
        Apply learned merge rules to a single word (list of chars → list of subwords).
        """
        tokens = list(word.lower())

        # Apply merges in the ORDER they were learned (important!)
        # This is the key insight: we replay the merge sequence
        for (a, b), merged in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(self, text: str) -> list:
        """
        Convert text to a list of token IDs.

        Backend analogy: the normalization middleware — maps free-text to
        canonical integer IDs that the model can process.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first")

        words = text.lower().split()
        ids = []
        for word in words:
            subwords = self._tokenize_word(word)
            for sw in subwords:
                ids.append(self.vocab.get(sw, 0))   # 0 for unknown
        return ids

    def decode(self, ids: list) -> str:
        """Convert token IDs back to a string."""
        tokens = [self.vocab_inv.get(i, "?") for i in ids]
        return " ".join(tokens)

    def tokenize(self, text: str) -> list:
        """Return the token strings (not IDs) — useful for inspection."""
        words = text.lower().split()
        result = []
        for word in words:
            result.extend(self._tokenize_word(word))
        return result

    @property
    def vocab_size(self):
        return len(self.vocab)


# ---------------------------------------------------------------------------
# 3. Demo: BPE on a small corpus
# ---------------------------------------------------------------------------

def demo_bpe():
    """
    Train BPE on a small corpus and show how it tokenizes various inputs.
    """
    print("\n--- BPE Tokenizer Demo ---")

    corpus = (
        "the quick brown fox jumps over the lazy dog "
        "the dog sat on the mat the cat sat on the mat "
        "the quick cat jumped over the brown dog "
        "tokenization is the process of converting text to tokens "
        "the tokenizer splits words into subword units "
    ) * 3   # repeat to make pairs more frequent

    bpe = BPETokenizer(vocab_size=80)
    merge_history = bpe.fit(corpus)

    print(f"\nVocabulary (sample of first 30 entries):")
    for token, idx in list(bpe.vocab.items())[:30]:
        print(f"  {idx:4d}: '{token}'")

    print(f"\n--- Tokenizing test phrases ---")
    test_phrases = [
        "the quick brown fox",
        "tokenization",
        "the cat sat",
        "jumping over",
        "unknown xyz word",
    ]
    for phrase in test_phrases:
        tokens = bpe.tokenize(phrase)
        ids = bpe.encode(phrase)
        print(f"  '{phrase}'")
        print(f"    tokens: {tokens}")
        print(f"    ids:    {ids}")
        print(f"    count:  {len(tokens)} tokens for {len(phrase.replace(' ',''))} chars")

    # Token count comparison
    print("\n--- Token count: char-level vs BPE ---")
    char_tok = CharTokenizer()
    char_tok.fit(corpus)
    for phrase in test_phrases[:3]:
        char_count = len(char_tok.encode(phrase))
        bpe_count  = len(bpe.encode(phrase))
        print(f"  '{phrase}': char={char_count} tokens, BPE={bpe_count} tokens")

    return bpe, merge_history


# ---------------------------------------------------------------------------
# 4. Visualization: BPE vocabulary growth
# ---------------------------------------------------------------------------

def visualize_bpe_growth(merge_history):
    """Plot vocabulary size vs merge step."""
    if not merge_history:
        return

    steps = [h[0] for h in merge_history]
    # Initial char vocab size (before merges) + merges done
    initial_size = merge_history[0][0]   # step 1 means we started at some base

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: vocab growth
    vocab_sizes = [h[0] for h in merge_history]
    axes[0].plot(vocab_sizes, linewidth=2)
    axes[0].set_xlabel("Merge step")
    axes[0].set_ylabel("Vocabulary size")
    axes[0].set_title("BPE: Vocabulary Grows with Each Merge")
    axes[0].grid(True, alpha=0.3)

    # Right: merge frequencies
    frequencies = [h[3] for h in merge_history]
    axes[1].plot(frequencies, linewidth=2, color="orange")
    axes[1].set_xlabel("Merge step")
    axes[1].set_ylabel("Pair frequency")
    axes[1].set_title("BPE: Merge Pair Frequency Decreases Over Steps")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig("/home/user/ML-for-LLM/module-12-tokenization/bpe_growth.png",
                dpi=150, bbox_inches="tight")
    print("\nSaved: bpe_growth.png")


def visualize_tokenization_comparison():
    """Bar chart: word count vs token count for various inputs."""
    examples = [
        ("English prose", "The quick brown fox jumps over the lazy dog"),
        ("Repeated words", "the the the the the cat the cat"),
        ("Long word",      "antidisestablishmentarianism"),
        ("Number",         "12345678"),
        ("Code snippet",   "for i in range(10): print(i)"),
    ]

    corpus = " ".join(e[1] for e in examples) * 5
    bpe = BPETokenizer(vocab_size=100)
    bpe.fit(corpus)

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [e[0] for e in examples]
    word_counts  = [len(e[1].split()) for e in examples]
    token_counts = [len(bpe.tokenize(e[1])) for e in examples]

    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, word_counts,  w, label="Word count",  color="#4C9BE8")
    ax.bar(x + w/2, token_counts, w, label="Token count (BPE)", color="#E87B4C")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Words vs BPE Tokens for Different Input Types")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig("/home/user/ML-for-LLM/module-12-tokenization/token_comparison.png",
                dpi=150, bbox_inches="tight")
    print("Saved: token_comparison.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Module 12: Tokenization")
    print("=" * 60)

    # 1. Character-level tokenizer
    char_tok = demo_char_tokenizer()

    # 2. BPE from scratch
    bpe, merge_history = demo_bpe()

    # 3. Visualizations
    visualize_bpe_growth(merge_history)
    visualize_tokenization_comparison()

    print("\n--- Summary ---")
    print("CharTokenizer: 1 char = 1 token. Simple, no OOV, but long sequences.")
    print("BPE:           learns merge rules from corpus.")
    print("               Common sequences → single tokens (compression).")
    print("               Rare words → split into subword pieces (no OOV).")
    print()
    print("Key insight: tokenization is the text→integer boundary.")
    print("  - 1 token ≠ 1 word (usually ~0.75 words per token for English)")
    print("  - Special tokens ([BOS], [EOS], [PAD]) are protocol sentinels")
    print("  - Tokenizers are model-specific — never mix token IDs across models")
