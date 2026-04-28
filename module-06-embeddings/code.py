"""
Module 06: Embeddings — Code Demonstrations
============================================
Covers:
  - Embedding table from scratch
  - Cosine similarity
  - Toy training loop: learn embeddings that predict word co-occurrence
  - PCA visualisation (before and after training)

Run this file directly: python code.py
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Embedding Table from Scratch
# ---------------------------------------------------------------------------

class EmbeddingTable:
    """
    A simple embedding table: maps integer token IDs to dense vectors.

    Internally: a matrix of shape [vocab_size, embedding_dim].
    Lookup: return the row at index token_id.

    Backend analogy: like a database table with an integer primary key
    where each row is a vector of floats.  The "table" is updated during
    training (the floats are learned, not stored).
    """

    def __init__(self, vocab_size, embedding_dim, rng=None):
        if rng is None:
            rng = np.random.default_rng(42)
        # Initialise with small random values — standard practice
        # Small init prevents exploding gradients early in training
        self.weights = rng.normal(0, 0.1, (vocab_size, embedding_dim))
        self.vocab_size    = vocab_size
        self.embedding_dim = embedding_dim

    def lookup(self, token_id):
        """Return the embedding vector for a single token ID."""
        return self.weights[token_id]

    def lookup_batch(self, token_ids):
        """Return embedding matrix for a list of token IDs. Shape: [n, embedding_dim]."""
        return self.weights[token_ids]

    def update(self, token_id, grad, lr):
        """
        Apply a gradient update to the embedding for token_id.
        Only the rows that were looked up get updated — sparse update.
        This is much more efficient than a dense update across all rows.
        """
        self.weights[token_id] -= lr * grad


def demo_embedding_table():
    """
    Create an embedding table and demonstrate basic lookup operations.
    """
    print("=" * 60)
    print("1. Embedding Table Basics")
    print("=" * 60)

    vocab = ["cat", "dog", "fish", "car", "bus", "truck", "kitten", "puppy"]
    vocab_size    = len(vocab)
    embedding_dim = 4
    word_to_id    = {w: i for i, w in enumerate(vocab)}

    table = EmbeddingTable(vocab_size, embedding_dim, rng=np.random.default_rng(0))

    print(f"  Vocab: {vocab}")
    print(f"  Embedding table shape: {table.weights.shape}  (vocab_size × embedding_dim)")
    print()

    # Lookup individual words
    for word in ["cat", "dog", "car"]:
        vec = table.lookup(word_to_id[word])
        print(f"  {word:<8} (id={word_to_id[word]}): {np.round(vec, 3)}")

    # Batch lookup
    ids = [word_to_id[w] for w in ["cat", "dog", "fish"]]
    batch = table.lookup_batch(ids)
    print(f"\n  Batch lookup for ['cat','dog','fish']:")
    print(f"  Shape: {batch.shape}")
    print(f"  Matrix:\n{np.round(batch, 3)}\n")


# ---------------------------------------------------------------------------
# 2. Cosine Similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors a and b.

    cos(θ) = (a · b) / (‖a‖ · ‖b‖)

    Returns a scalar in [-1, 1]:
        1.0  = same direction (identical meaning)
        0.0  = orthogonal (unrelated)
       -1.0  = opposite directions

    Backend analogy: comparing the *shape* of two time series after
    normalising for amplitude.  Two services with the same traffic pattern
    but different scales get cosine similarity close to 1.
    """
    dot    = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def cosine_similarity_matrix(E):
    """
    Compute pairwise cosine similarity for all rows of matrix E.
    Returns an (n, n) matrix.

    Efficient: normalise all rows first, then use matrix multiplication.
    norms shape: (n,1).  E_norm shape: (n, dim).  Result: (n, n).
    """
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)   # avoid division by zero
    E_norm = E / norms
    return E_norm @ E_norm.T


def demo_cosine_similarity():
    """
    Show cosine similarity on hand-crafted vectors, then on random embeddings.
    """
    print("=" * 60)
    print("2. Cosine Similarity")
    print("=" * 60)

    # Hand-crafted examples
    examples = [
        ("same",      np.array([1.0, 0.5, -0.3]),  np.array([1.0, 0.5, -0.3])),
        ("opposite",  np.array([1.0, 0.5, -0.3]),  np.array([-1.0, -0.5, 0.3])),
        ("orthogonal",np.array([1.0, 0.0, 0.0]),   np.array([0.0, 1.0, 0.0])),
        ("similar",   np.array([1.0, 0.8,  0.1]),  np.array([0.9, 0.9,  0.2])),
    ]

    for label, a, b in examples:
        sim = cosine_similarity(a, b)
        print(f"  {label:<12}: cos_sim = {sim:+.4f}")

    print()

    # Show cosine similarity is magnitude-invariant
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 4.0, 6.0])   # b = 2*a, same direction
    print(f"  Magnitude invariance: a={a}, b=2*a={b}")
    print(f"    cos_sim(a, b) = {cosine_similarity(a, b):.4f}  (should be 1.0)\n")


# ---------------------------------------------------------------------------
# 3. Toy Training Loop: Learn Embeddings from Co-Occurrence
# ---------------------------------------------------------------------------

def build_cooccurrence_matrix(corpus, vocab, word_to_id, window=2):
    """
    Build a co-occurrence count matrix from a corpus.

    For each word in the corpus, count how many times each other word
    appears within `window` positions of it.

    This is the raw signal that word2vec learns from — words that appear
    together frequently should get similar embeddings.

    Backend analogy: building a "service dependency matrix" by counting
    how often service A and service B appear in the same request trace.
    Services that appear together often are likely semantically related.
    """
    vocab_size = len(vocab)
    C = np.zeros((vocab_size, vocab_size))

    for sentence in corpus:
        ids = [word_to_id[w] for w in sentence if w in word_to_id]
        for center_pos, center_id in enumerate(ids):
            # Collect context IDs within the window
            for offset in range(-window, window + 1):
                context_pos = center_pos + offset
                if offset == 0 or context_pos < 0 or context_pos >= len(ids):
                    continue
                context_id = ids[context_pos]
                C[center_id, context_id] += 1.0

    return C


def train_embeddings(vocab, corpus, embedding_dim=4, n_epochs=200, lr=0.05, rng=None):
    """
    Train word embeddings using a simple co-occurrence prediction objective.

    Objective: for each (center, context) pair, make their dot product
    approximate the log co-occurrence count.  This is a simplified version
    of GloVe (Global Vectors for Word Representation).

    Loss: sum over all word pairs (i,j) of  (e_i · e_j - log(C_ij + 1))^2
    where C_ij is the co-occurrence count.

    We only train on pairs where C_ij > 0 (they actually co-occurred).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    word_to_id = {w: i for i, w in enumerate(vocab)}
    vocab_size  = len(vocab)

    # Initialise embedding table
    E = rng.normal(0, 0.1, (vocab_size, embedding_dim))

    # Build co-occurrence matrix
    C = build_cooccurrence_matrix(corpus, vocab, word_to_id, window=2)
    targets = np.log(C + 1.0)   # log-space target (smoother gradients)

    # Only train on non-zero co-occurrences
    pairs = [(i, j) for i in range(vocab_size) for j in range(vocab_size) if C[i, j] > 0]

    losses = []

    for epoch in range(n_epochs):
        total_loss = 0.0
        rng.shuffle(pairs)   # shuffle each epoch — standard practice

        for i, j in pairs:
            # Predicted score: dot product of center and context embeddings
            pred   = np.dot(E[i], E[j])
            target = targets[i, j]
            error  = pred - target

            # MSE gradient
            # dL/dE[i] = 2 * error * E[j]
            # dL/dE[j] = 2 * error * E[i]
            grad_i = 2.0 * error * E[j]
            grad_j = 2.0 * error * E[i]

            # Gradient descent update
            E[i] -= lr * grad_i
            E[j] -= lr * grad_j

            total_loss += error ** 2

        losses.append(total_loss / len(pairs))

    return E, losses


def demo_embeddings_training():
    """
    Train embeddings on a tiny corpus and show they cluster by meaning.

    Corpus: sentences about animals and vehicles.
    After training, animal words should be near each other in embedding space
    and vehicle words near each other — even though we never told the model
    these categories exist.
    """
    print("=" * 60)
    print("3. Training Embeddings from Co-Occurrence")
    print("=" * 60)

    vocab = ["cat", "dog", "kitten", "puppy", "car", "bus", "truck", "van"]
    word_to_id = {w: i for i, w in enumerate(vocab)}

    # Corpus: sentences that reveal semantic relationships through co-occurrence
    corpus = [
        ["cat",    "and",   "dog",    "are",   "pets"],
        ["kitten", "is",    "a",      "baby",  "cat"],
        ["puppy",  "is",    "a",      "baby",  "dog"],
        ["cat",    "chased", "the",   "dog"],
        ["kitten", "played", "with",  "puppy"],
        ["car",    "and",   "bus",    "drive", "on", "road"],
        ["truck",  "and",   "van",    "carry", "goods"],
        ["car",    "bus",   "truck",  "van",   "are", "vehicles"],
        ["bus",    "and",   "van",    "are",   "large"],
        ["dog",    "and",   "cat",    "are",   "animals"],
        ["kitten", "and",   "puppy",  "are",   "young", "animals"],
        ["truck",  "is",    "larger", "than",  "car"],
        ["bus",    "carries", "people", "like", "a", "van"],
        ["cat",    "dog",   "kitten", "puppy", "live", "at", "home"],
        ["car",    "truck", "bus",    "van",   "use",  "fuel"],
    ]

    rng = np.random.default_rng(17)

    print("  Training embeddings (4 dimensions)...")
    E_trained, losses = train_embeddings(
        vocab, corpus, embedding_dim=4, n_epochs=300, lr=0.05, rng=rng
    )
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss:   {losses[-1]:.4f}")

    # Show cosine similarities after training
    print("\n  Cosine similarity matrix (after training):")
    sim_matrix = cosine_similarity_matrix(E_trained)
    header = f"{'':8}" + "".join(f"{w:8}" for w in vocab)
    print(f"  {header}")
    for i, word in enumerate(vocab):
        row = f"  {word:<8}" + "".join(f"{sim_matrix[i,j]:8.3f}" for j in range(len(vocab)))
        print(row)

    # Check: animals should be more similar to animals than to vehicles
    animal_ids  = [word_to_id[w] for w in ["cat", "dog", "kitten", "puppy"]]
    vehicle_ids = [word_to_id[w] for w in ["car", "bus", "truck", "van"]]

    animal_sim  = np.mean([sim_matrix[i, j] for i in animal_ids
                           for j in animal_ids if i != j])
    vehicle_sim = np.mean([sim_matrix[i, j] for i in vehicle_ids
                           for j in vehicle_ids if i != j])
    cross_sim   = np.mean([sim_matrix[i, j] for i in animal_ids
                           for j in vehicle_ids])

    print(f"\n  Within-animal similarity:  {animal_sim:.3f}")
    print(f"  Within-vehicle similarity: {vehicle_sim:.3f}")
    print(f"  Animal vs vehicle:         {cross_sim:.3f}")
    print()

    return E_trained, vocab, losses


# ---------------------------------------------------------------------------
# 4. Visualise Embeddings with PCA
# ---------------------------------------------------------------------------

def pca_2d(E):
    """
    Project embedding matrix E (shape: n × dim) to 2D using PCA.

    PCA finds the two directions of maximum variance.  When we project
    embeddings onto these directions, similar embeddings tend to cluster
    visually.

    Steps:
      1. Mean-center the data
      2. Compute covariance matrix
      3. Get eigenvectors (principal components)
      4. Project data onto top-2 eigenvectors

    Backend analogy: like t-SNE/UMAP for log data — you're projecting
    high-dimensional service metrics to 2D to spot clusters visually.
    PCA is the linear version: the axes are interpretable combinations
    of the original dimensions.
    """
    # Mean-center
    E_centered = E - E.mean(axis=0)

    # Covariance matrix and eigen-decomposition
    cov = np.cov(E_centered.T)          # (dim, dim) covariance
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by descending eigenvalue (most variance first)
    idx = np.argsort(eigenvalues)[::-1]
    top2 = eigenvectors[:, idx[:2]]     # (dim, 2) — top 2 principal components

    # Project to 2D
    return E_centered @ top2            # (n, 2)


def demo_visualise_embeddings():
    """
    Compare embeddings BEFORE and AFTER training using PCA visualisation.
    Before: random — no clustering.
    After:  animals cluster together, vehicles cluster together.
    """
    print("=" * 60)
    print("4. Embedding Visualisation with PCA")
    print("=" * 60)

    vocab = ["cat", "dog", "kitten", "puppy", "car", "bus", "truck", "van"]
    word_to_id = {w: i for i, w in enumerate(vocab)}
    colors = ["red", "red", "red", "red", "blue", "blue", "blue", "blue"]
    labels_by_group = ["animal"] * 4 + ["vehicle"] * 4

    corpus = [
        ["cat",    "and",   "dog",    "are",   "pets"],
        ["kitten", "is",    "a",      "baby",  "cat"],
        ["puppy",  "is",    "a",      "baby",  "dog"],
        ["cat",    "chased", "the",   "dog"],
        ["kitten", "played", "with",  "puppy"],
        ["car",    "and",   "bus",    "drive", "on", "road"],
        ["truck",  "and",   "van",    "carry", "goods"],
        ["car",    "bus",   "truck",  "van",   "are", "vehicles"],
        ["bus",    "and",   "van",    "are",   "large"],
        ["dog",    "and",   "cat",    "are",   "animals"],
        ["kitten", "and",   "puppy",  "are",   "young", "animals"],
        ["truck",  "is",    "larger", "than",  "car"],
        ["bus",    "carries", "people", "like", "a", "van"],
        ["cat",    "dog",   "kitten", "puppy", "live", "at", "home"],
        ["car",    "truck", "bus",    "van",   "use",  "fuel"],
    ]

    rng = np.random.default_rng(17)

    # Before training: random embeddings
    E_before = rng.normal(0, 0.1, (len(vocab), 4))
    # After training: learned embeddings
    E_after, losses = train_embeddings(vocab, corpus, embedding_dim=4,
                                       n_epochs=300, lr=0.05, rng=rng)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot before
    coords_before = pca_2d(E_before)
    ax = axes[0]
    for i, (word, color) in enumerate(zip(vocab, colors)):
        ax.scatter(*coords_before[i], color=color, s=100, zorder=5)
        ax.annotate(word, coords_before[i], fontsize=10, ha="center",
                    xytext=(0, 8), textcoords="offset points")
    ax.set_title("BEFORE training\n(random embeddings — no clustering)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)

    # Plot after
    coords_after = pca_2d(E_after)
    ax = axes[1]
    for i, (word, color) in enumerate(zip(vocab, colors)):
        ax.scatter(*coords_after[i], color=color, s=100, zorder=5)
        ax.annotate(word, coords_after[i], fontsize=10, ha="center",
                    xytext=(0, 8), textcoords="offset points")
    ax.set_title("AFTER training\n(animals cluster in red, vehicles in blue)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)
    # Add legend
    ax.scatter([], [], color="red",  s=100, label="animals")
    ax.scatter([], [], color="blue", s=100, label="vehicles")
    ax.legend()

    # Plot training loss curve
    axes[2].plot(losses)
    axes[2].set_title("Training Loss over Epochs")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Mean Squared Error")
    axes[2].set_yscale("log")
    axes[2].grid(True)

    plt.suptitle("Embeddings: random noise → semantic clusters through training", fontsize=12)
    plt.tight_layout()
    plt.savefig("plot_04_embeddings_pca.png", dpi=100)
    print("  Saved: plot_04_embeddings_pca.png\n")


# ---------------------------------------------------------------------------
# 5. Cosine Similarity Heatmap — Before vs After Training
# ---------------------------------------------------------------------------

def demo_similarity_heatmap():
    """
    Visualise the full pairwise cosine similarity matrix before and after
    training.  After training, the block structure (animals vs vehicles)
    should be visible.
    """
    print("=" * 60)
    print("5. Cosine Similarity Heatmap Before vs After Training")
    print("=" * 60)

    vocab = ["cat", "dog", "kitten", "puppy", "car", "bus", "truck", "van"]

    corpus = [
        ["cat",    "and",   "dog",    "are",   "pets"],
        ["kitten", "is",    "a",      "baby",  "cat"],
        ["puppy",  "is",    "a",      "baby",  "dog"],
        ["cat",    "chased", "the",   "dog"],
        ["kitten", "played", "with",  "puppy"],
        ["car",    "and",   "bus",    "drive", "on", "road"],
        ["truck",  "and",   "van",    "carry", "goods"],
        ["car",    "bus",   "truck",  "van",   "are", "vehicles"],
        ["bus",    "and",   "van",    "are",   "large"],
        ["dog",    "and",   "cat",    "are",   "animals"],
        ["kitten", "and",   "puppy",  "are",   "young", "animals"],
        ["truck",  "is",    "larger", "than",  "car"],
        ["bus",    "carries", "people", "like", "a", "van"],
        ["cat",    "dog",   "kitten", "puppy", "live", "at", "home"],
        ["car",    "truck", "bus",    "van",   "use",  "fuel"],
    ]

    rng_before = np.random.default_rng(17)
    E_before   = rng_before.normal(0, 0.1, (len(vocab), 4))

    rng_after  = np.random.default_rng(17)
    E_after, _ = train_embeddings(vocab, corpus, embedding_dim=4,
                                  n_epochs=300, lr=0.05, rng=rng_after)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, E, title in [
        (axes[0], E_before, "BEFORE training\n(all similarities near zero)"),
        (axes[1], E_after,  "AFTER training\n(block structure: animals vs vehicles)"),
    ]:
        sim = cosine_similarity_matrix(E)
        im = ax.imshow(sim, vmin=-1, vmax=1, cmap="RdYlGn")
        ax.set_xticks(range(len(vocab)))
        ax.set_yticks(range(len(vocab)))
        ax.set_xticklabels(vocab, rotation=45, ha="right")
        ax.set_yticklabels(vocab)
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

        # Annotate cells with values
        for i in range(len(vocab)):
            for j in range(len(vocab)):
                ax.text(j, i, f"{sim[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black")

    plt.suptitle("Cosine Similarity Heatmap — Semantic Structure Emerges from Training")
    plt.tight_layout()
    plt.savefig("plot_05_similarity_heatmap.png", dpi=100)
    print("  Saved: plot_05_similarity_heatmap.png\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_embedding_table()
    print()
    demo_cosine_similarity()
    print()
    demo_embeddings_training()
    demo_visualise_embeddings()
    demo_similarity_heatmap()

    print("All demos complete.  PNG plots saved to the current directory.")
