"""
Module 04 — Code 04: Complete LLM Training Setup
=================================================
Assemble all optimizer components into a realistic LLM training loop.
Train a mini transformer on character-level text to demonstrate the full setup.

This file shows the exact training setup used in GPT/LLaMA training:
  AdamW + warmup + cosine decay + gradient clipping + gradient accumulation

Run: python code_04_llm_training_setup.py
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# COMPLETE OPTIMIZER COMPONENTS (from previous code files, assembled here)
# =============================================================================

class AdamW:
    """AdamW with decoupled weight decay. The LLM standard optimizer."""

    def __init__(self, lr=3e-4, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1):
        self.lr           = lr
        self.beta1        = beta1
        self.beta2        = beta2
        self.eps          = eps
        self.weight_decay = weight_decay
        self._states      = {}   # per-parameter states: {param_id: (m, v, t)}

    def step(self, param_id, params, grads):
        """Update a single parameter array."""
        if param_id not in self._states:
            self._states[param_id] = {
                'm': np.zeros_like(params),
                'v': np.zeros_like(params),
                't': 0,
            }
        state = self._states[param_id]
        state['t'] += 1
        t = state['t']

        # Moment updates (gradient only — no weight decay in moments)
        state['m'] = self.beta1 * state['m'] + (1 - self.beta1) * grads
        state['v'] = self.beta2 * state['v'] + (1 - self.beta2) * grads ** 2

        # Bias correction
        m_hat = state['m'] / (1 - self.beta1 ** t)
        v_hat = state['v'] / (1 - self.beta2 ** t)

        # Gradient step
        params = params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        # Decoupled weight decay
        params = params * (1 - self.lr * self.weight_decay)

        return params


def warmup_cosine_lr(step, warmup_steps, total_steps, lr_max, lr_min):
    """Warmup + cosine decay — used in every modern LLM."""
    if step < warmup_steps:
        return lr_max * (step / max(warmup_steps, 1))
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(progress, 1.0)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * progress))


def clip_gradient_norm(grads_list, max_norm=1.0):
    """
    Clip gradient by global norm.
    Returns clipped gradients and the pre-clip norm.
    """
    total_norm = np.sqrt(sum(np.sum(g**2) for g in grads_list))
    if total_norm > max_norm:
        factor = max_norm / total_norm
        grads_list = [g * factor for g in grads_list]
    return grads_list, float(total_norm)


# =============================================================================
# MINI LANGUAGE MODEL (pure numpy, for demonstration)
# =============================================================================

class MiniLM:
    """
    Tiny character-level language model.
    Architecture: embedding → single linear layer → output logits
    This is NOT a transformer (that's Module 06+), but uses the same
    training loop structure.
    """

    def __init__(self, vocab_size, embed_dim, context_len, seed=42):
        np.random.seed(seed)
        self.vocab_size  = vocab_size
        self.embed_dim   = embed_dim
        self.context_len = context_len
        d_in = embed_dim * context_len

        # Parameters
        scale = np.sqrt(1.0 / embed_dim)
        self.E = np.random.randn(vocab_size, embed_dim) * scale   # embedding table
        self.W = np.random.randn(vocab_size, d_in) * np.sqrt(1.0 / d_in)  # output weights
        self.b = np.zeros(vocab_size)

        # Gradient accumulators
        self.dE = np.zeros_like(self.E)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self._cache = {}

    def forward(self, token_ids):
        """
        token_ids: integer array of shape (context_len,)
        Returns logits: shape (vocab_size,)
        """
        # Embedding lookup
        embeds = self.E[token_ids]               # (context_len, embed_dim)
        x      = embeds.reshape(-1)              # flatten: (context_len * embed_dim,)

        # Linear output layer
        logits = self.W @ x + self.b             # (vocab_size,)

        self._cache = {'token_ids': token_ids, 'x': x, 'embeds': embeds}
        return logits

    def softmax(self, z):
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def loss_and_grad(self, logits, target_id):
        """Cross-entropy loss and gradient w.r.t. logits."""
        probs    = self.softmax(logits)
        loss     = -np.log(probs[target_id] + 1e-10)

        # Gradient of cross-entropy + softmax: dL/dz = probs - one_hot
        dlogits          = probs.copy()
        dlogits[target_id] -= 1.0

        return float(loss), dlogits

    def backward(self, dlogits):
        """Backprop through the linear + embedding layers."""
        x         = self._cache['x']
        token_ids = self._cache['token_ids']

        # Gradients for linear layer
        dW = np.outer(dlogits, x)   # (vocab_size, d_in)
        db = dlogits                 # (vocab_size,)
        dx = self.W.T @ dlogits     # (d_in,)

        # Accumulate
        self.dW += dW
        self.db += db

        # Gradient flows to embedding rows
        dx_reshaped = dx.reshape(self.context_len, self.embed_dim)
        for i, tid in enumerate(token_ids):
            self.dE[tid] += dx_reshaped[i]

    def zero_grad(self):
        self.dE[:] = 0
        self.dW[:] = 0
        self.db[:] = 0


# =============================================================================
# SECTION 1: Complete Training Loop
# =============================================================================

def section_complete_training_loop():
    print("=" * 55)
    print("SECTION 1: Complete LLM Training Loop")
    print("=" * 55)

    # --- Small text corpus (character-level) ---
    text = (
        "the cat sat on the mat the cat ate the rat the rat ran from the cat "
        "the dog chased the cat the cat climbed the tree the dog barked "
        "a bird flew over the tree the bird sat on a branch "
        "the cat watched the bird the bird watched the cat "
    ) * 10  # repeat to have more data

    chars    = sorted(set(text))
    vocab    = {c: i for i, c in enumerate(chars)}
    ivocab   = {i: c for c, i in vocab.items()}
    vocab_sz = len(chars)
    tokens   = [vocab[c] for c in text]

    print(f"\nCorpus: {len(text)} chars, {vocab_sz} unique chars")
    print(f"Vocabulary: {chars}")

    # --- Hyperparameters (LLM-style) ---
    context_len  = 8       # predict next char from 8 previous
    embed_dim    = 16
    lr_max       = 3e-3
    lr_min       = 3e-4
    warmup_steps = 100
    total_steps  = 2000
    batch_size   = 16      # examples per update
    max_norm     = 1.0
    weight_decay = 0.1

    model     = MiniLM(vocab_sz, embed_dim, context_len, seed=42)
    optimizer = AdamW(lr=lr_max, beta1=0.9, beta2=0.95,
                      eps=1e-8, weight_decay=weight_decay)

    losses      = []
    grad_norms  = []
    lr_history  = []
    n_tokens    = len(tokens)

    print(f"\nTraining for {total_steps} steps...")
    print(f"Config: AdamW(β1=0.9, β2=0.95, wd={weight_decay}) + warmup({warmup_steps}) + cosine")
    print(f"{'Step':>6}  {'Loss':>8}  {'Perplexity':>12}  {'LR':>10}  {'GradNorm':>10}  {'Clipped':>8}")
    print("-" * 65)

    for step in range(1, total_steps + 1):
        # --- Update learning rate ---
        current_lr    = warmup_cosine_lr(step, warmup_steps, total_steps, lr_max, lr_min)
        optimizer.lr  = current_lr
        lr_history.append(current_lr)

        # --- Mini-batch: accumulate gradients ---
        model.zero_grad()
        batch_loss = 0.0

        for _ in range(batch_size):
            # Sample a random context from the corpus
            start  = np.random.randint(0, n_tokens - context_len - 1)
            ctx    = np.array(tokens[start: start + context_len])
            target = tokens[start + context_len]

            # Forward + loss
            logits = model.forward(ctx)
            loss, dlogits = model.loss_and_grad(logits, target)
            batch_loss += loss

            # Backward (accumulates gradients)
            model.backward(dlogits)

        # Average gradients over batch
        model.dE /= batch_size
        model.dW /= batch_size
        model.db /= batch_size
        avg_loss = batch_loss / batch_size

        # --- Gradient clipping ---
        [model.dE, model.dW, model.db], norm = clip_gradient_norm(
            [model.dE, model.dW, model.db], max_norm=max_norm
        )
        clipped = norm > max_norm

        # --- AdamW parameter update ---
        model.E = optimizer.step('E', model.E, model.dE)
        model.W = optimizer.step('W', model.W, model.dW)
        model.b = optimizer.step('b', model.b, model.db)

        losses.append(avg_loss)
        grad_norms.append(norm)

        if step % 400 == 0 or step == 1:
            ppl = np.exp(avg_loss)
            print(f"{step:>6}  {avg_loss:>8.4f}  {ppl:>12.2f}  "
                  f"{current_lr:>10.2e}  {norm:>10.4f}  {'YES' if clipped else 'no':>8}")

    print(f"\nTraining complete!")
    print(f"  Initial loss:    {losses[0]:.4f}  (PPL={np.exp(losses[0]):.1f})")
    print(f"  Final loss:      {losses[-1]:.4f}  (PPL={np.exp(losses[-1]):.1f})")
    print(f"  Improvement:     {100*(losses[0]-losses[-1])/losses[0]:.1f}%")
    print(f"  Grad clips:      {sum(1 for n in grad_norms if n > max_norm)}/{total_steps} steps")

    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Smoothed loss
    smooth_window = 50
    smoothed = np.convolve(losses, np.ones(smooth_window)/smooth_window, mode='valid')
    axes[0].plot(losses, color='lightblue', alpha=0.5, linewidth=0.8, label='Raw')
    axes[0].plot(range(smooth_window-1, len(losses)), smoothed, 'b-', linewidth=2, label='Smoothed')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].set_title('Training Loss Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Learning rate schedule
    axes[1].plot(lr_history, 'g-', linewidth=2)
    axes[1].axvline(x=warmup_steps, color='red', linestyle='--', alpha=0.7, label='End of warmup')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('LR Schedule (Warmup + Cosine)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Gradient norm
    axes[2].plot(grad_norms, color='orange', alpha=0.6, linewidth=0.8)
    axes[2].axhline(y=max_norm, color='red', linestyle='--', linewidth=2,
                    label=f'Clip threshold ({max_norm})')
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Gradient Norm')
    axes[2].set_title('Gradient Norm (Clipping at 1.0)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Complete LLM Training Loop: AdamW + Warmup + Cosine + Clipping', fontsize=11)
    plt.tight_layout()
    plt.savefig('llm_training_loop.png', dpi=100)
    plt.close()
    print("\nSaved: llm_training_loop.png")

    return model, vocab, ivocab


# =============================================================================
# SECTION 2: Text Generation After Training
# =============================================================================

def section_generation(model, vocab, ivocab):
    print("\n" + "=" * 55)
    print("SECTION 2: Text Generation After Training")
    print("=" * 55)

    context_len = model.context_len

    def generate(prompt, n_chars=50, temperature=1.0):
        tokens = [vocab.get(c, 0) for c in prompt[-context_len:]]
        result = list(prompt)

        for _ in range(n_chars):
            ctx    = np.array(tokens[-context_len:])
            logits = model.forward(ctx)

            # Temperature + softmax
            logits_t = logits / temperature
            logits_t -= logits_t.max()
            probs = np.exp(logits_t) / np.sum(np.exp(logits_t))

            # Sample
            next_tok = int(np.random.choice(len(probs), p=probs))
            result.append(ivocab[next_tok])
            tokens.append(next_tok)

        return ''.join(result)

    np.random.seed(7)
    print("\nGenerating text with different temperatures:")
    for temp in [0.3, 0.7, 1.0, 1.5]:
        generated = generate("the cat", n_chars=60, temperature=temp)
        print(f"\n  T={temp}: '{generated}'")

    print("""
The model has learned basic n-gram patterns from the training text.
A real LLM uses a transformer (Module 06+) instead of a linear layer,
but the training loop is identical: AdamW + warmup + cosine + clipping.
""")


# =============================================================================
# SECTION 3: Training Diagnostics
# =============================================================================

def section_diagnostics():
    print("=" * 55)
    print("SECTION 3: Reading Training Diagnostics")
    print("=" * 55)

    print("""
What each metric tells you during LLM training:

┌─────────────────────────────────────────────────────────────┐
│ LOSS                                                         │
│   loss = 3.5  → PPL=33  (model guesses 1/33 tokens right)  │
│   loss = 2.5  → PPL=12  (much better)                      │
│   loss = 2.0  → PPL=7.4 (approaching good LLM territory)   │
│                                                              │
│   Healthy: steady decrease, slight noise is normal          │
│   Problem: sudden spike → bad batch or exploding gradient   │
│   Problem: plateau → lr too low or wrong architecture       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ LEARNING RATE                                                │
│   During warmup: lr should ramp linearly from 0             │
│   After warmup: smooth cosine decay                         │
│                                                              │
│   Problem: lr too high → loss diverges                      │
│   Problem: lr too low  → training stalls (loss plateau)     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ GRADIENT NORM                                                │
│   Healthy: 0.1–2.0 during stable training                   │
│   High (>1.0): clipping fires, normal occasionally           │
│   Very high (>10): something is wrong — check batch/data    │
│   Near zero: vanishing gradients, model not learning        │
│                                                              │
│   Constant clipping: lr might be too high                   │
│   Rare clipping:     training is stable                     │
└─────────────────────────────────────────────────────────────┘
""")

    # Simulate different training health scenarios
    np.random.seed(42)
    n = 500

    # Healthy training
    healthy_loss  = 3.0 * np.exp(-np.linspace(0, 3, n)) + 0.5 + np.random.randn(n) * 0.05
    healthy_gnorm = 0.8 * np.exp(-np.linspace(0, 2, n)) + 0.3 + np.abs(np.random.randn(n)) * 0.1

    # Unstable training (occasional spikes)
    unstable_loss = healthy_loss.copy()
    unstable_loss[[100, 250, 380]] += [2.0, 1.5, 1.0]  # loss spikes

    # Diverging training
    diverging_loss = 3.0 * (1 + 0.005 * np.arange(n)) + np.random.randn(n) * 0.3

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, losses, title, color in [
        (axes[0], healthy_loss,   'Healthy Training',   'blue'),
        (axes[1], unstable_loss,  'Unstable (spikes)',  'orange'),
        (axes[2], diverging_loss, 'Diverging (lr>>)',   'red'),
    ]:
        ax.plot(losses, color=color, linewidth=1.5, alpha=0.8)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Training Loss Patterns: What to Look For', fontsize=11)
    plt.tight_layout()
    plt.savefig('training_diagnostics.png', dpi=100)
    plt.close()
    print("Saved: training_diagnostics.png")


# =============================================================================
# RUN ALL SECTIONS
# =============================================================================

if __name__ == '__main__':
    model, vocab, ivocab = section_complete_training_loop()
    section_generation(model, vocab, ivocab)
    section_diagnostics()

    print("\n" + "=" * 55)
    print("Complete LLM training loop:")
    print("  1. forward()      → compute logits")
    print("  2. cross_entropy  → scalar loss")
    print("  3. backward()     → compute gradients")
    print("  4. clip_grad_norm → prevent explosions")
    print("  5. adamw.step()   → update weights")
    print("  6. scheduler.step()→ adjust lr")
    print("  Repeat for 100k–1M+ steps")
    print("=" * 55)
