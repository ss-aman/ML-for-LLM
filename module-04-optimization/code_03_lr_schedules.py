"""
Module 04 — Code 03: Learning Rate Schedules
=============================================
Implement and visualize all major LR schedules.
Show the effect of warmup + cosine decay on training.

Run: python code_03_lr_schedules.py
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# SCHEDULE IMPLEMENTATIONS
# =============================================================================

def constant_schedule(step, lr):
    return lr


def linear_warmup(step, warmup_steps, lr_max):
    """Linear warmup from 0 to lr_max over warmup_steps."""
    if step >= warmup_steps:
        return lr_max
    return lr_max * (step / warmup_steps)


def cosine_decay(step, total_steps, lr_max, lr_min=0.0):
    """Cosine decay from lr_max to lr_min over total_steps."""
    progress = min(step / total_steps, 1.0)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * progress))


def warmup_cosine_schedule(step, warmup_steps, total_steps, lr_max, lr_min):
    """
    Linear warmup + cosine decay.
    The LLM standard (GPT-3, LLaMA, LLaMA 2, Mistral, ...).

    Phase 1 (step < warmup_steps):  linear ramp 0 → lr_max
    Phase 2 (step >= warmup_steps): cosine decay lr_max → lr_min
    """
    if step < warmup_steps:
        return lr_max * (step / warmup_steps)
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * progress))


def step_decay(step, lr_init, drop_factor=0.1, drop_every=1000):
    """Multiply lr by drop_factor every drop_every steps."""
    return lr_init * (drop_factor ** (step // drop_every))


def linear_schedule(step, total_steps, lr_max, lr_min=0.0):
    """Linear decay from lr_max to lr_min."""
    progress = min(step / total_steps, 1.0)
    return lr_max * (1 - progress) + lr_min * progress


# =============================================================================
# SECTION 1: Visualizing All Schedules
# =============================================================================

def section_schedule_comparison():
    print("=" * 55)
    print("SECTION 1: Learning Rate Schedule Comparison")
    print("=" * 55)

    total_steps   = 10000
    warmup_steps  = 500
    lr_max        = 3e-4
    lr_min        = 3e-5
    steps         = np.arange(total_steps + 1)

    schedules = {
        'Constant':              [constant_schedule(s, lr_max) for s in steps],
        'Linear Decay':          [linear_schedule(s, total_steps, lr_max, lr_min) for s in steps],
        'Cosine Decay':          [cosine_decay(s, total_steps, lr_max, lr_min) for s in steps],
        'Step Decay':            [step_decay(s, lr_max, drop_factor=0.1, drop_every=3000) for s in steps],
        'Warmup + Cosine (LLM)': [warmup_cosine_schedule(s, warmup_steps, total_steps, lr_max, lr_min)
                                  for s in steps],
    }

    print(f"\nSchedule parameters:")
    print(f"  total_steps:  {total_steps:,}")
    print(f"  warmup_steps: {warmup_steps:,} ({100*warmup_steps/total_steps:.1f}% of training)")
    print(f"  lr_max:       {lr_max:.1e}")
    print(f"  lr_min:       {lr_min:.1e} ({100*lr_min/lr_max:.0f}% of max)")

    print(f"\n{'Schedule':25s}  {'Step 0':>10}  {'Step 500':>10}  {'Step 5000':>10}  {'Final':>10}")
    print("-" * 65)
    for name, lrs in schedules.items():
        print(f"  {name:25s}  {lrs[0]:>10.2e}  {lrs[500]:>10.2e}  {lrs[5000]:>10.2e}  {lrs[-1]:>10.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    colors = ['gray', 'red', 'green', 'orange', 'blue']
    for (name, lrs), color in zip(schedules.items(), colors):
        lw    = 3 if 'LLM' in name else 1.5
        alpha = 1.0 if 'LLM' in name else 0.7
        axes[0].plot(steps, lrs, color=color, label=name, linewidth=lw, alpha=alpha)

    axes[0].axvspan(0, warmup_steps, alpha=0.1, color='blue', label=f'Warmup ({warmup_steps} steps)')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Learning Rate')
    axes[0].set_title('Learning Rate Schedules')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Zoom into warmup region
    warmup_lrs = [warmup_cosine_schedule(s, warmup_steps, total_steps, lr_max, lr_min)
                  for s in range(warmup_steps * 3)]
    axes[1].plot(range(warmup_steps * 3), warmup_lrs, 'b-', linewidth=2)
    axes[1].axvline(x=warmup_steps, color='red', linestyle='--', alpha=0.7, label=f'End of warmup (step {warmup_steps})')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Warmup + Cosine: Zoom on Warmup Region')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lr_schedules.png', dpi=100)
    plt.close()
    print("\nSaved: lr_schedules.png")


# =============================================================================
# SECTION 2: Real LLM Schedule Parameters
# =============================================================================

def section_real_llm_schedules():
    print("\n" + "=" * 55)
    print("SECTION 2: Real LLM Schedule Parameters")
    print("=" * 55)

    llm_configs = [
        {
            "name": "GPT-3 (175B)",
            "lr_max": 6e-4,
            "lr_min": 6e-5,
            "warmup_tokens": 375e6,
            "total_tokens":  300e9,
            "batch_tokens":  3.2e6,
        },
        {
            "name": "LLaMA (7B)",
            "lr_max": 3e-4,
            "lr_min": 3e-5,
            "warmup_steps": 2000,
            "total_tokens":  1e12,
            "batch_tokens":  4e6,
        },
        {
            "name": "LLaMA 2 (70B)",
            "lr_max": 1e-4,
            "lr_min": 1e-5,
            "warmup_steps": 2000,
            "total_tokens":  2e12,
            "batch_tokens":  4e6,
        },
    ]

    print(f"\n{'Model':20s}  {'lr_max':>8}  {'lr_min':>8}  {'Warmup %':>10}  {'Total steps':>14}")
    print("-" * 70)
    for cfg in llm_configs:
        total_steps = int(cfg['total_tokens'] / cfg['batch_tokens'])
        if 'warmup_steps' in cfg:
            warmup_steps = cfg['warmup_steps']
        else:
            warmup_steps = int(cfg['warmup_tokens'] / cfg['batch_tokens'])
        warmup_pct = 100 * warmup_steps / total_steps
        print(f"  {cfg['name']:20s}  {cfg['lr_max']:>8.1e}  {cfg['lr_min']:>8.1e}  "
              f"{warmup_pct:>9.3f}%  {total_steps:>14,}")

    # Visualize the three schedules
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ['blue', 'green', 'red']

    for ax, cfg, color in zip(axes, llm_configs, colors):
        total_steps = int(cfg['total_tokens'] / cfg['batch_tokens'])
        if 'warmup_steps' in cfg:
            warmup_steps = cfg['warmup_steps']
        else:
            warmup_steps = int(cfg['warmup_tokens'] / cfg['batch_tokens'])

        steps = np.linspace(0, total_steps, 1000, dtype=int)
        lrs   = [warmup_cosine_schedule(s, warmup_steps, total_steps,
                                        cfg['lr_max'], cfg['lr_min'])
                 for s in steps]

        ax.plot(steps, lrs, color=color, linewidth=2)
        ax.axvline(x=warmup_steps, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title(cfg['name'])
        ax.grid(True, alpha=0.3)
        ax.text(warmup_steps, cfg['lr_max'] * 0.9, f'Warmup\n{warmup_steps:,} steps',
                fontsize=7, ha='left', color='gray')

    plt.suptitle('LLM Learning Rate Schedules (Real Configs)', fontsize=12)
    plt.tight_layout()
    plt.savefig('llm_lr_schedules.png', dpi=100)
    plt.close()
    print("\nSaved: llm_lr_schedules.png")
    print("""
Observations:
  - Larger models use smaller lr_max (more parameters, need smaller steps)
  - All use lr_min = 10% of lr_max
  - Warmup is typically < 0.1% of total training steps
  - Cosine decay covers the remaining 99.9%+ of training
""")


# =============================================================================
# SECTION 3: Effect of Warmup on Training Stability
# =============================================================================

def section_warmup_effect():
    print("=" * 55)
    print("SECTION 3: Why Warmup Is Needed")
    print("=" * 55)

    np.random.seed(42)

    # Simulate: training with and without warmup
    # We simulate an LM-like problem where early instability can destabilize training
    def simulate_training(schedule_fn, n_steps=500, noise_std=0.5):
        """
        Simplified simulation: loss ≈ true_loss + noise/lr_effective
        Higher lr early = bigger noise in loss
        """
        np.random.seed(0)
        w          = np.random.randn(10) * 2.0  # random init, far from optimum
        losses     = []
        grad_norms = []

        for step in range(n_steps):
            lr = schedule_fn(step)
            # "True" gradient + noise (noise scales with gradient magnitude)
            true_grad = 2 * w
            noise     = np.random.randn(10) * noise_std * np.linalg.norm(true_grad)
            grad      = true_grad + noise

            grad_norm = np.linalg.norm(grad)
            grad_norms.append(float(grad_norm))

            # Clip gradient (as in real LLM training)
            if grad_norm > 1.0:
                grad = grad / grad_norm  # clip to norm 1.0

            w = w - lr * grad
            losses.append(float(np.sum(w**2)))

        return losses, grad_norms

    total_steps  = 500
    warmup_steps = 50
    lr_max       = 0.1

    no_warmup_fn  = lambda s: lr_max
    with_warmup_fn = lambda s: warmup_cosine_schedule(
        s, warmup_steps, total_steps, lr_max, lr_max * 0.1
    )

    losses_nw, gnorms_nw = simulate_training(no_warmup_fn)
    losses_ww, gnorms_ww = simulate_training(with_warmup_fn)

    print(f"\nWith warmup ({warmup_steps} steps):")
    print(f"  Peak loss in first 50 steps: {max(losses_ww[:50]):.4f}")
    print(f"  Loss at step 100:            {losses_ww[100]:.4f}")
    print(f"  Final loss:                  {losses_ww[-1]:.6f}")

    print(f"\nWithout warmup:")
    print(f"  Peak loss in first 50 steps: {max(losses_nw[:50]):.4f}")
    print(f"  Loss at step 100:            {losses_nw[100]:.4f}")
    print(f"  Final loss:                  {losses_nw[-1]:.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    steps = range(total_steps)

    axes[0].plot(steps, losses_nw, 'r-', label='No warmup', linewidth=1.5, alpha=0.8)
    axes[0].plot(steps, losses_ww, 'b-', label=f'With warmup ({warmup_steps} steps)', linewidth=2)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Stability: Warmup vs No Warmup')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, min(max(losses_nw) * 1.1, 200))

    # LR schedules
    lrs_nw = [no_warmup_fn(s) for s in steps]
    lrs_ww = [with_warmup_fn(s) for s in steps]
    axes[1].plot(steps, lrs_nw, 'r-', label='No warmup', linewidth=1.5)
    axes[1].plot(steps, lrs_ww, 'b-', label='With warmup', linewidth=2)
    axes[1].axvspan(0, warmup_steps, alpha=0.1, color='blue')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('warmup_effect.png', dpi=100)
    plt.close()
    print("\nSaved: warmup_effect.png")
    print("""
Why warmup matters:
  At step 0, model weights are random → gradients can be very large
  Adam's v = 0 → bias correction amplifies early updates
  High lr + large initial gradients + uncalibrated Adam = instability

  Warmup lets:
  1. Adam's moment estimates accumulate (m, v become reliable)
  2. Model settle into a reasonable region before large steps
  3. Gradient norms stabilize before full learning rate is applied
""")


# =============================================================================
# RUN ALL SECTIONS
# =============================================================================

if __name__ == '__main__':
    section_schedule_comparison()
    section_real_llm_schedules()
    section_warmup_effect()

    print("\n" + "=" * 55)
    print("Key takeaways:")
    print("  1. Warmup: linear ramp prevents instability at init")
    print("  2. Cosine: slow then fast decay, stays near max longer")
    print("  3. lr_min = 10% of lr_max (industry standard)")
    print("  4. Warmup = ~0.1% of total steps for LLMs")
    print("  5. Larger models need smaller lr_max")
    print("=" * 55)
