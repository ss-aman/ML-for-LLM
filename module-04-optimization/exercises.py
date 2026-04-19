"""
Module 04 — Exercises: Optimization Algorithms
================================================
Implement each optimizer and schedule from scratch.
Do NOT look at solutions.py until you've tried.

Run: python exercises.py
"""

import numpy as np


# =============================================================================
# EXERCISE 1: SGD with Momentum
# Difficulty: Easy
# =============================================================================

class SGDMomentum:
    """
    SGD with momentum.

    Update rules:
        v = beta * v + grad          (velocity accumulates gradient history)
        w = w - lr * v               (update with velocity, not raw gradient)

    Args:
        lr:   learning rate
        beta: momentum coefficient (typically 0.9)

    State:
        self.v: velocity vector (None until first call, then same shape as params)

    Example:
        opt = SGDMomentum(lr=0.1, beta=0.9)
        w = np.array([5.0, 3.0])
        g = np.array([1.0, 2.0])
        w = opt.step(w, g)   # updates w using velocity
        w = opt.step(w, g)   # velocity accumulates
    """

    def __init__(self, lr=0.01, beta=0.9):
        self.lr   = lr
        self.beta = beta
        self.v    = None

    def step(self, params, grads):
        """
        Args:
            params: numpy array of current parameter values
            grads:  numpy array of gradients (same shape)
        Returns:
            updated params as numpy array
        """
        # TODO: initialize self.v if None, update velocity, update params
        raise NotImplementedError


# =============================================================================
# EXERCISE 2: Adam Optimizer
# Difficulty: Medium
# =============================================================================

class Adam:
    """
    Adam: Adaptive Moment Estimation.

    Update rules:
        m = beta1 * m + (1 - beta1) * grad       (first moment: direction)
        v = beta2 * v + (1 - beta2) * grad**2    (second moment: magnitude)
        m_hat = m / (1 - beta1**t)               (bias correction)
        v_hat = v / (1 - beta2**t)               (bias correction)
        w = w - lr * m_hat / (sqrt(v_hat) + eps)

    Args:
        lr:    learning rate (typically 0.001)
        beta1: first moment decay (typically 0.9)
        beta2: second moment decay (typically 0.999)
        eps:   numerical stability (typically 1e-8)

    State:
        self.m: first moment, initialized to zeros
        self.v: second moment, initialized to zeros
        self.t: step counter, starts at 0

    Example:
        opt = Adam(lr=0.001)
        w = np.array([5.0, -3.0])
        g = np.array([1.0,  2.0])
        w = opt.step(w, g)   # first step: bias correction is large
        w = opt.step(w, g)   # second step: less correction needed
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.m     = None
        self.v     = None
        self.t     = 0

    def step(self, params, grads):
        """
        Args:
            params: numpy array of current parameter values
            grads:  numpy array of gradients (same shape)
        Returns:
            updated params as numpy array
        """
        # TODO: init m, v if None; increment t; update m, v; apply bias correction; update params
        raise NotImplementedError


# =============================================================================
# EXERCISE 3: AdamW (Decoupled Weight Decay)
# Difficulty: Medium
# =============================================================================

class AdamW:
    """
    AdamW: Adam with decoupled weight decay.

    The key difference from Adam+L2:
        Adam+L2 adds weight_decay*w to gradient (goes through adaptive scaling)
        AdamW applies weight decay DIRECTLY to weights AFTER the gradient step

    Full update:
        [same as Adam for gradient step]
        w = w - lr * m_hat / (sqrt(v_hat) + eps)    # gradient step
        w = w - lr * weight_decay * w                # weight decay (decoupled)
        → equivalently: w = w * (1 - lr * weight_decay)

    Args:
        lr:           learning rate
        beta1:        first moment decay (0.9)
        beta2:        second moment decay (0.95 for LLMs, not 0.999)
        eps:          numerical stability (1e-8)
        weight_decay: regularization strength (0.1 for LLMs)

    Example:
        opt = AdamW(lr=3e-4, beta1=0.9, beta2=0.95, weight_decay=0.1)
        w = np.ones(4)
        g = np.array([0.1, 0.2, -0.1, 0.0])
        w = opt.step(w, g)
        # w should be slightly below 1.0 due to weight decay pulling toward 0
    """

    def __init__(self, lr=3e-4, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1):
        self.lr           = lr
        self.beta1        = beta1
        self.beta2        = beta2
        self.eps          = eps
        self.weight_decay = weight_decay
        self.m            = None
        self.v            = None
        self.t            = 0

    def step(self, params, grads):
        """
        Args:
            params: numpy array of current parameter values
            grads:  numpy array of gradients (same shape)
        Returns:
            updated params as numpy array
        """
        # TODO: Adam update (no weight decay in moments), then decoupled weight decay
        raise NotImplementedError


# =============================================================================
# EXERCISE 4: Warmup + Cosine LR Schedule
# Difficulty: Easy-Medium
# =============================================================================

def warmup_cosine_schedule(step, warmup_steps, total_steps, lr_max, lr_min):
    """
    Learning rate schedule used in all modern LLMs.

    Phase 1 — Linear warmup (step < warmup_steps):
        lr = lr_max * (step / warmup_steps)
        → starts at 0, linearly increases to lr_max

    Phase 2 — Cosine decay (step >= warmup_steps):
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * progress))
        → smoothly decreases from lr_max to lr_min

    Args:
        step:         current training step (integer)
        warmup_steps: number of warmup steps
        total_steps:  total training steps
        lr_max:       peak learning rate (after warmup)
        lr_min:       final learning rate (end of cosine decay)

    Returns:
        scalar learning rate for this step

    Examples:
        warmup_cosine_schedule(0, 100, 1000, 3e-4, 3e-5)   → 0.0
        warmup_cosine_schedule(50, 100, 1000, 3e-4, 3e-5)  → 1.5e-4  (mid warmup)
        warmup_cosine_schedule(100, 100, 1000, 3e-4, 3e-5) → 3e-4    (end of warmup)
        warmup_cosine_schedule(1000, 100, 1000, 3e-4, 3e-5)→ 3e-5    (end of training)
    """
    # TODO: linear warmup for step < warmup_steps, cosine decay otherwise
    raise NotImplementedError


# =============================================================================
# EXERCISE 5: Gradient Norm Clipping
# Difficulty: Easy
# =============================================================================

def clip_gradient_norm(grads_list, max_norm):
    """
    Clip gradients by their global norm.

    Algorithm:
        1. Compute global norm = sqrt(sum of squared values across ALL gradients)
        2. If global_norm > max_norm:
               scale each gradient by (max_norm / global_norm)
           Else:
               leave gradients unchanged

    Key property: direction is preserved, only magnitude is reduced.

    Args:
        grads_list: list of numpy arrays (one per parameter group)
        max_norm:   maximum allowed global norm

    Returns:
        (clipped_grads_list, global_norm)
        clipped_grads_list: list of (possibly scaled) gradient arrays
        global_norm:        the norm BEFORE clipping (float)

    Examples:
        grads = [np.array([3.0, 4.0])]   # norm = 5.0
        clipped, norm = clip_gradient_norm(grads, max_norm=1.0)
        # norm   → 5.0
        # clipped[0] → [0.6, 0.8]  (scaled by 1.0/5.0, direction preserved)

        grads = [np.array([0.3, 0.4])]   # norm = 0.5 < 1.0
        clipped, norm = clip_gradient_norm(grads, max_norm=1.0)
        # norm   → 0.5
        # clipped[0] → [0.3, 0.4]  (unchanged, norm already < max_norm)
    """
    # TODO: compute global norm, clip if needed, return (clipped, norm)
    raise NotImplementedError


# =============================================================================
# RUN ALL EXERCISES
# =============================================================================

def check():
    print("=" * 55)
    print("Module 04 — Exercise Results")
    print("=" * 55)

    passed  = 0
    total   = 0
    skipped = 0

    def test(name, got, expected, tol=1e-4):
        nonlocal passed, total
        total += 1
        if isinstance(expected, np.ndarray):
            ok = np.allclose(got, expected, atol=tol)
        elif isinstance(expected, bool):
            ok = bool(got) == expected
        elif isinstance(expected, (float, int)):
            ok = abs(float(got) - float(expected)) < tol
        else:
            ok = np.allclose(got, expected, atol=tol)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if ok:
            passed += 1

    # ── Exercise 1: SGDMomentum ───────────────────────────────────────────────
    print("\nExercise 1: SGDMomentum")
    try:
        # First step: v starts at 0, so result is just lr * grad
        np.random.seed(0)
        opt = SGDMomentum(lr=0.1, beta=0.9)
        w   = np.array([5.0, 3.0])
        g   = np.array([1.0, 2.0])

        w1 = opt.step(w.copy(), g)
        # v = 0*0.9 + g = g = [1,2]; w = [5,3] - 0.1*[1,2] = [4.9, 2.8]
        test("First step update", w1, np.array([4.9, 2.8]))

        # Second step: velocity accumulates
        w2 = opt.step(w1, g)
        # v = 0.9*[1,2] + [1,2] = [1.9, 3.8]; w = [4.9, 2.8] - 0.1*[1.9, 3.8]
        expected_w2 = w1 - 0.1 * (0.9 * g + g)
        test("Second step accumulates velocity", w2, expected_w2)

        # Velocity state is maintained
        test("Velocity is maintained (not None)", opt.v is not None, True)
    except NotImplementedError:
        skipped += 1
        print("  [SKIP] Not implemented yet")

    # ── Exercise 2: Adam ─────────────────────────────────────────────────────
    print("\nExercise 2: Adam")
    try:
        opt = Adam(lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8)
        w   = np.array([2.0, -1.0])
        g   = np.array([1.0,  1.0])

        # Manual first step
        m_1    = (1 - 0.9) * g         # [0.1, 0.1]
        v_1    = (1 - 0.999) * g**2    # [0.001, 0.001]
        m_hat  = m_1 / (1 - 0.9**1)    # [1.0, 1.0]
        v_hat  = v_1 / (1 - 0.999**1)  # [1.0, 1.0]
        w_exp  = w - 0.1 * m_hat / (np.sqrt(v_hat) + 1e-8)

        w1 = opt.step(w.copy(), g)
        test("Adam first step", w1, w_exp, tol=1e-6)
        test("t increments",    opt.t, 1)

        # After second step, t=2
        opt.step(w1, g)
        test("t increments to 2", opt.t, 2)

        # Bias correction: early m_hat should be larger than m
        opt2 = Adam(lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8)
        opt2.step(np.zeros(2), np.ones(2))
        test("Bias correction: m_hat > m at t=1",
             float(opt2.m[0] / (1 - 0.9**1)) > float(opt2.m[0]), True)
    except NotImplementedError:
        skipped += 1
        print("  [SKIP] Not implemented yet")

    # ── Exercise 3: AdamW ────────────────────────────────────────────────────
    print("\nExercise 3: AdamW")
    try:
        # With weight_decay > 0 and zero gradient, weights should shrink
        opt    = AdamW(lr=0.01, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1)
        w_ones = np.array([1.0, 1.0, 1.0])
        g_zero = np.zeros(3)

        # Gradient is zero, only weight decay acts
        # Gradient step does nothing meaningful (m_hat ≈ 0)
        # Weight decay: w = w * (1 - lr * weight_decay) = 1 * (1 - 0.001) = 0.999
        w1 = opt.step(w_ones.copy(), g_zero)
        test("Weight decay shrinks weights", all(w1 < 1.0), True)

        # With large weight decay, weights shrink more
        opt_wd = AdamW(lr=0.1, weight_decay=0.5)
        w2     = opt_wd.step(np.ones(3), np.zeros(3))
        test("Larger weight_decay → more shrinkage",
             all(w2 < w1), True)

        # AdamW does not apply weight decay through momentum
        # Test: gradient update and weight decay are separate
        opt3    = AdamW(lr=0.1, beta1=0.0, beta2=0.0, eps=1e-8, weight_decay=0.0)
        w_init  = np.array([1.0])
        g_init  = np.array([1.0])
        w3 = opt3.step(w_init.copy(), g_init)
        # With beta1=beta2=0, wd=0: like SGD with adaptive denominator
        test("Zero weight decay: no decay applied", float(w3[0]) < 1.0, True)
    except NotImplementedError:
        skipped += 1
        print("  [SKIP] Not implemented yet")

    # ── Exercise 4: warmup_cosine_schedule ───────────────────────────────────
    print("\nExercise 4: warmup_cosine_schedule")
    try:
        lr_max, lr_min  = 3e-4, 3e-5
        n_warmup        = 100
        n_total         = 1000

        test("step=0 → lr=0",       warmup_cosine_schedule(0, n_warmup, n_total, lr_max, lr_min), 0.0)
        test("step=50 → lr=lr_max/2",
             warmup_cosine_schedule(50, n_warmup, n_total, lr_max, lr_min), lr_max / 2, tol=1e-8)
        test("step=warmup → lr=lr_max",
             warmup_cosine_schedule(n_warmup, n_warmup, n_total, lr_max, lr_min), lr_max, tol=1e-8)
        test("step=total → lr=lr_min",
             warmup_cosine_schedule(n_total, n_warmup, n_total, lr_max, lr_min), lr_min, tol=1e-9)

        # Monotone during warmup
        lrs_warmup = [warmup_cosine_schedule(s, n_warmup, n_total, lr_max, lr_min)
                      for s in range(n_warmup + 1)]
        test("Monotone increase during warmup",
             all(lrs_warmup[i] <= lrs_warmup[i+1] for i in range(n_warmup)), True)

        # Monotone during decay
        lrs_decay = [warmup_cosine_schedule(s, n_warmup, n_total, lr_max, lr_min)
                     for s in range(n_warmup, n_total + 1)]
        test("Monotone decrease after warmup",
             all(lrs_decay[i] >= lrs_decay[i+1] for i in range(len(lrs_decay)-1)), True)
    except NotImplementedError:
        skipped += 1
        print("  [SKIP] Not implemented yet")

    # ── Exercise 5: clip_gradient_norm ───────────────────────────────────────
    print("\nExercise 5: clip_gradient_norm")
    try:
        # Basic clipping: [3,4] has norm 5, clip to 1 → [0.6, 0.8]
        grads = [np.array([3.0, 4.0])]
        clipped, norm = clip_gradient_norm(grads, max_norm=1.0)
        test("Returns correct norm (5.0)",  norm, 5.0)
        test("Clipped norm = max_norm",     np.linalg.norm(clipped[0]), 1.0)
        test("Direction preserved (0.6,0.8)", clipped[0], np.array([0.6, 0.8]))

        # No clipping needed: norm = 0.5 < 1.0
        grads2 = [np.array([0.3, 0.4])]
        clipped2, norm2 = clip_gradient_norm(grads2, max_norm=1.0)
        test("No clipping when norm < max_norm", norm2, 0.5)
        test("Values unchanged when not clipped", clipped2[0], np.array([0.3, 0.4]))

        # Multiple gradient arrays — global norm across all
        grads3 = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        clipped3, norm3 = clip_gradient_norm(grads3, max_norm=1.0)
        # global norm = sqrt(1+1) ≈ 1.414 > 1.0 → scale by 1/sqrt(2)
        test("Global norm across multiple arrays", norm3, np.sqrt(2.0), tol=1e-5)
        test("Each array scaled correctly",
             np.allclose(clipped3[0], np.array([1/np.sqrt(2), 0])), True)
    except NotImplementedError:
        skipped += 1
        print("  [SKIP] Not implemented yet")

    print(f"\n{'='*55}")
    if skipped > 0:
        print(f"Score: {passed}/{total} passed,  {skipped} exercise(s) not yet implemented.")
        print("Implement the TODO sections, then re-run.")
    elif passed == total and total > 0:
        print(f"Score: {passed}/{total} — All exercises complete! Move on to Module 05.")
    else:
        print(f"Score: {passed}/{total} — {total - passed} failing.")
    print("=" * 55)


if __name__ == '__main__':
    check()
