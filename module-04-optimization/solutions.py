"""
Module 04 — Solutions
======================
Reference solutions. Only look here after genuinely attempting exercises.

Run: python solutions.py
"""

import numpy as np


class SGDMomentum:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr   = lr
        self.beta = beta
        self.v    = None

    def step(self, params, grads):
        if self.v is None:
            self.v = np.zeros_like(params)
        self.v = self.beta * self.v + grads
        return params - self.lr * self.v


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.m     = None
        self.v     = None
        self.t     = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        self.m  = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v  = self.beta2 * self.v + (1 - self.beta2) * grads ** 2
        m_hat   = self.m / (1 - self.beta1 ** self.t)
        v_hat   = self.v / (1 - self.beta2 ** self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW:
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
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        self.m  = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v  = self.beta2 * self.v + (1 - self.beta2) * grads ** 2
        m_hat   = self.m / (1 - self.beta1 ** self.t)
        v_hat   = self.v / (1 - self.beta2 ** self.t)
        params  = params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        params  = params * (1 - self.lr * self.weight_decay)   # decoupled weight decay
        return params


def warmup_cosine_schedule(step, warmup_steps, total_steps, lr_max, lr_min):
    if step < warmup_steps:
        return lr_max * (step / warmup_steps)
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(progress, 1.0)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * progress))


def clip_gradient_norm(grads_list, max_norm):
    total_norm = np.sqrt(sum(np.sum(g**2) for g in grads_list))
    if total_norm > max_norm:
        factor     = max_norm / total_norm
        grads_list = [g * factor for g in grads_list]
    return grads_list, float(total_norm)


# =============================================================================
# CHECK (same tests as exercises.py)
# =============================================================================

def check():
    print("=" * 55)
    print("Module 04 — Solutions Check")
    print("=" * 55)

    passed = 0
    total  = 0

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

    print("\nExercise 1: SGDMomentum")
    opt = SGDMomentum(lr=0.1, beta=0.9)
    w   = np.array([5.0, 3.0])
    g   = np.array([1.0, 2.0])
    w1  = opt.step(w.copy(), g)
    test("First step update", w1, np.array([4.9, 2.8]))
    w2  = opt.step(w1, g)
    expected_w2 = w1 - 0.1 * (0.9 * g + g)
    test("Second step accumulates velocity", w2, expected_w2)
    test("Velocity is maintained (not None)", opt.v is not None, True)

    print("\nExercise 2: Adam")
    opt2  = Adam(lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8)
    w     = np.array([2.0, -1.0])
    g     = np.array([1.0,  1.0])
    m_1   = (1 - 0.9) * g
    v_1   = (1 - 0.999) * g**2
    m_hat = m_1 / (1 - 0.9**1)
    v_hat = v_1 / (1 - 0.999**1)
    w_exp = w - 0.1 * m_hat / (np.sqrt(v_hat) + 1e-8)
    w1    = opt2.step(w.copy(), g)
    test("Adam first step", w1, w_exp, tol=1e-6)
    test("t increments", opt2.t, 1)
    opt2.step(w1, g)
    test("t increments to 2", opt2.t, 2)
    opt3 = Adam(lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8)
    opt3.step(np.zeros(2), np.ones(2))
    test("Bias correction: m_hat > m at t=1",
         float(opt3.m[0] / (1 - 0.9**1)) > float(opt3.m[0]), True)

    print("\nExercise 3: AdamW")
    optW   = AdamW(lr=0.01, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1)
    w_ones = np.array([1.0, 1.0, 1.0])
    g_zero = np.zeros(3)
    w1     = optW.step(w_ones.copy(), g_zero)
    test("Weight decay shrinks weights", all(w1 < 1.0), True)
    opt_wd = AdamW(lr=0.1, weight_decay=0.5)
    w2     = opt_wd.step(np.ones(3), np.zeros(3))
    test("Larger weight_decay → more shrinkage", all(w2 < w1), True)
    opt3w    = AdamW(lr=0.1, beta1=0.0, beta2=0.0, eps=1e-8, weight_decay=0.0)
    w_init   = np.array([1.0])
    g_init   = np.array([1.0])
    w3       = opt3w.step(w_init.copy(), g_init)
    test("Zero weight decay: no decay applied", float(w3[0]) < 1.0, True)

    print("\nExercise 4: warmup_cosine_schedule")
    lr_max, lr_min  = 3e-4, 3e-5
    n_warmup        = 100
    n_total         = 1000
    test("step=0 → lr=0",     warmup_cosine_schedule(0, n_warmup, n_total, lr_max, lr_min), 0.0)
    test("step=50 → lr_max/2",warmup_cosine_schedule(50, n_warmup, n_total, lr_max, lr_min), lr_max/2, tol=1e-8)
    test("step=warmup → lr_max", warmup_cosine_schedule(n_warmup, n_warmup, n_total, lr_max, lr_min), lr_max, tol=1e-8)
    test("step=total → lr_min",  warmup_cosine_schedule(n_total, n_warmup, n_total, lr_max, lr_min), lr_min, tol=1e-9)
    lrs_warmup = [warmup_cosine_schedule(s, n_warmup, n_total, lr_max, lr_min) for s in range(n_warmup+1)]
    test("Monotone increase during warmup",
         all(lrs_warmup[i] <= lrs_warmup[i+1] for i in range(n_warmup)), True)
    lrs_decay = [warmup_cosine_schedule(s, n_warmup, n_total, lr_max, lr_min)
                 for s in range(n_warmup, n_total+1)]
    test("Monotone decrease after warmup",
         all(lrs_decay[i] >= lrs_decay[i+1] for i in range(len(lrs_decay)-1)), True)

    print("\nExercise 5: clip_gradient_norm")
    grads  = [np.array([3.0, 4.0])]
    clipped, norm = clip_gradient_norm(grads, max_norm=1.0)
    test("Returns correct norm (5.0)", norm, 5.0)
    test("Clipped norm = max_norm",    np.linalg.norm(clipped[0]), 1.0)
    test("Direction preserved",        clipped[0], np.array([0.6, 0.8]))
    grads2 = [np.array([0.3, 0.4])]
    clipped2, norm2 = clip_gradient_norm(grads2, max_norm=1.0)
    test("No clipping when norm < max_norm", norm2, 0.5)
    test("Values unchanged when not clipped", clipped2[0], np.array([0.3, 0.4]))
    grads3  = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    clipped3, norm3 = clip_gradient_norm(grads3, max_norm=1.0)
    test("Global norm across multiple arrays", norm3, np.sqrt(2.0), tol=1e-5)
    test("Each array scaled correctly",
         np.allclose(clipped3[0], np.array([1/np.sqrt(2), 0])), True)

    print(f"\n{'='*55}")
    print(f"Score: {passed}/{total}")
    print("=" * 55)


if __name__ == '__main__':
    check()
