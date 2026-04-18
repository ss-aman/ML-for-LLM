"""
Module 09 — RNNs to Transformers: Sequential Models from Scratch
=================================================================
Implements:
  - Simple RNN from scratch
  - Vanishing gradient demonstration (sequence length 50)
  - LSTM from scratch (all 4 gates, cell state)
  - Comparison: RNN vs LSTM on a long-sequence memory task

The memory task: at step 0 a special "signal" token is shown.
At step T-1 the network must output that signal.
This requires remembering a single value across T steps.

Backend analogy: Think of the memory task as a session-scoped event.
At the start of a session the client sends an important identifier.
20 requests later, a handler must recall that identifier.
An RNN is like a server with a single-byte session cookie: the byte
gets overwritten by each request. An LSTM is like a session store with
separate short-term and long-term fields and an explicit eviction policy.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Utility: sigmoid and tanh with numerically stable implementations
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    # Clamp to avoid overflow in exp
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


# ---------------------------------------------------------------------------
# Simple RNN (Elman network)
# ---------------------------------------------------------------------------

class SimpleRNN:
    """
    Single-layer Elman RNN:
        h_t = tanh(W_h @ h_{t-1} + W_x @ x_t + b)
        y_t = W_out @ h_t + b_out    (only computed at the final step)

    Parameters:
        input_size:  dimension of each input token x_t
        hidden_size: dimension of hidden state h_t

    Backend analogy: A stateful request handler. Each call to step() is one
    incoming request; the hidden state is the session object updated in place.
    The session is fixed-size regardless of how many requests have been seen.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        scale = 0.1
        self.W_h = rng.standard_normal((hidden_size, hidden_size)) * scale
        self.W_x = rng.standard_normal((hidden_size, input_size)) * scale
        self.b = np.zeros((hidden_size, 1))
        self.W_out = rng.standard_normal((output_size, hidden_size)) * scale
        self.b_out = np.zeros((output_size, 1))

        self.hidden_size = hidden_size
        self.input_size = input_size

    def forward(self, xs: list, h0: np.ndarray = None) -> tuple:
        """
        xs: list of T input vectors, each shape (input_size, 1)
        Returns:
          - final output y (output_size, 1)
          - list of all hidden states [h_0, h_1, ..., h_T] for backprop
          - list of all pre-activation values (for backprop through tanh)
        """
        T = len(xs)
        h = h0 if h0 is not None else np.zeros((self.hidden_size, 1))

        hs = [h]          # h_0 through h_T
        pre_acts = []     # pre-activation values (before tanh)

        for t in range(T):
            z = self.W_h @ h + self.W_x @ xs[t] + self.b
            pre_acts.append(z)
            h = tanh(z)
            hs.append(h)

        y = self.W_out @ hs[-1] + self.b_out
        return y, hs, pre_acts

    def backward(self, xs: list, hs: list, pre_acts: list,
                 d_y: np.ndarray) -> dict:
        """
        Backpropagation Through Time (BPTT).

        d_y: gradient of loss w.r.t. final output y, shape (output_size, 1)
        Returns:
          - dict of gradients for all parameters
          - list of gradient norms at each time step (for vanishing-gradient plot)
        """
        T = len(xs)

        # Gradients for output layer
        dW_out = d_y @ hs[-1].T
        db_out = d_y.copy()
        d_h = self.W_out.T @ d_y   # gradient flows from output into final hidden state

        # Accumulated parameter gradients
        dW_h = np.zeros_like(self.W_h)
        dW_x = np.zeros_like(self.W_x)
        db = np.zeros_like(self.b)

        grad_norms = []  # record gradient norm entering each time step

        # Backprop through time (from T-1 down to 0)
        for t in range(T - 1, -1, -1):
            grad_norms.append(float(np.linalg.norm(d_h)))

            # Gradient through tanh: d(tanh)/dz = 1 - tanh(z)^2
            d_z = d_h * (1.0 - hs[t + 1] ** 2)

            # Accumulate weight gradients
            dW_h += d_z @ hs[t].T
            dW_x += d_z @ xs[t].T
            db += d_z

            # Gradient for previous hidden state
            d_h = self.W_h.T @ d_z

        grad_norms.reverse()  # time step 0 first

        return {
            "dW_h": dW_h, "dW_x": dW_x, "db": db,
            "dW_out": dW_out, "db_out": db_out,
        }, grad_norms

    def update(self, grads: dict, lr: float):
        self.W_h -= lr * grads["dW_h"]
        self.W_x -= lr * grads["dW_x"]
        self.b -= lr * grads["db"]
        self.W_out -= lr * grads["dW_out"]
        self.b_out -= lr * grads["db_out"]


# ---------------------------------------------------------------------------
# LSTM (Long Short-Term Memory)
# ---------------------------------------------------------------------------

class LSTM:
    """
    Single-layer LSTM:
        combined = concat(h_{t-1}, x_t)       # (hidden+input, 1)
        f_t = sigmoid(W_f @ combined + b_f)   # forget gate
        i_t = sigmoid(W_i @ combined + b_i)   # input gate
        g_t = tanh   (W_g @ combined + b_g)   # cell gate (candidate values)
        o_t = sigmoid(W_o @ combined + b_o)   # output gate
        c_t = f_t * c_{t-1} + i_t * g_t       # cell state update
        h_t = o_t * tanh(c_t)                  # hidden state

    Backend analogy: A session cache with explicit eviction/admission/read
    policies. Each gate is an independent policy function that decides
    "how much" to forget, admit, or expose at each step.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        combined = input_size + hidden_size
        scale = 0.1

        # One weight matrix per gate (forget, input, cell/gate, output)
        self.W_f = rng.standard_normal((hidden_size, combined)) * scale
        self.W_i = rng.standard_normal((hidden_size, combined)) * scale
        self.W_g = rng.standard_normal((hidden_size, combined)) * scale
        self.W_o = rng.standard_normal((hidden_size, combined)) * scale

        # Forget gate bias initialised to 1 — a common trick so the gate
        # starts open (remember everything) rather than closed
        self.b_f = np.ones((hidden_size, 1))
        self.b_i = np.zeros((hidden_size, 1))
        self.b_g = np.zeros((hidden_size, 1))
        self.b_o = np.zeros((hidden_size, 1))

        self.W_out = rng.standard_normal((output_size, hidden_size)) * scale
        self.b_out = np.zeros((output_size, 1))

        self.hidden_size = hidden_size
        self.input_size = input_size

    def forward(self, xs: list, h0=None, c0=None) -> tuple:
        """
        Forward pass through all T time steps.
        Returns final output y and all cached states for backprop.
        """
        T = len(xs)
        h = h0 if h0 is not None else np.zeros((self.hidden_size, 1))
        c = c0 if c0 is not None else np.zeros((self.hidden_size, 1))

        # Cache everything needed for backprop
        cache = {"hs": [h], "cs": [c], "xs": xs,
                 "fs": [], "is_": [], "gs": [], "os": []}

        for t in range(T):
            combined = np.vstack([h, xs[t]])   # (hidden+input, 1)

            f = sigmoid(self.W_f @ combined + self.b_f)   # forget gate
            i = sigmoid(self.W_i @ combined + self.b_i)   # input gate
            g = tanh(self.W_g @ combined + self.b_g)      # candidate values
            o = sigmoid(self.W_o @ combined + self.b_o)   # output gate

            c = f * cache["cs"][-1] + i * g               # update cell state
            h = o * tanh(c)                                # update hidden state

            cache["fs"].append(f)
            cache["is_"].append(i)
            cache["gs"].append(g)
            cache["os"].append(o)
            cache["hs"].append(h)
            cache["cs"].append(c)

        y = self.W_out @ h + self.b_out
        return y, cache

    def backward(self, cache: dict, d_y: np.ndarray) -> tuple:
        """
        Backprop through LSTM. Returns parameter gradients and per-step gradient norms.
        """
        T = len(cache["xs"])
        xs = cache["xs"]
        hs = cache["hs"]
        cs = cache["cs"]
        fs, is_, gs, os = cache["fs"], cache["is_"], cache["gs"], cache["os"]

        # Gradients for output layer
        dW_out = d_y @ hs[-1].T
        db_out = d_y.copy()

        # Initialise gradients
        dW_f = np.zeros_like(self.W_f)
        dW_i = np.zeros_like(self.W_i)
        dW_g = np.zeros_like(self.W_g)
        dW_o = np.zeros_like(self.W_o)
        db_f = np.zeros_like(self.b_f)
        db_i = np.zeros_like(self.b_i)
        db_g = np.zeros_like(self.b_g)
        db_o = np.zeros_like(self.b_o)

        d_h = self.W_out.T @ d_y    # gradient from output into final h_T
        d_c = np.zeros_like(cs[0])  # gradient through cell state

        grad_norms = []

        for t in range(T - 1, -1, -1):
            grad_norms.append(float(np.linalg.norm(d_h)))

            f, i, g, o = fs[t], is_[t], gs[t], os[t]
            c, c_prev = cs[t + 1], cs[t]
            h = hs[t + 1]

            # Gradient through h_t = o * tanh(c)
            d_o = d_h * tanh(c)
            d_c_from_h = d_h * o * (1.0 - tanh(c) ** 2)

            # Accumulate cell state gradient
            d_c = d_c_from_h + d_c

            # Gradient through c_t = f * c_{t-1} + i * g
            d_f = d_c * c_prev
            d_i = d_c * g
            d_g = d_c * i
            d_c_prev = d_c * f   # gradient into previous cell state

            # Gradient through gate activations
            d_zf = d_f * f * (1.0 - f)       # through sigmoid
            d_zi = d_i * i * (1.0 - i)
            d_zg = d_g * (1.0 - g ** 2)      # through tanh
            d_zo = d_o * o * (1.0 - o)

            combined = np.vstack([hs[t], xs[t]])

            # Accumulate weight gradients
            dW_f += d_zf @ combined.T
            dW_i += d_zi @ combined.T
            dW_g += d_zg @ combined.T
            dW_o += d_zo @ combined.T
            db_f += d_zf
            db_i += d_zi
            db_g += d_zg
            db_o += d_zo

            # Gradient for previous hidden state (from all four gates)
            H = self.hidden_size
            d_combined = (self.W_f.T @ d_zf + self.W_i.T @ d_zi +
                          self.W_g.T @ d_zg + self.W_o.T @ d_zo)
            d_h = d_combined[:H]     # upper part of combined = h_{t-1}
            d_c = d_c_prev

        grad_norms.reverse()

        grads = {
            "dW_f": dW_f, "dW_i": dW_i, "dW_g": dW_g, "dW_o": dW_o,
            "db_f": db_f, "db_i": db_i, "db_g": db_g, "db_o": db_o,
            "dW_out": dW_out, "db_out": db_out,
        }
        return grads, grad_norms

    def update(self, grads: dict, lr: float):
        for key in ["W_f", "W_i", "W_g", "W_o", "b_f", "b_i", "b_g", "b_o",
                    "W_out", "b_out"]:
            setattr(self, key, getattr(self, key) - lr * grads["d" + key])


# ---------------------------------------------------------------------------
# Task generator: remember the first token after T steps
# ---------------------------------------------------------------------------

def make_memory_task(T: int, n_samples: int, seed: int = 0) -> tuple:
    """
    Generate sequences for the long-range memory task.

    Each sequence has T steps.
    - Step 0: input is either [1] (signal=1) or [-1] (signal=0), rest of vector is zeros
    - Steps 1..T-1: input is zeros (just noise)
    - Target: the network must output the signal seen at step 0

    Backend analogy: At request #0 of a session, a client sends a flag.
    At request #T, a handler must recall what flag was set at request #0,
    ignoring all the no-op requests in between.
    """
    rng = np.random.default_rng(seed)
    input_size = 2   # [signal_value, step_indicator]
    xs_all = []
    ys_all = []

    for _ in range(n_samples):
        signal = rng.integers(0, 2)   # 0 or 1
        xs = []
        for t in range(T):
            if t == 0:
                xs.append(np.array([[float(signal)], [1.0]]))   # signal at step 0
            else:
                xs.append(np.array([[0.0], [0.0]]))             # silence
        xs_all.append(xs)
        ys_all.append(np.array([[float(signal)]]))

    return xs_all, ys_all, input_size


# ---------------------------------------------------------------------------
# Training loop (generic — works for both RNN and LSTM)
# ---------------------------------------------------------------------------

def train_model(model, xs_all, ys_all, epochs: int, lr: float,
                model_type: str = "rnn") -> tuple:
    """
    Train the given model. Returns loss history and average final gradient norms.

    model_type: "rnn" or "lstm" — controls which forward/backward API to use.
    """
    n = len(xs_all)
    losses = []
    all_grad_norms = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_grad_norms = None

        for i in range(n):
            xs = xs_all[i]
            y_true = ys_all[i]

            # Forward
            if model_type == "rnn":
                y_pred, hs, pre_acts = model.forward(xs)
                d_y = 2 * (y_pred - y_true) / n
                grads, grad_norms = model.backward(xs, hs, pre_acts, d_y)
            else:  # lstm
                y_pred, cache = model.forward(xs)
                d_y = 2 * (y_pred - y_true) / n
                grads, grad_norms = model.backward(cache, d_y)

            loss = float(np.mean((y_pred - y_true) ** 2))
            epoch_loss += loss

            model.update(grads, lr)

            if epoch_grad_norms is None:
                epoch_grad_norms = grad_norms
            else:
                epoch_grad_norms = [a + b for a, b in zip(epoch_grad_norms, grad_norms)]

        avg_loss = epoch_loss / n
        losses.append(avg_loss)
        all_grad_norms.append([g / n for g in epoch_grad_norms])

        if epoch % (epochs // 5) == 0:
            print(f"  Epoch {epoch:4d} | Loss: {avg_loss:.5f}")

    return losses, all_grad_norms


# ---------------------------------------------------------------------------
# Demonstration 1: Vanishing gradient in a plain RNN
# ---------------------------------------------------------------------------

def demo_vanishing_gradient(seq_len: int = 50):
    """
    Show that gradient norms decay exponentially as we go back in time in a plain RNN.

    Backend analogy: Like passing an amplitude through 50 multiplications by 0.9.
    The signal at step 1 is barely distinguishable from noise by the time it
    arrives back at step 50.
    """
    print("=" * 60)
    print(f"DEMO 1: Vanishing gradients in RNN (sequence length = {seq_len})")
    print("=" * 60)

    rnn = SimpleRNN(input_size=2, hidden_size=16, output_size=1, seed=42)
    xs, ys, _ = make_memory_task(T=seq_len, n_samples=1, seed=0)

    y_pred, hs, pre_acts = rnn.forward(xs[0])
    d_y = 2 * (y_pred - ys[0])
    _, grad_norms = rnn.backward(xs[0], hs, pre_acts, d_y)

    print(f"\n  Gradient norm at each time step (step 0 = earliest):")
    for t, gn in enumerate(grad_norms):
        bar = "#" * max(0, int(gn * 500))
        print(f"  Step {t:2d}: {gn:.2e}  {bar}")

    ratio = grad_norms[0] / (grad_norms[-1] + 1e-30)
    print(f"\n  Gradient at step 0 is {ratio:.1f}x smaller than at step {seq_len-1}")
    print(f"  → Early tokens receive negligible gradient signal\n")

    return grad_norms


# ---------------------------------------------------------------------------
# Demonstration 2: RNN vs LSTM on the memory task
# ---------------------------------------------------------------------------

def demo_rnn_vs_lstm(T: int = 20, n_samples: int = 100, epochs: int = 300):
    """
    Train both an RNN and an LSTM on the memory task (remember token from step 0).
    The RNN should struggle; the LSTM should solve it.

    Backend analogy: Both systems must maintain a session flag for T requests.
    The RNN overwrites its session with each request. The LSTM explicitly chooses
    to keep the flag set (forget_gate ≈ 1, input_gate ≈ 0 for that slot) until
    the final step.
    """
    print("=" * 60)
    print(f"DEMO 2: RNN vs LSTM — remember token from step 0 (T={T})")
    print("=" * 60)

    xs_train, ys_train, input_size = make_memory_task(T, n_samples, seed=1)

    hidden = 16

    print("\n  Training RNN...")
    rnn = SimpleRNN(input_size=input_size, hidden_size=hidden, output_size=1, seed=0)
    rnn_losses, rnn_grads = train_model(rnn, xs_train, ys_train,
                                        epochs=epochs, lr=0.01, model_type="rnn")

    print("\n  Training LSTM...")
    lstm = LSTM(input_size=input_size, hidden_size=hidden, output_size=1, seed=0)
    lstm_losses, lstm_grads = train_model(lstm, xs_train, ys_train,
                                          epochs=epochs, lr=0.01, model_type="lstm")

    print(f"\n  RNN  final loss: {rnn_losses[-1]:.4f}  (chance ≈ 0.25)")
    print(f"  LSTM final loss: {lstm_losses[-1]:.4f}  (target < 0.05)")

    return rnn_losses, lstm_losses, rnn_grads, lstm_grads


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_all(vanish_norms, rnn_losses, lstm_losses, rnn_grads, lstm_grads):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Vanishing gradient
    axes[0].semilogy(vanish_norms, marker='o', markersize=3, color='crimson')
    axes[0].set_title("Vanishing Gradient in RNN\n(sequence length 50)")
    axes[0].set_xlabel("Time step (0 = earliest)")
    axes[0].set_ylabel("Gradient norm (log scale)")
    axes[0].grid(True, alpha=0.3)

    # 2. RNN vs LSTM loss curves
    axes[1].plot(rnn_losses, label="RNN", color='crimson')
    axes[1].plot(lstm_losses, label="LSTM", color='steelblue')
    axes[1].axhline(0.25, color='gray', linestyle='--', alpha=0.5, label='random chance')
    axes[1].set_title("RNN vs LSTM\non Long-Range Memory Task")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Gradient norm at step 0 across training (shows LSTM maintains signal)
    rnn_step0 = [g[0] for g in rnn_grads]
    lstm_step0 = [g[0] for g in lstm_grads]
    axes[2].semilogy(rnn_step0, label="RNN grad@step0", color='crimson')
    axes[2].semilogy(lstm_step0, label="LSTM grad@step0", color='steelblue')
    axes[2].set_title("Gradient at Step 0 During Training\n(LSTM maintains signal, RNN loses it)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Gradient norm (log scale)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("module-09-rnn-lstm.png", dpi=120)
    plt.show()
    print("Plot saved to module-09-rnn-lstm.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    vanish_norms = demo_vanishing_gradient(seq_len=50)
    rnn_losses, lstm_losses, rnn_grads, lstm_grads = demo_rnn_vs_lstm(
        T=20, n_samples=100, epochs=300
    )
    plot_all(vanish_norms, rnn_losses, lstm_losses, rnn_grads, lstm_grads)
