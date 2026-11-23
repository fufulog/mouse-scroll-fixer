"""NumPy GRU jitter model: streaming inference, batched forward/backward, Adam.

Same math and parameter layout as torch.nn.GRU (gate order r, z, n) plus a
linear head, so torch checkpoints convert 1:1. No torch needed at runtime or
for fine-tuning.
"""
import math

import numpy as np

from dataset import FeatureExtractor

NAMES = ("w_ih", "w_hh", "b_ih", "b_hh", "w_out", "b_out")


def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))


def load(path):
    with np.load(path) as z:
        return {k: z[k].astype(np.float64) for k in NAMES}


def save(path, params, **meta):
    np.savez(path, **params, **{f"meta_{k}": np.array(v) for k, v in meta.items()})


def from_torch_state(sd):
    g = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in sd.items()}
    return {"w_ih": g["gru.weight_ih_l0"], "w_hh": g["gru.weight_hh_l0"],
            "b_ih": g["gru.bias_ih_l0"], "b_hh": g["gru.bias_hh_l0"],
            "w_out": g["head.weight"][0], "b_out": g["head.bias"].reshape(())}


class Streaming:
    """One report at a time, for the live filter."""

    def __init__(self, params):
        self.p = params
        self.h = np.zeros(params["w_hh"].shape[1])
        self.fe = FeatureExtractor()

    def step(self, t_ns, v):
        x = np.array(self.fe.step(t_ns, v))
        p, H = self.p, len(self.h)
        gi, gh = p["w_ih"] @ x + p["b_ih"], p["w_hh"] @ self.h + p["b_hh"]
        r = sigmoid(gi[:H] + gh[:H])
        z = sigmoid(gi[H:2 * H] + gh[H:2 * H])
        n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
        self.h = (1 - z) * n + z * self.h
        return float(sigmoid(p["w_out"] @ self.h + p["b_out"]))


def forward(p, x):
    """x: (B, T, D) -> logits (B, T), cache for backward."""
    B, T, _ = x.shape
    H = p["w_hh"].shape[1]
    h = np.zeros((B, H))
    cache, logits = [], np.empty((B, T))
    gi_all = x @ p["w_ih"].T + p["b_ih"]
    for t in range(T):
        gi, gh = gi_all[:, t], h @ p["w_hh"].T + p["b_hh"]
        r = sigmoid(gi[:, :H] + gh[:, :H])
        z = sigmoid(gi[:, H:2 * H] + gh[:, H:2 * H])
        n = np.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
        h_new = (1 - z) * n + z * h
        cache.append((h, r, z, n, gh[:, 2 * H:]))
        h = h_new
        logits[:, t] = h @ p["w_out"] + p["b_out"]
    return logits, (x, cache, h)


def loss_and_grads(p, x, y, w, pos_weight):
    """Weighted BCE-with-logits (torch semantics), normalised by sum(w)."""
    logits, (x, cache, h_last) = forward(p, x)
    s = sigmoid(logits)
    wsum = w.sum()
    loss = -(pos_weight * y * np.log(s + 1e-12) + (1 - y) * np.log(1 - s + 1e-12))
    loss = float((loss * w).sum() / wsum)
    dlogit = (s * (pos_weight * y + 1 - y) - pos_weight * y) * w / wsum  # (B, T)

    B, T, _ = x.shape
    H = p["w_hh"].shape[1]
    g = {k: np.zeros_like(v) for k, v in p.items()}
    dh = np.zeros((B, H))
    hs = [c[0] for c in cache[1:]] + [h_last]  # h after each step
    for t in reversed(range(T)):
        h_prev, r, z, n, ghn = cache[t]
        g["w_out"] += dlogit[:, t] @ hs[t]
        g["b_out"] += dlogit[:, t].sum()
        dh = dh + np.outer(dlogit[:, t], p["w_out"])
        dz = dh * (h_prev - n)
        dn = dh * (1 - z)
        dh_prev = dh * z
        da_n = dn * (1 - n * n)
        dghn = da_n * r
        da_r = da_n * ghn * r * (1 - r)
        da_z = dz * z * (1 - z)
        da_i = np.concatenate([da_r, da_z, da_n], axis=1)
        da_h = np.concatenate([da_r, da_z, dghn], axis=1)
        g["w_ih"] += da_i.T @ x[:, t]
        g["b_ih"] += da_i.sum(0)
        g["w_hh"] += da_h.T @ h_prev
        g["b_hh"] += da_h.sum(0)
        dh = dh_prev + da_h @ p["w_hh"]
    return loss, g


class Adam:
    def __init__(self, params, lr, clip=1.0, wd=0.0):
        self.lr, self.clip, self.wd, self.t = lr, clip, wd, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, b1=0.9, b2=0.999, eps=1e-8):
        norm = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
        scale = min(1.0, self.clip / (norm + 1e-12))
        self.t += 1
        for k in params:
            if self.wd:
                params[k] *= 1.0 - self.lr * self.wd
            gk = grads[k] * scale
            self.m[k] = b1 * self.m[k] + (1 - b1) * gk
            self.v[k] = b2 * self.v[k] + (1 - b2) * gk * gk
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + eps)
