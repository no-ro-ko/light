"""Обычная сеть для сравнения: окно 5×5 → два скрытых слоя → вероятности сортов.

Это то же самое, что свёрточная сеть с окном 5×5 (одинаковые веса во всех местах мира),
обучаемая обратным распространением ошибки (Adam). Чтобы не быть «соломенным чучелом»:
  • сеть начинает с привычки «клетка остаётся как есть» (пропускная связь с центра);
  • вес примеров (все различные положения поровну / корень из числа повторов / как в жизни),
    скорость обучения и момент остановки выбираются по отложенной части СОБСТВЕННОГО опыта
    (последние 10% тактов), а не по проверкам.
Всё на numpy, без видеокарты.
"""
from __future__ import annotations

import numpy as np

from .patches import NF, NF_BASE, NPOS, POOL4, Samples, keys, patches
from .world import K

NIN = NPOS * K + 8 + 2 + 2 * K   # 280 входов: окно (250) + действие (8) + метроном (2) + «рядом есть X» (20)
CBLOCK = slice(12 * K, 13 * K)   # где в входе лежит «сорт центральной клетки»


def onehot(X: np.ndarray, pooled: bool = True) -> np.ndarray:
    b = len(X)
    Xi = X[:, :NF_BASE].astype(np.int64)
    idx = np.empty((b, NF_BASE), np.int64)
    idx[:, :NPOS] = np.arange(NPOS) * K + Xi[:, :NPOS]
    idx[:, NPOS] = NPOS * K + Xi[:, NPOS]
    idx[:, NPOS + 1] = NPOS * K + 8 + Xi[:, NPOS + 1]
    oh = np.zeros((b, NIN), np.float32)
    oh[np.arange(b)[:, None], idx] = 1.0
    if pooled:
        oh[:, NPOS * K + 10:] = X[:, POOL4:]
    return oh


class PatchMLP:
    def __init__(self, width: int = 64, lr: float = 3e-3, steps: int = 3000, weighting: str = "unit",
                 batch: int = 1024, wd: float = 1e-5, seed: int = 0, eval_every: int = 250,
                 patience: int = 4, val_frac: float = 0.1, pooled: bool = True):
        self.width, self.lr, self.steps, self.weighting = width, lr, steps, weighting
        self.batch, self.wd, self.seed, self.eval_every = batch, wd, seed, eval_every
        self.patience, self.val_frac, self.pooled = patience, val_frac, pooled
        rng = np.random.default_rng(seed)
        w = width
        self.p = {
            "E": (rng.normal(size=(NIN, w)) * np.sqrt(2 / NF_BASE)).astype(np.float32), "b1": np.zeros(w, np.float32),
            "W2": (rng.normal(size=(w, w)) * np.sqrt(2 / w)).astype(np.float32), "b2": np.zeros(w, np.float32),
            "W3": (rng.normal(size=(w, K)) * 0.01).astype(np.float32), "b3": np.zeros(K, np.float32),
            "S": (4.0 * np.eye(K)).astype(np.float32),
        }
        self.info: dict = {}

    @property
    def n_params(self) -> int:
        return int(sum(v.size for v in self.p.values()))

    # ----------------------------------------------------------- прямой и обратный ход
    def _forward(self, oh, p=None):
        p = p or self.p
        z1 = oh @ p["E"] + p["b1"]
        h1 = np.maximum(z1, 0)
        z2 = h1 @ p["W2"] + p["b2"]
        h2 = np.maximum(z2, 0)
        logits = h2 @ p["W3"] + p["b3"] + oh[:, CBLOCK] @ p["S"]
        return z1, h1, z2, h2, logits

    @staticmethod
    def _logsoftmax(z):
        z = z - z.max(1, keepdims=True)
        return z - np.log(np.exp(z).sum(1, keepdims=True))

    def _nll(self, X, y, w, chunk=8192):
        tot = 0.0
        for i in range(0, len(X), chunk):
            lp = self._logsoftmax(self._forward(onehot(X[i:i + chunk], self.pooled))[4])
            tot += float((-lp[np.arange(len(lp)), y[i:i + chunk]] * w[i:i + chunk]).sum())
        return tot / w.sum()

    def fit(self, S0, A, PH, S1):
        T = len(S0)
        nv = max(1, int(T * self.val_frac))
        tr = Samples.collect(S0[:T - nv], A[:T - nv], PH[:T - nv], S1[:T - nv])
        va = Samples.collect(S0[T - nv:], A[T - nv:], PH[T - nv:], S1[T - nv:])
        alpha = {"unit": 0.0, "sqrt": 0.5, "count": 1.0}[self.weighting]
        cum = np.cumsum(tr.w ** alpha)
        vw = va.w ** 0.5
        rng = np.random.default_rng(self.seed + 1)
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        v2 = {k: np.zeros_like(v) for k, v in self.p.items()}
        best, best_p, bad, t = np.inf, {k: x.copy() for k, x in self.p.items()}, 0, 0
        for step in range(1, self.steps + 1):
            idx = np.searchsorted(cum, rng.random(self.batch) * cum[-1])
            idx = np.minimum(idx, len(cum) - 1)
            oh, y = onehot(tr.X[idx], self.pooled), tr.y[idx]
            z1, h1, z2, h2, logits = self._forward(oh)
            pr = np.exp(self._logsoftmax(logits))
            pr[np.arange(self.batch), y] -= 1.0
            d = pr / self.batch
            g = {"W3": h2.T @ d, "b3": d.sum(0), "S": oh[:, CBLOCK].T @ d}
            dz2 = (d @ self.p["W3"].T) * (z2 > 0)
            g["W2"], g["b2"] = h1.T @ dz2, dz2.sum(0)
            dz1 = (dz2 @ self.p["W2"].T) * (z1 > 0)
            g["E"], g["b1"] = oh.T @ dz1, dz1.sum(0)
            lr = self.lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * step / self.steps)))
            for k in self.p:
                gk = g[k] + (self.wd * self.p[k] if k in ("E", "W2", "W3") else 0)
                m[k] = 0.9 * m[k] + 0.1 * gk
                v2[k] = 0.999 * v2[k] + 0.001 * gk * gk
                self.p[k] -= (lr * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(v2[k] / (1 - 0.999 ** step)) + 1e-8)).astype(np.float32)
            if step % self.eval_every == 0 or step == self.steps:
                val = self._nll(va.X, va.y, vw)
                if val < best - 1e-5:
                    best, bad = val, 0
                    best_p = {k: x.copy() for k, x in self.p.items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            t = step
        self.p = best_p
        self.info = {"steps": t, "val": float(best), "unique_train": len(tr), "unique_val": len(va)}
        return self

    # ----------------------------------------------------------- предсказание
    def logp(self, g0, a, ph, chunk: int = 8192):
        n, h, w = g0.shape
        X = patches(g0, a, ph).reshape(-1, NF)
        uk, first, inv = np.unique(keys(X), return_index=True, return_inverse=True)
        U = X[first]
        out = np.empty((len(U), K), np.float32)
        for i in range(0, len(U), chunk):
            out[i:i + chunk] = self._logsoftmax(self._forward(onehot(U[i:i + chunk], self.pooled))[4])
        return out[inv].reshape(n, h, w, K)
