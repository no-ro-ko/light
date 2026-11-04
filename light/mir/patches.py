"""Окна вокруг клеток и сжатие повторов (общее для всех обучающихся моделей).

Для каждой клетки берётся окно 5×5 вокруг неё (за краем мира — «земля»), действие «я» и
метроном: всего 27 чисел. Мир местный (до 2 клеток), поэтому по этим 27 числам следующее
состояние центральной клетки определяется однозначно. Одинаковых окон в опыте огромное
число (пустое небо!), поэтому повторы сворачиваются в «окно + сколько раз встретилось».
"""
from __future__ import annotations

import numpy as np

from .world import GROUND, K

RAD = 2
NPOS = 25            # клеток в окне
CENTER = 12          # номер центральной клетки
NF = 27              # 25 клеток + действие + метроном
IA, IP = 25, 26      # где в векторе действие и метроном
OFFS = [(dy, dx) for dy in range(-RAD, RAD + 1) for dx in range(-RAD, RAD + 1)]
DIST = np.array([max(abs(dy), abs(dx)) for dy, dx in OFFS] + [3, 3])  # «расстояние» признака

_MUL = (np.random.default_rng(12345).integers(1, 2 ** 62, NF, dtype=np.uint64) * 2 + 1).astype(np.uint64)


def patches(g: np.ndarray, a: np.ndarray, ph: np.ndarray) -> np.ndarray:
    """(N,H,W) -> (N,H,W,27) uint8."""
    n, h, w = g.shape
    P = np.pad(g, ((0, 0), (RAD, RAD), (RAD, RAD)), constant_values=GROUND)
    X = np.empty((n, h, w, NF), np.uint8)
    p = 0
    for dy in range(-RAD, RAD + 1):
        for dx in range(-RAD, RAD + 1):
            X[..., p] = P[:, RAD + dy:RAD + dy + h, RAD + dx:RAD + dx + w]
            p += 1
    X[..., IA] = np.asarray(a)[:, None, None]
    X[..., IP] = np.asarray(ph)[:, None, None]
    return X


def changed_bits(g0: np.ndarray, g1: np.ndarray) -> np.ndarray:
    """Для каждой клетки: какие клетки её окна изменились за такт (25 бит) -> (N,H,W) int32."""
    n, h, w = g0.shape
    ch = np.pad(g0 != g1, ((0, 0), (RAD, RAD), (RAD, RAD)))
    out = np.zeros((n, h, w), np.int32)
    p = 0
    for dy in range(-RAD, RAD + 1):
        for dx in range(-RAD, RAD + 1):
            out |= ch[:, RAD + dy:RAD + dy + h, RAD + dx:RAD + dx + w].astype(np.int32) << p
            p += 1
    return out


def keys(X: np.ndarray) -> np.ndarray:
    """Хеш окна (uint64); переполнение при умножении — намеренное."""
    with np.errstate(over="ignore"):
        return (X.astype(np.uint64) * _MUL).sum(-1, dtype=np.uint64)


class Samples:
    """Уникальные (окно → что стало с центром) с числом повторов и порядком первого появления."""

    def __init__(self, X, y, w, chg, first):
        self.X, self.y, self.w, self.chg, self.first = X, y, w, chg, first

    def __len__(self):
        return len(self.y)

    @staticmethod
    def collect(S0, A, PH, S1, chunk: int = 1500) -> "Samples":
        parts = []
        T = len(S0)
        for t0 in range(0, T, chunk):
            s0, a, ph, s1 = S0[t0:t0 + chunk], A[t0:t0 + chunk], PH[t0:t0 + chunk], S1[t0:t0 + chunk]
            X = patches(s0, a, ph).reshape(-1, NF)
            y = s1.reshape(-1)
            c = changed_bits(s0, s1).reshape(-1)
            k = keys(X) * np.uint64(K) + y.astype(np.uint64)
            _, i, cnt = np.unique(k, return_index=True, return_counts=True)
            order = np.argsort(i)  # по времени первого появления
            i, cnt = i[order], cnt[order]
            parts.append((k[i], X[i], y[i], cnt, c[i], t0 * s0.shape[1] * s0.shape[2] + i))
        k = np.concatenate([p[0] for p in parts])
        X = np.concatenate([p[1] for p in parts])
        y = np.concatenate([p[2] for p in parts])
        cnt = np.concatenate([p[3] for p in parts])
        c = np.concatenate([p[4] for p in parts])
        first = np.concatenate([p[5] for p in parts])
        _, i, inv = np.unique(k, return_index=True, return_inverse=True)
        order = np.argsort(i)
        rank = np.empty(len(order), np.int64)
        rank[order] = np.arange(len(order))
        w = np.bincount(rank[inv], weights=cnt, minlength=len(order)).astype(np.float64)
        i = i[order]
        return Samples(X[i], y[i], w, c[i], first[i])
