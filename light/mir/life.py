"""Опыт существа: жизни, прожитые в песочнице, подряд.

Блуждание — самое простое поведение: существо чаще продолжает то, что делало, иногда
меняет действие наугад. Разумный выбор действий (любопытство) — в curious.py.
Опыт хранится как поток тактов «жизнь за жизнью»; бюджет T — первые T тактов потока.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .world import H, NA, R_ALL, W, Sandbox

ACTION_P = np.array([.06, .17, .17, .10, .20, .10, .10, .10])
LIFE_LEN = 50
CACHE = Path(__file__).resolve().parents[2] / "mir_cache"


def wander(T: int, seed: int, life_len: int = LIFE_LEN, rules: int = R_ALL, cache: bool = True):
    """T тактов блуждания: (S0, A, PH, S1) — кадр до, действие, метроном, кадр после."""
    path = CACHE / f"wander_{seed}_{T}_{life_len}_{rules}.npz"
    if cache and path.exists():
        z = np.load(path)
        return z["S0"], z["A"], z["PH"], z["S1"]
    n = -(-T // life_len)
    env = Sandbox(n, seed, rules=rules)
    rng = np.random.default_rng(seed + 7919)
    prev = rng.choice(NA, n, p=ACTION_P)
    S0 = np.empty((life_len, n, H, W), np.uint8)
    S1 = np.empty_like(S0)
    A = np.empty((life_len, n), np.int64)
    PH = np.empty((life_len, n), np.int64)
    for t in range(life_len):
        a = np.where(rng.random(n) < .5, prev, rng.choice(NA, n, p=ACTION_P))
        prev = a
        g0, ph0, g1 = env.step(a)
        S0[t], S1[t], A[t], PH[t] = g0, g1, a, ph0
    out = tuple(x.swapaxes(0, 1).reshape(n * life_len, *x.shape[2:])[:T] for x in (S0, A, PH, S1))
    out = (out[0], out[1], out[2], out[3])
    if cache:
        CACHE.mkdir(exist_ok=True)
        np.savez(path, S0=out[0], A=out[1], PH=out[2], S1=out[3])
    return out
