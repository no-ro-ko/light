"""Закон в весах нейрона: местное правило обучения вместо поиска формул.

Нейрон получает на вход логарифмы измеренных величин (так кодируют интенсивность
многие сенсорные нейроны — закон Вебера–Фехнера) и учится по правилу дельты
(Видроу–Хофф / перцептрон): вес меняется пропорционально ошибке и входу — локально,
без обратного распространения через слои.

- Плавание: логистический нейрон на [log масса, log объём] → веса выходят
  противоположными (w, −w): решает ОТНОШЕНИЕ массы к объёму. Понятие «плотность»
  появляется в весах само.
- Количественный закон (маятник): линейный нейрон на [log длина, log масса] →
  веса (½, 0): период ∝ √длина, масса не влияет.
"""
from __future__ import annotations

import math

import numpy as np


class LogisticNeuron:
    def __init__(self, n: int, lr: float = 0.5, seed: int = 0):
        self.w = np.zeros(n + 1)
        self.lr = lr
        self.rng = np.random.default_rng(seed)

    def out(self, x: np.ndarray) -> float:
        return 1 / (1 + math.exp(-float(np.clip(self.w @ np.append(x, 1.0), -30, 30))))

    def learn(self, X: np.ndarray, y: np.ndarray, epochs: int = 300) -> None:
        for _ in range(epochs):
            for i in self.rng.permutation(len(X)):
                xi = np.append(X[i], 1.0)
                self.w += self.lr * (y[i] - self.out(X[i])) * xi      # правило дельты

    def explain(self, names: list[str]) -> str:
        w = self.w[:-1] / abs(self.w[0])
        terms = " · ".join(f"{n}^{v:+.2f}" for n, v in zip(names, w))
        th = math.exp(-self.w[-1] / self.w[0]) if self.w[0] else float("nan")
        return f"решает величина {terms}; граница ≈ {th:.2f}"


class LinearNeuron:
    def __init__(self, n: int, lr: float = 0.05, seed: int = 0):
        self.w = np.zeros(n + 1)
        self.lr = lr
        self.rng = np.random.default_rng(seed)

    def learn(self, X: np.ndarray, y: np.ndarray, epochs: int = 400) -> None:
        for _ in range(epochs):
            for i in self.rng.permutation(len(X)):
                xi = np.append(X[i], 1.0)
                self.w += self.lr * (y[i] - self.w @ xi) * xi        # правило Видроу–Хоффа

    def explain(self, names: list[str], target: str) -> str:
        parts = [f"{n}^{v:.2f}" for n, v in zip(names, self.w[:-1]) if abs(v) > 0.05]
        idle = [n for n, v in zip(names, self.w[:-1]) if abs(v) <= 0.05]
        return (f"{target} ≈ {math.exp(self.w[-1]):.3f} · {' · '.join(parts)}"
                + (f"; не влияет: {', '.join(idle)}" if idle else ""))
