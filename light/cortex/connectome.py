"""Связи между областями («белое вещество»).

- Прямая связь (снизу вверх): выход L5 нижней области → вход L4 верхней.
- Обратная связь (сверху вниз): активность верхней области → апикальные дендриты
  (L1) нижней. Эти связи обучаются по Хеббу: если верхняя картина и нижние колонки
  часто активны вместе, верхняя начинает «подсказывать» нижней, чего ждать.

Так строится иерархия: буквы → слова → фразы, или сенсорная → ассоциативная кора.
"""
from __future__ import annotations

import numpy as np

from .region import CorticalRegion


class Connectome:
    def __init__(self):
        self.regions: dict[str, CorticalRegion] = {}
        self.feeds: dict[str, list[str]] = {}            # верхняя ← список нижних
        self.feedback: dict[tuple[str, str], np.ndarray] = {}

    def add(self, region: CorticalRegion) -> CorticalRegion:
        self.regions[region.name] = region
        self.feeds.setdefault(region.name, [])
        return region

    def connect(self, lower: str, higher: str) -> None:
        lo, hi = self.regions[lower], self.regions[higher]
        self.feeds[higher].append(lower)
        expected = sum(self.regions[n].N for n in self.feeds[higher])
        if hi.input_size != expected:
            raise ValueError(f"{higher}.input_size должен быть {expected}")
        self.feedback[(lower, higher)] = np.zeros((lo.C, hi.N))

    def step(self, inputs: dict[str, np.ndarray | None], learn: bool = True) -> dict[str, float]:
        surprise = {}
        for name, region in self.regions.items():         # снизу вверх (порядок добавления)
            if self.feeds[name]:
                inp = np.concatenate([self.regions[n].output() for n in self.feeds[name]])
            else:
                inp = inputs.get(name)
            surprise[name] = region.step(inp, learn)
        for (lower, higher), w in self.feedback.items():  # сверху вниз
            lo, hi = self.regions[lower], self.regions[higher]
            if learn:
                w *= 0.98
                w += 0.02 * np.outer(lo.active_cols, hi.active)
            bias = w @ hi.active
            if bias.max() > 0:
                lo.set_apical(0.5 * bias / bias.max())
        return surprise
