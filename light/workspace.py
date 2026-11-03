"""Глобальное рабочее пространство: узкая «сцена внимания».

Кандидаты конкурируют за место; победители рассылаются всем узлам.
Привыкание снижает вес повторяющихся мыслей, чтобы не было зацикливания.
"""
from __future__ import annotations

from collections import Counter

from .signals import Signal


def _key(s: Signal) -> str:
    return " ".join(s.text().lower().split())[:120]


class Workspace:
    def __init__(self, capacity: int = 2, habituation: float = 0.5, threshold: float = 0.3):
        self.capacity = capacity
        self.threshold = threshold  # ниже порога мысль не доходит до сцены
        self.habituation = habituation
        self.candidates: list[Signal] = []
        self.stream: list[Signal] = []  # всё, что побывало на сцене («поток сознания»)
        self._seen: Counter[str] = Counter()

    def propose(self, signals: list[Signal]) -> None:
        self.candidates.extend(signals)

    def score(self, s: Signal) -> float:
        return s.salience * (self.habituation ** self._seen[_key(s)])

    def select(self) -> list[Signal]:
        """Базальные ганглии внимания: отобрать лучших, остальных забыть."""
        ranked = sorted(self.candidates, key=self.score, reverse=True)
        winners = [s for s in ranked[: self.capacity] if self.score(s) >= self.threshold]
        self.candidates = []
        for s in winners:
            self._seen[_key(s)] += 1
        self.stream.extend(winners)
        return winners

    def recent(self, n: int = 6) -> list[Signal]:
        return self.stream[-n:]
