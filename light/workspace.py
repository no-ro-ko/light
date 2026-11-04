"""Глобальное рабочее пространство: узкая «сцена внимания».

Кандидаты конкурируют за место; победители рассылаются всем узлам.
Привыкание снижает вес повторяющихся мыслей, чтобы не было зацикливания.
"""
from __future__ import annotations

from collections import Counter

from .signals import Signal, topic_words


def _key(s: Signal) -> str:
    return " ".join(s.text().lower().split())[:120]


class Workspace:
    def __init__(self, capacity: int = 2, habituation: float = 0.5, threshold: float = 0.3):
        self.capacity = capacity
        self.threshold = threshold  # ниже порога мысль не доходит до сцены
        self.buffer_ticks = 3       # сколько тактов внешний стимул/действие ждёт очереди
        self.habituation = habituation
        self.candidates: list[Signal] = []
        self.stream: list[Signal] = []  # всё, что побывало на сцене («поток сознания»)
        self._seen: Counter[str] = Counter()
        self.inhibited: list[list] = []  # [тема, сколько тактов ещё подавлять]

    def propose(self, signals: list[Signal]) -> None:
        self.candidates.extend(signals)

    def inhibit(self, topic: set[str], ticks: int = 8) -> None:
        """Подавить тему (так «я-модель» выводит внимание из зацикливания)."""
        self.inhibited.append([set(topic), ticks])

    def is_inhibited(self, text: str) -> bool:
        words = topic_words(text)
        return any(words and len(words & t) / len(words | t) > 0.4 for t, _ in self.inhibited)

    def score(self, s: Signal) -> float:
        # привыкание — только к собственным мыслям; внешний вопрос не игнорируется
        repeats = 0 if s.kind in ("percept", "action") else self._seen[_key(s)]
        base = s.salience * (self.habituation ** repeats)
        if s.kind not in ("percept", "self", "action") and self.is_inhibited(s.text()):
            base *= 0.2
        return base

    def select(self) -> list[Signal]:
        """Базальные ганглии внимания: отобрать лучших, остальных забыть."""
        ranked = sorted(self.candidates, key=self.score, reverse=True)
        winners = [s for s in ranked[: self.capacity] if self.score(s) >= self.threshold]
        # Сенсорный буфер: проигравшие восприятия и готовые действия не забываются сразу,
        # а ждут своей очереди несколько тактов (как иконическая/эхоическая память).
        kept = []
        for s in ranked:
            if s in winners or s.kind not in ("percept", "action"):
                continue
            s.data["waited"] = s.data.get("waited", 0) + 1
            if s.data["waited"] <= self.buffer_ticks:
                kept.append(s)
        self.candidates = kept
        for s in winners:
            self._seen[_key(s)] += 1
        self.stream.extend(winners)
        for item in self.inhibited:
            item[1] -= 1
        self.inhibited = [i for i in self.inhibited if i[1] > 0]
        return winners

    def recent(self, n: int = 6) -> list[Signal]:
        return self.stream[-n:]
