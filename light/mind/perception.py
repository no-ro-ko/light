"""Память категорий: как ребёнок сам замечает, что «вот такие» — одного сорта.

Никаких меток. Каждый увиденный объект сравнивается с уже известными образцами
(прототипами). Похож — уточняет прототип (среднее по встреченным). Не похож ни на
что — заводится НОВАЯ категория: структура растёт сама.

Новая категория сначала «пробная»: если за долгое время к ней ничего не добавилось
(скорее всего, это был странный или наполовину заслонённый экземпляр), во сне она
сливается с ближайшей устойчивой. Так исправляются ошибки роста, как у детей,
которые сначала путаются, а потом уточняют свои понятия.
"""
from __future__ import annotations

import numpy as np


class CategoryMemory:
    def __init__(self, dim: int, radius: float = 0.75, stable_after: int = 3, patience: int = 60):
        self.dim = dim
        self.radius = radius              # дальше этого — «что-то новое»
        self.stable_after = stable_after  # столько встреч — категория устойчива
        self.patience = patience          # столько наблюдений ждём пробную категорию
        self.protos: list[np.ndarray] = []
        self.counts: list[int] = []
        self.born: list[int] = []
        self.alive: list[bool] = []
        self.pinned: set[int] = set()     # категории, получившие имя, не сливаются
        self.seen = 0
        self.log: list[str] = []

    def ids(self) -> list[int]:
        return [i for i, a in enumerate(self.alive) if a]

    def distances(self, x: np.ndarray) -> dict[int, float]:
        return {i: float(np.linalg.norm(self.protos[i] - x)) for i in self.ids()}

    def nearest(self, x: np.ndarray, exclude: set[int] = frozenset()) -> tuple[int | None, float]:
        d = {i: v for i, v in self.distances(x).items() if i not in exclude}
        if not d:
            return None, float("inf")
        i = min(d, key=d.get)
        return i, d[i]

    def similarity(self, d: float) -> float:
        return float(np.exp(-(d / self.radius) ** 2))

    @staticmethod
    def partial_distance(x: np.ndarray, p: np.ndarray) -> float:
        """Расстояние с поправкой на заслонение: отсутствие признака — слабое свидетельство
        (его могло быть не видно), а лишний признак — сильное."""
        diff = x - p
        w = np.where(diff < 0, 0.35, 1.0)
        return float(np.sqrt((w * diff ** 2).sum()))

    def categorize(self, x: np.ndarray, learn: bool = True) -> tuple[int, bool]:
        """→ (категория, создана ли новая)."""
        i, d = self.nearest(x)
        if i is not None and d <= self.radius:
            if learn:
                self.counts[i] += 1
                rate = max(1.0 / self.counts[i], 0.02)
                self.protos[i] = self.protos[i] + rate * (x - self.protos[i])
                self.seen += 1
            return i, False
        if i is not None and self.partial_distance(x, self.protos[i]) <= self.radius:
            if learn:                       # похоже, заслонённый представитель: узнаём,
                self.counts[i] += 1         # но образец по нему не уточняем
                self.seen += 1
            return i, False
        if not learn:
            return (i if i is not None else -1), False
        self.protos.append(x.copy())
        self.counts.append(1)
        self.born.append(self.seen)
        self.alive.append(True)
        self.seen += 1
        new = len(self.protos) - 1
        self.log.append(f"новая категория #{new}")
        return new, True

    def add(self, proto: np.ndarray, count: int = 1) -> int:
        self.protos.append(proto.copy())
        self.counts.append(count)
        self.born.append(self.seen)
        self.alive.append(True)
        return len(self.protos) - 1

    def is_stable(self, i: int) -> bool:
        return self.counts[i] >= self.stable_after or i in self.pinned

    def consolidate(self) -> list[tuple[int, int]]:
        """Сон: пробные категории, которые так и не подтвердились, сливаются с ближайшей."""
        merges = []
        for i in self.ids():
            if i in self.pinned or self.seen - self.born[i] < self.patience:
                continue
            j, d = self.nearest(self.protos[i], exclude={i})
            if j is None:
                continue
            unconfirmed = not self.is_stable(i) and d < 2 * self.radius
            satellite = self.counts[i] * 10 < self.counts[j] and d < 1.6 * self.radius
            if unconfirmed or satellite:
                self.counts[j] += self.counts[i]
                self.alive[i] = False
                merges.append((i, j))
                self.log.append(f"категория #{i} слита с #{j} (расстояние {d:.2f})")
        return merges

    def prototype(self, i: int) -> np.ndarray:
        return self.protos[i]
