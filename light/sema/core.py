"""СЕМА — нейросеть смыслов. Основа: распределённые коды и ассоциативная память.

Понятие — это не ячейка таблицы, а рисунок активности большой популяции нейронов:
вектор из D значений ±1 (у каждого понятия свой случайный рисунок). Такие рисунки:
- почти не похожи друг на друга (в большой размерности случайные векторы ортогональны);
- устойчивы к повреждению: если часть нейронов умерла, рисунок всё ещё узнаётся.

Три нейронные операции (как в векторно-символических архитектурах Канервы, Плейта,
Элиасмита — сочетание здесь своё):
- СВЯЗЫВАНИЕ «роль ⊗ участник» — поэлементное умножение (мультипликативное
  взаимодействие на дендритах): агент⊗лиса не похож ни на агента, ни на лису;
- НАЛОЖЕНИЕ — сумма с порогом: событие = агент⊗лиса + действие⊗хвалить + объект⊗ворона;
- РАЗВЯЗЫВАНИЕ — то же умножение: событие⊗агент ≈ лиса (с шумом) → ОЧИСТКА в
  аттракторной памяти понятий (ближайший сохранённый рисунок).

Память событий — плотная ассоциативная сеть (современная сеть Хопфилда): каждый
«нейрон памяти» хранит один рисунок в своих синапсах; запрос (частичный намёк)
активирует похожие — и восстанавливается целое. Обучение — одноразовое, локальное
(запомнить = записать рисунок в синапсы), без градиентного спуска.

Повреждение: часть нейронов можно «убить» — все сравнения идут только по живым.
"""
from __future__ import annotations

import numpy as np

ROLES = ("агент", "действие", "объект", "отрицание")


class Space:
    def __init__(self, dim: int = 2048, seed: int = 0):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.codes: dict[str, np.ndarray] = {}
        self.alive = np.ones(dim, dtype=bool)
        self._matrix: tuple[list[str], np.ndarray] | None = None
        for r in ROLES:
            self.code("#" + r)
        self.code("#НЕ")

    # ---------- коды ----------
    def code(self, name: str) -> np.ndarray:
        if name not in self.codes:
            self.codes[name] = np.where(self.rng.random(self.dim) < 0.5, -1.0, 1.0).astype(np.float32)
            self._matrix = None
        return self.codes[name]

    def role(self, r: str) -> np.ndarray:
        return self.codes["#" + r]

    @staticmethod
    def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a * b

    def bundle(self, vs: list[np.ndarray]) -> np.ndarray:
        s = np.sum(vs, axis=0) + self.rng.normal(0, 0.01, self.dim)   # ничьи — случайно
        return np.where(s >= 0, 1.0, -1.0).astype(np.float32)

    def sim(self, a: np.ndarray, b: np.ndarray) -> float:
        m = self.alive
        return float(a[m] @ b[m]) / max(int(m.sum()), 1)

    # ---------- очистка (аттракторная память понятий) ----------
    def cleanup(self, v: np.ndarray, exclude_roles: bool = True) -> tuple[str, float]:
        if self._matrix is None:
            names = [n for n in self.codes if not (exclude_roles and n.startswith("#"))]
            self._matrix = (names, np.stack([self.codes[n] for n in names]))
        names, M = self._matrix
        m = self.alive
        s = (M[:, m] @ v[m]) / max(int(m.sum()), 1)
        i = int(np.argmax(s))
        return names[i], float(s[i])

    # ---------- повреждение ----------
    def lesion(self, fraction: float, seed: int = 0) -> None:
        """Убить долю нейронов (размерностей) — проверка постепенной деградации."""
        rng = np.random.default_rng(seed)
        self.alive = rng.random(self.dim) >= fraction


class EventMemory:
    """Ассоциативная память событий (плотная сеть Хопфилда, один нейрон — один след)."""

    def __init__(self, space: Space):
        self.s = space
        self.items: list[np.ndarray] = []
        self.weight: list[float] = []
        self._M: np.ndarray | None = None

    def event(self, agent: str, action: str, obj: str | None = None, neg: bool = False) -> np.ndarray:
        s = self.s
        parts = [s.bind(s.role("агент"), s.code(agent)), s.bind(s.role("действие"), s.code(action))]
        if obj:
            parts.append(s.bind(s.role("объект"), s.code(obj)))
        if neg:
            parts.append(s.bind(s.role("отрицание"), s.code("#НЕ")))
        return s.bundle(parts)

    def store(self, v: np.ndarray, weight: float = 1.0) -> None:
        self.items.append(v)
        self.weight.append(weight)
        self._M = None

    def recall(self, probe: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
        if not self.items:
            return []
        if self._M is None:
            self._M = np.stack(self.items)
        m = self.s.alive
        sims = (self._M[:, m] @ probe[m]) / max(int(m.sum()), 1)
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order]

    def decode(self, v: np.ndarray) -> dict:
        """Развязать событие по ролям и очистить: кто, что, кого/что, отрицание ли."""
        s = self.s
        out = {}
        for r in ("агент", "действие", "объект"):
            name, strength = s.cleanup(s.bind(v, s.role(r)))
            out[r] = name if strength > 0.2 else None
            out[r + "_сила"] = strength
        out["не"] = s.sim(s.bind(v, s.role("отрицание")), s.code("#НЕ")) > 0.2
        return out
