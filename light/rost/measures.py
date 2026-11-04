"""Законы из записей измерений в текстах («длина 0.5 м, масса 1 кг, период 1.42 с»).

Узел-закон вырастает, когда о каком-то наборе величин накопилось достаточно записей
и нашлось постоянное сочетание: период / длина^½ = const (а масса не входит — значит,
не влияет). Это тот же поиск инварианта, что у программы-открывателя BACON, но по
прочитанным данным. Если данных мало или они узкие — система просит новых измерений
(вопрос с объяснением).
"""
from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict

import numpy as np

EXPONENTS = [-2, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3]


def _num(t: str) -> float:
    return float(t.replace(",", "."))


def extract(toks: list[str], lemma, is_function) -> dict[str, float]:
    """«длина 0.5 м масса 1 кг период 1.42 с» → {длина: 0.5, масса: 1, период: 1.42}."""
    row, name, after_number = {}, None, False
    for t in toks:
        if t[0].isdigit():
            if name is not None:
                row[name] = _num(t)
                name = None
            after_number = True
            continue
        if after_number and len(t) <= 3:  # короткое слово сразу после числа — единица («м», «кг»)
            after_number = False
            continue
        after_number = False
        if not is_function(t):
            name = lemma(t)
    return row if len(row) >= 2 else {}


class Law:
    def __init__(self, names: tuple, target: str, exps: dict, const: float, spread: float,
                 rows: int, form: str = "произведение", power: float = 1.0):
        self.names, self.target, self.exps = names, target, exps
        self.const, self.spread, self.rows = const, spread, rows
        self.form, self.power = form, power
        self.ranges: dict[str, tuple] = {}

    def predict(self, x: dict) -> float:
        if self.form == "произведение":
            return self.const * math.prod(x[k] ** a for k, a in self.exps.items() if a)
        return (self.const * sum(x[k] ** self.power for k, a in self.exps.items() if a)) ** (1 / self.power)

    def solve(self, var: str, y: float, x: dict) -> float | None:
        a = self.exps.get(var, 0)
        if not a:
            return None
        if self.form == "произведение":
            rest = self.const * math.prod(x[k] ** b for k, b in self.exps.items() if b and k != var)
            return (y / rest) ** (1 / a)
        rest = sum(x[k] ** self.power for k, b in self.exps.items() if b and k != var)
        v = y ** self.power / self.const - rest
        return v ** (1 / self.power) if v > 0 else None

    def text(self, name=lambda w: w) -> str:
        def pw(a):
            return {1: "", 0.5: "^½", -0.5: "^-½", 2: "²", 3: "³", -1: "⁻¹", -2: "⁻²", 1.5: "^1½"}.get(a, f"^{a:g}")
        if self.form == "произведение":
            terms = " · ".join(f"{name(k)}{pw(a)}" for k, a in self.exps.items() if a)
            s = f"{name(self.target)} = {self.const:.4g} · {terms}"
        else:
            terms = " + ".join(f"{name(k)}{pw(self.power)}" for k, a in self.exps.items() if a)
            s = f"{name(self.target)}{pw(self.power)} = {self.const:.4g} · ({terms})"
        idle = [name(k) for k, a in self.exps.items() if not a]
        return s + (f"; не влияет: {', '.join(idle)}" if idle else "")


class MeasureField:
    def __init__(self, min_rows: int = 5, noise: float = 0.01):
        self.rows: dict[frozenset, list[dict]] = defaultdict(list)
        self.last: dict[frozenset, Counter] = defaultdict(Counter)   # какая величина пишется последней
        self.sources: dict[frozenset, set] = defaultdict(set)
        self.laws: dict[frozenset, Law] = {}
        self.min_rows, self.noise = min_rows, noise
        self.journal: list[str] = []

    def add(self, row: dict, order: list[str], source: str) -> None:
        key = frozenset(row)
        self.rows[key].append(row)
        self.last[key][order[-1]] += 1
        self.sources[key].add(source)

    def fit(self, name=lambda w: w) -> list[str]:
        notes = []
        for key, rows in self.rows.items():
            if len(rows) < self.min_rows:
                continue
            target = self.last[key].most_common(1)[0][0]
            others = sorted(k for k in key if k != target)
            Y = np.array([r[target] for r in rows])
            X = [{k: r[k] for k in others} for r in rows]
            best = None
            for exps in itertools.product(EXPONENTS, repeat=len(others)):
                if not any(exps):
                    continue
                c = Y / np.array([math.prod(x[k] ** a for k, a in zip(others, exps)) for x in X])
                spread = float(np.std(c) / abs(np.mean(c)))
                score = spread + 0.002 * (sum(1 for a in exps if a) + 0.1 * sum(abs(a) for a in exps))
                if best is None or score < best[0]:
                    best = (score, Law(tuple(sorted(key)), target, dict(zip(others, exps)),
                                       float(np.mean(c)), spread, len(rows)))
            law = best[1]
            if law.spread > 3 * self.noise:
                for r in range(1, len(others) + 1):
                    for subset in itertools.combinations(others, r):
                        for pw in (1, 2, 3):
                            s = np.array([sum(x[k] ** pw for k in subset) for x in X])
                            c = Y ** pw / s
                            spread = float(np.std(c) / abs(np.mean(c)))
                            if spread < law.spread:
                                law = Law(tuple(sorted(key)), target, {k: (1 if k in subset else 0) for k in others},
                                          float(np.mean(c)), spread, len(rows), "сумма степеней", pw)
            tol = 3 * self.noise * (law.power if law.form == "сумма степеней" else 1)
            if law.spread <= tol:
                law.ranges = {k: (min(r[k] for r in rows), max(r[k] for r in rows)) for k in key}
                old = self.laws.get(key)
                self.laws[key] = law
                if old is None or old.text() != law.text():
                    note = f"нашла закон по {len(rows)} записям: {law.text(name)} (разброс {law.spread:.1%})"
                    self.journal.append(note)
                    notes.append(note)
        return notes

    def law_for(self, names: set[str]) -> Law | None:
        for key, law in self.laws.items():
            if names <= set(key):
                return law
        return None

    def questions(self, name=lambda w: w) -> list[str]:
        out = []
        for key, rows in self.rows.items():
            law = self.laws.get(key)
            if law is None and len(rows) < self.min_rows:
                out.append(f"Мало записей о том, как связаны {', '.join(name(k) for k in sorted(key))} "
                           f"({len(rows)}): дайте ещё измерений — и я поищу закон")
            elif law:
                k = next((v for v, a in law.exps.items() if a), None)
                if k:
                    lo, hi = law.ranges[k]
                    out.append(f"Чему равен(а) «{name(law.target)}», если {name(k)} = {hi * 10:g}? "
                               f"Хочу проверить закон далеко за пределами того, что мне давали ({lo:g}–{hi:g})")
        return out
