"""Открытие количественных законов — задачи по физике, геометрии, математике,
которым система не обучена.

Как работает настоящий экспериментатор (и программа-открыватель BACON, 1970-е):
1. Сама планирует, что измерить: меняет ОДНУ величину, остальные держит постоянными
   (контролируемый эксперимент), берёт значения в широком диапазоне.
2. Каждый замер — это ВОПРОС к миру или учителю («каков период маятника длиной 0.5 м?»).
3. Ищет инвариант: такое сочетание величин y / (x₁^a · x₂^b · …), которое остаётся
   постоянным; если произведение степеней не подходит — сумму степеней (так находится
   теорема Пифагора: c² = a² + b²).
4. Величины с показателем 0 — «не влияют» (масса маятника); это тоже открытие.
5. Проверяет закон на новых, специально выбранных точках (в том числе за пределами
   измеренного диапазона) — и только тогда принимает.
6. Решает задачи: прямые («каков будет период при длине 10 м?») и обратные
   («какой длины нужен маятник, чтобы период был 3 с?»), объясняет «как»
   («увеличишь длину в 4 раза — период вырастет в 2 раза»).
7. Замечает одинаковые постоянные в разных задачах (3.14 в площади и в окружности).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

EXPONENTS = [-2, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3]


@dataclass
class Problem:
    """Задача-мир: переменные (с диапазонами и единицами), искомая величина и скрытый закон."""
    name: str
    target: str
    variables: dict[str, tuple[float, float]]
    oracle: Callable[..., float]
    units: dict[str, str] = field(default_factory=dict)
    noise: float = 0.01

    def question(self, x: dict) -> str:
        parts = ", ".join(f"{k} = {v:.3g}{(' ' + self.units[k]) if k in self.units else ''}" for k, v in x.items())
        return f"Чему равно «{self.target}», если {parts}?"


@dataclass
class FoundLaw:
    target: str
    form: str                       # «произведение» | «сумма степеней»
    exps: dict[str, float]
    const: float
    spread: float
    power: float = 1.0

    def predict(self, x: dict) -> float:
        if self.form == "произведение":
            return self.const * math.prod(x[k] ** a for k, a in self.exps.items() if a)
        s = sum(x[k] ** self.power for k, a in self.exps.items() if a)
        return (self.const * s) ** (1 / self.power)

    def solve(self, var: str, y: float, x: dict) -> float | None:
        """Обратная задача: какое значение var даст y при остальных x."""
        a = self.exps.get(var, 0)
        if not a:
            return None
        if self.form == "произведение":
            rest = self.const * math.prod(x[k] ** b for k, b in self.exps.items() if b and k != var)
            return (y / rest) ** (1 / a)
        rest = sum(x[k] ** self.power for k, b in self.exps.items() if b and k != var)
        val = y ** self.power / self.const - rest
        return val ** (1 / self.power) if val > 0 else None

    def text(self) -> str:
        if self.form == "произведение":
            terms = " · ".join(f"{k}{_pow(a)}" for k, a in self.exps.items() if a)
            law = f"{self.target} = {self.const:.4g} · {terms}"
        else:
            terms = " + ".join(f"{k}{_pow(self.power)}" for k, a in self.exps.items() if a)
            law = f"{self.target}{_pow(self.power)} = {self.const:.4g} · ({terms})"
        idle = [k for k, a in self.exps.items() if not a]
        return law + (f"; не влияет: {', '.join(idle)}" if idle else "")

    def how(self) -> list[str]:
        out = []
        if self.form == "произведение":
            for k, a in self.exps.items():
                if not a:
                    out.append(f"«{k}» на «{self.target}» не влияет — менять его бесполезно")
                    continue
                f = 4 if abs(a) == 0.5 else 2
                out.append(f"увеличишь «{k}» в {f} раза — «{self.target}» "
                           f"{'вырастет' if a > 0 else 'уменьшится'} в {f ** abs(a):.3g} раза")
        else:
            out.append(f"«{self.target}» складывается из вкладов: {self.text()}")
        return out


def _pow(a: float) -> str:
    if a == 1:
        return ""
    sup = {"-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", ".": "·"}
    if a == 0.5:
        return "^½"
    if a == -0.5:
        return "^-½"
    if a == 1.5:
        return "^1½"
    return "".join(sup.get(ch, ch) for ch in f"{a:g}")


KNOWN_NUMBERS = {"π": math.pi, "2π": 2 * math.pi, "4/3·π": 4 / 3 * math.pi, "π²": math.pi ** 2}


class LawFinder:
    def __init__(self, problem: Problem, seed: int = 0):
        self.p = problem
        self.rng = np.random.default_rng(seed)
        self.data: list[tuple[dict, float]] = []
        self.asked: list[str] = []
        self.law: FoundLaw | None = None
        self.journal: list[str] = []

    # ---------- план опытов ----------
    def plan(self, levels: int = 5) -> list[dict]:
        """Контролируемый эксперимент: меняю по одной величине, остальные — посередине."""
        base = {k: math.sqrt(lo * hi) for k, (lo, hi) in self.p.variables.items()}
        pts = [dict(base)]
        for k, (lo, hi) in self.p.variables.items():
            for v in np.geomspace(lo, hi, levels):
                x = dict(base)
                x[k] = float(v)
                pts.append(x)
        return pts

    def ask(self, x: dict, oracle: Callable | None = None) -> float:
        q = self.p.question(x)
        self.asked.append(q)
        y = (oracle or self.p.oracle)(**x) * (1 + self.rng.normal(0, self.p.noise))
        self.data.append((x, y))
        return y

    # ---------- поиск закона ----------
    def fit(self) -> FoundLaw | None:
        names = list(self.p.variables)
        X = [x for x, _ in self.data]
        Y = np.array([y for _, y in self.data])
        best = None
        for exps in itertools.product(EXPONENTS, repeat=len(names)):
            if not any(exps):
                continue
            c = Y / np.array([math.prod(x[k] ** a for k, a in zip(names, exps)) for x in X])
            spread = float(np.std(c) / abs(np.mean(c)))
            complexity = sum(1 for a in exps if a) + 0.1 * sum(abs(a) for a in exps)
            score = spread + 0.002 * complexity
            if best is None or score < best[0]:
                best = (score, FoundLaw(self.p.target, "произведение", dict(zip(names, exps)),
                                        float(np.mean(c)), spread))
        law = best[1]
        if law.spread > 3 * self.p.noise:
            self.journal.append(f"произведения степеней не подходят (разброс {law.spread:.0%}) — "
                                "пробую сумму степеней")
            for r in range(1, len(names) + 1):
                for subset in itertools.combinations(names, r):
                    for pw in (1, 2, 3):
                        s = np.array([sum(x[k] ** pw for k in subset) for x in X])
                        c = Y ** pw / s
                        spread = float(np.std(c) / abs(np.mean(c)))
                        if spread < law.spread:
                            law = FoundLaw(self.p.target, "сумма степеней",
                                           {k: (1 if k in subset else 0) for k in names},
                                           float(np.mean(c)), spread, pw)
        # погрешность замера растёт при возведении в степень: y² ошибается вдвое сильнее y
        tolerance = 3 * self.p.noise * max(1.0, law.power if law.form == "сумма степеней" else 1.0)
        self.law = law if law.spread <= tolerance else None
        if self.law:
            self.journal.append(f"нашла закон: {self.law.text()} (разброс {law.spread:.1%})")
        else:
            self.journal.append(f"закона не нашла (лучший разброс {law.spread:.0%})")
        return self.law

    def verify(self, n: int = 5, stretch: float = 3.0) -> float:
        """Проверка на новых точках, в том числе далеко за пределами измеренного."""
        if not self.law:
            return float("inf")
        errs = []
        for _ in range(n):
            x = {k: float(np.exp(self.rng.uniform(np.log(lo / stretch), np.log(hi * stretch))))
                 for k, (lo, hi) in self.p.variables.items()}
            y = self.ask(x)
            errs.append(abs(self.law.predict(x) - y) / abs(y))
        worst = max(errs)
        self.journal.append(f"проверка на {n} новых точках (до ×{stretch:g} за пределами опытов): "
                            f"наибольшая ошибка {worst:.1%}")
        return worst

    def discover(self) -> FoundLaw | None:
        for x in self.plan():
            self.ask(x)
        self.fit()
        if self.law and self.verify() > 0.05:
            self.journal.append("проверка не пройдена — закон отвергнут")
            self.law = None
        return self.law

    def recognize_constant(self, others: dict[str, float]) -> str | None:
        """Не встречалось ли мне это число раньше (здесь или в других задачах)?"""
        if not self.law:
            return None
        c = self.law.const
        if abs(c - 1) < 0.02:
            return "1 — никакого множителя, величины связаны напрямую"
        for name, v in {**KNOWN_NUMBERS, **others}.items():
            if abs(c - v) / abs(v) < 0.015:
                return name
        return None


def related_constants(found: dict[str, float], tol: float = 0.01) -> list[str]:
    """Гипотеза: постоянные РАЗНЫХ задач связаны простым соотношением — возможно, за ними
    стоит одна общая величина. Беру только ещё не объяснённые постоянные (π, 2π и т.п.
    уже объяснены) и строгий допуск — иначе это нумерология, а не наука."""
    out = []
    explained = list(KNOWN_NUMBERS.values()) + [1.0]
    names = [k for k, v in found.items() if all(abs(v / e - 1) > 0.015 for e in explained)]
    specials = {"1": 1.0, "2": 2.0, "π": math.pi, "2π": 2 * math.pi, "π²": math.pi ** 2,
                "2π²": 2 * math.pi ** 2, "4π²": 4 * math.pi ** 2}
    for a, b in itertools.combinations(names, 2):
        best = None
        for p, q in itertools.product((1, 2, -1, -2), repeat=2):
            val = found[a] ** p * found[b] ** q
            for kname, k in specials.items():
                err = abs(val / k - 1)
                if err < tol and (best is None or abs(p) + abs(q) < best[0]):
                    best = (abs(p) + abs(q), f"гипотеза: c({a}){_pow(p)} · c({b}){_pow(q)} ≈ {kname} "
                                             f"(совпадение {err:.2%}) — возможно, за обеими задачами "
                                             f"стоит одна общая величина; нужна проверка")
        if best:
            out.append(best[1])
    return out


def problem_book(g: float = 9.81) -> list[Problem]:
    """Задачи, которым система не обучена. Законы скрыты в oracle."""
    return [
        Problem("маятник", "период", {"длина": (0.1, 3.0), "масса груза": (0.1, 5.0)},
                lambda **x: 2 * math.pi * math.sqrt(x["длина"] / g), {"длина": "м", "масса груза": "кг"}),
        Problem("падение", "путь", {"время": (0.2, 4.0), "масса": (0.1, 10.0)},
                lambda **x: g * x["время"] ** 2 / 2, {"время": "с", "масса": "кг"}),
        Problem("рычаг", "груз для равновесия", {"груз слева": (0.5, 10.0), "плечо слева": (0.1, 2.0),
                                                 "плечо справа": (0.1, 2.0)},
                lambda **x: x["груз слева"] * x["плечо слева"] / x["плечо справа"]),
        Problem("закон Ома", "ток", {"напряжение": (1.0, 220.0), "сопротивление": (1.0, 1000.0)},
                lambda **x: x["напряжение"] / x["сопротивление"], {"напряжение": "В", "сопротивление": "Ом"}),
        Problem("прямоугольник", "площадь", {"ширина": (0.5, 20.0), "высота": (0.5, 20.0)},
                lambda **x: x["ширина"] * x["высота"]),
        Problem("круг", "площадь круга", {"радиус": (0.1, 10.0)}, lambda **x: math.pi * x["радиус"] ** 2),
        Problem("окружность", "длина окружности", {"радиус": (0.1, 10.0)},
                lambda **x: 2 * math.pi * x["радиус"]),
        Problem("шар", "объём шара", {"радиус": (0.1, 5.0)}, lambda **x: 4 / 3 * math.pi * x["радиус"] ** 3),
        Problem("прямоугольный треугольник", "гипотенуза", {"катет a": (1.0, 20.0), "катет b": (1.0, 20.0)},
                lambda **x: math.hypot(x["катет a"], x["катет b"])),
        Problem("планеты (Кеплер)", "период обращения", {"расстояние до Солнца": (0.3, 30.0)},
                lambda **x: x["расстояние до Солнца"] ** 1.5, {"расстояние до Солнца": "а.е."}),
    ]
