"""Любопытство: умение поставить правильный вопрос.

«Правильный» вопрос — тот, ответ на который сильнее всего изменит знания:

    ценность = неопределённость × влияние

- неопределённость: «не знаю» и противоречие — 1.0; гипотеза — тем больше, чем она
  ближе к 50/50; осторожное «нет, наверное» (закрытый мир) — 0.5;
- влияние: сколько других выводов зависит от ответа. Узнав, кто такая «птица»,
  система сразу поймёт что-то и про воробья, и про пингвина, и про любую новую птицу.

Откуда берутся вопросы:
1. Безымянная категория, которая попадается часто: «Как это называется?»
2. Понятие без рода: «Рыба — это кто?» (от рода наследуется всё общее знание).
3. Непроверенная гипотеза: «Я думаю, что все животные дышат, потому что… Это правда?»
4. Противоречие и сомнение в выводе.

Каждый вопрос несёт объяснение — почему система его задаёт.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from .mind import Mind


@dataclass
class Question:
    kind: str                     # name | parent | prop
    x: str | None
    y: str | None
    text: str
    why: str
    value: float
    look: np.ndarray | None = field(default=None, repr=False)
    cat: int | None = None

    def key(self) -> tuple:
        return (self.kind, self.cat if self.kind == "name" else self.x, self.y)

    def for_parent(self) -> dict:
        if self.kind == "name":
            return {"type": "name", "look": self.look}
        if self.kind == "parent":
            return {"type": "parent", "x": self.x}
        return {"type": "prop", "x": self.x, "p": self.y}


UNCERTAINTY = {"не знаю": 1.0, "противоречие": 1.0, "по умолчанию": 0.5}


class Curiosity:
    def __init__(self, mind: Mind, seed: int = 0):
        self.mind = mind
        self.asked: set[tuple] = set()
        self.rng = random.Random(seed)
        self.last_look: dict[int, np.ndarray] = {}
        self.last_seen: dict[int, int] = {}

    def saw(self, cats: list[int], looks: list[np.ndarray]) -> None:
        for c, lk in zip(cats, looks):
            self.last_look[c] = lk
            self.last_seen[c] = self.mind.experiences

    # ---------- кандидаты ----------
    def general_knowledge(self) -> int:
        """Сколько общих (родовых) свойств я знаю — столько можно унаследовать."""
        m = self.mind
        return sum(1 for (x, _), (pos, neg) in m.told_prop.items() if pos != neg and m.children(x))

    def candidates(self) -> list[Question]:
        m = self.mind
        out: list[Question] = []
        lx = m.lexicon
        gen = self.general_knowledge()
        # 1. безымянные категории
        for c in m.perception.ids():
            if c in lx.name_of or not m.perception.is_stable(c) or c not in self.last_look:
                continue
            n = m.perception.counts[c]
            like, _ = m.nearest_named(m.perception.prototype(c))
            hint = f"; похоже на «{lx.name_of[like]}», но это что-то другое" if like is not None else ""
            # узнав имя, я смогу узнавать и рассуждать про всё, что касается этой вещи;
            # чем меньше она похожа на знакомое, тем больше изменит ответ (новизна),
            # и о том, что перед глазами прямо сейчас, спрашивают в первую очередь
            answerable = len(m.predicates()) + sum(1 for y in m.nouns() if m.children(y))
            d = float(np.linalg.norm(m.perception.prototype(c) - m.perception.prototype(like))) \
                if like is not None else 2.0
            novelty = 1 - np.exp(-d / m.perception.radius)
            fresh = 1.5 if m.experiences - self.last_seen.get(c, -999) <= 20 else 1.0
            out.append(Question("name", None, None, "Как называется вот это?",
                                f"я видела такое уже {n} раз, но не знаю, как это называется{hint}",
                                0.5 * answerable * min(1.0, n / 5) * novelty * fresh,
                                self.last_look[c], c))
        # 2. понятия без рода
        for x in sorted(lx.basic_nouns() | {y for y in m.nouns() if m.children(y)}):
            if m.parents(x) or self._is_top(x):
                continue
            impact = 1 + len(m.descendants(x)) + gen
            out.append(Question("parent", x, None, f"{x} — это кто?",
                                f"я не знаю, к какому роду относится «{x}», а от рода я "
                                f"узнала бы сразу многое (общих свойств известно: {gen})",
                                impact))
        # 3–4. гипотезы, сомнения, пробелы
        for x in sorted(m.nouns()):
            for p in sorted(m.predicates()):
                v = m.judge(x, p)
                if v.source in ("слышала", "видела") or (v.source == "вывод" and v.conf >= 0.75):
                    continue
                if v.source == "гипотеза":
                    u = 1 - abs(2 * v.conf - 1) + 0.2
                else:
                    u = UNCERTAINTY.get(v.source, 0.4)
                impact = 1 + len(m.descendants(x))
                out.append(self._prop_question(x, p, v, u * impact))
        return [q for q in out if q.key() not in self.asked]

    def _is_top(self, x: str) -> bool:
        return ("parent", x, None) in self.asked

    @staticmethod
    def _prop_question(x: str, p: str, v, value: float) -> Question:
        if v.source == "гипотеза":
            text = f"Я думаю, что {v.chain[-1].removeprefix('наверное, ')}. Это правда: {x} {p}?"
            why = "это моя догадка: " + "; ".join(v.chain[:-1])
        elif v.source == "противоречие":
            text = f"{x} {p} или нет?"
            why = "у меня противоречие: " + (v.conflict or v.chain[0])
        elif v.source == "по умолчанию":
            text = f"{x} {p}? Думаю, что нет, но не уверена."
            why = v.chain[0]
        else:
            text = f"{x} {p}?"
            why = v.chain[0]
        return Question("prop", x, p, text, why, value)

    # ---------- мысленный эксперимент ----------
    def beliefs(self, scope: set[str]) -> dict:
        """Мои нынешние ответы (и уверена ли я в них) про всё, что входит в scope."""
        m = self.mind
        supers = [y for y in m.nouns() if m.children(y)]
        out = {}
        for x in scope:
            for p in m.predicates():
                v = m.judge(x, p)
                out[(x, p)] = (v.answer(), v.conf >= 0.7)
            for y in supers:
                if y != x:
                    v = m.judge_isa(x, y)
                    out[(x, y)] = (v.answer(), v.conf >= 0.7)
        return out

    def _gain(self, before: dict, scope: set[str], kind: str, x: str, y: str, value: bool) -> float:
        """Ценность ответа: сколько моих ОТВЕТОВ он изменит (информация ценна, только если
        может что-то изменить); простое прибавление уверенности весит в 10 раз меньше."""
        with self.mind.supposing(kind, x, y, value):
            after = self.beliefs(scope)
        gain = 0.0
        for k, (ans, sure) in before.items():
            a2, s2 = after.get(k, (None, False))
            gain += 1.0 if a2 != ans else (0.1 if s2 != sure else 0.0)
        return gain

    def expected_gain(self, q: Question) -> float:
        """Сколько моих выводов, скорее всего, изменится или станет уверенными после ответа."""
        m = self.mind
        if q.kind == "name":
            return q.value                         # без имени не о чем даже рассуждать
        scope = {q.x} | m.descendants(q.x)
        before = self.beliefs(scope)
        if q.kind == "prop":
            p_yes = self.prior_yes(q.x, q.y)
            return (p_yes * self._gain(before, scope, "prop", q.x, q.y, True)
                    + (1 - p_yes) * self._gain(before, scope, "prop", q.x, q.y, False))
        # род: перебираю, каким он может оказаться, — пропорционально правдоподобию
        options = self._plausible_parents(q.x)
        if not options:
            return 0.0
        return sum(w * self._gain(before, scope, "isa", q.x, y, True) for y, w in options)

    def prior_yes(self, x: str, p: str) -> float:
        """Насколько вероятно, что ответ будет «да».

        Для рода — по тем его представителям, о которых я что-то знаю: «плавают только
        рыбы и пингвины, а коровы и кошки — нет; вряд ли плавает всё живое»."""
        m = self.mind
        kids = m.descendants(x)
        if kids:
            yes = no = 0.0
            for k in kids:
                d = m.direct(k, p)
                if d:
                    yes, no = (yes + 1, no) if d.value else (yes, no + 1)
                elif m.well_known(k):
                    no += 0.5           # хорошо знакомое, но ничего такого за ним не замечала
            return (yes + 0.5) / (yes + no + 1)
        v = m.judge(x, p)
        return 0.5 if v.value is None else (v.conf if v.value else 1 - v.conf)

    def _plausible_parents(self, x: str) -> list[tuple[str, float]]:
        """Каким может оказаться род x — и насколько это правдоподобно.

        Род должен включать всех представителей x: если кошки, коровы и люди — точно
        не игрушки, то «живое» не может быть разновидностью игрушки. Поэтому про самое
        общее понятие («живое») спрашивать о роде почти бессмысленно — и система это видит."""
        m = self.mind
        below = {x} | m.descendants(x)
        supers = [y for y in m.nouns() if m.children(y) and y not in below]
        if not supers:
            return []
        members = m.descendants(x) or {x}
        cats = m.lexicon.ext.get(x, [])
        w = {}
        for y in supers:
            f = 1.0
            for k in members:
                v = m.judge_isa(k, y)
                f *= 1.0 if v.value else (1 - v.conf if v.value is False else 0.5)
            if cats and not m.children(x):
                f *= self._closeness(cats[0], y)   # конкретный вид: на кого похож внешне
            w[y] = f
        total = sum(w.values()) + 0.3              # 0.3 — «рода нет или он мне незнаком»
        return [(y, v / total) for y, v in w.items() if v / total > 0.01]

    def _closeness(self, cat: int, y: str) -> float:
        m = self.mind
        members = [c for k in m.descendants(y) for c in m.lexicon.ext.get(k, [])
                   if m.lexicon.name_of.get(c) == k]
        if not members:
            return 0.05
        proto = m.perception.prototype(cat)
        d = min(np.linalg.norm(m.perception.prototype(c) - proto) for c in members)
        return float(np.exp(-d))

    # ---------- выбор ----------
    def best(self, shortlist: int = 10) -> Question | None:
        """Выбрать вопрос с наибольшим ожидаемым приростом знания."""
        qs = self.candidates()
        if not qs:
            return None
        names = [q for q in qs if q.kind == "name"]
        parents = [q for q in qs if q.kind == "parent"]
        props = sorted((q for q in qs if q.kind == "prop"), key=lambda q: -q.value)[:shortlist]
        scored = []
        for q in names + parents + props:
            q.value = round(self.expected_gain(q), 2)
            scored.append(q)
        return max(scored, key=lambda q: q.value)

    def random(self) -> Question | None:
        """Для сравнения: случайный вопрос из того же круга понятий, без расчёта ценности."""
        m = self.mind
        pool: list[Question] = []
        for c in m.perception.ids():
            if c not in m.lexicon.name_of and c in self.last_look:
                pool.append(Question("name", None, None, "Как называется вот это?", "", 0,
                                     self.last_look[c], c))
        for x in sorted(m.nouns()):
            pool.append(Question("parent", x, None, f"{x} — это кто?", "", 0))
            for p in sorted(m.predicates()):
                pool.append(Question("prop", x, p, f"{x} {p}?", "", 0))
        pool = [q for q in pool if q.key() not in self.asked]
        return self.rng.choice(pool) if pool else None

    def mark(self, q: Question) -> None:
        self.asked.add(q.key())
