"""«Росток» — сеть, которая рождается пустой и растёт по материалу.

Чем она отличается от обычных нейросетей (в том числе больших языковых моделей):
- НЕТ заранее созданной матрицы параметров и случайных начальных чисел. В момент
  рождения в сети нет ни одного узла и ни одной связи.
- Узел (ансамбль) появляется только тогда, когда опыт его потребовал: встретилось
  то, на что не похож ни один существующий узел. Его «настройка» — след реально
  пережитого (первый пример, затем уточняемое среднее), а не случайное число.
- Связь появляется при первом совпадении (видела это — слышала то) и крепнет с
  повторением (правило Хебба). Никакого градиентного спуска.
- Размер и сложность сети определяет материал: простой мир — маленькая сеть,
  богатый — большая; когда материал освоен, рост сам останавливается.
- Во сне сеть САМА СЕБЯ УЛУЧШАЕТ: пробует перестроиться (разделить узел, слить два,
  убрать лишний, стать внимательнее или терпимее к различиям), проверяет каждое
  изменение на собственной памяти и оставляет только то, что сделало её лучше.
  Оценка — без учителя: «угадала бы я слово, которое прозвучало, когда показали
  на вещь?» плюс небольшая цена за каждый лишний узел (бритва Оккама).
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\w+", text.lower()))


@dataclass
class Moment:
    """След пережитого: что было видно, что прозвучало, на что показали."""
    looks: list
    words: tuple
    focus: int | None


class Units:
    """Поле узлов-образов. Пустое при рождении; каждый узел — из опыта."""

    def __init__(self, radius: float):
        self.radius = radius
        self.protos: list[np.ndarray] = []
        self.counts: list[int] = []
        self.born: list[int] = []

    def __len__(self) -> int:
        return len(self.protos)

    def copy(self) -> "Units":
        u = Units(self.radius)
        u.protos = [p.copy() for p in self.protos]
        u.counts = list(self.counts)
        u.born = list(self.born)
        return u

    def radius_of(self, i: int) -> float:
        # пока примеров мало, представление размыто — границы шире
        return self.radius * (1 + 0.6 / math.sqrt(self.counts[i]))

    def nearest(self, x: np.ndarray) -> tuple[int | None, float]:
        if not self.protos:
            return None, float("inf")
        d = np.linalg.norm(np.stack(self.protos) - x, axis=1)
        i = int(np.argmin(d))
        return i, float(d[i])

    def assign(self, x: np.ndarray, t: int, learn: bool = True) -> tuple[int, bool]:
        i, d = self.nearest(x)
        if i is not None and d <= self.radius_of(i):
            if learn:
                self.counts[i] += 1
                self.protos[i] = self.protos[i] + (x - self.protos[i]) / min(self.counts[i], 50)
            return i, False
        if i is not None and (not learn or self._occluded(x, i)):
            if learn:
                self.counts[i] += 1
            return i, False
        self.protos.append(x.copy())            # РОСТ: новый узел — след этого опыта
        self.counts.append(1)
        self.born.append(t)
        return len(self.protos) - 1, True

    def _occluded(self, x: np.ndarray, i: int) -> bool:
        """Похоже на знакомое, но часть признаков не видна (заслонено)."""
        diff = x - self.protos[i]
        w = np.where(diff < 0, 0.35, 1.0)
        return float(np.sqrt((w * diff ** 2).sum())) <= self.radius_of(i)


class Rostok:
    def __init__(self, radius: float = 0.75, unit_cost: float = 0.002, sleep_every: int = 200,
                 self_improve: bool = True):
        self.units = Units(radius)
        self.unit_cost = unit_cost
        self.sleep_every = sleep_every
        self.self_improve = self_improve
        self.words: Counter = Counter()                 # узлы-слова (появляются при первом звучании)
        self.links: dict[int, Counter] = defaultdict(Counter)   # связи образ → слово
        self.memory: list[Moment] = []
        self.journal: list[str] = []
        self.history: list[dict] = []
        self.t = 0

    # ================= жизнь =================
    def experience(self, looks: list[np.ndarray], text: str = "", focus: int | None = None) -> list[int]:
        words = tokenize(text)
        units = [self.units.assign(x, self.t)[0] for x in looks]
        self.words.update(set(words))
        self._link(units, words, focus)
        self.memory.append(Moment([x.copy() for x in looks], words, focus))
        self.t += 1
        if self.sleep_every and self.t % self.sleep_every == 0:
            self.sleep()
        return units

    def _link(self, units: list[int], words: tuple, focus: int | None,
              links: dict | None = None) -> None:
        links = self.links if links is None else links
        if not words or not units:
            return
        attention = {units[focus]: 1.0} if focus is not None else {u: 1 / len(units) for u in units}
        for u, a in attention.items():
            for w in set(words):
                links[u][w] += a                         # связь появилась / окрепла

    # ================= умения =================
    def name(self, look: np.ndarray, units: Units | None = None, links: dict | None = None,
             words: Counter | None = None) -> tuple[str | None, float]:
        """Как это называется? — слово, сильнее всего и избирательнее всего связанное с узлом."""
        units = units or self.units
        links = self.links if links is None else links
        words = words or self.words
        u, _ = units.nearest(look)
        if u is None or not links.get(u):
            return None, 0.0
        total = sum(words.values())
        lu = links[u]
        su = sum(lu.values())
        best, score = None, 0.0
        for w, s in lu.items():
            p = s / su
            q = words[w] / total
            v = p * math.log(p / q) if p > q else 0.0     # слово, звучащее ИМЕННО при этом
            if v > score:
                best, score = w, v
        return best, score

    def which(self, word: str) -> int | None:
        """Что значит слово? — узел-образ, к которому оно привязано сильнее всего."""
        best, s = None, 0.0
        for u, lu in self.links.items():
            if lu.get(word, 0) / max(sum(lu.values()), 1e-9) > s:
                best, s = u, lu[word] / sum(lu.values())
        return best

    # ================= самооценка (без учителя) =================
    def _rebuild(self, units: Units, part: int | None = None) -> dict:
        """Заново связать образы и слова по памяти — для пробной перестройки.
        part=0 — только по чётным воспоминаниям (вторая половина остаётся для проверки)."""
        links: dict[int, Counter] = defaultdict(Counter)
        for k, m in enumerate(self.memory):
            if part is not None and k % 2 != part:
                continue
            us = [units.nearest(x)[0] for x in m.looks]
            self._link(us, m.words, m.focus, links)
        return links

    def evaluate(self, units: Units) -> float:
        """Честная самопроверка: перестройку строю по одной половине воспоминаний,
        а проверяю на другой — чтобы не подгонять себя под знакомые случаи."""
        links = self._rebuild(units, part=0)
        held = [m for k, m in enumerate(self.memory) if k % 2 == 1]
        return self.self_error(units, links, moments=held) + self.unit_cost * len(units)

    def self_error(self, units: Units | None = None, links: dict | None = None, last: int = 800,
                   moments: list | None = None) -> float:
        """Доля случаев «показали на вещь — а я не угадала ни одного прозвучавшего слова»."""
        units = units or self.units
        links = self.links if links is None else links
        source = self.memory[-last:] if moments is None else moments
        pointed = [m for m in source if m.focus is not None and m.words]
        if not pointed:
            return 0.0
        miss = 0
        for m in pointed:
            w, _ = self.name(m.looks[m.focus], units, links)
            miss += w not in m.words
        return miss / len(pointed)

    def score(self, units: Units | None = None, links: dict | None = None) -> float:
        units = units or self.units
        return self.self_error(units, links) + self.unit_cost * len(units)

    # ================= сон: самоэксперименты над собственной структурой =================
    def sleep(self, max_changes: int = 6) -> list[str]:
        accepted = []
        if self.self_improve:
            for _ in range(max_changes):
                base = self.evaluate(self.units)
                best = None
                for desc, trial in self._candidates():
                    s = self.evaluate(trial)
                    if s < base - 1e-4 and (best is None or s < best[0]):
                        best = (s, desc, trial)
                if best is None:
                    break
                s, desc, trial = best
                self.units, self.links = trial, self._rebuild(trial)
                note = (f"[{self.t}] {desc}: самооценка {base:.3f} → {s:.3f} "
                        f"(узлов {len(trial)})")
                self.journal.append(note)
                accepted.append(note)
        self.history.append({"t": self.t, "узлов": len(self.units), "устойчивых": self.stable_units(),
                             "слов": len(self.words),
                             "связей": sum(len(v) for v in self.links.values()),
                             "ошибка": round(self.self_error(), 3)})
        return accepted

    def _candidates(self):
        """Что можно попробовать изменить в себе."""
        U = self.units
        pointed = [m for m in self.memory if m.focus is not None and m.words]
        spec = self._specific_words()
        # 1) разделить узел, если его упорно называют двумя разными словами
        by_unit: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for m in pointed:
            u, _ = U.nearest(m.looks[m.focus])
            w = max(m.words, key=lambda x: spec.get(x, 0.0))
            if spec.get(w, 0) > 0.3:
                by_unit[u][w].append(m.looks[m.focus])
        for u, groups in by_unit.items():
            total = sum(len(v) for v in groups.values())
            big = {w: v for w, v in groups.items() if len(v) >= 4 and len(v) >= 0.15 * total}
            if len(big) >= 2:
                order = sorted(big, key=lambda w: -len(big[w]))
                means = {w: np.mean(big[w], axis=0) for w in order}
                # делить стоит, только если это и ВЫГЛЯДИТ по-разному (пингвины ≠ воробьи),
                # а не просто названо родовым словом («мяч» и «игрушка» — те же мячи)
                if min(np.linalg.norm(means[order[0]] - means[w]) for w in order[1:]) < 0.8 * U.radius:
                    continue
                trial = U.copy()
                trial.protos[u] = means[order[0]]
                for w in order[1:]:
                    trial.protos.append(means[w])
                    trial.counts.append(len(big[w]))
                    trial.born.append(self.t)
                yield f"разделила узел #{u}: его называют по-разному ({', '.join(order)})", trial
        # 2) слить два близких узла, которые называют одинаково
        names = {i: self.name(U.protos[i])[0] for i in range(len(U))}
        for i in range(len(U)):
            for j in range(i + 1, len(U)):
                d = float(np.linalg.norm(U.protos[i] - U.protos[j]))
                if d < 1.6 * U.radius and names[i] == names[j]:
                    yield f"слила узлы #{i} и #{j}: похожи и зовутся одинаково («{names[i]}»)", \
                        self._merged(U, i, j)
        # 3) убрать узел-случайность (мало примеров, давно не пополнялся)
        for i in range(len(U)):
            if U.counts[i] <= 2 and self.t - U.born[i] > 100:
                j, _ = self._nearest_other(U, i)
                if j is not None:
                    yield f"убрала узел #{i}: случайный (всего {U.counts[i]} примера)", self._merged(U, j, i)
        # 4) стать внимательнее или терпимее к различиям
        for k in (0.85, 1.15):
            yield (f"изменила внимательность к различиям: {U.radius:.2f} → {U.radius * k:.2f}",
                   self._regrow(U.radius * k))

    def _specific_words(self) -> dict[str, float]:
        """Насколько слово привязано к чему-то одному (служебные слова — ни к чему)."""
        by_word: dict[str, Counter] = defaultdict(Counter)
        for u, lu in self.links.items():
            for w, s in lu.items():
                by_word[w][u] += s
        return {w: max(c.values()) / sum(c.values()) for w, c in by_word.items()}

    @staticmethod
    def _nearest_other(U: Units, i: int) -> tuple[int | None, float]:
        best, bd = None, float("inf")
        for j in range(len(U)):
            if j != i:
                d = float(np.linalg.norm(U.protos[i] - U.protos[j]))
                if d < bd:
                    best, bd = j, d
        return best, bd

    @staticmethod
    def _merged(U: Units, keep: int, drop: int) -> Units:
        t = U.copy()
        n1, n2 = t.counts[keep], t.counts[drop]
        t.protos[keep] = (t.protos[keep] * n1 + t.protos[drop] * n2) / (n1 + n2)
        t.counts[keep] = n1 + n2
        del t.protos[drop], t.counts[drop], t.born[drop]
        return t

    def _regrow(self, radius: float) -> Units:
        """Вырасти заново по всей памяти с другой внимательностью к различиям."""
        u = Units(radius)
        for i, m in enumerate(self.memory):
            for x in m.looks:
                u.assign(x, i)
        return u

    def stable_units(self, min_count: int = 20) -> int:
        """Устойчивые узлы — структура; мелкие временные узелки сеть сама убирает во сне."""
        return sum(1 for c in self.units.counts if c >= min_count)

    # ================= отчёт о себе =================
    def report(self) -> str:
        h = self.history[-1] if self.history else {}
        return (f"Я родилась пустой. Сейчас во мне {self.stable_units()} устойчивых узлов-образов "
                f"(и {len(self.units) - self.stable_units()} временных), "
                f"{len(self.words)} слов и {sum(len(v) for v in self.links.values())} связей — "
                f"всё это выросло из {self.t} пережитых моментов. Ошибаюсь в {h.get('ошибка', 0):.0%} "
                f"случаев. Изменений в себе я приняла: {len(self.journal)}.")
