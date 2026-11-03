"""Учёный: научный подход вместо «размышлений вообще».

Цикл, как в настоящей науке:
1. Наблюдение и измерение — масса, объём, исход опыта (плавает/тонет).
2. Гипотезы-соперники. Простые: «решает материал», «тяжёлое тонет», «большое тонет».
   Если простые величины не объясняют данные, система ВВОДИТ НОВУЮ ВЕЛИЧИНУ из
   измеренных — отношение или произведение (как программа-открыватель законов BACON).
3. Выбор опыта: проверяется тот предмет, на котором гипотезы-соперники расходятся
   сильнее всего (решающий эксперимент), — а не любой подряд.
4. Опровержение: одного надёжного контрпримера достаточно, чтобы отвергнуть закон
   («железо тонет» — опровергнуто железным корабликом).
5. Бритва Оккама: из объясняющих данные законов выбирается самый простой.
6. Закон → предсказания для невиданных материалов, ответы на «почему» и «как»
   (план с расчётом), проверка плана опытом.
7. Чтение незнакомого текста: неизвестное слово сопоставляется с собственными
   измерениями; утверждения текста проверяются опытом, ошибки замечаются.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..mind.perception import CategoryMemory


@dataclass
class Obs:
    look: np.ndarray
    cat: int
    label: str
    cup: bool
    m: float
    v: float
    floats: bool | None = None


@dataclass
class Law:
    name: str             # человеческое описание величины
    kind: str             # «порог» | «по материалу» | «материал и форма»
    a: int = 0            # показатели степени: величина = масса^a · объём^b
    b: int = 0
    theta: float = 0.0    # граница
    below_floats: bool = True
    errors: int = 0
    complexity: float = 1.0
    table: dict = field(default_factory=dict)

    def q(self, m: float, v: float) -> float:
        return (m ** self.a) * (v ** self.b)

    def predict(self, o: Obs) -> bool | None:
        if self.kind == "порог":
            return (self.q(o.m, o.v) < self.theta) == self.below_floats
        key = o.cat if self.kind == "по материалу" else (o.cat, o.cup)
        return self.table.get(key)

    def text(self) -> str:
        if self.kind == "порог":
            side = "меньше" if self.below_floats else "больше"
            return f"плавает, если {self.name} {side} {self.theta:.2f}"
        if self.kind == "по материалу":
            return "плавает или тонет — смотря из чего сделан"
        return "плавает или тонет — смотря из чего сделан и какой формы"


QUANT = {(1, 0): "масса", (0, 1): "объём", (1, -1): "масса на единицу объёма",
         (-1, 1): "объём на единицу массы", (1, 1): "масса × объём"}


class Scientist:
    def __init__(self, dim: int):
        self.perception = CategoryMemory(dim, radius=0.6)
        self.data: list[Obs] = []
        self.labels: dict[int, str] = {}
        self.journal: list[str] = []
        self.law: Law | None = None
        self.derived = False              # введены ли новые (производные) величины
        self.severe_tests = 0             # проверки, где соперники предсказывали иное
        self.meaning: dict[str, str] = {}  # понятые незнакомые слова

    # ---------- наблюдение ----------
    def look_at(self, look: np.ndarray, label: str, m: float, v: float) -> Obs:
        c, _ = self.perception.categorize(look)
        self.labels.setdefault(c, label)
        return Obs(look, c, label, bool(look[self._cup_index] > 0.5), m, v)

    _cup_index = -1

    def set_cup_feature(self, idx: int) -> None:
        self._cup_index = idx

    def record(self, o: Obs, floats: bool) -> list[str]:
        """Записать исход опыта и пересмотреть законы. Возвращает записи журнала."""
        before = self.law
        rivals = self.plausible()
        o.floats = floats
        self.data.append(o)
        notes = []
        if before and before.predict(o) is not None and before.predict(o) != floats:
            notes.append(f"ОПРОВЕРЖЕНИЕ: закон «{before.text()}» предсказал иное для "
                         f"{o.label}{' (чаша)' if o.cup else ''}: {self._fmt(o)}")
        preds = {r.predict(o) for r in rivals if r.predict(o) is not None}
        if len(preds) > 1 and self.law is not None:
            self.severe_tests += 1
        self.law = self.best()
        if self.law and (before is None or self.law.text() != before.text()):
            notes.append(f"принимаю закон: «{self.law.text()}» (ошибок {self.law.errors} "
                         f"на {len(self.data)} опытах)")
        self.journal += notes
        return notes

    @staticmethod
    def _fmt(o: Obs) -> str:
        return (f"масса {o.m:.2f}, объём {o.v:.2f} → "
                f"{'плавает' if o.floats else 'тонет'}")

    # ---------- гипотезы ----------
    def candidates(self) -> list[Law]:
        data = [o for o in self.data if o.floats is not None]
        if not data:
            return []
        out = [self._by_material(data, False), self._by_material(data, True)]
        for (a, b) in [(1, 0), (0, 1)]:
            out.append(self._threshold(data, a, b))
        # таблица случаев — ещё не закон: если нет ПРОСТОГО объяснения без ошибок,
        # ищу новую величину, составленную из измеренных
        simple_ok = any(h.errors == 0 and h.complexity <= 2 for h in out)
        if not simple_ok and not self.derived and len(data) >= 4:
            self.derived = True
            self.journal.append("ни масса, ни объём по отдельности не объясняют опыты, а «смотря из "
                                "чего сделан» — это таблица случаев, а не закон; ввожу новые "
                                "величины: масса/объём и масса×объём")
        if self.derived:
            for (a, b) in [(1, -1), (1, 1)]:
                out.append(self._threshold(data, a, b))
        return out

    def _threshold(self, data: list[Obs], a: int, b: int) -> Law:
        vals = sorted(((o.m ** a) * (o.v ** b), o.floats) for o in data)
        best = None
        for below in (True, False):
            cuts = [vals[0][0] - 1e-9] + [(vals[i][0] + vals[i + 1][0]) / 2
                                          for i in range(len(vals) - 1)] + [vals[-1][0] + 1e-9]
            for th in cuts:
                err = sum(((q < th) == below) != f for q, f in vals)
                if best is None or err < best[0]:
                    best = (err, th, below)
        err, th, below = best
        derived = (a, b) not in [(1, 0), (0, 1)]
        return Law(QUANT[(a, b)], "порог", a, b, th, below, err, 2.0 if derived else 1.0)

    def _by_material(self, data: list[Obs], with_shape: bool) -> Law:
        groups: dict = defaultdict(list)
        for o in data:
            groups[(o.cat, o.cup) if with_shape else o.cat].append(o.floats)
        table = {k: Counter(v).most_common(1)[0][0] for k, v in groups.items()}
        err = sum(len(v) - Counter(v).most_common(1)[0][1] for v in groups.values())
        return Law("", "материал и форма" if with_shape else "по материалу",
                   errors=err, complexity=float(len(table)), table=table)

    def score(self, h: Law) -> float:
        return 10 * h.errors + h.complexity       # ошибки дороги, сложность — немного

    def best(self) -> Law | None:
        c = self.candidates()
        return min(c, key=self.score) if c else None

    def plausible(self) -> list[Law]:
        c = self.candidates()
        if not c:
            return []
        s0 = min(self.score(h) for h in c)
        return [h for h in c if self.score(h) <= s0 + 8]

    # ---------- выбор опыта ----------
    def choose(self, shelf: list[Obs]) -> int:
        """Решающий эксперимент: где соперники расходятся; иначе — самое незнакомое."""
        rivals = self.plausible()
        best_i, best_v = 0, -1.0
        for i, o in enumerate(shelf):
            preds = [r.predict(o) for r in rivals]
            known = [p for p in preds if p is not None]
            split = (min(known.count(True), known.count(False)) / max(len(known), 1)) if known else 0
            unknown = preds.count(None) / max(len(preds), 1)
            near = 0.0
            if self.law and self.law.kind == "порог":
                q = self.law.q(o.m, o.v)
                near = math.exp(-abs(math.log(max(q, 1e-9) / self.law.theta)) * 3)
            v = 2 * split + unknown + 0.3 * near
            if v > best_v:
                best_i, best_v = i, v
        return best_i

    # ---------- применение закона ----------
    def predict(self, o: Obs) -> tuple[bool | None, str]:
        law = self.law
        if law is None:
            return None, "у меня ещё нет закона"
        p = law.predict(o)
        if p is None:
            return None, f"закон «{law.text()}» ничего не говорит про такое ({o.label})"
        if law.kind == "порог":
            q = law.q(o.m, o.v)
            return p, (f"{law.name} = {o.m:.2f}/{o.v:.2f} = {q:.2f}, это "
                       f"{'меньше' if q < law.theta else 'больше'} {law.theta:.2f} → "
                       f"{'плавает' if p else 'тонет'}")
        return p, f"предметы из такого материала у меня {'плавали' if p else 'тонули'}"

    def material_constants(self) -> dict[str, tuple[float, float]]:
        """Новое знание: у сплошных предметов из одного материала масса/объём постоянна."""
        out = {}
        groups = defaultdict(list)
        for o in self.data:
            if not o.cup:
                groups[o.label].append(o.m / o.v)
        for lab, qs in groups.items():
            if len(qs) >= 2:
                mean = float(np.mean(qs))
                out[lab] = (mean, float(np.std(qs) / mean))
        return out

    def how_to_float(self, o: Obs, float_density: float | None = None) -> tuple[list[dict], str]:
        """Ответ на «как сделать, чтобы это плавало?» — план с расчётом по закону."""
        law = self.law
        if not law or law.kind != "порог" or (law.a, law.b) != (1, -1) or not law.below_floats:
            return [], "я пока не знаю закона, по которому можно рассчитать ответ"
        th = law.theta * 0.97                       # с запасом: закон известен с точностью опытов
        q = o.m / o.v
        if q < th:
            return [], f"оно и так плавает: {o.m:.2f}/{o.v:.2f} = {q:.2f} < {law.theta:.2f}"
        lines = [f"сейчас масса на единицу объёма {q:.2f}, а плавает то, у чего меньше {law.theta:.2f}"]
        lines.append("разрезать пополам бесполезно: масса и объём уменьшатся одинаково, "
                     "отношение останется прежним")
        plans = []
        if float_density is not None and float_density < th:
            need = (o.m - th * o.v) / (th - float_density)
            vol = need * 1.15
            plans.append({"action": "поплавок", "volume": vol})
            lines.append(f"можно привязать поплавок (масса на объём {float_density:.2f}) объёмом "
                         f"больше {need:.2f}: (масса + {float_density:.2f}·x)/(объём + x) < {law.theta:.2f}"
                         f" → x > {need:.2f}; беру {vol:.2f}")
        keep = th / q * 0.9
        plans.append({"action": "выдолбить", "keep": keep})
        lines.append(f"или выдолбить, оставив не больше {th / q:.0%} материала "
                     f"(объём тот же, масса меньше); беру {keep:.0%}")
        return plans, "; ".join(lines)

    # ---------- чтение незнакомого текста ----------
    @staticmethod
    def _stem(w: str) -> str:
        return w[:max(3, len(w) - 2)]

    def _material_of(self, word: str) -> str | None:
        for lab in set(self.labels.values()):
            if self._stem(lab) == self._stem(word) or lab.startswith(word[:-1]) or word.startswith(lab[:-1]):
                return lab
        return None

    def read(self, lines: list[str]) -> dict:
        """Понять неизвестное слово по собственным измерениям и проверить текст опытом."""
        claims_big, claims_cmp, rules = [], [], []
        unknown = Counter()
        for line in lines:
            line = line.lower().strip()
            m = re.match(r"у (\w+) (большая|маленькая) (\w+)$", line)
            if m and self._material_of(m.group(1)):
                claims_big.append((self._material_of(m.group(1)), m.group(2) == "большая", line))
                unknown[m.group(3)] += 1
                continue
            m = re.match(r"(\w+) (\w+) (больше|меньше) (\w+) (\w+)$", line)
            if m and self._stem(m.group(1)) == self._stem(m.group(4)) \
                    and self._material_of(m.group(2)) and self._material_of(m.group(5)):
                claims_cmp.append((self._material_of(m.group(2)), self._material_of(m.group(5)),
                                   m.group(3) == "больше", line))
                unknown[m.group(1)] += 1
                continue
            m = re.match(r"если (\w+) предмета (больше|меньше) (\w+) воды то он (тонет|плавает)$", line)
            if m and self._stem(m.group(1)) == self._stem(m.group(3)):
                rules.append((m.group(2) == "больше", m.group(4) == "плавает", line))
                unknown[m.group(1)] += 1
        if not unknown:
            return {}
        word = unknown.most_common(1)[0][0]
        per = {}
        for lab in set(self.labels.values()):
            obs = [o for o in self.data if o.label == lab and not o.cup]
            if obs:
                per[lab] = {"масса": np.mean([o.m for o in obs]), "объём": np.mean([o.v for o in obs]),
                            "масса на единицу объёма": np.mean([o.m / o.v for o in obs])}
        if not per:
            return {}
        med = {k: np.median([v[k] for v in per.values()]) for k in next(iter(per.values()))}
        report = {}
        for k in med:
            agree = total = 0
            for lab, big, _ in claims_big:
                if lab in per:
                    total += 1
                    agree += (per[lab][k] > med[k]) == big
            for a, b, more, _ in claims_cmp:
                if a in per and b in per:
                    total += 1
                    agree += (per[a][k] > per[b][k]) == more
            report[k] = agree / max(total, 1)
        # про материал говорят как про свойство материала: оно не должно зависеть от размера
        # куска — проверяю, насколько величина постоянна у предметов из одного материала
        spread = {}
        for k in med:
            cvs = []
            for lab in per:
                vals = [o.m if k == "масса" else o.v if k == "объём" else o.m / o.v
                        for o in self.data if o.label == lab and not o.cup]
                if len(vals) >= 2:
                    cvs.append(np.std(vals) / np.mean(vals))
            spread[k] = float(np.mean(cvs)) if cvs else 1.0
        result_fit = {k: report[k] - 0.5 * min(spread[k], 1.0) for k in report}
        meaning = max(result_fit, key=result_fit.get)
        self.meaning[word] = meaning
        result = {"word": word, "meaning": meaning, "fit": report, "spread": spread,
                  "checked": [], "disagree": []}
        k = meaning
        for lab, big, line in claims_big:
            if lab in per:
                ok = (per[lab][k] > med[k]) == big
                (result["checked"] if ok else result["disagree"]).append(
                    f"«{line}»: по моим измерениям {k} у {lab} ≈ {per[lab][k]:.2f}")
        for a, b, more, line in claims_cmp:
            if a in per and b in per:
                ok = (per[a][k] > per[b][k]) == more
                (result["checked"] if ok else result["disagree"]).append(
                    f"«{line}»: у {a} ≈ {per[a][k]:.2f}, у {b} ≈ {per[b][k]:.2f}")
        result["unchecked"] = [line for lab, _, line in claims_big if lab not in per] + \
                              [line for a, b, _, line in claims_cmp if a not in per or b not in per]
        if self.law and self.law.kind == "порог" and (self.law.a, self.law.b) == (1, -1) \
                and meaning == "масса на единицу объёма":
            # мой закон: плавает, если масса/объём < θ. Текст: плавает, если плотность < плотности воды.
            # Значит, «плотность воды» — это и есть моя граница θ.
            for more, floats, line in rules:
                consistent = (more and not floats) or (not more and floats)
                result["water"] = self.law.theta
                (result["checked"] if consistent else result["disagree"]).append(
                    f"«{line}»: совпадает с моим законом, если «{word} воды» ≈ {self.law.theta:.2f}")
        return result
