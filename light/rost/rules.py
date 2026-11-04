"""Поле правил «Ростка»: правила вырастают из историй, сеть сама их проверяет и уточняет.

Тоже рождается пустым. Узел-правило («X толкает Y → Y падает») появляется только когда
одна и та же схема перехода встретилась хотя бы в двух разных историях; конкретные
герои при этом заменяются переменными — так возникает обобщение.

Во сне сеть САМА ПРОВЕРЯЕТ свои правила на всей памяти историй:
- правило, которое часто не сбывается, она пробует УТОЧНИТЬ условием, найденным в тех
  случаях, где оно сбылось («толкнёшь — упадёт, только если оно шатается»);
- если уточнить не удаётся — ОТКАЗЫВАЕТСЯ от него как от совпадения;
- каждое изменение — с записью в журнал: что изменила и насколько стало лучше.

Предсказание следующего события в НОВОЙ истории (с новыми героями) — проверка того,
что выросло обобщение, а не заучивание.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Ev:
    agent: str
    action: str
    obj: str | None = None

    def roles(self) -> dict[str, str]:
        return {"агент": self.agent, "объект": self.obj} if self.obj else {"агент": self.agent}


@dataclass(frozen=True)
class Rule:
    trigger: str                      # действие-повод
    effect: str                       # действие-следствие
    maps: tuple                       # ((роль следствия, (откуда: «повод»|«условие», роль)), ...)
    cond: tuple | None = None         # (действие условия, его роль, роль повода, с которой совпадает)

    def text(self) -> str:
        v = {}
        letter = lambda key: v.setdefault(key, "XYZW"[len(v)])  # noqa: E731
        trig = f"{letter(('повод', 'агент'))} {self.trigger}"
        if any(src == ("повод", "объект") for _, src in self.maps) or self.cond and self.cond[2] == "объект":
            trig += f" {letter(('повод', 'объект'))}"
        cond = ""
        if self.cond:
            caction, crole, trole = self.cond
            v[("условие", crole)] = v.get(("повод", trole)) or letter(("повод", trole))
            other = "объект" if crole == "агент" else "агент"
            cond = f", а {v[('условие', 'агент')] if crole == 'агент' else letter(('условие', 'агент'))} {caction}"
            if any(src == ("условие", other) for _, src in self.maps) or crole == "объект":
                cond += f" {v.get(('условие', 'объект')) or letter(('условие', 'объект'))}"
        eff = " ".join([v.get(dict(self.maps)["агент"], "?") if "агент" in dict(self.maps) else "?", self.effect]
                       + ([v.get(dict(self.maps)["объект"]) or letter(dict(self.maps)["объект"])]
                          if "объект" in dict(self.maps) else []))
        return f"{trig}{cond} → {eff}"


def _common(a: str, b: str) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


class RuleField:
    def __init__(self, min_support: int = 2, trust: float = 0.8, give_up: float = 0.6):
        self.min_support = min_support
        self.trust = trust            # правило надёжно, если сбывается не реже
        self.give_up = give_up        # ниже — совпадение (если уточнить не удалось)
        self.entities: list[str] = []
        self.verbs: list[str] = []
        self.stories: list[list[Ev]] = []
        self.seen: Counter = Counter()           # сколько раз встречалась схема перехода
        self.rules: dict[Rule, dict] = {}        # узлы-правила (растут по мере опыта)
        self.journal: list[str] = []

    # ---------- чтение (формы слов сводятся: «вазу» → «ваза») ----------
    def _norm(self, w: str, pool: list[str]) -> str:
        for c in pool:
            n = _common(w, c)
            if w == c or (n >= 3 and n >= min(len(w), len(c)) - 1 and n >= 0.6 * max(len(w), len(c))):
                return c
        pool.append(w)
        return w

    def parse(self, sentence: str) -> Ev | None:
        t = re.findall(r"\w+", sentence.lower())
        if len(t) not in (2, 3):
            return None
        return Ev(self._norm(t[0], self.entities), self._norm(t[1], self.verbs),
                  self._norm(t[2], self.entities) if len(t) == 3 else None)

    # ---------- рост ----------
    def read(self, sentences: list[str]) -> list[Ev]:
        story = [e for e in (self.parse(s) for s in sentences) if e]
        self.stories.append(story)
        for t in range(1, len(story)):
            r = self._abstract(story[:t], story[t])
            if r:
                self.seen[r] += 1
                if self.seen[r] >= self.min_support and r not in self.rules:
                    self.rules[r] = {"born": len(self.stories)}
                    self.journal.append(f"[история {len(self.stories)}] выросло правило: {r.text()}")
        return story

    @staticmethod
    def _abstract(before: list[Ev], effect: Ev) -> Rule | None:
        """Обобщить переход: героев заменить ссылками на роли повода (или условия)."""
        trigger = before[-1]
        tr = trigger.roles()
        maps, cond = [], None
        for role, filler in effect.roles().items():
            src = next((("повод", r) for r, f in tr.items() if f == filler), None)
            if src is None:
                for e in reversed(before[:-1]):
                    er = e.roles()
                    r = next((r for r, f in er.items() if f == filler), None)
                    if r:
                        link = next(((r2, rt) for r2, f2 in er.items() for rt, ft in tr.items()
                                     if r2 != r and f2 == ft), None)
                        if link is None:
                            break
                        cond = (e.action, link[0], link[1])
                        src = ("условие", r)
                        break
            if src is None:
                return None                      # участник взялся ниоткуда — не следствие
            maps.append((role, src))
        return Rule(trigger.action, effect.action, tuple(sorted(maps)), cond)

    # ---------- применение ----------
    @staticmethod
    def fire(rule: Rule, trigger: Ev, state: list[Ev]) -> Ev | None:
        if trigger.action != rule.trigger:
            return None
        tr = trigger.roles()
        cond_ev = None
        if rule.cond:
            caction, crole, trole = rule.cond
            cond_ev = next((e for e in reversed(state) if e.action == caction
                            and e.roles().get(crole) == tr.get(trole)), None)
            if cond_ev is None:
                return None
        filled = {}
        for role, (where, r) in rule.maps:
            src = tr if where == "повод" else cond_ev.roles()
            if r not in src:
                return None
            filled[role] = src[r]
        if "агент" not in filled:
            return None
        return Ev(filled["агент"], rule.effect, filled.get("объект"))

    def check(self, rule: Rule, stories: list[list[Ev]] | None = None) -> tuple[int, int]:
        """Самопроверка правила на памяти историй: (сколько раз было применимо, сколько сбылось)."""
        applicable = hit = 0
        for st in stories if stories is not None else self.stories:
            for t in range(len(st)):
                p = self.fire(rule, st[t], st[:t])
                if p is not None:
                    applicable += 1
                    hit += t + 1 < len(st) and st[t + 1] == p
        return applicable, hit

    def predict(self, prefix: list[Ev]) -> tuple[Ev | None, Rule | None]:
        best = None
        for rule, info in self.rules.items():
            p = self.fire(rule, prefix[-1], prefix[:-1])
            if p is not None:
                rel = info.get("надёжность", 1.0)
                if best is None or rel > best[2]:
                    best = (p, rule, rel)
        return (best[0], best[1]) if best else (None, None)

    # ---------- сон: самопроверка, уточнение, отказ ----------
    def sleep(self) -> list[str]:
        notes = []
        for rule in list(self.rules):
            n, k = self.check(rule)
            rel = k / n if n else 0.0
            self.rules[rule]["надёжность"] = rel
            if rel >= self.trust:
                continue
            better = self._specialize(rule)
            if better:
                n2, k2 = self.check(better)
                del self.rules[rule]
                self.rules[better] = {"born": len(self.stories), "надёжность": k2 / n2}
                notes.append(f"уточнила правило «{rule.text()}» (сбывалось {k} из {n}) → "
                             f"«{better.text()}» (сбывается {k2} из {n2})")
            elif rel < self.give_up:
                del self.rules[rule]
                notes.append(f"отказалась от правила «{rule.text()}»: сбывалось {k} из {n} — совпадение")
        self.journal += notes
        return notes

    def _specialize(self, rule: Rule) -> Rule | None:
        """Найти условие, которое было во всех случаях, где правило сбылось."""
        if rule.cond:
            return None
        common = None
        for st in self.stories:
            for t in range(len(st) - 1):
                if self.fire(rule, st[t], st[:t]) == st[t + 1]:
                    tr = st[t].roles()
                    here = {(e.action, cr, trl) for e in st[:t] for cr, cf in e.roles().items()
                            for trl, tf in tr.items() if cf == tf and e.action != rule.trigger}
                    common = here if common is None else common & here
        best = None
        for c in sorted(common or []):
            cand = Rule(rule.trigger, rule.effect, rule.maps, c)
            n, k = self.check(cand)
            if n and k / n >= self.trust and (best is None or (k, -n) > best[0]):
                best = ((k, -n), cand)
        return best[1] if best else None


# ---------- материал для проверки ----------
STORIES = [
    ["мама хвалит сына", "сын радуется"],
    ["учитель хвалит ученика", "ученик радуется", "ученик поёт"],
    ["бабушка хвалит внучку", "внучка радуется", "внучка поёт"],
    ["девочка держит конфету", "девочка поёт", "конфета падает"],
    ["мальчик держит ложку", "мальчик поёт", "ложка падает"],
    ["кот хочет рыбу", "рыба падает", "кот берёт рыбу"],
    ["собака хочет кость", "кость падает", "собака берёт кость"],
    ["ваза шатается", "кот толкает вазу", "ваза падает"],
    ["чашка шатается", "мальчик толкает чашку", "чашка падает"],
    ["папа толкает дверь", "дверь открывается"],
    ["девочка толкает мяч", "мяч катится"],
    ["мальчик толкает тележку", "тележка едет"],
    ["папа держит газету", "папа читает"],
    ["кошка держит мышь", "кошка бежит"],
    ["мальчик хочет сок", "мальчик просит сок"],
    ["девочка хочет куклу", "девочка плачет"],
]
# новые истории — новые герои и предметы: предсказать следующее событие
NEW = [
    ["лиса хочет сыр", "ворона держит сыр", "лиса хвалит ворону", "ворона радуется",
     "ворона поёт", "сыр падает", "лиса берёт сыр"],
    ["лампа шатается", "пёс толкает лампу", "лампа падает"],
    ["пёс толкает санки", "санки едут"],
    ["дедушка хвалит внука", "внук радуется"],
]
