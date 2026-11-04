"""Знания «Ростка»: понятия и связи, выросшие из прочитанного, и рассуждение по ним.

Понятие-узел появляется, когда о нём впервые что-то сказано. Связь «X — это Y» или
«X умеет P» появляется при первом упоминании и помнит, ОТКУДА она (файл и фраза).

Рассуждение (с объяснением каждого шага):
1. прямое знание — «так написано в …»;
2. наследование: ближайший род, о котором что-то известно (исключения работают сами:
   «пингвин не летает» ближе, чем «птицы летают»);
3. обобщение по примерам («в основном живые не плавают — кроме рыб и пингвинов»);
4. закрытый мир — осторожное «нет, наверное»;
5. иначе — честное «не знаю».

Противоречия между текстами не затирают знание: появляется СОМНЕНИЕ с обоими
источниками и вопрос «как правильно?». Если новое прочитанное опровергает её вывод —
запись «я думала …, а в … написано …».
"""
from __future__ import annotations

from collections import defaultdict

from ..mind.mind import Verdict


def _neg(v: bool) -> str:
    return "" if v else "не "


class Knowledge:
    def __init__(self):
        self.isa: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        self.prop: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        self.src: dict[tuple, list[str]] = defaultdict(list)
        self.concepts: set[str] = set()
        self.predicates: set[str] = set()
        self.doubts: list[str] = []
        self.corrections: list[str] = []
        self._cache: dict = {}
        self.name = lambda w: w                     # как показывать слово (даст язык)

    def __getstate__(self):
        d = dict(self.__dict__)
        d["isa"], d["prop"], d["src"] = dict(self.isa), dict(self.prop), dict(self.src)
        d["name"] = None
        d["_cache"] = {}
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.isa = defaultdict(lambda: [0, 0], d["isa"])
        self.prop = defaultdict(lambda: [0, 0], d["prop"])
        self.src = defaultdict(list, d["src"])
        self.name = lambda w: w

    # ---------- запоминание ----------
    def tell(self, kind: str, x: str, y: str, pol: bool, source: str) -> str:
        """Запомнить прочитанное. → «new» | «again» | «doubt»."""
        if x == y:
            return "again"
        table = self.isa if kind == "isa" else self.prop
        n = self.name
        before = self.judge_isa(x, y) if kind == "isa" else self.judge(x, y)
        pos, neg = table.get((x, y), [0, 0])
        if (pos if not pol else neg) > 0 and (neg if not pol else pos) == 0:
            other = self.src.get((kind, x, y, not pol), ["?"])[0]
            self.doubts.append(f"в «{other}» сказано «{self._say(kind, x, y, not pol)}», а в «{source}» — "
                               f"«{self._say(kind, x, y, pol)}». Как правильно?")
            table[(x, y)][0 if pol else 1] += 1
            self.src[(kind, x, y, pol)].append(source)
            self._cache.clear()
            return "doubt"
        if before.value is not None and before.value != pol and before.conf >= 0.5 \
                and before.source not in ("прочитала",):
            self.corrections.append(f"я думала: {self._say(kind, x, y, before.value)} "
                                    f"({before.source}: {before.chain[0]}), а в «{source}» написано: "
                                    f"{self._say(kind, x, y, pol)}")
        new = (x, y) not in table or table[(x, y)] == [0, 0]
        table[(x, y)][0 if pol else 1] += 1
        self.src[(kind, x, y, pol)].append(source)
        self.concepts.add(x)
        if kind == "isa":
            self.concepts.add(y)
        else:
            self.predicates.add(y)
        self._cache.clear()
        return "new" if new else "again"

    def _say(self, kind: str, x: str, y: str, v: bool) -> str:
        n = self.name
        if kind == "isa":
            return f"{n(x)} — это {n(y)}" if v else f"{n(x)} не {n(y)}"
        return f"{n(x)} {_neg(v)}{n(y)}"

    def is_category(self, w: str) -> bool:
        return any(y == w and v[0] > v[1] for (x, y), v in self.isa.items())

    # ---------- структура ----------
    def parents(self, x: str) -> list[str]:
        key = ("p", x)
        if key not in self._cache:
            self._cache[key] = [y for (a, y), (pos, neg) in self.isa.items() if a == x and pos > neg]
        return self._cache[key]

    def children(self, y: str) -> list[str]:
        if "kids" not in self._cache:
            kids = defaultdict(list)
            for (a, b), (pos, neg) in self.isa.items():
                if pos > neg:
                    kids[b].append(a)
            self._cache["kids"] = kids
        return self._cache["kids"].get(y, [])

    def descendants(self, y: str) -> set[str]:
        out, todo = set(), [y]
        while todo:
            for k in self.children(todo.pop()):
                if k not in out:
                    out.add(k)
                    todo.append(k)
        return out

    def ancestors(self, x: str) -> list[list[tuple[str, list]]]:
        out, seen, level = [], {x}, [(x, [])]
        while level:
            nxt = []
            for node, path in level:
                for y in self.parents(node):
                    if y not in seen:
                        seen.add(y)
                        nxt.append((y, path + [(node, y)]))
            if nxt:
                out.append(nxt)
            level = nxt
        return out

    # ---------- рассуждение ----------
    def direct(self, x: str, p: str) -> Verdict | None:
        pos, neg = self.prop.get((x, p), (0, 0))
        if pos == neg:
            return None
        v = pos > neg
        where = self.src.get(("prop", x, p, v), ["?"])[0]
        return Verdict(v, 0.95, [f"в «{where}» написано: {self._say('prop', x, p, v)}"], "прочитала")

    def judge(self, x: str, p: str) -> Verdict:
        own = self.direct(x, p)
        if own:
            return own
        n = self.name
        for level in self.ancestors(x):
            found = [(y, path, self.direct(y, p)) for y, path in level]
            found = [f for f in found if f[2]]
            if not found:
                continue
            if len({f[2].value for f in found}) > 1:
                return Verdict(None, 0.5, [f"роды «{n(x)}» говорят разное"], "противоречие")
            y, path, v = found[0]
            chain = [f"{n(a)} — это {n(b)}" for a, b in path] + v.chain
            chain.append(f"значит, {n(x)} {_neg(v.value)}{n(p)}")
            return Verdict(v.value, v.conf * 0.97 ** len(path), chain, "вывод")
        kids = [(k, self.direct(k, p)) for k in self.children(x)]
        kids = [(k, v) for k, v in kids if v]
        if len(kids) >= 2:
            yes = [k for k, v in kids if v.value]
            no = [k for k, v in kids if not v.value]
            major, val = (yes, True) if len(yes) >= len(no) else (no, False)
            share = len(major) / len(kids)
            if share >= 0.75:
                return Verdict(val, 0.5 + 0.3 * share,
                               [f"из известных мне «{n(x)}» {len(major)} из {len(kids)} — «{_neg(val)}{n(p)}»",
                                f"наверное, «{n(x)}» — «{_neg(val)}{n(p)}»"], "гипотеза")
        if (self.parents(x) or x in self.concepts) and p in self.predicates:
            return Verdict(False, 0.6, [f"нигде не написано и не следует, что {n(x)} {n(p)}"],
                           "по умолчанию")
        return Verdict(None, 0.0, [f"я ничего не знаю о том, {n(p)} ли {n(x)}"], "не знаю")

    def judge_isa(self, x: str, y: str) -> Verdict:
        n = self.name
        pos, neg = self.isa.get((x, y), (0, 0))
        if neg > pos:
            return Verdict(False, 0.95, [f"в «{self.src[('isa', x, y, False)][0]}» написано: "
                                         f"{n(x)} не {n(y)}"], "прочитала")
        for level in self.ancestors(x):
            for node, path in level:
                if node == y:
                    chain = [f"{n(a)} — это {n(b)}" for a, b in path]
                    if len(path) == 1:
                        return Verdict(True, 0.95, [f"в «{self.src[('isa', x, y, True)][0]}» написано: "
                                                    f"{n(x)} — это {n(y)}"], "прочитала")
                    return Verdict(True, 0.95 * 0.97 ** len(path), chain + [f"значит, {n(x)} — это {n(y)}"],
                                   "вывод")
        if (self.parents(x) or x in self.concepts) and self.children(y):
            return Verdict(False, 0.6, [f"нигде не написано и не следует, что {n(x)} — это {n(y)}"],
                           "по умолчанию")
        return Verdict(None, 0.0, [f"я не знаю, {n(y)} ли {n(x)}"], "не знаю")

    def known(self, x: str, y: str) -> tuple[str, bool] | None:
        """Для выучивания оборотов: какая связь между x и y уже известна."""
        if (x, y) in self.isa and self.isa[(x, y)] != [0, 0]:
            pos, neg = self.isa[(x, y)]
            return ("isa", pos > neg)
        if (x, y) in self.prop and self.prop[(x, y)] != [0, 0]:
            pos, neg = self.prop[(x, y)]
            return ("prop", pos > neg)
        return None
