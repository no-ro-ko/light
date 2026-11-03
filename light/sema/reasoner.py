"""Рассуждение нейронными операциями: знания — в ассоциативной памяти событий,
вывод — циклы «запрос → узнавание → развязывание → очистка → новый фокус».

«Дышит ли пингвин?»: фокус = пингвин.
  1) запрос «пингвин, дышит, ?» к памяти — сильного отклика нет;
  2) запрос «пингвин, это, ?» → узнан след «пингвин это птица» → развязать объект → «птица»;
  3) новый фокус = птица → «птица, дышит?» — нет; «птица это ?» → «животное»;
  4) «животное, дышит?» — есть след → ответ «да», а путь фокуса — объяснение.
Ближайший род проверяется раньше дальнего, поэтому исключения («пингвин не летает»)
работают сами собой. Уверенность — произведение сил узнавания на каждом шаге: при
повреждении нейронов она падает постепенно, а не обрывается.
"""
from __future__ import annotations

from .core import EventMemory, Space

IS = "это"


class NeuralReasoner:
    def __init__(self, dim: int = 2048, seed: int = 0, threshold: float = 0.35):
        self.space = Space(dim, seed)
        self.mem = EventMemory(self.space)
        self.threshold = threshold
        self.predicates: set[str] = set()
        self.agents: set[str] = set()

    # ---------- запоминание ----------
    def tell(self, x: str, y: str, value: bool = True, isa: bool = False) -> None:
        if isa:
            self.mem.store(self.mem.event(x, IS, y, neg=not value))
        else:
            self.mem.store(self.mem.event(x, y, None, neg=not value))
            self.predicates.add(y)
        self.agents.add(x)

    def load_from(self, mind) -> int:
        """Перенести в нейронную память то, что разум узнал из опыта (сказанное,
        показанное и увиденное своими глазами)."""
        n = 0
        for (x, y), (pos, neg) in mind.told_prop.items():
            if pos != neg:
                self.tell(x, y, pos > neg)
                n += 1
        for (x, y), (pos, neg) in mind.told_isa.items():
            if pos != neg:
                self.tell(x, y, pos > neg, isa=True)
                n += 1
        for x in mind.lexicon.basic_nouns():
            for y, how, _ in mind.parents(x):
                if how == "мне показывали":
                    self.tell(x, y, True, isa=True)
                    n += 1
            for p in mind.predicates():
                v = mind.seen(x, p)
                if v:
                    self.tell(x, p, v.value)
                    n += 1
        return n

    # ---------- нейронные запросы ----------
    def _probe(self, x: str, action: str):
        s = self.space
        return s.bundle([s.bind(s.role("агент"), s.code(x)), s.bind(s.role("действие"), s.code(action))])

    def fact(self, x: str, action: str) -> tuple[bool | None, float, dict | None]:
        """Есть ли в памяти след «x action»? → (да/нет/неизвестно, сила, развязанный след)."""
        if x not in self.space.codes or action not in self.space.codes:
            return None, 0.0, None
        hits = self.mem.recall(self._probe(x, action), k=1)
        if not hits or hits[0][1] < self.threshold:
            return None, hits[0][1] if hits else 0.0, None
        d = self.mem.decode(self.mem.items[hits[0][0]])
        if d["агент"] != x or d["действие"] != action:
            return None, hits[0][1], None
        return (not d["не"]), hits[0][1], d

    def parents(self, x: str) -> list[tuple[str, float]]:
        if x not in self.space.codes:
            return []
        out = []
        for i, strength in self.mem.recall(self._probe(x, IS), k=4):
            if strength < self.threshold:
                break
            d = self.mem.decode(self.mem.items[i])
            if d["агент"] == x and d["действие"] == IS and d["объект"] and not d["не"]:
                out.append((d["объект"], strength))
        return out

    # ---------- рассуждение ----------
    def judge(self, x: str, p: str, isa: bool = False) -> tuple[bool | None, float, list[str]]:
        """Ответ с объяснением — путь фокуса по нейронной памяти."""
        chain: list[str] = []
        level = [(x, 1.0, [])]
        seen = {x}
        for _ in range(6):
            nxt = []
            for node, conf, path in level:
                if isa and node == p and node != x:
                    return True, conf, path + [f"значит, {x} — это {p}"]
                if not isa:
                    v, strength, _ = self.fact(node, p)
                    if v is not None:
                        step = f"помню: {node} {'' if v else 'не '}{p} (сила {strength:.2f})"
                        return v, conf * min(1.0, strength / 0.5), path + [step] + [
                            f"значит, {x} {'' if v else 'не '}{p}"]
                for parent, strength in self.parents(node):
                    if parent not in seen:
                        seen.add(parent)
                        nxt.append((parent, conf * min(1.0, strength / 0.5) * 0.98,
                                    path + [f"{node} — это {parent} (сила {strength:.2f})"]))
            if not nxt:
                break
            level = nxt
        # закрытый мир: сказуемое мне известно, а путь к нему из x не нашёлся
        known_x = bool(self.parents(x)) or x in self.agents
        if known_x and (isa or p in self.predicates):
            return False, 0.6, chain + [f"не нашла в памяти пути от «{x}» к «{p}»"]
        return None, 0.0, chain + [f"ничего не знаю про «{x}»"]
