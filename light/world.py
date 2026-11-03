"""«Детская» — синтетический мир, в котором система учится как ребёнок.

Скрытая от системы истина: какие бывают существа и предметы, как они выглядят,
что делают, кто к какому роду относится и какими свойствами обладает.

Система получает только то, что получил бы ребёнок:
- ВИД объектов в сцене — набор признаков (размер, шерсть, перья, колёса, ...) с
  индивидуальными отличиями и шумом; никаких названий;
- СОБЫТИЯ — что объект делает прямо сейчас (звук «гав», в воздухе, в воде, ...);
- РЕЧЬ «родителя» — короткие фразы, часто неоднозначные: в сцене несколько
  объектов, а о каком из них речь — не сказано.

Часть истинных фактов родитель НЕ произносит никогда — по ним и проверяется,
умеет ли система приходить к выводам, а не повторять услышанное.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

LOOK = ["размер", "четыре_ноги", "две_ноги", "шерсть", "перья", "чешуя", "крылья", "клюв",
        "плавники", "висячие_уши", "усы", "рога", "вымя", "лицо", "одежда", "ствол", "листья",
        "лепестки", "круглый", "колёса", "черенок", "чёрно_белый"]

EVENTS = ["звук_гав", "звук_мяу", "звук_му", "звук_чирик", "речь", "в_воздухе", "в_воде",
          "катится", "едет", "звук_уху"]

# вид: (типичный облик, вероятности событий в сцене)
KINDS: dict[str, tuple[dict[str, float], dict[str, float]]] = {
    "собака": ({"размер": .4, "четыре_ноги": 1, "шерсть": 1, "висячие_уши": 1, "усы": .3},
               {"звук_гав": .5}),
    "кошка": ({"размер": .25, "четыре_ноги": 1, "шерсть": 1, "усы": 1}, {"звук_мяу": .5}),
    "корова": ({"размер": .9, "четыре_ноги": 1, "шерсть": .5, "рога": 1, "вымя": 1}, {"звук_му": .5}),
    "воробей": ({"размер": .05, "две_ноги": 1, "перья": 1, "крылья": 1, "клюв": 1},
                {"звук_чирик": .4, "в_воздухе": .5}),
    "пингвин": ({"размер": .35, "две_ноги": 1, "перья": 1, "крылья": .5, "клюв": 1, "чёрно_белый": 1},
                {"в_воде": .5}),
    "рыба": ({"размер": .1, "чешуя": 1, "плавники": 1}, {"в_воде": .9}),
    "человек": ({"размер": .7, "две_ноги": 1, "лицо": 1, "одежда": 1}, {"речь": .6}),
    "дерево": ({"размер": 1, "ствол": 1, "листья": 1}, {}),
    "цветок": ({"размер": .05, "листья": .6, "лепестки": 1}, {}),
    "мяч": ({"размер": .1, "круглый": 1}, {"катится": .4}),
    "машинка": ({"размер": .15, "колёса": 1}, {"едет": .5}),
    "яблоко": ({"размер": .05, "круглый": .8, "черенок": 1}, {}),
    # вид, которого нет в обычной жизни системы, — для проверки встречи с неизвестным
    "сова": ({"размер": .2, "две_ноги": 1, "перья": 1, "крылья": 1, "клюв": 1},
             {"в_воздухе": .5, "звук_уху": .5}),
}
EVERYDAY = [k for k in KINDS if k != "сова"]

ISA = {"собака": ["животное"], "кошка": ["животное"], "корова": ["животное"],
       "воробей": ["птица"], "пингвин": ["птица"], "сова": ["птица"], "птица": ["животное"],
       "рыба": ["животное"], "животное": ["живое"], "человек": ["живое"],
       "дерево": ["растение"], "цветок": ["растение"], "растение": ["живое"],
       "мяч": ["игрушка"], "машинка": ["игрушка"], "яблоко": ["еда"]}

# собственные свойства (наследуются вниз; False — исключение)
PROPS = {"собака": {"лает": True}, "кошка": {"мяукает": True}, "корова": {"мычит": True},
         "воробей": {"чирикает": True}, "пингвин": {"плавает": True, "летает": False},
         "рыба": {"плавает": True}, "человек": {"говорит": True, "дышит": True},
         "птица": {"летает": True}, "животное": {"дышит": True}, "живое": {"растёт": True},
         "мяч": {"катится": True}, "машинка": {"едет": True}, "сова": {"ухает": True}}

# слова-действия, которые можно увидеть/услышать (остальные свойства — только из речи)
EVENT_WORDS = {"лает": "звук_гав", "мяукает": "звук_мяу", "мычит": "звук_му",
               "чирикает": "звук_чирик", "говорит": "речь", "летает": "в_воздухе",
               "плавает": "в_воде", "катится": "катится", "едет": "едет", "ухает": "звук_уху"}

FRAMES = ["смотри", "вот", "это", "какой", "ой"]
SUPERS = sorted({p for ps in ISA.values() for p in ps})
PREDICATES = sorted({p for ps in PROPS.values() for p in ps})


def ancestors(x: str) -> list[str]:
    out, frontier = [], list(ISA.get(x, []))
    while frontier:
        y = frontier.pop(0)
        if y not in out:
            out.append(y)
            frontier += ISA.get(y, [])
    return out


def is_a(x: str, y: str) -> bool:
    return y in ancestors(x)


def holds(x: str, p: str) -> bool:
    """Истина мира: ближайшее по иерархии указание; если нигде не сказано — нет."""
    for y in [x] + ancestors(x):
        if p in PROPS.get(y, {}):
            return PROPS[y][p]
    return False


def kids_of(y: str) -> list[str]:
    return [x for x, ps in ISA.items() if y in ps]


# Вопросы для экзамена. Их ответы родитель НИКОГДА не произносит.
EXAM_PROPS = [("пингвин", "дышит"), ("воробей", "дышит"), ("собака", "дышит"),
              ("кошка", "дышит"), ("корова", "дышит"), ("рыба", "дышит"),
              ("кошка", "растёт"), ("пингвин", "растёт"), ("дерево", "растёт"),
              ("цветок", "растёт"), ("человек", "растёт"),
              ("пингвин", "летает"), ("собака", "летает"),
              ("корова", "летает"), ("кошка", "лает"), ("рыба", "говорит"),
              ("мяч", "дышит"), ("машинка", "растёт"), ("яблоко", "дышит")]
EXAM_ISA = [("пингвин", "животное"), ("воробей", "живое"), ("дерево", "живое"),
            ("мяч", "животное"), ("рыба", "животное"), ("машинка", "живое")]


def exam() -> list[tuple[str, str, bool]]:
    """(подлежащее, сказуемое или род, истина)."""
    return ([(x, p, holds(x, p)) for x, p in EXAM_PROPS]
            + [(x, y, is_a(x, y)) for x, y in EXAM_ISA])


@dataclass
class Obj:
    kind: str
    look: np.ndarray
    events: set[str] = field(default_factory=set)


class Nursery:
    """Мир + родитель. `withhold` — факты, которые родитель никогда не произносит."""

    def __init__(self, seed: int = 0, kinds: list[str] | None = None,
                 withhold: set[tuple[str, str]] | None = None, generic_rate: float = 0.25,
                 point_rate: float = 0.5):
        self.rng = random.Random(seed)
        self.np = np.random.default_rng(seed)
        self.kinds = kinds or EVERYDAY
        self.withhold = set(withhold if withhold is not None else
                            {(x, p) for x, p, _ in exam()} | {("рыба", "животное")})
        self.generic_rate = generic_rate
        self.point_rate = point_rate
        self.facts = self._facts()

    # ---------- восприятие ----------
    def instance(self, kind: str, events: bool = True) -> Obj:
        proto, ev = KINDS[kind]
        look = np.array([proto.get(f, 0.0) for f in LOOK])
        look = look + self.np.normal(0, 0.08, len(LOOK))
        look[0] += self.np.normal(0, 0.1)                # особи разного размера
        look[self.np.random(len(LOOK)) < 0.04] = 0.0     # часть признаков не видна (заслонено)
        look = np.clip(look, 0, 1)
        happening = {e for e, p in ev.items() if events and self.rng.random() < p}
        return Obj(kind, look, happening)

    def scene(self, n: int | None = None) -> list[Obj]:
        n = n or self.rng.choice([1, 2, 2, 3])
        return [self.instance(self.rng.choice(self.kinds)) for _ in range(n)]

    # ---------- речь ----------
    def _facts(self) -> list[tuple[str, str, bool]]:
        facts = [(x, y, True) for x, ys in ISA.items() for y in ys]
        facts += [(x, p, v) for x, ps in PROPS.items() for p, v in ps.items()]
        known = set(self.kinds) | {a for k in self.kinds for a in ancestors(k)}
        return [(x, y, v) for x, y, v in facts if x in known and (x, y) not in self.withhold]

    def say_fact(self, x: str, y: str, v: bool) -> str:
        if y in SUPERS or y in KINDS:
            return f"{x} это {y}" if v else f"{x} не {y}"
        return f"{x} {y}" if v else f"{x} не {y}"

    def allowed(self, x: str, y: str) -> bool:
        return (x, y) not in self.withhold

    def episode(self) -> tuple[list[Obj], str, int | None]:
        """→ (сцена, фраза, на какой объект родитель показал или None).

        Совместное внимание: примерно в половине случаев родитель показывает,
        о чём говорит (как реальные родители — взглядом или жестом)."""
        scene = self.scene()
        ti = self.rng.randrange(len(scene))
        target = scene[ti]
        point = ti if self.rng.random() < self.point_rate else None
        r = self.rng.random()
        if r < self.generic_rate:                         # общее знание, не о сцене
            return scene, self.say_fact(*self.rng.choice(self.facts)), None
        r = self.rng.random()
        verbs = [w for w, e in EVENT_WORDS.items() if e in target.events
                 and holds(target.kind, w) and self.allowed(target.kind, w)]
        if verbs and r < 0.4:                             # что объект делает
            return scene, f"{target.kind} {self.rng.choice(verbs)}", point
        name = target.kind
        ups = [a for a in ancestors(target.kind) if self.allowed(target.kind, a)]
        if r > 0.8 and ups:                               # иногда — родовым словом
            name = self.rng.choice(ups)
        return scene, f"{self.rng.choice(FRAMES)} {name}", point

    # ---------- ответы на вопросы ребёнка ----------
    def answer(self, q: dict) -> tuple[list[Obj], str]:
        """q: {"type": "name", "obj": Obj} | {"type": "prop", "x", "p"} | {"type": "isa", "x", "y"}."""
        if q["type"] == "name":
            return [q["obj"]], f"это {q['obj'].kind}"
        x = q["x"]
        if x not in KINDS and x not in SUPERS:
            return [], "не знаю такого"
        if q["type"] == "prop":
            v = holds(x, q["p"])
            return [], f"да {x} {q['p']}" if v else f"нет {x} не {q['p']}"
        if q["type"] == "isa":
            v = is_a(x, q["y"])
            return [], f"да {x} это {q['y']}" if v else f"нет {x} не {q['y']}"
        if q["type"] == "parent":                         # «кто такой x?»
            ps = ISA.get(x)
            return [], f"{x} это {ps[0]}" if ps else f"{x} это {x}"
        return [], "не понимаю"
