"""«Мастерская» — мир, о котором системе ничего не рассказывают.

Скрытая истина — настоящая физика плавания: предмет плавает, если его масса,
делённая на объём (плотность), меньше, чем у воды (1.0). У полого предмета внутри
воздух, поэтому железный кораблик плавает, а железный гвоздь тонет.

Что доступно системе (как ребёнку в мастерской или учёному в лаборатории):
- облик предмета (цвет, блеск, шероховатость, форма-чаша...) — набор признаков;
- надпись на ящике, из которого взят предмет (название материала);
- измерения: взвесить (масса), опустить в мерный стакан (объём) — с небольшой ошибкой;
- опыт: опустить в воду и посмотреть — плавает или тонет;
- действия: разрезать пополам, выдолбить, привязать поплавок, добавить груз.

Никаких формул, законов и слова «плотность» система не получает.
Есть и «учебник» — незнакомый текст, где встречается неизвестное ей слово.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, replace

import numpy as np

WATER = 1.0

# материал: (плотность — скрыта от системы, облик)
MATERIALS: dict[str, tuple[float, dict[str, float]]] = {
    "пробка": (0.24, {"светлый": .6, "пористый": 1, "коричневый": .4}),
    "дерево": (0.6, {"коричневый": 1, "волокна": 1}),
    "воск": (0.9, {"жёлтый": .8, "гладкий": 1, "мягкий": .6}),
    "лёд": (0.92, {"прозрачный": .8, "холодный": 1, "гладкий": .6}),
    "резина": (1.2, {"чёрный": 1, "мягкий": 1}),
    "пластилин": (1.4, {"яркий": 1, "мягкий": 1}),
    "стекло": (2.5, {"прозрачный": 1, "блестит": .7, "гладкий": 1}),
    "камень": (2.7, {"серый": 1, "шершавый": 1}),
    "железо": (7.9, {"серый": .6, "блестит": 1, "холодный": .6}),
    "медь": (8.9, {"рыжий": 1, "блестит": 1}),
}
# материалы, которых система никогда не видела, — для проверки понимания
NOVEL: dict[str, tuple[float, dict[str, float]]] = {
    "пенопласт": (0.05, {"белый": 1, "пористый": 1}),
    "алюминий": (2.7, {"светлый": .8, "блестит": 1}),
    "свинец": (11.3, {"серый": 1, "тёмный": 1, "мягкий": .5}),
    "янтарь": (1.1, {"жёлтый": 1, "прозрачный": .6}),
}
ALL = {**MATERIALS, **NOVEL}
LOOK = sorted({f for _, look in ALL.values() for f in look} | {"чаша"})


@dataclass
class Thing:
    material: str
    hollow: bool          # чаша/кораблик: внутри воздух
    size: float           # внешний объём самого предмета
    wall: float = 1.0     # доля материала (для полых < 1)
    extra_m: float = 0.0  # привязанный груз/поплавок
    extra_v: float = 0.0
    look: np.ndarray = field(default=None, repr=False)
    note: str = ""

    @property
    def mass(self) -> float:
        return ALL[self.material][0] * self.size * self.wall + self.extra_m

    @property
    def volume(self) -> float:
        return self.size + self.extra_v

    def floats(self) -> bool:
        return self.mass / self.volume < WATER


class Workshop:
    def __init__(self, seed: int = 0, materials: dict | None = None, hollow_rate: float = 0.25):
        self.rng = random.Random(seed)
        self.np = np.random.default_rng(seed)
        self.materials = materials or MATERIALS
        self.hollow_rate = hollow_rate
        self.drops = 0                                   # сколько опытов «в воду» сделано

    # ---------- предметы ----------
    def look_of(self, material: str, hollow: bool) -> np.ndarray:
        proto = ALL[material][1]
        x = np.array([proto.get(f, 0.0) for f in LOOK])
        x[LOOK.index("чаша")] = 1.0 if hollow else 0.0
        x = x + self.np.normal(0, 0.07, len(LOOK))
        return np.clip(x, 0, 1)

    def make(self, material: str | None = None, hollow: bool | None = None,
             size: float | None = None, wall: float | None = None) -> Thing:
        material = material or self.rng.choice(list(self.materials))
        hollow = (self.rng.random() < self.hollow_rate) if hollow is None else hollow
        size = size if size is not None else round(self.rng.uniform(0.2, 5.0), 2)
        if hollow:
            wall = wall if wall is not None else round(self.rng.uniform(0.04, 0.35), 3)
        else:
            wall = 1.0
        return Thing(material, hollow, size, wall, look=self.look_of(material, hollow))

    def shelf(self, n: int = 30, materials: dict | None = None) -> list[Thing]:
        mats = list((materials or self.materials))
        return [self.make(self.rng.choice(mats)) for _ in range(n)]

    def label(self, t: Thing) -> str:
        """Надпись на ящике."""
        return t.material

    # ---------- измерения и опыт ----------
    def weigh(self, t: Thing) -> float:
        return t.mass * (1 + self.np.normal(0, 0.01))

    def measure_volume(self, t: Thing) -> float:
        return t.volume * (1 + self.np.normal(0, 0.02))

    def drop(self, t: Thing) -> bool:
        """Опустить в воду: плавает (True) или тонет (False)."""
        self.drops += 1
        return t.floats()

    # ---------- действия ----------
    def cut(self, t: Thing) -> Thing:
        return replace(t, size=t.size / 2, extra_m=t.extra_m / 2, extra_v=t.extra_v / 2,
                       note=t.note + " разрезан пополам;")

    def hollow_out(self, t: Thing, keep: float) -> Thing:
        """Выдолбить, оставив долю материала keep (форма — чаша)."""
        look = t.look.copy()
        look[LOOK.index("чаша")] = 1.0
        return replace(t, hollow=True, wall=t.wall * keep, look=look,
                       note=t.note + f" выдолблен (осталось {keep:.0%});")

    def attach_float(self, t: Thing, volume: float) -> Thing:
        """Привязать пробковый поплавок заданного объёма."""
        return replace(t, extra_m=t.extra_m + ALL["пробка"][0] * volume, extra_v=t.extra_v + volume,
                       note=t.note + f" привязан поплавок {volume:.2f};")

    def add_weight(self, t: Thing, mass: float) -> Thing:
        """Привязать железный груз заданной массы."""
        return replace(t, extra_m=t.extra_m + mass, extra_v=t.extra_v + mass / ALL["железо"][0],
                       note=t.note + f" привязан груз {mass:.2f};")


TEXTBOOK = [
    "у пробки маленькая плотность",
    "у железа большая плотность",
    "у меди большая плотность",
    "у дерева маленькая плотность",
    "плотность камня больше плотности дерева",
    "плотность железа больше плотности стекла",
    "плотность воска меньше плотности резины",
    "если плотность предмета больше плотности воды то он тонет",
    "если плотность предмета меньше плотности воды то он плавает",
]
WRONG_TEXT = [
    "у пробки большая плотность",
    "плотность дерева больше плотности железа",
]
