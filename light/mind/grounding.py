"""Связать слово с ощущением: «что значит зелёный?»

Сценарий слепого, который сначала узнаёт о цвете из слов, а потом получает глаза:

1. В тексте встречается незнакомое слово («дом стоит в зелёном саду») → система сама
   спрашивает: «Что значит „зелёный“?»
2. Ответ — определение («зелёный — это свет с длиной волны от 500 до 565 нанометров») и
   примеры («листья зелёные»). Из этого — выводы и ПРЕДСКАЗАНИЯ: «если посмотреть на лист
   прибором, пик будет между 500 и 565 нм».
3. Подключается новый модуль восприятия:
   - спектрометр, у которого известно, какой длине волны соответствует каждый канал, —
     тогда цвет узнаётся ПО ОПРЕДЕЛЕНИЮ: «пик 540 нм — это в диапазоне зелёного»;
   - камера с непонятными каналами (три числа, смысл которых неизвестен) — тогда цвет
     узнаётся ПО ПРИМЕРАМ: «мне говорили, что листья зелёные, — значит, вот так зелёный
     выглядит для камеры».
4. При первом восприятии система сама отмечает: «вот что подразумевалось под зелёным»,
   проверяет свои предсказания, узнаёт цвет у новых предметов и спрашивает, чтобы проверить.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

ADJ_END = re.compile(r"(ый|ий|ой|ая|яя|ое|ее|ые|ие|ом|ем|ого|его|ую|юю|ым|им|ых|их)$")
DEF = re.compile(r"^(\w+)\s*(?:—|-)?\s*это свет с длиной волны от (\d+) до (\d+) (?:нанометров|нм)")
EXAMPLE = re.compile(r"^(\w+) (\w+)$")


SOFT = {"ий", "яя", "ее", "ие", "его", "ем", "им", "их", "юю"}


def stem(w: str) -> str:
    w = w.lower().replace("ё", "е")
    return ADJ_END.sub("", w) if len(w) > 4 else w


def lemma(w: str) -> str:
    """Начальная форма прилагательного по встреченной форме: «синее» → «синий»."""
    m = ADJ_END.search(w)
    if not m or len(w) <= 4:
        return w
    return w[:m.start()] + ("ий" if m.group(1) in SOFT else "ый")


def same_thing(a: str, b: str) -> bool:
    """Разные формы одного слова: «лист» и «листья», «трава» и «травы»."""
    a, b = a.lower().replace("ё", "е"), b.lower().replace("ё", "е")
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n >= 4 and n >= 0.6 * min(len(a), len(b)) or a == b


@dataclass
class Concept:
    word: str
    quantity: str | None = None
    lo: float | None = None
    hi: float | None = None
    examples: set = field(default_factory=set)
    told: list = field(default_factory=list)
    prototypes: dict = field(default_factory=dict)   # датчик → как выглядит для этого датчика
    grounded: dict = field(default_factory=dict)     # датчик → как связала (по определению/примерам)


@dataclass
class Sensor:
    name: str
    channels: list | None      # что означает каждый канал (длина волны), если известно


class Grounder:
    def __init__(self, known_words: set[str] | None = None):
        self.known = {stem(w) for w in (known_words or set())}
        self.concepts: dict[str, Concept] = {}
        self.sensors: dict[str, Sensor] = {}
        self.asked: set[str] = set()
        self.journal: list[str] = []
        self.seen_color: dict[str, str] = {}             # предмет → цвет, который я увидела сама

    # ---------- язык ----------
    def concept(self, word: str) -> Concept:
        return self.concepts.setdefault(stem(word), Concept(word))

    def read(self, text: str) -> list[str]:
        """Прочитать текст; вернуть вопросы, которые хочется задать."""
        questions = []
        for line in re.split(r"[.!?\n]+", text.lower()):
            line = line.strip().replace(",", "")
            if not line:
                continue
            m = DEF.match(line)
            if m:
                c = self.concept(m.group(1))
                c.word = m.group(1)                     # определение даёт начальную форму
                c.quantity, c.lo, c.hi = "длина волны", float(m.group(2)), float(m.group(3))
                c.told.append(line)
                self.journal.append(f"поняла: «{c.word}» — свет {c.lo:.0f}–{c.hi:.0f} нм")
                continue
            m = EXAMPLE.match(line)
            if m and ADJ_END.search(m.group(2)):
                thing, adj = m.group(1), m.group(2)
                c = self.concept(lemma(adj))
                c.examples.add(thing)
                c.told.append(line)
                self.journal.append(f"запомнила: {thing} — {c.word}")
                if c.lo is None and stem(adj) not in self.asked:
                    questions.append(self._ask(adj))
                continue
            for w in line.split():
                s = stem(w)
                if ADJ_END.search(w) and len(w) > 4 and s not in self.known and \
                        (s not in self.concepts or self.concepts[s].lo is None) and s not in self.asked:
                    questions.append(self._ask(w))
        return questions

    def _ask(self, w: str) -> str:
        self.asked.add(stem(w))
        q = f"Что значит «{lemma(w)}»?"
        self.journal.append(f"встретила незнакомое слово «{w}» — спрашиваю: {q}")
        return q

    def predictions(self) -> list[str]:
        out = []
        for c in self.concepts.values():
            if c.lo is not None:
                for ex in sorted(c.examples):
                    out.append(f"если посмотреть на «{ex}» прибором, пик будет между "
                               f"{c.lo:.0f} и {c.hi:.0f} нм (потому что {ex} — {c.word})")
        return out

    def conclusions(self) -> list[str]:
        out = []
        cs = [c for c in self.concepts.values() if c.lo is not None]
        for c in cs:
            if len(c.examples) >= 2:
                out.append(f"{', '.join(sorted(c.examples))} — одного цвета: все отражают свет "
                           f"{c.lo:.0f}–{c.hi:.0f} нм")
        for a in cs:
            for b in cs:
                if a is not b:
                    for ex in sorted(a.examples):
                        out.append(f"{ex} — не {b.word}: его свет {a.lo:.0f}–{a.hi:.0f} нм, "
                                   f"а {b.word} — {b.lo:.0f}–{b.hi:.0f} нм")
                        break
        return out

    # ---------- новые органы чувств ----------
    def attach(self, name: str, channels: list | None = None) -> str:
        self.sensors[name] = Sensor(name, channels)
        msg = (f"подключён «{name}»: {len(channels)} каналов, каждому соответствует длина волны"
               if channels else f"подключён «{name}»: каналы есть, но что они значат — не знаю")
        self.journal.append(msg)
        return msg

    def perceive(self, sensor: str, reading: np.ndarray, thing: str | None = None) -> dict:
        """Посмотреть на предмет новым органом чувств. thing — если сказали, что это."""
        s = self.sensors[sensor]
        out = {"color": None, "how": None, "notes": []}
        if s.channels is not None:                      # знаю физический смысл каналов
            w = np.clip(reading, 0, None)
            peak = float(np.sum(np.array(s.channels) * w) / np.sum(w))
            out["peak"] = peak
            for c in self.concepts.values():
                if c.lo is not None and c.lo <= peak <= c.hi:
                    out["color"], out["how"] = c.word, "по определению"
                    if sensor not in c.grounded:
                        c.grounded[sensor] = "по определению"
                        out["notes"].append(f"впервые вижу «{c.word}» сама: пик {peak:.0f} нм — это "
                                            f"в описанном мне диапазоне {c.lo:.0f}–{c.hi:.0f} нм. "
                                            f"Вот что подразумевалось под словом «{c.word}»")
                    c.prototypes.setdefault(sensor, reading.copy())
            if thing:
                self._check_prediction(thing, out, peak)
                if out["color"] is None:
                    out["question"] = (f"Как называется цвет {thing}? Его свет — около {peak:.0f} нм, "
                                       f"а такой мне ещё не называли")
        else:                                           # смысл каналов неизвестен
            if thing:
                for c in self.concepts.values():
                    if any(same_thing(thing, ex) for ex in c.examples):
                        first = sensor not in c.prototypes
                        c.prototypes[sensor] = reading.copy() if first else 0.8 * c.prototypes[sensor] + 0.2 * reading
                        c.grounded.setdefault(sensor, "по примерам")
                        out["color"], out["how"] = c.word, "мне говорили"
                        if first:
                            out["notes"].append(f"мне говорили, что {thing} — {c.word}. Значит, вот так "
                                                f"«{c.word}» выглядит для «{sensor}»: {np.round(reading, 2)}")
            if out["color"] is None:
                best, dist = self._nearest(sensor, reading)
                if best is not None and dist < 0.25:
                    out["color"], out["how"] = best.word, "похоже на то, что мне называли"
                    if thing:
                        out["notes"].append(f"{thing}, наверное, {best.word}: для «{sensor}» он выглядит "
                                            f"как {', '.join(sorted(best.examples))} (отличие {dist:.2f})")
                        out["question"] = f"{thing} — {best.word}? Хочу проверить свою догадку"
                elif thing:
                    out["question"] = f"Какого цвета {thing}? Я такого ещё не видела"
        if thing and out["color"]:
            self.seen_color[stem(thing)] = out["color"]
        return out

    def _nearest(self, sensor: str, reading: np.ndarray):
        best, dist = None, float("inf")
        for c in self.concepts.values():
            p = c.prototypes.get(sensor)
            if p is not None:
                d = float(np.linalg.norm(p - reading) / (np.linalg.norm(p) + 1e-9))
                if d < dist:
                    best, dist = c, d
        return best, dist

    def _check_prediction(self, thing: str, out: dict, peak: float) -> None:
        for c in self.concepts.values():
            if any(same_thing(thing, ex) for ex in c.examples) and c.lo is not None:
                ok = c.lo <= peak <= c.hi
                out["notes"].append(f"предсказание «{thing}: пик между {c.lo:.0f} и {c.hi:.0f} нм» "
                                    f"{'подтвердилось' if ok else 'НЕ подтвердилось'} ({peak:.0f} нм)")

    def link_sensors(self, a: str, b: str, reading_a: np.ndarray, reading_b: np.ndarray) -> str | None:
        """Один и тот же предмет двумя органами чувств: перенести знание с одного на другой."""
        ra = self.perceive(a, reading_a)
        if ra["color"]:
            c = self.concept(ra["color"])
            first = b not in c.prototypes
            c.prototypes[b] = reading_b.copy() if first else 0.8 * c.prototypes[b] + 0.2 * reading_b
            c.grounded.setdefault(b, f"через «{a}»")
            if first:
                return (f"«{a}» говорит, что это {c.word}; запоминаю, как {c.word} выглядит для «{b}»")
        return None


# ---------- простая физика света для проверок ----------
WAVES = np.arange(380, 750, 10)
COLOR_OF = {"лист": 545, "трава": 535, "огурец": 550, "небо": 470, "море": 480,
            "помидор": 650, "клубника": 640, "банан": 578, "лимон": 575, "мяч": 620}
CAMERA_SENS = [(450, 40), (540, 40), (610, 45)]     # чувствительность каналов камеры — скрыта


def spectrum(peak: float, rng: np.random.Generator, width: float = 25) -> np.ndarray:
    s = np.exp(-((WAVES - peak) ** 2) / (2 * width ** 2))
    return np.clip(s + rng.normal(0, 0.02, len(WAVES)), 0, None)


def camera(spec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    raw = np.array([np.sum(spec * np.exp(-((WAVES - mu) ** 2) / (2 * sd ** 2))) for mu, sd in CAMERA_SENS])
    raw = raw / (np.sum(raw) + 1e-9)
    return np.clip(raw + rng.normal(0, 0.01, 3), 0, None)


DEFINITIONS = {
    "зелёный": "зелёный — это свет с длиной волны от 500 до 565 нанометров, глаз воспринимает его как зелёный цвет",
    "синий": "синий — это свет с длиной волны от 450 до 495 нанометров",
    "красный": "красный — это свет с длиной волны от 625 до 740 нанометров",
    "жёлтый": "жёлтый — это свет с длиной волны от 565 до 590 нанометров",
}
FACTS = ["листья зелёные", "трава зелёная", "небо синее", "помидор красный", "лимон жёлтый"]
