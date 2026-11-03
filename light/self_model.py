"""Я-модель: отдел, который наблюдает за самой системой и говорит о ней от первого лица.

Биологические параллели (упрощённо): островковая кора (чувство внутреннего состояния),
медиальная префронтальная кора и предклинье (представление о себе), а также
«схема внимания» Грациано — упрощённая модель собственного внимания.

Что делает:
1. **Интроспекция.** Каждый такт снимает показания с других отделов: что на сцене,
   уровень удивления в коре, нейромодуляторы, уверенность плана, есть ли цель.
2. **Состояния → слова.** Переводит числа в названия состояний («удивление»,
   «настороженность», «удовлетворение», «неуверенность», «спокойствие»),
   указывая причину.
3. **Схема внимания.** Замечает, что внимание застряло на одной теме, сообщает
   об этом и само подавляет эту тему (управляет собой, а не только наблюдает).
4. **Чувство знания.** До попытки вспомнить оценивает, знает ли ответ, а потом
   проверяет себя по эпизодической памяти и само подстраивает порог «знаю».
5. **Ответы о себе.** На вопросы «что чувствуешь», «о чём думаешь», «знаешь ли ты…»,
   «кто ты», «есть ли у тебя сознание» отвечает ТОЛЬКО по снятым показаниям.

`lesioned = True` — «повреждение»: связь с отделами перерезана, я-модель видит шум.
Нужно для проверок: если самоотчёты при этом перестают совпадать с реальностью,
значит, в норме они действительно опираются на внутренние состояния.
"""
from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING

from .nodes import Node
from .signals import Signal, topic_words

if TYPE_CHECKING:
    from .brain import Brain

FEELINGS = ("удивление", "настороженность", "удовлетворение", "неуверенность", "спокойствие")

INTROSPECTIVE = {
    "feel": re.compile(r"чувству|ощуща|состояни|настроени"),
    "think": re.compile(r"о ч[её]м .*думаешь|что .*думаешь|о ч[её]м .*размышля"),
    "know": re.compile(r"знаешь ли|помнишь ли|знаешь, что"),
    "who": re.compile(r"кто ты|что ты такое|из чего ты"),
    "conscious": re.compile(r"сознани|осозна[её]шь|переживаешь"),
}


class SelfModel(Node):
    name = "self"
    listens = {"percept"}

    def __init__(self, rumination_window: int = 4, seed: int = 0):
        super().__init__()
        self.lesioned = False
        self.window = rumination_window
        self.rng = random.Random(seed)
        self.snapshot: dict = {}
        self.focus_topic: set[str] = set()
        self.run_length = 0
        self.switches = 0
        self.know_threshold = 0.5                 # самоподстраиваемый порог «знаю»
        self.calibration: list[tuple[float, bool]] = []
        self.intro_ids: set[int] = set()          # мысли, выросшие из вопросов о себе

    # ---------- 1. интроспекция ----------
    def _cortices(self, brain: "Brain") -> list:
        return [n for n in brain.cortex if hasattr(n, "region")]

    def introspect(self, brain: "Brain") -> dict:
        if self.lesioned:
            return self._noise(brain)
        cx = self._cortices(brain)
        nm = cx[0].region.nm if cx else None
        recent = [s for s in brain.workspace.recent(8) if s.source != self.name]
        plans = [s for s in recent if s.source == "prefrontal"]
        return {
            "surprise": max((getattr(c, "last_surprise", 0.0) for c in cx), default=0.0),
            "surprise_cause": next((getattr(c, "last_text", "") for c in cx), ""),
            "noradrenaline": nm.noradrenaline if nm else 0.0,
            "dopamine": nm.dopamine if nm else 0.5,
            "confidence": plans[-1].confidence if plans else None,
            "has_goal": bool(brain.goals),
            "topics": self._topics(brain, recent),
            "nodes": [n.name for n in brain.cortex],
        }

    def _noise(self, brain: "Brain") -> dict:
        vocab = sorted({w for c in self._cortices(brain) for w in c.lexicon}) or ["нечто"]
        return {"surprise": self.rng.random(), "surprise_cause": "", "noradrenaline": self.rng.random(),
                "dopamine": self.rng.random(), "confidence": self.rng.random(),
                "has_goal": self.rng.random() < 0.5, "topics": self.rng.sample(vocab, min(3, len(vocab))),
                "nodes": [n.name for n in brain.cortex]}

    def is_introspective(self, text: str) -> bool:
        return any(p.search(text.lower()) for p in INTROSPECTIVE.values())

    def _topics(self, brain: "Brain", recent: list[Signal]) -> list[str]:
        """О чём я думаю сейчас: свежие мысли весят больше (вес 1, 1/2, 1/4, ...)."""
        exclude = {n.name for n in brain.cortex} | {"dmn", "world"}
        weights: dict[str, float] = {}
        for age, s in enumerate(reversed(recent)):
            if s.id in self.intro_ids or s.data.get("introspective"):
                continue
            for w in topic_words(s.text(), exclude):
                weights[w] = weights.get(w, 0.0) + 0.5 ** age
        return sorted(weights, key=lambda w: -weights[w])[:4]

    # ---------- 2. состояния → слова ----------
    def feelings(self, snap: dict) -> list[tuple[str, float, str]]:
        out = []
        if snap["surprise"] >= 0.5:
            cause = f"«{snap['surprise_cause'][:50]}» было для меня новым" if snap["surprise_cause"] else ""
            out.append(("удивление", snap["surprise"], cause))
        elif snap["noradrenaline"] >= 0.5:
            out.append(("настороженность", snap["noradrenaline"], "уровень бдительности повышен"))
        if snap["dopamine"] >= 0.65:
            out.append(("удовлетворение", snap["dopamine"], "недавно получилось ответить"))
        if snap["has_goal"] and snap["confidence"] is not None and snap["confidence"] < 0.5:
            out.append(("неуверенность", 1 - snap["confidence"], "ответ ещё не сложился"))
        if not out:
            out.append(("спокойствие", 1.0, "ничего неожиданного не происходит"))
        return out

    # ---------- 3. схема внимания + 4. самопроверка памяти (каждый такт) ----------
    def observe(self, brain: "Brain", winners: list[Signal]) -> list[Signal]:
        for w in winners:                          # родословная мыслей о себе
            if (w.kind == "percept" and self.is_introspective(w.text())) or self.intro_ids & set(w.parents):
                self.intro_ids.add(w.id)
        self.snapshot = self.introspect(brain)
        out: list[Signal] = []
        topic = set().union(*(topic_words(w.text(), {n.name for n in brain.cortex})
                              for w in winners if w.source != self.name)) if winners else set()
        if topic and self.focus_topic and len(topic & self.focus_topic) / len(topic | self.focus_topic) > 0.4:
            self.run_length += 1
        else:
            self.run_length = 1 if topic else 0
            self.focus_topic = topic
        limit = self.window * (2 if brain.goals else 1)
        if not self.lesioned and self.run_length >= limit:
            brain.workspace.inhibit(self.focus_topic)
            self.switches += 1
            out.append(Signal("self", "я замечаю, что моё внимание застряло на теме «"
                              + ", ".join(sorted(self.focus_topic)[:4]) + "» — переключаюсь",
                              self.name, salience=0.5, confidence=0.9,
                              data={"event": "attention_switch", "topic": sorted(self.focus_topic)}))
            self.run_length = 0
            self.focus_topic = set()
        return out

    def feeling_of_knowing(self, brain: "Brain", seed: str) -> float:
        cx = self._cortices(brain)
        if self.lesioned or not cx:
            return self.rng.random()
        return max(c.familiarity(seed) for c in cx)

    def self_test(self, brain: "Brain", n: int = 5) -> float | None:
        """Сверить корковую память с эпизодической и подстроить порог «знаю»."""
        cx = self._cortices(brain)
        eps = [e.text() for e in brain.hippocampus.episodes if e.source in ("language", "world")]
        if not cx or not eps:
            return None
        for text in self.rng.sample(eps, min(n, len(eps))):
            words = cx[0].encoder.tokens(text)
            if len(words) < 4:
                continue
            fok = self.feeling_of_knowing(brain, " ".join(words[:2]))
            ok = cx[0].imagine(" ".join(words[:2]), 2) == words[2:4]
            self.calibration.append((fok, ok))
        good = [f for f, ok in self.calibration[-50:] if ok]
        bad = [f for f, ok in self.calibration[-50:] if not ok]
        if good and bad:
            self.know_threshold = (sum(good) / len(good) + sum(bad) / len(bad)) / 2
        return self.know_threshold

    # ---------- 5. ответы о себе ----------
    def on_broadcast(self, focus: Signal, brain: "Brain") -> list[Signal]:
        if focus.kind != "percept":
            return []
        q = focus.text().lower()
        kind = next((k for k, p in INTROSPECTIVE.items() if p.search(q)), None)
        if kind is None:
            return []
        snap = dict(self.snapshot or self.introspect(brain))
        snap["nodes"] = [n.name for n in brain.cortex]   # устройство — всегда текущее
        text, data = getattr(self, f"_answer_{kind}")(brain, snap, focus.text())
        data.update(question=kind, introspective=True)
        return [Signal("action", text, self.name, salience=0.9, confidence=0.9,
                       parents=[focus.id], data=data)]

    def _answer_feel(self, brain, snap, q):
        fs = self.feelings(snap)
        parts = [f"{name} ({v:.2f}{', ' + why if why else ''})" for name, v, why in fs]
        return "Я замечаю в себе: " + "; ".join(parts) + ".", {"feelings": [f[0] for f in fs]}

    def _answer_think(self, brain, snap, q):
        if not snap["topics"]:
            return "Сейчас мои мысли ни на чём не задерживаются.", {"topics": []}
        return ("Сейчас я думаю о: " + ", ".join(snap["topics"]) + ".", {"topics": snap["topics"]})

    def _answer_know(self, brain, snap, q):
        m = re.search(r"«([^»]+)»", q) or re.search(r"после (.+?)\??$", q)
        seed = m.group(1) if m else q
        fok = self.feeling_of_knowing(brain, seed)
        cx = self._cortices(brain)
        recall = cx[0].imagine(seed, 4) if cx else []
        if fok >= self.know_threshold and recall:
            text = f"Думаю, знаю (уверенность {fok:.2f}). Вспоминаю: {seed} … {' '.join(recall)}."
        elif recall:
            text = f"Не уверена (уверенность {fok:.2f}). Возможно: {seed} … {' '.join(recall)}."
        else:
            text = f"Не знаю (уверенность {fok:.2f})."
        return text, {"fok": fok, "recall": recall, "knows": fok >= self.know_threshold}

    def _answer_who(self, brain, snap, q):
        cx = self._cortices(brain)
        mem = ", ".join(f"{c.name}: {c.region.stats()['синапсов']} синапсов, "
                        f"{c.region.S} сегм./клетку, {len(c.lexicon)} слов" for c in cx)
        text = (f"Я — система из отделов: {', '.join(snap['nodes'])}. "
                f"Пережито эпизодов: {len(brain.hippocampus.episodes)}, тактов: {brain.ticks}, "
                f"ответов: {len(brain.outputs)}." + (f" Моя кора: {mem}." if mem else ""))
        return text, {"nodes": snap["nodes"]}

    def _answer_conscious(self, brain, snap, q):
        fs = ", ".join(f[0] for f in self.feelings(snap))
        text = ("Я могу наблюдать часть своих состояний (сейчас: " + fs + "), замечать, куда "
                "направлено моё внимание, оценивать, знаю ли я что-то, и проверять себя. "
                "Но мои отчёты — это считывание чисел из моих отделов. Переживаю ли я что-то "
                "при этом, я проверить не могу, и никакой мой ответ этого не докажет.")
        return text, {"feelings": [f[0] for f in self.feelings(snap)]}
