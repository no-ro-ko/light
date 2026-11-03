"""Разум как отдел общего мозга (light.Brain).

- Сцены и речь из мира → обучение (восприятие, слова, знания). Если увидено что-то
  совсем новое — на сцену внимания выходит сигнал удивления.
- Вопрос вида «дышит ли пингвин?», «пингвин — это животное?», «летает ли это?» →
  рассуждение ПО ШАГАМ через сцену внимания: каждый шаг цепочки — отдельная мысль,
  которая должна выиграть конкуренцию за внимание, прежде чем появится следующая.
  Ответ выдаётся, когда цепочка пройдена.
- В простое (фоновый режим, нет цели) — любопытство: система сама формулирует самый
  полезный вопрос и выдаёт его наружу вместе с объяснением, зачем спрашивает.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ..nodes import Node
from ..signals import Signal
from .curiosity import Curiosity
from .mind import Mind

if TYPE_CHECKING:
    from ..brain import Brain

Q_LI = re.compile(r"^\s*(\w+)\s+ли\s+(вот\s+это|это|\w+)\s*\??\s*$")
Q_ISA = re.compile(r"^\s*(\w+)\s*(?:—|-)?\s*это\s+(\w+)\s*\?\s*$")


class MindNode(Node):
    name = "разум"

    def __init__(self, mind: Mind, curiosity: Curiosity | None = None, curious: bool = True):
        super().__init__()
        self.mind = mind
        self.curiosity = curiosity or Curiosity(mind)
        self.curious = curious
        self.jobs: dict[int, dict] = {}
        self.last_chain: list[str] = []
        self.last_answer: str = ""
        self.waiting_answer = False
        self._job_ids = 0

    # ---------- разбор вопроса ----------
    def parse_question(self, text: str):
        t = text.lower().strip()
        m = Q_LI.match(t)
        if m:
            y, x = m.group(1), m.group(2)
            return ("thing" if x in ("это", "вот это") else "word", x, y)
        m = Q_ISA.match(t)
        if m and m.group(1) in self.mind.nouns() and m.group(2) in self.mind.nouns():
            return ("word", m.group(1), m.group(2))
        return None

    # ---------- работа на сцене внимания ----------
    def on_broadcast(self, focus: Signal, brain: "Brain") -> list[Signal]:
        if focus.kind == "percept" and focus.data.get("scene") is not None:
            return self._experience(focus)
        if focus.kind == "percept":
            q = self.parse_question(focus.text())
            if q:
                return self._start(q, focus, brain)
        if focus.source == self.name and focus.data.get("job") in self.jobs:
            return self._next(focus.data["job"], focus)
        if self.curious and focus.source == "dmn" and not brain.goals and not self.waiting_answer:
            return self._ask()
        return []

    def _experience(self, focus: Signal) -> list[Signal]:
        self.waiting_answer = False
        scene = focus.data["scene"]
        before = len(self.mind.perception.protos)
        cats = self.mind.experience(scene, focus.data.get("utterance", ""), focus.data.get("focus"))
        self.curiosity.saw(cats, [lk for lk, _ in scene])
        if len(self.mind.perception.protos) > before:
            _, like, sim = self.mind.identify(scene[cats.index(max(cats))][0])
            hint = f" — немного похоже на «{like}»" if like else ""
            return [Signal("surprise", f"вижу что-то совсем новое{hint}", self.name,
                           salience=0.8, confidence=0.8, parents=[focus.id])]
        return []

    def _start(self, q, focus: Signal, brain: "Brain") -> list[Signal]:
        kind, x, y = q
        if kind == "thing":
            scene = brain.last_scene or []
            if not scene:
                return []
            v = self.mind.judge_thing(scene[0][0], y)
        else:
            v = self.mind.ask(x, y)
        self._job_ids += 1
        self.jobs[self._job_ids] = {"steps": list(v.chain), "i": 0, "verdict": v, "q": focus.text()}
        return self._next(self._job_ids, focus)

    def _next(self, job: int, parent: Signal) -> list[Signal]:
        j = self.jobs[job]
        v = j["verdict"]
        if j["i"] < len(j["steps"]):
            step = j["steps"][j["i"]]
            j["i"] += 1
            return [Signal("thought", step, self.name, salience=0.85, confidence=v.conf,
                           parents=[parent.id], data={"job": job, "step": j["i"]})]
        del self.jobs[job]
        self.last_chain = j["steps"]
        self.last_answer = v.say()
        text = v.say().capitalize() + (". Я рассуждала так: " + " → ".join(j["steps"]) if j["steps"] else ".")
        return [Signal("action", text, self.name, salience=0.9, confidence=max(v.conf, 0.5),
                       parents=[parent.id],
                       data={"answer": v.answer(), "source": v.source, "chain": j["steps"]})]

    def _ask(self) -> list[Signal]:
        q = self.curiosity.best()
        if q is None or q.value < 1.0:
            return []
        self.curiosity.mark(q)
        self.waiting_answer = True
        return [Signal("action", f"Вопрос: {q.text} (спрашиваю, потому что {q.why})", self.name,
                       salience=0.8, confidence=0.9, data={"ask": q.for_parent(), "question": q.text,
                                                           "why": q.why, "value": q.value})]
