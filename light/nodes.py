"""Узлы-«отделы». Каждый узел — роль; модель подключается через adapter."""
from __future__ import annotations

import random
from typing import TYPE_CHECKING

from .adapters import MockAdapter, ModelAdapter
from .signals import Signal, core_text

if TYPE_CHECKING:
    from .brain import Brain


class Node:
    name = "node"
    instruction = ""
    listens: set[str] = set()  # на какие виды сигналов реагирует

    def __init__(self, adapter: ModelAdapter | None = None):
        self.adapter: ModelAdapter = adapter or MockAdapter(self.name)

    def think(self, inputs: list[Signal]) -> str:
        return self.adapter.run(self.instruction, inputs)

    def on_broadcast(self, focus: Signal, brain: "Brain") -> list[Signal]:
        return []


# ---------- Сенсорные узлы ----------

class SensoryNode(Node):
    """Сырой сигнал своей модальности → восприятие в текстовом виде."""
    modality = "text"

    def perceive(self, raw: Signal) -> Signal:
        if raw.modality == "text" and self.modality == "text":
            desc = raw.text()
        else:
            desc = self.think([raw])
        return Signal("percept", desc, self.name, salience=0.7, confidence=0.8, parents=[raw.id])


class VisualCortex(SensoryNode):
    name, modality = "visual", "image"
    instruction = "Опиши изображение: объекты, действия, необычное. Кратко."


class AuditoryCortex(SensoryNode):
    name, modality = "auditory", "audio"
    instruction = "Преобразуй звук в текст и отметь интонацию, эмоции, источник."


class LanguageCortex(SensoryNode):
    name, modality = "language", "text"


# ---------- Центральные узлы ----------

class Amygdala(Node):
    """Оценка значимости. Без модели — простые правила; с моделью можно уточнять."""
    name = "amygdala"

    def appraise(self, s: Signal, goals: list[str]) -> None:
        bonus = {"question": 0.2, "error": 0.25, "surprise": 0.25, "self": 0.1, "percept": 0.15, "action": 0.3, "plan": 0.1}.get(s.kind, 0.0)
        text = s.text().lower()
        if any(w in text for g in goals for w in g.lower().split() if len(w) > 3):
            bonus += 0.15  # связано с текущей целью
        bonus += 0.1 * (1 - s.confidence)  # неуверенность требует внимания
        if not goals and s.kind not in ("percept", "question", "surprise", "self"):
            bonus -= 0.25  # нет цели — «пережёвывание» затухает, остаётся место фоновым вопросам
        s.salience = max(0.0, min(1.0, s.salience + bonus))


class Hippocampus(Node):
    """Эпизодическая память. Поиск по совпадению слов; в реальной системе —
    эмбеддинги + векторная база."""
    name = "hippocampus"

    def __init__(self, adapter=None):
        super().__init__(adapter)
        self.episodes: list[Signal] = []
        self.semantic: list[str] = []  # консолидированные знания после «сна»

    def store(self, s: Signal) -> None:
        self.episodes.append(s)

    def recall(self, s: Signal, k: int = 1) -> list[Signal]:
        words = {w for w in s.text().lower().split() if len(w) > 3}
        scored = [(len(words & set(e.text().lower().split())), e)
                  for e in self.episodes if e.id != s.id and e.id not in s.parents]
        scored = [x for x in scored if x[0] > 0]
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:k]]

    def on_broadcast(self, focus, brain):
        return [Signal("memory", f"вспомнил: {core_text(m.text())}", self.name, salience=0.4,
                       confidence=0.6, parents=[focus.id, m.id]) for m in self.recall(focus)]

    def consolidate(self, summarizer: Node) -> str | None:
        """«Сон»: сжать недавние эпизоды в обобщённое знание."""
        if len(self.episodes) < 3:
            return None
        fact = summarizer.adapter.run(
            "Сожми эти эпизоды в 1–2 устойчивых вывода.", self.episodes[-10:])
        self.semantic.append(fact)
        return fact


class AssociationCortex(Node):
    """Склеивает восприятие и воспоминания в идею."""
    name = "association"
    instruction = "Свяжи входные данные в одну новую мысль или гипотезу. Кратко."
    listens = {"percept", "memory", "question"}

    def on_broadcast(self, focus, brain):
        if focus.kind not in self.listens:
            return []
        ctx = brain.workspace.recent(4)
        return [Signal("thought", self.think(ctx + [focus]), self.name,
                       salience=0.5, confidence=0.5, parents=[focus.id])]


class PrefrontalCortex(Node):
    """Цели, план, решение «отвечать или думать дальше»."""
    name = "prefrontal"
    instruction = ("Ты исполнительный центр. Дано: цели и текущий поток мыслей. "
                   "Сформулируй следующий шаг рассуждения или итоговый ответ.")
    listens = {"thought", "percept", "error", "question"}

    def on_broadcast(self, focus, brain):
        if focus.kind not in self.listens:
            return []
        goals = [Signal("plan", g, "goals") for g in brain.goals]
        step = self.think(goals + brain.workspace.recent(6))
        depth = sum(1 for s in brain.workspace.stream[brain.goal_start:] if s.source == self.name)
        # уверенность растёт с проработкой и падает при ошибках критика
        errors = sum(1 for s in brain.workspace.recent(4) if s.kind == "error")
        conf = min(0.95, 0.3 + 0.15 * depth - 0.2 * errors)
        out = [Signal("plan", step, self.name, salience=0.5, confidence=max(0.1, conf), parents=[focus.id])]
        if brain.goals and conf >= brain.answer_threshold:
            out.append(Signal("action", step, self.name, salience=0.6, confidence=conf, parents=[focus.id]))
        return out


class Critic(Node):
    """Поясная кора + мозжечок: ищет противоречия и ошибки."""
    name = "critic"
    instruction = "Найди ошибку, противоречие или пробел. Если всё в порядке — ответь ОК."
    listens = {"thought", "plan"}

    def on_broadcast(self, focus, brain):
        if focus.kind not in self.listens:
            return []
        verdict = self.think([focus])
        if verdict.strip().upper().startswith("ОК") or isinstance(self.adapter, MockAdapter):
            return []
        return [Signal("error", verdict, self.name, salience=0.6, confidence=0.6, parents=[focus.id])]


class DefaultModeNetwork(Node):
    """Фоновый режим: мысли без внешнего стимула."""
    name = "dmn"
    instruction = "Свободно поразмышляй над этим: что тут непонятно, с чем это связано?"

    def __init__(self, adapter=None, seed: int = 0):
        super().__init__(adapter)
        self.rng = random.Random(seed)

    def wander(self, brain: "Brain") -> Signal | None:
        pool = brain.open_questions or [e.text() for e in brain.hippocampus.episodes[-20:]
                                        if e.source != "self"]
        pool = [p for p in pool if not brain.workspace.is_inhibited(p)]  # схема внимания
        if not pool:
            return None
        seed_text = self.rng.choice(pool)
        idea = self.think([Signal("memory", seed_text, self.name)])
        return Signal("question", idea, self.name, salience=0.45, confidence=0.3)


class MotorCortex(Node):
    """Брока / моторная кора: формирует ответ наружу (текст, речь, картинка)."""
    name = "motor"
    instruction = "Сформулируй окончательный ответ пользователю ясно и кратко."

    def act(self, action: Signal) -> Signal:
        # самоотчёт выдаётся дословно: его смысл — точность, а не красота
        text = action.text() if action.source == "self" or isinstance(self.adapter, MockAdapter) \
            else self.think([action])
        return Signal("output", text, self.name, confidence=action.confidence,
                      parents=[action.id], data=action.data)
