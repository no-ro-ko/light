"""Сборка узлов и когнитивный цикл."""
from __future__ import annotations

from typing import Callable

from .nodes import (Amygdala, AssociationCortex, AuditoryCortex, Critic, DefaultModeNetwork,
                    Hippocampus, LanguageCortex, MotorCortex, Node, PrefrontalCortex, SensoryNode,
                    VisualCortex)
from .self_model import SelfModel
from .signals import Signal
from .workspace import Workspace


class Brain:
    def __init__(self, capacity: int = 2, answer_threshold: float = 0.7,
                 sleep_every: int = 10, gate: Callable[[Signal], bool] | None = None,
                 log: Callable[[str], None] = print, self_aware: bool = True):
        self.workspace = Workspace(capacity)
        self.sensory: dict[str, SensoryNode] = {
            n.modality: n for n in (VisualCortex(), AuditoryCortex(), LanguageCortex())}
        self.amygdala = Amygdala()
        self.hippocampus = Hippocampus()
        self.dmn = DefaultModeNetwork()
        self.motor = MotorCortex()
        self.cortex: list[Node] = [self.hippocampus, AssociationCortex(), PrefrontalCortex(), Critic()]
        self.self_model: SelfModel | None = SelfModel() if self_aware else None
        if self.self_model:
            self.cortex.append(self.self_model)
        self.goals: list[str] = []
        self.open_questions: list[str] = []
        self.inbox: list[Signal] = []
        self.outputs: list[Signal] = []
        self.answer_threshold = answer_threshold
        self.sleep_every = sleep_every
        self.gate = gate or (lambda s: True)  # базальные ганглии: разрешить действие?
        self.log = log
        self.ticks = 0
        self.goal_start = 0  # с какого места потока идёт работа над текущей целью

    # --- доступ к узлам для подключения моделей ---
    def node(self, name: str) -> Node:
        for n in [*self.sensory.values(), self.amygdala, self.dmn, self.motor, *self.cortex]:
            if n.name == name:
                return n
        raise KeyError(name)

    def attach(self, node: Node) -> Node:
        """Подключить новый отдел (например, корковую сеть light.cortex.CorticalNode)."""
        self.cortex.append(node)
        return node

    # --- вход ---
    def hear(self, content, modality: str = "text", goal: bool = True) -> None:
        raw = Signal("percept", content, "world", modality=modality)
        self.inbox.append(raw)
        if self.self_model and modality == "text" and self.self_model.is_introspective(str(content)):
            goal = False  # на вопрос о себе отвечают наблюдением за собой, а не обдумыванием
        if goal and modality == "text":
            self.goal_start = len(self.workspace.stream)
            self.goals.append(str(content))
            self.open_questions.append(str(content))

    # --- один такт ---
    def tick(self) -> list[Signal]:
        self.ticks += 1
        # 1. восприятие
        for raw in self.inbox:
            self.workspace.propose([self.sensory[raw.modality].perceive(raw)])
        self.inbox = []
        # 2. фоновый режим, если сцена пуста
        if not self.workspace.candidates:
            spontaneous = self.dmn.wander(self)
            if spontaneous:
                self.workspace.propose([spontaneous])
        # 3. оценка значимости
        for s in self.workspace.candidates:
            self.amygdala.appraise(s, self.goals)
        # 4. конкуренция за сцену
        winners = self.workspace.select()
        if not winners:  # остатки мыслей угасли — шанс фоновому режиму
            spontaneous = self.dmn.wander(self)
            if spontaneous:
                self.amygdala.appraise(spontaneous, self.goals)
                self.workspace.propose([spontaneous])
                winners = self.workspace.select()
        produced: list[Signal] = []
        for focus in winners:
            self.log(f"  [сцена] {focus}")
            self.hippocampus.store(focus)  # 6. запись в память
            if focus.kind == "action":      # 7. действие через шлюз
                if self.gate(focus):
                    out = self.motor.act(focus)
                    self.outputs.append(out)
                    self.log(f"  [ВЫХОД] {out.text()}")
                    for n in self.cortex:  # дофамин: ответ состоялся
                        if hasattr(n, "reward"):
                            n.reward(0.3)
                    self.goals.clear()
                    self.open_questions.clear()
                continue
            for node in self.cortex:        # 5. рассылка всем узлам
                produced.extend(node.on_broadcast(focus, self))
        if self.self_model:                 # я-модель наблюдает такт целиком
            produced.extend(self.self_model.observe(self, winners))
        self.workspace.propose(produced)
        # сон
        if self.sleep_every and self.ticks % self.sleep_every == 0:
            fact = self.hippocampus.consolidate(self.node("association"))
            if fact:
                self.log(f"  [сон] консолидировано: {fact}")
            for n in self.cortex:
                if hasattr(n, "sleep"):
                    self.log(f"  [сон] {n.sleep(self.hippocampus.episodes[-20:])}")
            if self.self_model:
                thr = self.self_model.self_test(self)
                if thr is not None:
                    self.log(f"  [сон] самопроверка памяти: порог «знаю» = {thr:.2f}")
        return winners

    def run(self, max_ticks: int = 20, stop_on_output: bool = False) -> list[Signal]:
        """Крутить цикл. Бюджет тактов — обязательный ограничитель."""
        for _ in range(max_ticks):
            self.log(f"такт {self.ticks + 1}")
            n_out = len(self.outputs)
            winners = self.tick()
            if stop_on_output and len(self.outputs) > n_out:
                break
            if not winners and not self.inbox:
                self.log("  (тишина: думать не о чем)")
                break
        return self.outputs
