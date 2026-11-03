"""Узел-переходник: корковая сеть как «отдел» общего мозга (light.Brain).

Что делает на сцене внимания:
- читает текст (восприятие, вопросы, воспоминания) слово за словом;
- работает как таламический фильтр: наверх с высокой важностью идёт только
  НЕОЖИДАННОЕ (сигнал `surprise`), привычное почти не шумит;
- предлагает ожидаемое продолжение (`thought`);
- на фоновые вопросы (режим без внешнего входа) — «воображает» продолжение,
  прокручивая собственные ожидания при низком ацетилхолине;
- принимает награду (дофамин) и засыпает вместе с мозгом.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..nodes import Node
from ..signals import Signal
from .encoder import WordEncoder
from .region import CorticalRegion

if TYPE_CHECKING:
    from ..brain import Brain


class CorticalNode(Node):
    listens = {"percept", "question"}  # свои же воспоминания о своём удивлении не перечитываем

    def __init__(self, name: str = "cortex", region: CorticalRegion | None = None,
                 encoder: WordEncoder | None = None, surprise_level: float = 0.5):
        super().__init__()
        self.name = name
        self.encoder = encoder or WordEncoder()
        self.region = region or CorticalRegion(name, self.encoder.size)
        self.surprise_level = surprise_level
        self.lexicon: dict[str, np.ndarray] = {}  # слово → типичные колонки (выход «к речи»)

    # ---------- чтение и декодирование ----------
    def read(self, text: str, learn: bool = True) -> list[float]:
        self.region.reset()
        anomalies = []
        for w in self.encoder.tokens(text):
            anomalies.append(self.region.step(self.encoder.encode(w), learn))
            cols = self.region.active_cols.astype(float)
            prev = self.lexicon.get(w)
            self.lexicon[w] = cols if prev is None else 0.8 * prev + 0.2 * cols
        return anomalies

    def decode(self, col_strength: np.ndarray, top: int = 3) -> list[str]:
        if not self.lexicon or col_strength.max() <= 0:
            return []
        x = np.minimum(col_strength, 1)
        scores = {w: float(v @ x) / (v.sum() or 1) for w, v in self.lexicon.items()}
        best = sorted(scores, key=scores.get, reverse=True)[:top]
        return [w for w in best if scores[w] >= 0.5]

    def expect(self) -> list[str]:
        return self.decode(self.region.prediction())

    def imagine(self, seed: str, steps: int = 6) -> list[str]:
        """Внутренняя речь: продолжить мысль без внешнего входа."""
        nm = self.region.nm
        ach = nm.acetylcholine
        nm.acetylcholine = 0.1
        self.read(seed, learn=False)
        out = []
        for _ in range(steps):
            words = self.expect()
            if not words:
                break
            out.append(words[0])
            self.region.step(self.encoder.encode(words[0]), learn=False)
        nm.acetylcholine = ach
        return out

    # ---------- связь с мозгом ----------
    def on_broadcast(self, focus: Signal, brain: "Brain") -> list[Signal]:
        if focus.kind not in self.listens or focus.source == self.name:
            return []
        nm = self.region.nm
        nm.noradrenaline = max(nm.noradrenaline, focus.salience * 0.5)  # важное → бдительность
        text = focus.text()
        if focus.source == "dmn":
            known = [w for w in self.encoder.tokens(text) if w in self.lexicon]
            tries = [" ".join(known[i:i + 2]) for i in range(len(known))]
            seed, cont = max(((s, self.imagine(s)) for s in tries), key=lambda x: len(x[1]),
                             default=("", []))
            if not cont:
                return []
            return [Signal("thought", f"воображаю: {seed} … {' '.join(cont)}", self.name,
                           salience=0.6, confidence=0.4, parents=[focus.id])]
        anomalies = self.read(text)
        if not anomalies:
            return []
        out = []
        surprise = float(np.mean(anomalies[1:] or anomalies))
        if surprise >= self.surprise_level:
            out.append(Signal("surprise", f"новое для меня: «{text[:60]}» (удивление {surprise:.2f})",
                              self.name, salience=0.3 + 0.5 * surprise, confidence=0.8,
                              parents=[focus.id]))
        nxt = self.expect()
        if nxt:
            out.append(Signal("thought", f"ожидаю дальше: {', '.join(nxt)}", self.name,
                              salience=0.35, confidence=1 - surprise, parents=[focus.id]))
        return out

    def reward(self, value: float) -> None:
        self.region.nm.dopamine = min(1.0, self.region.nm.dopamine + value)

    def sleep(self, episodes: list[Signal]) -> str:
        """Повтор пережитого при низком ацетилхолине, затем ослабление и обрезка."""
        nm = self.region.nm
        ach = nm.acetylcholine
        nm.acetylcholine = 0.3
        for e in episodes:
            if e.source in ("world", "language"):
                self.read(e.text(), learn=True)
        nm.acetylcholine = ach
        pruned = self.region.downscale()
        return f"{self.name}: повтор {len(episodes)} эпизодов, обрезано связей: {pruned}"
