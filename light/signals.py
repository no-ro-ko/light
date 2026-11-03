"""Единый формат обмена между узлами: сигнал (мысль)."""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any

_ids = itertools.count(1)

KINDS = {"percept", "thought", "memory", "question", "plan", "error", "action", "emotion", "output"}


@dataclass
class Signal:
    kind: str
    content: Any
    source: str
    modality: str = "text"
    salience: float = 0.5
    confidence: float = 0.5
    parents: list[int] = field(default_factory=list)
    id: int = field(default_factory=lambda: next(_ids))
    time: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown signal kind: {self.kind}")

    def text(self) -> str:
        return self.content if isinstance(self.content, str) else repr(self.content)

    def __str__(self) -> str:
        return f"#{self.id} [{self.kind}/{self.source} s={self.salience:.2f} c={self.confidence:.2f}] {self.text()}"
