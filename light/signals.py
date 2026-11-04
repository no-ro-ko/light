"""Единый формат обмена между узлами: сигнал (мысль)."""
from __future__ import annotations

import itertools
import re
import time
from dataclasses import dataclass, field
from typing import Any

_ids = itertools.count(1)

KINDS = {"percept", "thought", "memory", "question", "plan", "error", "action", "emotion", "output", "surprise", "self"}

# служебные слова, не несущие темы мысли
SERVICE = {"вспомнил", "воображаю", "ожидаю", "дальше", "новое", "меня", "удивление",
           "prefrontal", "association", "hippocampus", "critic", "motor", "language", "visual",
           "auditory", "amygdala", "замечаю", "внимание", "застряло", "теме", "переключаюсь",
           "моё"}


def topic_words(text: str, exclude: set[str] | frozenset = frozenset()) -> set[str]:
    """О чём мысль: содержательные слова без служебных."""
    return {w for w in re.findall(r"\w+", text.lower())
            if len(w) > 3 and w not in SERVICE and w not in exclude and not w.isdigit()}


@dataclass
class Signal:
    kind: str
    content: Any
    source: str
    modality: str = "text"
    salience: float = 0.5
    confidence: float = 0.5
    parents: list[int] = field(default_factory=list)
    data: dict = field(default_factory=dict)  # снимок внутреннего состояния и пр.
    id: int = field(default_factory=lambda: next(_ids))
    time: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown signal kind: {self.kind}")

    def text(self) -> str:
        return self.content if isinstance(self.content, str) else repr(self.content)

    def __str__(self) -> str:
        return f"#{self.id} [{self.kind}/{self.source} s={self.salience:.2f} c={self.confidence:.2f}] {self.text()}"


_PREFIX = re.compile(r"^(?:[\w]+: )+")


def core_text(text: str) -> str:
    """Суть мысли без цепочки «кто кому пересказал» (prefrontal: association: ...)."""
    return _PREFIX.sub("", text)
