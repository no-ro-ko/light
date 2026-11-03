"""Адаптеры: как подключить конкретную модель к узлу.

Узел знает только интерфейс `ModelAdapter.run(instruction, inputs)`.
Какая модель стоит за ним — текстовая, визуальная, звуковая — неважно.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from typing import Protocol

from .signals import Signal


class ModelAdapter(Protocol):
    def run(self, instruction: str, inputs: list[Signal]) -> str: ...


class MockAdapter:
    """Заглушка без модели: детерминированно пересказывает вход. Для проверки цикла."""

    def __init__(self, role: str):
        self.role = role

    def run(self, instruction: str, inputs: list[Signal]) -> str:
        last = inputs[-1].text() if inputs else ""
        short = last if len(last) <= 80 else last[:77] + "..."
        return f"{self.role}: {short}"


class HTTPChatAdapter:
    """Любой сервер с OpenAI-совместимым API /chat/completions
    (Ollama, vLLM, LM Studio, облачные шлюзы). Изображения передаются
    как data-URL, если модель их поддерживает."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 temperature: float = 0.7, timeout: float = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("LIGHT_API_KEY", "")
        self.temperature = temperature
        self.timeout = timeout

    def _part(self, s: Signal) -> dict:
        if s.modality == "image" and isinstance(s.content, str) and os.path.exists(s.content):
            with open(s.content, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        return {"type": "text", "text": f"[{s.kind} от {s.source}] {s.text()}"}

    def run(self, instruction: str, inputs: list[Signal]) -> str:
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": [self._part(s) for s in inputs] or "(нет входа)"},
            ],
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)["choices"][0]["message"]["content"].strip()
