"""Демонстрация цикла на заглушках (без моделей).

Подключение реальной модели (любой OpenAI-совместимый сервер, напр. Ollama):

    from light import HTTPChatAdapter
    llm = HTTPChatAdapter("http://localhost:11434/v1", "qwen2.5:14b")
    for name in ["association", "prefrontal", "critic", "dmn", "motor"]:
        brain.node(name).adapter = llm
    brain.node("visual").adapter = HTTPChatAdapter("http://localhost:11434/v1", "llava")
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from light import Brain  # noqa: E402

brain = Brain()
brain.hear("Почему небо голубое, а закат красный?")
brain.run(max_ticks=8)

print("\n--- внешнего ввода нет: фоновое обдумывание ---")
brain.run(max_ticks=4)
