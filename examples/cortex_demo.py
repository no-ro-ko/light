"""Корковая сеть CLN как отдел общего мозга.

1) «Детство»: сеть читает несколько фраз и учится ожидать продолжение.
2) Подключаем её к мозгу и задаём знакомую и незнакомую фразу.
3) Без внешнего ввода фоновый режим заставляет её «воображать».
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from light import Brain  # noqa: E402
from light.cortex import CorticalNode  # noqa: E402

corpus = ["небо голубое потому что свет рассеивается в воздухе",
          "закат красный потому что свет проходит длинный путь",
          "кот спит на тёплом окне",
          "собака бежит по зелёной траве"]

cortex = CorticalNode("кора")
print("Обучение (среднее удивление по фразам):")
for epoch in range(6):
    s = [np.mean(cortex.read(t)[1:]) for t in corpus]
    print(f"  эпоха {epoch}: " + "  ".join(f"{x:.2f}" for x in s))
print("Воображение «закат» →", " ".join(cortex.imagine("закат")))
print("Состояние области:", cortex.region.stats())

brain = Brain(sleep_every=6)
brain.attach(cortex)

print("\n=== знакомое ===")
brain.hear("кот спит на тёплом окне")
brain.run(max_ticks=6, stop_on_output=True)

print("\n=== незнакомое ===")
brain.hear("кот бежит по красному воздуху")
brain.run(max_ticks=6, stop_on_output=True)

print("\n=== тишина: фоновое обдумывание и сон ===")
brain.run(max_ticks=12)
