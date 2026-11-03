# light

Модульная архитектура «мозга» из подключаемых моделей (текстовых, визуальных, звуковых),
которая не только отвечает на вопросы, но и крутит собственный цикл обдумывания.

- Описание архитектуры: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Каркас на Python (без зависимостей): `light/`
- Собственная корковая нейросеть CLN (колонки, слои, нейромодуляторы, сон): [docs/CORTEX.md](docs/CORTEX.md)
- Демо на заглушках: `python3 examples/demo.py`
- Демо с корой: `python3 examples/cortex_demo.py` (нужен `numpy`)
- Тесты: `python3 -m unittest discover tests`
