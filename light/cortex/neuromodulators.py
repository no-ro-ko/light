"""Нейромодуляторы — «химия» мозга. Это не сигналы, а глобальные регуляторы режима работы.

Каждый — число 0..1, влияющее на ВСЮ область сразу (как вещество, разлитое в коре):

- дофамин (DA)      — «это было хорошо»: усиливает закрепление того, что только что сработало;
- норадреналин (NE) — «внимание, неожиданность!»: растёт от удивления, ускоряет
                      выращивание новых связей и расширяет поиск (больше активных колонок);
- ацетилхолин (ACh) — «смотри наружу или внутрь»: высокий — верим входу (бодрствование),
                      низкий — верим своим ожиданиям (воображение, сон);
- серотонин (5-HT)  — «терпение»: как долго держится контекст сверху (обратная связь).

Упрощение: в мозге их эффекты гораздо разнообразнее и зависят от типа рецепторов.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Neuromodulators:
    dopamine: float = 0.5
    noradrenaline: float = 0.2
    acetylcholine: float = 0.9
    serotonin: float = 0.6

    def clamp(self) -> None:
        for k in ("dopamine", "noradrenaline", "acetylcholine", "serotonin"):
            setattr(self, k, min(1.0, max(0.0, getattr(self, k))))

    def relax(self, rate: float = 0.1) -> None:
        """Возврат к фоновому уровню (вещества разрушаются и выводятся)."""
        base = Neuromodulators()
        for k in ("dopamine", "noradrenaline", "serotonin"):
            v = getattr(self, k)
            setattr(self, k, v + rate * (getattr(base, k) - v))
