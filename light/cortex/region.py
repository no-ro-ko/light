"""Корковая область из колонок со слоями (Columnar Laminar Network, CLN).

Единица сети — не нейрон и не слой, а КОРТИКАЛЬНАЯ КОЛОНКА: вертикальная группа
клеток, внутри которой есть слои с разными функциями (как в неокортексе):

  L4    — вход «снизу»: колонки конкурируют за входной сигнал (боковое торможение,
          k победителей). Проксимальные синапсы, обучение по Хеббу.
  L2/3  — контекст: в каждой колонке K клеток. Какая именно клетка активна,
          зависит от того, что было ДО — так одно слово в разных фразах кодируется
          по-разному. Клетки имеют дендритные сегменты (≈ «мини-детекторы» на ветвях
          дендритов): если сегмент узнал прошлую картину — клетка «предвозбуждена»,
          то есть ПРЕДСКАЗЫВАЕТ, что её колонка будет активна.
  L5    — выход в другие области: устойчивый (растянутый во времени) код.
  L6    — обратная связь: что область ожидает на следующем шаге (уходит в «таламус»,
          который сравнивает ожидание с реальностью).
  L1/апикальные дендриты — контекст сверху: подсказки от более высоких областей.

Один вызов `step()` — один тета-цикл: сначала ожидание (L6) уже выставлено,
затем приходит вход, сравнивается с ожиданием, получаем «удивление» и обучаемся.

Обучение только локальное (без обратного распространения ошибки):
усиление/ослабление существующих синапсов + рост новых и обрезка мёртвых
(структурная пластичность). Скорость обучения задают нейромодуляторы.
Гомеостаз: редко побеждающие колонки получают «бонус возбудимости».

Сон: повтор пережитого при низком ацетилхолине + общее ослабление всех
синапсов (гипотеза синаптического гомеостаза), слабые связи отмирают.
"""
from __future__ import annotations

import numpy as np

from .neuromodulators import Neuromodulators

CONNECTED = 0.5  # порог «синапс подключён»


class CorticalRegion:
    def __init__(self, name: str, input_size: int, columns: int = 256, cells: int = 8,
                 active_columns: int = 12, segments: int = 8, synapses: int = 32,
                 activation_threshold: int = 7, match_threshold: int = 4, seed: int = 0):
        self.name = name
        self.input_size = input_size
        self.C, self.K, self.S, self.M = columns, cells, segments, synapses
        self.N = columns * cells
        self.k = active_columns
        self.theta = activation_threshold
        self.min_match = match_threshold
        self.rng = np.random.default_rng(seed)
        self.nm = Neuromodulators()

        # L4: проксимальные синапсы. Каждая колонка «видит» ~60% входа.
        self.potential = self.rng.random((self.C, input_size)) < 0.6
        self.ff_perm = np.where(self.potential, self.rng.normal(CONNECTED, 0.1, (self.C, input_size)), 0.0)
        self.duty = np.full(self.C, self.k / self.C)
        self.boost = np.ones(self.C)

        # L2/3: дендритные сегменты. seg_pre = индекс пресинаптической клетки (-1 = пусто).
        self.seg_pre = np.full((self.N, self.S, self.M), -1, dtype=np.int32)
        self.seg_perm = np.zeros((self.N, self.S, self.M))
        self.seg_used = np.zeros((self.N, self.S))

        # L1: контекст сверху (по колонкам)
        self.apical = np.zeros(self.C)
        self.pooled = np.zeros(self.N)  # L5: растянутый во времени выход

        self.t = 0
        self.reset()

    # ---------- состояние ----------
    def reset(self) -> None:
        """Пауза/тишина: текущий контекст гаснет (как граница между фразами)."""
        self.active = np.zeros(self.N, dtype=bool)
        self.winners = np.zeros(self.N, dtype=bool)
        self.active_cols = np.zeros(self.C, dtype=bool)
        self._compute_segments()
        self.anomaly = 0.0

    def _compute_segments(self) -> None:
        ext = np.append(self.active, False)          # индекс -1 → False
        hit = ext[self.seg_pre]
        valid = self.seg_pre >= 0
        self.seg_overlap = (hit & (self.seg_perm >= CONNECTED)).sum(-1)
        # Для выбора клетки при «вспышке» сравниваем только с клетками-победителями:
        # иначе новая фраза прилипает к старой через общую вспышку и контексты сливаются.
        win = np.append(self.winners, False)[self.seg_pre]
        self.seg_potential = (win & valid).sum(-1)
        self.seg_active = self.seg_overlap >= self.theta
        self.predictive = self.seg_active.any(1)               # любое ожидание (в т.ч. размытое)
        # «Точное» ожидание: сегмент узнал именно клетки-победители, т.е. конкретный контекст.
        self.seg_strong = (win & (self.seg_perm >= CONNECTED)).sum(-1) >= self.theta
        self.predictive_strong = self.seg_strong.any(1)

    # ---------- L6: ожидание ----------
    def prediction(self) -> np.ndarray:
        """Сила ожидания по колонкам (сколько клеток колонки предвозбуждено)."""
        return self.predictive.reshape(self.C, self.K).sum(1).astype(float)

    # ---------- один тета-цикл ----------
    def step(self, inp: np.ndarray | None, learn: bool = True) -> float:
        nm = self.nm
        pred_cols = self.prediction()
        prev_active, prev_winners = self.active.copy(), self.winners.copy()

        # --- L4: конкуренция колонок ---
        if inp is not None:
            connected = (self.ff_perm >= CONNECTED) & self.potential
            overlap = (connected & inp).sum(1).astype(float)
            scale = overlap.max() or 1.0
            drive = (nm.acetylcholine * overlap * self.boost
                     + (1 - nm.acetylcholine) * scale * (np.minimum(pred_cols, 1) + self.apical))
        else:  # воображение: вход отсутствует, работают только ожидания и контекст сверху
            drive = pred_cols + self.apical
        drive = drive + self.rng.random(self.C) * 1e-3  # разрыв ничьих
        k = int(round(self.k * (1 + 0.5 * nm.noradrenaline)))
        top = np.argsort(drive)[-k:]
        top = top[drive[top] > 1e-2]
        cols = np.zeros(self.C, dtype=bool)
        cols[top] = True

        # --- L2/3: какие клетки внутри колонок ---
        active = np.zeros(self.N, dtype=bool)
        winners = np.zeros(self.N, dtype=bool)
        learn_segs: list[tuple[int, int]] = []
        new_segs: list[int] = []
        bursting = 0
        start = not prev_winners.any()                  # начало последовательности
        for c in top:
            cells = np.arange(c * self.K, (c + 1) * self.K)
            strong = cells[self.predictive_strong[cells]]
            weak = cells[self.predictive[cells]]
            if strong.size:                             # точное ожидание сбылось
                active[strong] = winners[strong] = True
                for cell in strong:
                    for s in np.flatnonzero(self.seg_strong[cell]):
                        learn_segs.append((cell, s))
                continue
            if weak.size:                               # размытое ожидание: слово ждали,
                active[weak] = True                     # но контекст новый → своя клетка
            else:                                       # «вспышка»: неожиданно → вся колонка
                bursting += 1
                active[cells] = True
            if start:                                   # стартовая клетка колонки
                winners[cells[0]] = True
                continue
            pot = self.seg_potential[cells]
            if pot.max() >= self.min_match:
                i, s = np.unravel_index(np.argmax(pot), pot.shape)
                winners[cells[i]] = active[cells[i]] = True
                learn_segs.append((cells[i], s))
            else:
                used = (self.seg_pre[cells] >= 0).any(-1).sum(-1).astype(float)
                used[0] += 100                          # стартовую клетку не занимаем
                cell = cells[np.argmin(used + self.rng.random(self.K) * 0.1)]
                winners[cell] = active[cell] = True
                new_segs.append(cell)
        self.anomaly = bursting / max(1, len(top))

        # --- обучение (локальное, с модуляцией) ---
        if learn and prev_winners.any():
            inc = 0.10 * (0.5 + nm.dopamine)
            dec = 0.03
            ext = np.append(prev_active, False)
            pw = np.flatnonzero(prev_winners)
            grow = int(round(16 * (0.75 + 0.5 * nm.noradrenaline)))
            for cell, s in learn_segs:
                self._adapt(cell, s, ext, pw, inc, dec, grow)
            for cell in new_segs:
                s = int(np.argmin(self.seg_used[cell]))
                self.seg_pre[cell, s] = -1
                self.seg_perm[cell, s] = 0
                self._adapt(cell, s, ext, pw, inc, dec, grow)
            # ошибочные ожидания: клетка предсказала, но колонка не включилась
            wrong = self.seg_active & ~cols.repeat(self.K)[:, None]
            for cell, s in zip(*np.nonzero(wrong)):
                pre = self.seg_pre[cell, s]
                hit = (pre >= 0) & ext[pre]
                self.seg_perm[cell, s][hit] -= 0.01
        if learn and inp is not None:
            self._learn_proximal(cols, inp)

        # --- гомеостаз и нейромодуляторы ---
        self.duty = 0.99 * self.duty + 0.01 * cols
        self.boost = np.exp(3.0 * (self.k / self.C - self.duty))
        nm.noradrenaline += 0.3 * (self.anomaly - nm.noradrenaline)
        self.apical *= nm.serotonin
        nm.relax(0.05)
        nm.clamp()

        self.active, self.winners, self.active_cols = active, winners, cols
        self.pooled = 0.6 * self.pooled + active
        self._compute_segments()                     # L6: ожидание на следующий цикл
        self.t += 1
        return self.anomaly

    def _adapt(self, cell, s, ext, prev_winners, inc, dec, grow) -> None:
        pre, perm = self.seg_pre[cell, s], self.seg_perm[cell, s]
        valid = pre >= 0
        hit = valid & ext[pre]
        perm[hit] += inc
        perm[valid & ~hit] -= dec
        np.clip(perm, 0, 1, out=perm)
        dead = valid & (perm <= 0)
        pre[dead] = -1                                  # обрезка
        need = grow - int(hit.sum())
        free = np.flatnonzero(pre < 0)
        cand = np.setdiff1d(prev_winners, pre[pre >= 0])
        n = min(need, free.size, cand.size)
        if n > 0:                                       # рост новых синапсов
            pre[free[:n]] = self.rng.choice(cand, n, replace=False)
            perm[free[:n]] = 0.55 if self.nm.noradrenaline > 0.5 else 0.45
        self.seg_used[cell, s] = self.t

    def _learn_proximal(self, cols, inp) -> None:
        rows = np.flatnonzero(cols)
        delta = np.where(inp, 0.05, -0.02)
        self.ff_perm[rows] = np.clip(self.ff_perm[rows] + delta * self.potential[rows], 0, 1)

    # ---------- порты для других областей ----------
    def output(self) -> np.ndarray:
        """L5: устойчивый выход — клетки, активные недавно."""
        return self.pooled > 0.5

    def set_apical(self, bias: np.ndarray) -> None:
        self.apical = np.maximum(self.apical, bias)

    # ---------- сон ----------
    def downscale(self, factor: float = 0.97, prune: float = 0.1) -> int:
        """Синаптический гомеостаз во сне: всё ослабляется, слабое отмирает."""
        valid = self.seg_pre >= 0
        self.seg_perm[valid] *= factor
        dead = valid & (self.seg_perm < prune)
        self.seg_pre[dead] = -1
        self.seg_perm[dead] = 0
        self._compute_segments()
        return int(dead.sum())

    def stats(self) -> dict:
        return {"синапсов": int((self.seg_pre >= 0).sum()),
                "сегментов": int((self.seg_pre >= 0).any(-1).sum()),
                "удивление": round(self.anomaly, 2),
                **{k: round(v, 2) for k, v in vars(self.nm).items()}}
