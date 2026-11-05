"""Дневник: быстрая кора недавнего опыта поверх медленной коры, с арбитром между ними.

Зачем. Кора копит счёт и ничего не забывает: это устойчиво, но когда закон мира меняется, прежние
тысячи свидетельств перевешивают новые, и она переучивается очень долго (docs/MIR.md, разд. 10.5).
Обычная сеть переучивается быстро, но забывает остальное. В мозге это решают две системы:
быстрая запоминает недавнее, медленная хранит устойчивое (гиппокамп и неокортекс).

Как устроено.
  • Дневник хранит последние `horizon` тактов опыта (окна, исходы, сколько раз встречались).
  • По этому буферу он растит СВОЮ быструю кору (`fast`): такое же дерево сегментов, но обученное только на
    недавнем, со свежим весом больше (e^{-возраст/tau}). За ~1000 тактов она выучивает и новый закон.
  • Арбитр. Для каждого нового опыта, ДО того как медленная кора его выучит, считается, чьё предсказание
    лучше: давнее (медленная кора) или недавнее (быстрая). Разница логарифмов правдоподобия копится в
    листьях быстрой коры (они чистые именно по недавнему поведению, поэтому «изменившиеся» окна отделены
    от «спокойных»). π = σ(prior + доказательство): при нехватке доказательств π≈0 и ответ совпадает с ответом
    медленной коры; если закон поменялся, π→1 и работает быстрая; если закон вернулся, π снова падает:
    давняя память цела.
  • Закрепление (сон, `consolidate=True`): в листьях, где быстрая уверенно не согласна с медленной, давние
    свидетельства, противоречащие недавним, «увольняются» (вес × `rho`), и медленная кора принимает новый
    закон навсегда. Цена: возврат старого закона придётся учить заново.

Честно о происхождении. Это «быстрые и медленные веса» (Хинтон, Плаут 1987) и «дополняющие системы
обучения» (Макклелланд, Макнотон, О'Рейли 1995) в простом исполнении, а не новый алгоритм. Здесь важно
измерить, нужен ли отдельный быстрый отдел или хватает простого забывания в коре.
"""
from __future__ import annotations

import numpy as np

from .patches import NF, Samples, keys, patches
from .seg import SegCortex
from .world import K


class Diary:
    def __init__(self, cortex: SegCortex, horizon: int = 1000, tau: float = 300.0, prior: float = -4.0,
                 temp: float = 1.0, n_min: float = 2.0, alpha: float = 0.3, floor: float = 1e-5,
                 refit_every: int = 200, min_recent: float = 0.1, consolidate: bool = False, rho: float = 0.05,
                 pi_cons: float = 0.95):
        self.cortex = cortex
        self.horizon, self.tau, self.prior, self.temp = horizon, tau, prior, temp
        self.n_min, self.alpha, self.floor = n_min, alpha, floor
        self.refit_every, self.min_recent = refit_every, min_recent
        self.consolidate, self.rho, self.pi_cons = consolidate, rho, pi_cons
        self.fast = SegCortex(tol=cortex.tol, pooled=cortex.pooled, min_unique=cortex.min_unique,
                              cochange=cortex.cochange, alpha=alpha, floor=floor)
        self.info: dict = {}
        self.reset_buffer()

    def reset_buffer(self):
        self.bX = np.zeros((0, NF), np.uint8)
        self.by = np.zeros(0, np.uint8)
        self.bw = np.zeros(0)
        self.bchg = np.zeros(0, np.int32)
        self.bt = np.zeros(0, np.int64)
        self.bd = np.zeros(0)
        self.fast.reset()
        self._fver = 0
        self._last_refit = -10 ** 9
        self._fmass = np.zeros(0)
        self._ck = None
        self._cv = None

    # ------------------------------------------------------------------ быстрая кора по буферу
    def _refit_fast(self, now: int):
        """Вырастить быструю кору заново по недавнему буферу (свежее весит больше)."""
        rec = np.exp(-np.maximum(now - self.bt, 0) / self.tau)
        use = rec >= self.min_recent
        f = self.fast
        f.reset()
        if use.any():
            S = Samples(self.bX[use], self.by[use], self.bw[use] * rec[use], self.bchg[use], np.arange(int(use.sum())))
            f.clock = now
            f._add(S, now)
            f._grow()
            f._ver += 1
            f._fmass_leaf = np.bincount(f.leaf[f.leaf >= 0], weights=f.w[f.leaf >= 0], minlength=len(f.feat))
            self._fmass = f._fmass_leaf
        else:
            self._fmass = np.zeros(0)
        self._fver += 1
        self._last_refit = now

    def _fast_logp(self, U: np.ndarray):
        """Для окон: лист быстрой коры, её log-вероятности и «есть ли в листе свежие данные»."""
        f = self.fast
        if not len(f.y):
            return None
        node, lp = f._window_logp(U)
        has = np.zeros(len(U), bool)
        ok = node >= 0
        has[ok] = self._fmass[node[ok]] >= 1.0
        return node, lp, has

    # ------------------------------------------------------------------ жизнь
    def observe(self, S0, A, PH, S1):
        """Записать порцию опыта. Вызывать ДО того, как её выучит медленная кора: так давнее и недавнее честно соревнуются."""
        S = Samples.collect(S0, A, PH, S1)
        cor = self.cortex
        t_end = cor.clock + len(S0)
        d = np.zeros(len(S))
        fl = self._fast_logp(S.X) if len(cor.y) else None
        if fl is not None:
            _, lp_f, has = fl
            _, lp_s = cor._window_logp(S.X)
            idx = np.nonzero(has)[0]
            d[idx] = lp_f[idx, S.y[idx]] - lp_s[idx, S.y[idx]]
        self.bX = np.concatenate([self.bX, S.X])
        self.by = np.concatenate([self.by, S.y])
        self.bw = np.concatenate([self.bw, S.w])
        self.bchg = np.concatenate([self.bchg, S.chg])
        self.bt = np.concatenate([self.bt, np.full(len(S), t_end, np.int64)])
        self.bd = np.concatenate([self.bd, d])
        keep = (t_end - self.bt) <= self.horizon
        if not keep.all():
            self.bX, self.by, self.bw, self.bchg, self.bt, self.bd = (
                a[keep] for a in (self.bX, self.by, self.bw, self.bchg, self.bt, self.bd))
        if t_end - self._last_refit >= self.refit_every:
            self._refit_fast(t_end)
        return self

    def partial_fit(self, S0, A, PH, S1):
        self.observe(S0, A, PH, S1)
        self.cortex.partial_fit(S0, A, PH, S1)
        return self

    def fit(self, S0, A, PH, S1, warm: int = 1000, chunk: int = 200):
        """Медленная кора учится разом на старом; последние `warm` тактов проживаются порциями — дневнику есть что помнить."""
        T = len(S0)
        w = min(warm, T)
        if T - w > 0:
            self.cortex.fit(S0[:T - w], A[:T - w], PH[:T - w], S1[:T - w])
        else:
            self.cortex.reset()
        self.reset_buffer()
        for i in range(T - w, T, chunk):
            self.partial_fit(S0[i:i + chunk], A[i:i + chunk], PH[i:i + chunk], S1[i:i + chunk])
        return self

    # ------------------------------------------------------------------ арбитр
    def _tables(self, now: int | None = None):
        cor, f = self.cortex, self.fast
        now = cor.clock if now is None else now
        key = (now, self._fver, len(self.bX))
        if self._ck == key:
            return self._cv
        self._ck = key
        if not len(self.bX) or not len(f.y):
            self._cv = None
            return None
        node = f._route(self.bX)
        ok = node >= 0
        n_nodes = len(f.feat)
        rec = np.exp(-np.maximum(now - self.bt, 0) / self.tau)
        nd = node[ok]
        ev = np.bincount(nd, weights=(self.bw * rec * self.bd)[ok], minlength=n_nodes)
        raw = np.bincount(nd, weights=self.bw[ok], minlength=n_nodes)
        z = np.clip(self.prior + ev / self.temp, -50, 50)
        pi = np.where(raw >= self.n_min, 1 / (1 + np.exp(-z)), 0.0)
        pi[:len(self._fmass)] *= (self._fmass >= 1.0)
        if len(pi) > len(self._fmass):
            pi[len(self._fmass):] = 0.0
        self._cv = {"pi": pi, "E": ev, "raw": raw}
        return self._cv

    # ------------------------------------------------------------------ предсказание
    def logp(self, g0, a, ph):
        n, h, w = g0.shape
        X = patches(g0, a, ph).reshape(-1, NF)
        uk, first, inv = np.unique(keys(X), return_index=True, return_inverse=True)
        U = X[first]
        _, lp = self.cortex._window_logp(U)
        tab = self._tables()
        fl = self._fast_logp(U) if tab is not None else None
        if fl is not None:
            node_f, lp_f, has = fl
            pi = np.zeros(len(U))
            sel = (node_f >= 0) & has
            pi[sel] = tab["pi"][node_f[sel]]
            use = pi > 1e-4
            if use.any():
                p = pi[use, None] * np.exp(lp_f[use]) + (1 - pi[use, None]) * np.exp(lp[use])
                lp[use] = np.log(np.maximum(p, self.floor)).astype(np.float32)
        return lp[inv].reshape(n, h, w, K)

    # ------------------------------------------------------------------ сон
    def sleep(self, rebuild: bool = True):
        """Сон: медленное дерево вырастает заново по всему накопленному; при `consolidate` устаревшее, противоречащее
        недавнему, увольняется в тех листьях, где быстрая кора уверенно не согласна с медленной."""
        cor = self.cortex
        if rebuild:
            cor.rebuild()
        self._refit_fast(cor.clock)
        if self.consolidate:
            tab = self._tables(cor.clock)
            if tab is not None and len(cor.y):
                node = self.fast._route(cor.X)
                ok = node >= 0
                flagged = np.zeros(len(cor.y), bool)
                flagged[ok] = tab["pi"][node[ok]] >= self.pi_cons
                if flagged.any():
                    f = self.fast
                    cnt = np.bincount(f.leaf[f.leaf >= 0] * K + f.y[f.leaf >= 0].astype(np.int64),
                                      weights=f.w[f.leaf >= 0], minlength=len(f.feat) * K).reshape(len(f.feat), K)
                    top = cnt.argmax(1)
                    wrong = np.zeros(len(cor.y), bool)
                    wrong[ok] = cor.y[ok].astype(np.int64) != top[node[ok]]
                    stale = cor.last < cor.clock - self.horizon
                    mask = flagged & wrong & stale
                    cor.retire_rows(mask, self.rho)
                    self.info["retired_rows"] = int(mask.sum())
        return self

    def stats(self) -> dict:
        tab = self._tables()
        if tab is None:
            return {"buffer": int(len(self.bX)), "active_leaves": 0, "active_mass": 0.0}
        act = tab["pi"] > 0.5
        return {"buffer": int(len(self.bX)), "active_leaves": int(act.sum()),
                "active_mass": float(tab["raw"][act].sum() / max(tab["raw"].sum(), 1.0))}
