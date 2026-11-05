"""Любопытство и «дни» существа: бодрствование (действую, собираю опыт) и сон (учусь).

Каждый «день» существо живёт в нескольких новых мирах по 50 тактов; ночью кора учится на
всём собранном за день (и сон выводит общие правила). Утром действия выбираются уже с
учётом новых знаний.

Две политики (миры в каждый день одинаковы, различаются только действия):
  • блуждание — чаще продолжает прежнее действие, иногда выбирает наугад;
  • любопытство — для каждого из 8 действий спрашивает свою кору, НАСКОЛЬКО ОНА СОМНЕВАЕТСЯ
    в исходе (сумма энтропий предсказаний по всем клеткам) и делит это на (1 + сколько раз
    ЭТА ситуация вокруг «я» уже пробовалась с этим действием) — привыкание: после первой
    пробы неизвестное перестаёт быть интересным, даже если ночь ещё не наступила. Если одно
    действие заметно «интереснее» остальных — делает его, иначе блуждает. Мир детерминирован,
    шума нет, поэтому сомнение коры — это просто «я этого ещё не знаю» (ловушки «телевизора с
    помехами» здесь нет; в мире с настоящим шумом нужен «прогресс в понимании»).
  Каждые NAP тактов — короткая дремота: кора доучивается на только что пережитом.

Первая версия любопытства (без привыкания, учёба только ночью) на черновом наборе
оказалась ХУЖЕ блуждания: существо зациклилось на прыжках (572 против 146 за 3000 тактов) и
почти не ходило, не копало и не толкало. См. docs/MIR.md.
"""
from __future__ import annotations

import time

import numpy as np

from . import probes, sealed, world as Wd
from .life import ACTION_P
from .world import NA, Sandbox

DAY_LIVES = [2, 4, 14] + [20] * 29        # жизни по дням: накопленные такты 100, 300, 1000, 2000 ...
LIFE_LEN = 50
NAP = 10                                    # через сколько тактов дремота (доучивание)
CHECKPOINTS = [100, 300, 1000, 3000, 10000, 30000]


def entropy_sum(lp: np.ndarray) -> np.ndarray:
    p = np.exp(lp)
    return -(p * lp).sum(-1).sum((1, 2))


class Wanderer:
    name = "блуждание"

    def __init__(self, n: int, rng):
        self.rng, self.prev = rng, rng.choice(NA, n, p=ACTION_P)

    def choose(self, cortex, g, ph):
        a = np.where(self.rng.random(len(g)) < .5, self.prev, self.rng.choice(NA, len(g), p=ACTION_P))
        self.prev = a
        return a


_MUL25 = (np.random.default_rng(777).integers(1, 2 ** 62, 25, dtype=np.uint64) * 2 + 1).astype(np.uint64)


def around_agent(g: np.ndarray) -> np.ndarray:
    """Хеш окна 5×5 вокруг «я» для каждого мира."""
    n = len(g)
    P = np.pad(g, ((0, 0), (2, 2), (2, 2)), constant_values=Wd.GROUND)
    flat = (g >= Wd.AGENT_L).reshape(n, -1).argmax(1)
    ay, ax = flat // g.shape[2], flat % g.shape[2]
    rows = ay[:, None] + np.arange(5)[None, :]            # в координатах с отступом 2
    cols = ax[:, None] + np.arange(5)[None, :]
    win = P[np.arange(n)[:, None, None], rows[:, :, None], cols[:, None, :]].reshape(n, 25)
    with np.errstate(over="ignore"):
        return (win.astype(np.uint64) * _MUL25).sum(1, dtype=np.uint64)


def coarse_key(g: np.ndarray, face_kind: bool = True) -> np.ndarray:
    """Грубая «обстановка»: какие сорта есть в 3×3 вокруг «я» (набор, без числа и мест) + куда смотрю."""
    n = len(g)
    P = np.pad(g, ((0, 0), (1, 1), (1, 1)), constant_values=Wd.GROUND)
    flat = (g >= Wd.AGENT_L).reshape(n, -1).argmax(1)
    ay, ax = flat // g.shape[2], flat % g.shape[2]
    rows = ay[:, None] + np.arange(3)[None, :]
    cols = ax[:, None] + np.arange(3)[None, :]
    win = P[np.arange(n)[:, None, None], rows[:, :, None], cols[:, None, :]].reshape(n, 9)
    mask = np.zeros(n, np.int64)
    for v in range(Wd.K):
        mask |= (win == v).any(1).astype(np.int64) << v
    return mask * 2 + (g[np.arange(n), ay, ax] == Wd.AGENT_R)


class Curious(Wanderer):
    name = "любопытство"

    def __init__(self, n: int, rng, thr: float = 0.1, seen: dict | None = None, habituate: bool = True,
                 mix: float = 1.0, novelty_only: bool = False, coarse: bool = False):
        super().__init__(n, rng)
        self.thr, self.habituate = thr, habituate
        self.mix, self.novelty_only, self.coarse = mix, novelty_only, coarse
        self.seen = seen if seen is not None else {}

    def choose(self, cortex, g, ph):
        n = len(g)
        base = super().choose(cortex, g, ph)
        if cortex is None or not len(cortex.y):
            return base
        keys = coarse_key(g) if self.coarse else around_agent(g)
        cnt = np.array([[self.seen.get((int(k), a), 0) for a in range(NA)] for k in keys])
        if self.novelty_only:                   # только «как редко я это пробовал»
            sc = 1.0 / (1.0 + cnt) + 1e-3 * self.rng.random((n, NA))
            use = np.ones(n, bool)
        else:
            sc = np.empty((n, NA))
            for a in range(NA):
                sc[:, a] = entropy_sum(cortex.logp(g, np.full(n, a), ph))
            sc -= sc.min(1, keepdims=True)       # общая для всех действий часть сокращается
            if self.habituate:
                sc = sc / (1.0 + cnt)
            use = sc.max(1) > self.thr
        use &= self.rng.random(n) < self.mix
        pick = sc.argmax(1)
        a = np.where(use, pick, base)
        for k, x in zip(keys, a):
            self.seen[(int(k), int(x))] = self.seen.get((int(k), int(x)), 0) + 1
        self.prev = a
        return a


class Approach(Wanderer):
    """Тянется к предметам, с которыми меньше всего возился: идёт к ближайшему предмету самого
    «непробованного» сорта (из тех, что видит), поворачивается к нему и действует. Счёт проб ведётся
    по сортам на всю жизнь. Это простейшая форма интереса к предметам, а не к собственному сомнению."""
    name = "тянусь к непробованному"
    KINDS = (Wd.SAND, Wd.STONE, Wd.WOOD, Wd.ICE, Wd.WATER)

    def __init__(self, n, rng, tried: dict | None = None, mix: float = 0.8, **_):
        super().__init__(n, rng)
        self.tried = tried if tried is not None else {k: 0 for k in self.KINDS}
        self.mix = mix

    def choose(self, cortex, g, ph):
        out = super().choose(cortex, g, ph).copy()
        for i in range(len(g)):
            if self.rng.random() >= self.mix:
                continue
            gi = g[i]
            ay, ax = np.argwhere(gi >= Wd.AGENT_L)[0]
            face = -1 if gi[ay, ax] == Wd.AGENT_L else 1
            present = [k for k in self.KINDS if (gi == k).any()]
            if not present:
                continue
            m = min(self.tried[k] for k in present)
            kind = int(self.rng.choice([k for k in present if self.tried[k] == m]))
            cells = np.argwhere(gi == kind)
            d = np.abs(cells[:, 0] - ay) + np.abs(cells[:, 1] - ax)
            ty, tx = cells[d.argmin()]
            if (ty, tx) == (ay, ax + face):                 # прямо перед носом
                out[i] = Wd.ACT
                self.tried[kind] += 1
            elif ty == ay and abs(tx - ax) == 1:            # рядом, но стою спиной: поворачиваюсь
                out[i] = Wd.LEFT if tx < ax else Wd.RIGHT
            else:
                step = Wd.LEFT if tx < ax else Wd.RIGHT
                ahead = gi[ay, ax + (-1 if step == Wd.LEFT else 1)]
                out[i] = Wd.JUMP if (ahead != Wd.AIR and self.rng.random() < .5) else step
        self.prev = out
        return out


def law_witnesses(S0, A, PH, S1) -> dict:
    """Для каждого закона — индексы тактов, в которых он реально сработал."""
    out = {}
    for bit, name in Wd.RULE_NAMES.items():
        alt = Wd.step(S0, A, PH, Wd.R_ALL & ~bit)
        out[name] = np.nonzero((alt != S1).reshape(len(S0), -1).any(1))[0]
    return out


def live(policy: str, cfg: dict, seed: int, T_max: int = 30000, mode: str = "dev",
         checkpoints=CHECKPOINTS, tag: str = "", pol: dict | None = None) -> list[dict]:
    """Прожить дни с политикой; на контрольных точках проверить кору (после сна)."""
    from .seg import SegCortex
    cortex = SegCortex(**{k: v for k, v in cfg.items() if k != "consolidate"})
    rng = np.random.default_rng(seed + 424242)
    Pol = {"random": Wanderer, "curious": Curious, "curious0": Curious, "approach": Approach}[policy]
    pol_kw = pol
    seen: dict = {}                                  # память о пробованном — на всю жизнь существа
    tried = {k: 0 for k in Approach.KINDS}
    acts = np.zeros(NA, np.int64)
    S0s, As, PHs, S1s = [], [], [], []
    total, day, results = 0, 0, []
    todo = list(checkpoints)
    t0 = time.time()
    while total < T_max:
        lives = DAY_LIVES[min(day, len(DAY_LIVES) - 1)]
        env = Sandbox(lives, seed=seed * 1000 + day)           # миры дня одинаковы для обеих политик
        if policy == "random":
            pol = Pol(lives, rng)
        elif policy == "approach":
            pol = Pol(lives, np.random.default_rng(seed * 7 + day), tried=tried, **(pol_kw or {}))
        elif policy == "curious0":                               # первая версия: без привыкания
            pol = Pol(lives, np.random.default_rng(seed * 7 + day), habituate=False)
        else:
            pol = Pol(lives, np.random.default_rng(seed * 7 + day), seen=seen, **(pol_kw or {}))
        d0, da, dp, d1 = [], [], [], []
        for t in range(LIFE_LEN):
            a = pol.choose(cortex, env.g, env.ph)
            acts += np.bincount(a, minlength=NA)
            g0, ph0, g1 = env.step(a)
            d0.append(g0), da.append(a), dp.append(ph0), d1.append(g1)
            if policy != "curious0" and (t + 1) % NAP == 0:     # дремота: доучиваемся на последних тактах
                k = len(d0) - NAP
                cortex.partial_fit(np.concatenate(d0[k:]), np.concatenate(da[k:]), np.concatenate(dp[k:]),
                                   np.concatenate(d1[k:]))
        S0 = np.concatenate(d0)                                  # такт за тактом (общие часы опыта)
        A = np.concatenate(da)
        PH = np.concatenate(dp)
        S1 = np.concatenate(d1)
        if policy == "curious0":
            cortex.partial_fit(S0, A, PH, S1)                    # только ночью
        S0s.append(S0), As.append(A), PHs.append(PH), S1s.append(S1)
        total += len(A)
        day += 1
        while todo and total >= todo[0]:
            T = todo.pop(0)
            if cfg.get("consolidate"):
                cortex.sleep()
            rec = {"policy": policy, "cfg": cfg, "seed": seed, "T": total, "days": day,
                   "elapsed_s": round(time.time() - t0, 1), "info": dict(cortex.info),
                   "actions": {Wd.ACTION_NAMES[i]: int(c) for i, c in enumerate(acts)}}
            if "dev" in mode or mode == "both":
                r = probes.evaluate(cortex, sealed.dev())
                rec["dev"] = {"macro": round(r["macro"], 4), "families": {k: round(v, 3) for k, v in r["families"].items()}}
            if "sealed" in mode or mode == "both":
                r = sealed.evaluate_sealed(cortex, f"curious:{policy}:{cfg}:seed={seed}:T={total}:{tag}")
                rec["sealed"] = {"macro": round(r["macro"], 4), "families": {k: round(v, 3) for k, v in r["families"].items()}}
            results.append(rec)
    S0, A, PH, S1 = (np.concatenate(x) for x in (S0s, As, PHs, S1s))
    wit = law_witnesses(S0, A, PH, S1)
    second = {name: (int(idx[1]) + 1 if len(idx) > 1 else None) for name, idx in wit.items()}
    count = {name: int(len(idx)) for name, idx in wit.items()}
    for r in results:
        r["second_witness_at"] = second
        r["witness_counts_at_end"] = count
    return results
