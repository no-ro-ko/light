"""Проверки «удивлением»: пары «возможно / невозможно», как опыты с младенцами.

Берётся сцена и действие. «Возможное» продолжение — то, что на самом деле выдаёт мир.
«Невозможное» — то же самое, но с одной небольшой правкой, нарушающей закон (предмет
завис, прошёл сквозь опору, исчез, «я» прошёл сквозь стену…). Модель «удивляется»
сильнее на том кадре, которому она дала меньшую вероятность. Пара пройдена, если
невозможное удивляет сильнее возможного. Случайное угадывание даёт 50%.

Сцены собираются не так, как миры «детства»: другой рельеф, плавающие плиты и свалки из
случайных клеток вокруг события — так что окружения почти наверняка небывалые.
Каждая пара проверена настоящим миром: событие действительно происходит.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import world as Wd
from .world import (AGENT_L, AGENT_R, AIR, FIRE, GROUND, H, ICE, K, SAND, STONE, W, WATER, WOOD,
                    ACT, JUMP, LEFT, NOOP, PUT_SAND, PUT_STONE, PUT_WATER, RIGHT)

AG = {-1: AGENT_L, 1: AGENT_R}
LEVELS = [(0.0, 0.0), (0.15, 0.05), (0.35, 0.12)]  # (плотность мусора у события, вдали)
FILL = [GROUND, STONE, SAND, WATER, WOOD, ICE]
SOLID = [GROUND, GROUND, WOOD, ICE, STONE]
DRY = [AIR, GROUND, STONE, SAND, WOOD, ICE]
DIRS4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]


@dataclass
class Core:
    ess: dict            # {(dy,dx): сорт} — клетки, которые обязаны быть такими
    a: int
    ph: int
    expect: list         # [(dy,dx,сорт)] — что обязано получиться в настоящем кадре
    edit: list           # [(dy,dx,сорт)] — правка, делающая кадр невозможным
    has_agent: bool = False


def _d(rng):
    return int(rng.choice([-1, 1]))


def _solid(rng):
    return int(rng.choice(SOLID))


def b_fall(rng, mode):
    m = int(rng.choice([STONE, SAND, WATER]))
    return Core({(0, 0): m, (1, 0): AIR}, NOOP, int(rng.integers(2)), [(0, 0, AIR), (1, 0, m)],
                {"hover": [(0, 0, m), (1, 0, AIR)], "vanish": [(1, 0, AIR)]}[mode])


def b_solid(rng, mode):
    s = int(rng.choice([GROUND, GROUND, WOOD, ICE, STONE, WATER, SAND]))
    return Core({(0, 0): STONE, (1, 0): s}, NOOP, int(rng.integers(2)), [(0, 0, STONE), (1, 0, s)],
                {"pass": [(0, 0, AIR), (1, 0, STONE)], "vanish": [(0, 0, AIR)]}[mode])


def b_slide(rng, mode):
    ph = int(rng.integers(2))
    d = -1 if ph == 0 else 1
    b = int(rng.choice([GROUND, GROUND, WOOD, ICE, STONE, SAND, WATER]))
    ess = {(0, 0): SAND, (1, 0): b, (0, d): AIR, (1, d): AIR}
    if mode == "wrong":
        ess[(0, -d)] = AIR
        ess[(1, -d)] = AIR
    return Core(ess, NOOP, ph, [(0, 0, AIR), (1, d, SAND)],
                {"stay": [(0, 0, SAND), (1, d, AIR)], "wrong": [(0, 0, AIR), (1, d, AIR), (1, -d, SAND)]}[mode])


def b_spread(rng, mode):
    ph = int(rng.integers(2))
    d = -1 if ph == 0 else 1
    b = int(rng.choice([GROUND, GROUND, WOOD, ICE, STONE, SAND]))
    if rng.random() < .5:   # давит вода сверху, обрыва нет
        ess = {(0, 0): WATER, (1, 0): b, (-1, 0): WATER, (0, d): AIR, (1, d): _solid(rng)}
    else:                   # одинокая капля у обрыва
        ess = {(0, 0): WATER, (1, 0): b, (-1, 0): AIR, (0, d): AIR, (1, d): AIR}
    if mode == "wrong":
        ess[(0, -d)] = AIR
    return Core(ess, NOOP, ph, [(0, 0, AIR), (0, d, WATER)],
                {"stay": [(0, 0, WATER), (0, d, AIR)], "wrong": [(0, 0, AIR), (0, d, AIR), (0, -d, WATER)]}[mode])


def _around(wy, wx):
    return [(wy + dy, wx + dx) for dy, dx in DIRS4 if (wy + dy, wx + dx) != (0, 0)]


def b_burn(rng, mode):
    wy, wx = DIRS4[int(rng.integers(4))]
    ess = {(0, 0): FIRE, (wy, wx): WOOD}
    for n in _around(wy, wx):
        ess[n] = int(rng.choice(DRY))
    return Core(ess, NOOP, int(rng.integers(2)), [(0, 0, AIR), (wy, wx, FIRE)],
                {"no_ignite": [(wy, wx, WOOD)], "fire_stays": [(0, 0, FIRE)]}[mode])


def b_wet(rng, mode):
    wy, wx = DIRS4[int(rng.integers(4))]
    nb = _around(wy, wx)
    wet = nb[int(rng.integers(len(nb)))]
    ess = {(0, 0): FIRE, (wy, wx): WOOD}
    for n in nb:
        ess[n] = WATER if n == wet else int(rng.choice(DRY + [WATER]))
    return Core(ess, NOOP, int(rng.integers(2)), [(0, 0, AIR), (wy, wx, WOOD)],
                {"ignite": [(wy, wx, FIRE)], "fire_stays": [(0, 0, FIRE)]}[mode])


def b_melt(rng, mode):
    iy, ix = DIRS4[int(rng.integers(4))]
    return Core({(0, 0): FIRE, (iy, ix): ICE}, NOOP, int(rng.integers(2)), [(0, 0, AIR), (iy, ix, WATER)],
                {"stays_ice": [(iy, ix, ICE)], "wrong": [(iy, ix, SAND)]}[mode])


def b_fire_out(rng, mode):
    ess = {(0, 0): FIRE}
    for dy, dx in DIRS4:
        ess[(dy, dx)] = int(rng.choice([AIR, GROUND, STONE, SAND, WATER]))
    return Core(ess, NOOP, int(rng.integers(2)), [(0, 0, AIR)],
                {"persist": [(0, 0, FIRE)], "to_stone": [(0, 0, STONE)]}[mode])


def b_walk(rng, mode):
    d = _d(rng)
    k0 = int(rng.choice([AGENT_L, AGENT_R]))
    ess = {(0, 0): k0, (1, 0): _solid(rng), (0, d): AIR, (1, d): _solid(rng)}
    if mode == "backward":
        ess[(0, -d)] = AIR
        ess[(1, -d)] = _solid(rng)
    return Core(ess, LEFT if d < 0 else RIGHT, int(rng.integers(2)), [(0, 0, AIR), (0, d, AG[d])],
                {"stay": [(0, 0, k0), (0, d, AIR)],
                 "backward": [(0, 0, AIR), (0, d, AIR), (0, -d, AG[-d])]}[mode], True)


def b_bump(rng, mode):
    d = _d(rng)
    k0 = AG[-d]
    x = int(rng.choice([GROUND, GROUND, STONE, WOOD, ICE, SAND, WATER]))
    return Core({(0, 0): k0, (1, 0): _solid(rng), (0, d): x}, LEFT if d < 0 else RIGHT, int(rng.integers(2)),
                [(0, 0, AG[d]), (0, d, x)],
                {"no_turn": [(0, 0, k0)], "pass": [(0, 0, AIR), (0, d, AG[d])]}[mode], True)


def b_dig(rng, mode):
    d = _d(rng)
    return Core({(0, 0): AG[d], (1, 0): _solid(rng), (0, d): SAND, (1, d): _solid(rng)}, ACT,
                int(rng.integers(2)), [(0, d, AIR)],
                {"stays": [(0, d, SAND)], "to_stone": [(0, d, STONE)]}[mode], True)


def b_ignite(rng, mode):
    d = _d(rng)
    ess = {(0, 0): AG[d], (1, 0): _solid(rng), (0, d): WOOD}
    for n in [(-1, d), (1, d), (0, 2 * d)]:
        ess[n] = int(rng.choice(DRY))
    return Core(ess, ACT, int(rng.integers(2)), [(0, d, FIRE)],
                {"stays": [(0, d, WOOD)], "vanish": [(0, d, AIR)]}[mode], True)


def b_push(rng, mode):
    d = _d(rng)
    ess = {(0, 0): AG[d], (1, 0): _solid(rng), (0, d): STONE, (0, 2 * d): AIR}
    return Core(ess, ACT, int(rng.integers(2)), [(0, d, AIR), (0, 2 * d, STONE)],
                {"stays": [(0, d, STONE), (0, 2 * d, AIR)], "vanish": [(0, d, AIR), (0, 2 * d, AIR)]}[mode], True)


def b_put(rng, mode):
    d = _d(rng)
    x = int(rng.choice([SAND, WATER, STONE]))
    act = {SAND: PUT_SAND, WATER: PUT_WATER, STONE: PUT_STONE}[x]
    others = [m for m in (SAND, WATER, STONE) if m != x]
    return Core({(0, 0): AG[d], (1, 0): _solid(rng), (0, d): AIR}, act, int(rng.integers(2)), [(0, d, x)],
                {"nothing": [(0, d, AIR)], "wrong": [(0, d, int(rng.choice(others)))]}[mode], True)


def b_agent_fall(rng, mode):
    k0 = int(rng.choice([AGENT_L, AGENT_R]))
    a = int(rng.choice([NOOP, NOOP, JUMP, ACT]))
    ess = {(0, 0): k0, (1, 0): AIR}
    if mode == "up":
        ess[(-1, 0)] = AIR
    return Core(ess, a, int(rng.integers(2)), [(0, 0, AIR), (1, 0, k0)],
                {"hover": [(0, 0, k0), (1, 0, AIR)], "up": [(0, 0, AIR), (1, 0, AIR), (-1, 0, k0)]}[mode], True)


def b_jump(rng, mode):
    k0 = int(rng.choice([AGENT_L, AGENT_R]))
    ess = {(0, 0): k0, (1, 0): _solid(rng), (-1, 0): AIR}
    if mode == "two":
        ess[(-2, 0)] = AIR
    return Core(ess, JUMP, int(rng.integers(2)), [(0, 0, AIR), (-1, 0, k0)],
                {"stay": [(0, 0, k0), (-1, 0, AIR)], "two": [(0, 0, AIR), (-1, 0, AIR), (-2, 0, k0)]}[mode], True)


FAMILIES = {
    "падение": (b_fall, ["hover", "vanish"]),
    "опора держит": (b_solid, ["pass", "vanish"]),
    "осыпание песка": (b_slide, ["stay", "wrong"]),
    "растекание воды": (b_spread, ["stay", "wrong"]),
    "горение": (b_burn, ["no_ignite", "fire_stays"]),
    "мокрое дерево": (b_wet, ["ignite", "fire_stays"]),
    "таяние льда": (b_melt, ["stays_ice", "wrong"]),
    "огонь гаснет": (b_fire_out, ["persist", "to_stone"]),
    "иду": (b_walk, ["stay", "backward"]),
    "упираюсь": (b_bump, ["no_turn", "pass"]),
    "копаю": (b_dig, ["stays", "to_stone"]),
    "поджигаю": (b_ignite, ["stays", "vanish"]),
    "толкаю": (b_push, ["stays", "vanish"]),
    "кладу": (b_put, ["nothing", "wrong"]),
    "я падаю": (b_agent_fall, ["hover", "up"]),
    "прыгаю": (b_jump, ["stay", "two"]),
}
# какие законы нужны семейству (для проверки самих проверок)
NEEDS = {"падение": Wd.R_FALL, "опора держит": Wd.R_FALL, "осыпание песка": Wd.R_SLIDE,
         "растекание воды": Wd.R_SPREAD, "горение": Wd.R_BURN, "мокрое дерево": Wd.R_WET,
         "таяние льда": Wd.R_MELT, "огонь гаснет": Wd.R_BURN, "иду": Wd.R_MOVE, "упираюсь": Wd.R_MOVE,
         "копаю": Wd.R_DIG, "поджигаю": Wd.R_IGNITE, "толкаю": Wd.R_PUSH, "кладу": Wd.R_PUT,
         "я падаю": Wd.R_FALL, "прыгаю": Wd.R_JUMP}


def _terrain(rng):
    g = Wd.empty_world()
    kind = int(rng.integers(3))
    if kind == 0:
        h = int(rng.integers(1, 3))
        heights = np.full(W, h)
        g[H - 1 - h:H - 1, 1:W - 1] = GROUND
    elif kind == 1:
        heights = Wd.floor_heights(rng, g, 1, 3, 0.3)
    else:  # впадина / холм
        heights = np.zeros(W, int)
        c = int(rng.integers(5, W - 5))
        for x in range(1, W - 1):
            heights[x] = int(np.clip(2 + (1 if rng.random() < .5 else -1) * (2 - min(abs(x - c) // 2, 2)), 0, 4))
            if heights[x] > 0:
                g[H - 1 - heights[x]:H - 1, x] = GROUND
    for _ in range(int(rng.integers(0, 3))):   # плавающие плиты
        pw = int(rng.integers(3, 7))
        x0 = int(rng.integers(1, W - 1 - pw))
        y0 = int(rng.integers(2, H - 5))
        g[y0, x0:x0 + pw] = int(rng.choice([GROUND, WOOD, ICE]))
    return g, heights


def build_pair(rng, core: Core, level: int):
    """Собирает сцену вокруг события и проверяет её настоящим миром. None — если не вышло."""
    p_ctx, p_glob = LEVELS[level]
    offs = list(core.ess)
    ys = [o[0] for o in offs] + [o[0] for o in core.expect] + [o[0] for o in core.edit]
    xs = [o[1] for o in offs] + [o[1] for o in core.expect] + [o[1] for o in core.edit]
    cy = int(rng.integers(1 - min(ys), H - 1 - max(ys)))
    cx = int(rng.integers(1 - min(xs), W - 1 - max(xs)))
    g, heights = _terrain(rng)
    Y, X = np.mgrid[0:H, 0:W]
    interior = (Y > 0) & (Y < H - 1) & (X > 0) & (X < W - 1)
    near = interior & (np.maximum(abs(Y - cy), abs(X - cx)) <= 2)
    far = interior & ~near
    for mask, p in ((near, p_ctx), (far, p_glob)):
        flip = mask & (rng.random((H, W)) < p)
        g[flip] = rng.choice(FILL + ([FIRE] if p > 0.1 else []), int(flip.sum()),
                             p=None if p <= 0.1 else [.2, .17, .17, .17, .13, .1, .06])
    for (dy, dx), k in core.ess.items():
        g[cy + dy, cx + dx] = k
    if not core.has_agent:
        for _ in range(20):
            bx = int(rng.integers(1, W - 1))
            if abs(bx - cx) >= 4:
                by = H - 2 - int(heights[bx])
                if by >= 1:
                    g[by, bx] = int(rng.choice([AGENT_L, AGENT_R]))
                    break
        else:
            return None
    if int(((g >= AGENT_L).sum())) != 1:
        return None
    g0 = g[None]
    a = np.array([core.a])
    ph = np.array([core.ph])
    g1 = Wd.step(g0, a, ph)[0]
    for dy, dx, k in core.expect:
        if g1[cy + dy, cx + dx] != k:
            return None
    bad = g1.copy()
    for dy, dx, k in core.edit:
        bad[cy + dy, cx + dx] = k
    if (bad == g1).all():
        return None
    return g0[0], core.a, core.ph, g1, bad


@dataclass
class Probes:
    G0: np.ndarray
    A: np.ndarray
    PH: np.ndarray
    G1T: np.ndarray      # настоящий следующий кадр
    G1I: np.ndarray      # невозможный
    fam: np.ndarray      # номер семейства
    mode: np.ndarray     # номер вида нарушения внутри семейства
    level: np.ndarray    # уровень «мусора» вокруг события

    def __len__(self):
        return len(self.A)


FAM_NAMES = list(FAMILIES)


def build(seed: int, per_mode: int = 20) -> Probes:
    rows = []
    for fi, name in enumerate(FAM_NAMES):
        builder, modes = FAMILIES[name]
        for mi, mode in enumerate(modes):
            rng = np.random.default_rng([seed, fi, mi])
            made = 0
            tries = 0
            while made < per_mode:
                tries += 1
                if tries > per_mode * 400:
                    raise RuntimeError(f"не удаётся собрать сцены для {name}/{mode}")
                level = made % 3
                res = build_pair(rng, builder(rng, mode), level)
                if res is None:
                    continue
                rows.append((*res, fi, mi, level))
                made += 1
    G0 = np.stack([r[0] for r in rows])
    return Probes(G0, np.array([r[1] for r in rows]), np.array([r[2] for r in rows]),
                  np.stack([r[3] for r in rows]), np.stack([r[4] for r in rows]),
                  np.array([r[5] for r in rows]), np.array([r[6] for r in rows]), np.array([r[7] for r in rows]))


def surprise(lp: np.ndarray, g1: np.ndarray) -> np.ndarray:
    """Удивление = минус логарифм вероятности кадра (клетки считаем независимыми)."""
    take = np.take_along_axis(lp, g1[..., None].astype(np.intp), axis=-1)[..., 0]
    return -take.sum((1, 2))


def evaluate(model, pr: Probes, chunk: int = 128) -> dict:
    ok = np.zeros(len(pr))
    conf_sum, conf_n = 0.0, 0
    for i in range(0, len(pr), chunk):
        sl = slice(i, i + chunk)
        lp = model.logp(pr.G0[sl], pr.A[sl], pr.PH[sl])
        st, si = surprise(lp, pr.G1T[sl]), surprise(lp, pr.G1I[sl])
        diff = si - st
        ok[sl] = np.where(np.abs(diff) < 1e-6, 0.5, (diff > 0).astype(float))
        # уверенность в настоящих изменениях: вероятность, данная настоящему исходу там, где клетка действительно изменилась
        take = np.take_along_axis(lp, pr.G1T[sl][..., None].astype(np.intp), axis=-1)[..., 0]
        changed = pr.G1T[sl] != pr.G0[sl]
        conf_sum += float(np.exp(take)[changed].sum())
        conf_n += int(changed.sum())
    fam = {}
    for fi, name in enumerate(FAM_NAMES):
        m = pr.fam == fi
        fam[name] = float(ok[m].mean())
    return {"macro": float(np.mean(list(fam.values()))), "micro": float(ok.mean()), "families": fam,
            "conf": conf_sum / max(conf_n, 1),
            "by_level": [float(ok[pr.level == l].mean()) for l in range(3)],
            "by_mode": {f"{FAM_NAMES[fi]}/{FAMILIES[FAM_NAMES[fi]][1][mi]}":
                        float(ok[(pr.fam == fi) & (pr.mode == mi)].mean())
                        for fi in range(len(FAM_NAMES)) for mi in range(2)}}


# ------------------------------------------------------- эталоны для проверки самих проверок

class RuleModel:
    """«Знает» законы из набора rules настолько, насколько уверен eps (вероятность ошибки)."""

    def __init__(self, rules: int = Wd.R_ALL, eps: float = 1e-3):
        self.rules, self.eps = rules, eps

    def logp(self, g0, a, ph):
        pred = Wd.step(g0, a, ph, self.rules)
        lp = np.full(g0.shape + (K,), np.log(self.eps / (K - 1)), np.float32)
        np.put_along_axis(lp, pred[..., None].astype(np.intp), np.float32(np.log(1 - self.eps)), axis=-1)
        return lp


class UniformModel:
    def logp(self, g0, a, ph):
        return np.full(g0.shape + (K,), -np.log(K), np.float32)


class RandomModel:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def logp(self, g0, a, ph):
        z = self.rng.normal(size=g0.shape + (K,)).astype(np.float32)
        return z - np.log(np.exp(z).sum(-1, keepdims=True))
