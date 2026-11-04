"""«Песочница» — маленький мир из клеток с разными законами.

Скрытая истина: из чего сделаны клетки и как они действуют друг на друга. Существо
видит только картинку клеток (какой «сорт» в каждой), свой выбор действия и
«метроном» (чётность такта). Названий и правил оно не получает.

Законы (все честные и детерминированные — один и тот же кадр всегда даёт один и тот же
следующий; радиус действия — до двух клеток):
  падение      камень, песок, вода и «я» падают вниз, если снизу воздух;
  осыпание     песок, которому некуда падать, соскальзывает по диагонали (в сторону
               задаёт метроном: чётный такт — влево, нечётный — вправо);
  растекание   вода, лежащая на твёрдой опоре, течёт вбок в ту же сторону — но только если
               на ней лежит ещё вода (давит) или рядом обрыв; капля на ровном месте стоит;
  огонь        живёт один такт; поджигает соседнее СУХОЕ дерево («сухое» — когда рядом,
               по сторонам света, нет воды); соседний лёд превращает в воду;
  «я»          иду влево/вправо (если клетка свободна, иначе просто поворачиваюсь),
               прыгаю на клетку вверх (если есть опора), действую на клетку перед собой:
               копаю песок, поджигаю дерево, толкаю камень; кладу песок/воду/камень.

Один такт считается целиком из кадра до такта (реакции, действия «я» и движения
вычисляются параллельно из него), поэтому мир — настоящая функция кадра.
"""
from __future__ import annotations

import numpy as np

AIR, GROUND, STONE, SAND, WATER, WOOD, FIRE, ICE, AGENT_L, AGENT_R = range(10)
K = 10
NAMES = ["воздух", "земля", "камень", "песок", "вода", "дерево", "огонь", "лёд", "я←", "я→"]
GLYPHS = " #OS~wFi<>"

NOOP, LEFT, RIGHT, JUMP, ACT, PUT_SAND, PUT_WATER, PUT_STONE = range(8)
NA = 8
ACTION_NAMES = ["ничего", "влево", "вправо", "прыжок", "действие", "песок", "вода", "камень"]

H, W = 12, 20

# группы законов — чтобы можно было «выключать» их (для проверки самих проверок)
(R_FALL, R_SLIDE, R_SPREAD, R_BURN, R_WET, R_MELT, R_MOVE, R_JUMP, R_DIG, R_IGNITE,
 R_PUSH, R_PUT) = (1 << i for i in range(12))
R_ALL = (1 << 12) - 1
RULE_NAMES = {R_FALL: "падение", R_SLIDE: "осыпание", R_SPREAD: "растекание",
              R_BURN: "горение", R_WET: "мокрое дерево", R_MELT: "таяние", R_MOVE: "ходьба",
              R_JUMP: "прыжок", R_DIG: "копание", R_IGNITE: "поджигание",
              R_PUSH: "толкание", R_PUT: "кладу"}


def step(g: np.ndarray, a: np.ndarray, ph: np.ndarray, rules: int = R_ALL) -> np.ndarray:
    """Один такт для пачки миров: g (N,H,W), действия a (N,), метроном ph (N,) -> новый g."""
    g = np.asarray(g, dtype=np.uint8)
    a = np.asarray(a)
    ph = np.asarray(ph)
    N, h, w = g.shape
    idx = np.arange(N)
    P = np.pad(g, ((0, 0), (2, 2), (2, 2)), constant_values=GROUND)

    def nb(dy, dx):
        return P[:, 2 + dy:2 + dy + h, 2 + dx:2 + dx + w]

    def at(y, x):
        return P[idx, 2 + y, 2 + x]

    new = g.copy()
    aff = np.zeros(g.shape, bool)  # клетки, которых коснулось «я» (физика их не трогает)

    # ---------- реакции: огонь, дерево, лёд ----------
    if rules & (R_BURN | R_MELT):
        up, dn, lf, rt = nb(-1, 0), nb(1, 0), nb(0, -1), nb(0, 1)
        fire_adj = (up == FIRE) | (dn == FIRE) | (lf == FIRE) | (rt == FIRE)
        if rules & R_BURN:
            wet = np.zeros(g.shape, bool)
            if rules & R_WET:
                wet = (up == WATER) | (dn == WATER) | (lf == WATER) | (rt == WATER)
            new[g == FIRE] = AIR
            new[(g == WOOD) & fire_adj & ~wet] = FIRE
        if rules & R_MELT:
            new[(g == ICE) & fire_adj] = WATER

    # ---------- «я» ----------
    isag = g >= AGENT_L
    flat = isag.reshape(N, -1).argmax(1)
    ay, ax = flat // w, flat % w
    kind = g[idx, ay, ax]
    face = np.where(kind == AGENT_L, -1, 1)

    def write(mask, y, x, val):
        if not mask.any():
            return
        i = idx[mask]
        new[i, y[mask], x[mask]] = val if np.isscalar(val) else val[mask]
        aff[i, y[mask], x[mask]] = True

    walk = (a == LEFT) | (a == RIGHT)
    d = np.where(a == LEFT, -1, np.where(a == RIGHT, 1, 0))
    if rules & R_MOVE:
        nk = np.where(d < 0, AGENT_L, AGENT_R).astype(np.uint8)
        free = at(ay, ax + d) == AIR
        go, turn = walk & free, walk & ~free
        write(go, ay, ax, AIR)
        write(go, ay, ax + d, nk)
        write(turn, ay, ax, nk)
    if rules & R_JUMP:
        j = (a == JUMP) & (at(ay + 1, ax) != AIR) & (at(ay - 1, ax) == AIR)
        write(j, ay, ax, AIR)
        write(j, ay - 1, ax, kind)

    fx = ax + face
    fk = at(ay, fx)
    act = a == ACT
    if rules & R_DIG:
        write(act & (fk == SAND), ay, fx, AIR)
    if rules & R_IGNITE:
        wet_f = np.zeros(N, bool)
        if rules & R_WET:
            wet_f = ((at(ay - 1, fx) == WATER) | (at(ay + 1, fx) == WATER)
                     | (at(ay, fx - 1) == WATER) | (at(ay, fx + 1) == WATER))
        write(act & (fk == WOOD) & ~wet_f, ay, fx, FIRE)
    if rules & R_PUSH:
        m = act & (fk == STONE) & (at(ay, fx + face) == AIR)
        write(m, ay, fx, AIR)
        write(m, ay, fx + face, STONE)
    if rules & R_PUT:
        for code, mat in ((PUT_SAND, SAND), (PUT_WATER, WATER), (PUT_STONE, STONE)):
            write((a == code) & (fk == AIR), ay, fx, mat)

    # ---------- движение: падение, осыпание, растекание ----------
    Pa = np.pad(aff, ((0, 0), (2, 2), (2, 2)))

    def affs(dy, dx):
        return Pa[:, 2 + dy:2 + dy + h, 2 + dx:2 + dx + w]

    movers = (g == STONE) | (g == SAND) | (g == WATER) | (g >= AGENT_L)
    claimed = np.zeros(g.shape, bool)  # куда уже кто-то идёт
    dnk = nb(1, 0)

    def go(ok, dy, dx):
        n, y, x = np.nonzero(ok)
        if n.size == 0:
            return
        new[n, y + dy, x + dx] = g[n, y, x]
        new[n, y, x] = AIR
        claimed[n, y + dy, x + dx] = True

    def cl(dy, dx):
        Pc = np.pad(claimed, ((0, 0), (2, 2), (2, 2)))
        return Pc[:, 2 + dy:2 + dy + h, 2 + dx:2 + dx + w]

    if rules & R_FALL:
        go(movers & (dnk == AIR) & ~aff & ~affs(1, 0), 1, 0)
    ph3 = ph[:, None, None]
    if rules & R_SLIDE:
        sand = (g == SAND) & (dnk != AIR) & ~aff
        for sd, pv in ((-1, 0), (1, 1)):
            ok = (sand & (ph3 == pv) & (nb(0, sd) == AIR) & (nb(1, sd) == AIR)
                  & ~affs(1, sd) & ~cl(1, sd))
            go(ok, 1, sd)
    if rules & R_SPREAD:
        wat = (g == WATER) & (dnk != AIR) & (dnk != WATER) & ~aff  # лежит на твёрдом, не на воде
        for sd, pv in ((-1, 0), (1, 1)):
            # течёт, если на ней лежит другая вода (давит) или рядом обрыв
            ok = (wat & (ph3 == pv) & (nb(0, sd) == AIR) & ((nb(-1, 0) == WATER) | (nb(1, sd) == AIR))
                  & ~affs(0, sd) & ~cl(0, sd))
            go(ok, 0, sd)
    return new


class Sandbox:
    """Пачка параллельных миров. Каждый такт: step(действия) -> (кадр до, метроном до, кадр после)."""

    def __init__(self, n: int, seed: int = 0, gen=None, rules: int = R_ALL):
        self.rng = np.random.default_rng(seed)
        gen = gen or gen_train
        self.g = np.stack([gen(self.rng) for _ in range(n)])
        self.ph = np.zeros(n, np.int64)
        self.rules = rules

    def step(self, a):
        g0, ph0 = self.g, self.ph
        self.g = step(g0, np.asarray(a), ph0, self.rules)
        self.ph = 1 - ph0
        return g0, ph0, self.g


# ---------------------------------------------------------------- рождение миров

def empty_world() -> np.ndarray:
    g = np.zeros((H, W), np.uint8)
    g[0, :] = g[-1, :] = GROUND
    g[:, 0] = g[:, -1] = GROUND
    return g


def floor_heights(rng, g: np.ndarray, lo: int = 0, hi: int = 3, wiggle: float = 0.5):
    """Неровный пол: возвращает высоты по столбцам (над нижней границей) и рисует их."""
    f = int(rng.integers(lo, hi + 1))
    heights = np.zeros(W, int)
    for x in range(1, W - 1):
        if rng.random() < wiggle:
            f = int(np.clip(f + rng.choice([-1, 1]), lo, hi))
        heights[x] = f
        if f > 0:
            g[H - 1 - f:H - 1, x] = GROUND
    return heights


def gen_train(rng) -> np.ndarray:
    """Обычный мир «детства»: пол с бугорками и разбросанные глыбы разных веществ."""
    g = empty_world()
    heights = floor_heights(rng, g)
    for _ in range(int(rng.integers(3, 8))):
        kind = int(rng.choice([SAND, STONE, WATER, WOOD, ICE], p=[.25, .2, .2, .2, .15]))
        bw, bh = int(rng.integers(1, 5)), int(rng.integers(1, 4))
        x0 = int(rng.integers(1, W - 1 - bw + 1))
        y0 = int(rng.integers(1, H - 1 - bh + 1))
        reg = g[y0:y0 + bh, x0:x0 + bw]
        reg[reg == AIR] = kind
    # «наборы», в которых происходит что-то интересное: дерево рядом со льдом или водой
    for other in (ICE, WATER, SAND):
        if rng.random() < .35:
            x = int(rng.integers(1, W - 3))
            top = H - 2 - int(heights[x])
            top2 = H - 2 - int(heights[x + 1])
            g[top, x] = WOOD
            if rng.random() < .5 and top > 1:
                g[top - 1, x] = WOOD
            g[top2, x + 1] = other
    x = int(rng.integers(1, W - 1))
    y = H - 2 - int(heights[x])
    g[y, x] = AGENT_L if rng.random() < .5 else AGENT_R
    return g


def render(g: np.ndarray) -> str:
    return "\n".join("".join(GLYPHS[v] for v in row) for row in g)
