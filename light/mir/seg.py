"""Растущая кора из сегментов.

Что это. Для каждого сорта клетки (песок, вода, воздух…) — своё «дерево сегментов».
Сегмент — условие на нескольких соседей («центр — песок, снизу — воздух, метроном — чётный»)
вместе с накопленным ожиданием «что станет с центром». Всё обучение — счёт совпадений
и рост; обратного распространения ошибки нет.

Как растёт. Новый опыт раскладывается по сегментам. Если у сегмента ожидание не всегда
сбывается (в нём «перемешаны» разные исходы), он делится на два: «условие выполнено / нет».
Условие выбирают так:
  1. оно должно собрать не меньше двух РАЗНЫХ свидетельств редкого исхода — по одному случаю
     правило не рождается (иначе выучим случайность: «горит, потому что в углу лёд»);
  2. среди подходящих лучшее — то, что сильнее всего «чистит» сегмент;
  3. при равенстве — условие на клетке, которая сама изменилась вместе с центром (причина
     обычно рядом с тем, что происходит), затем на более близкой к центру клетке.
Опыт, в котором центр не меняется, тоже учит (это «отрицательные примеры»: песок с землёй
снизу не падает), поэтому условия вроде «снизу — НЕ воздух» получаются сами.

Честно о происхождении. Это близко к классическим инкрементальным деревьям решений и
системам правил (ACS/LCS); новое здесь не алгоритм, а то, как он встроен в «организм»:
сегменты — единицы, общие для всех мест мира, растут по ошибке, их можно читать как законы
(`rules_text`), а память о неожиданном и сон (дневник) подключаются отдельно.
"""
from __future__ import annotations

import time

import numpy as np

from .patches import CENTER, DIST, NF, NF_BASE, OFFS, POOL4, POOL8, Samples, keys, patches
from .world import K, NAMES

POS_NAMES = {(-1, 0): "сверху", (1, 0): "снизу", (0, -1): "слева", (0, 1): "справа",
             (-1, -1): "сверху-слева", (-1, 1): "сверху-справа", (1, -1): "снизу-слева", (1, 1): "снизу-справа"}
ACTION_NAMES_RU = ["ничего", "влево", "вправо", "прыжок", "действие", "песок", "вода", "камень"]


def pos_name(p: int) -> str:
    if p >= POOL8:
        return f"вокруг (8) есть {NAMES[p - POOL8]}"
    if p >= POOL4:
        return f"рядом по сторонам есть {NAMES[p - POOL4]}"
    if p == 25:
        return "я делаю"
    if p == 26:
        return "метроном"
    dy, dx = OFFS[p]
    if (dy, dx) in POS_NAMES:
        return POS_NAMES[(dy, dx)]
    return f"({dy:+d},{dx:+d})"


def cond_words(p: int, v: int, eq: bool) -> str:
    """Условие «признак p равен v» (eq=True) или «не равен v» (eq=False) словами."""
    if p >= POOL4:   # двоичные «рядом есть X»: значение 1 — есть, 0 — нет
        present = (v == 1) == eq
        return pos_name(p) if present else "НЕТ: " + pos_name(p)
    val = NAMES[v] if p < 25 else (ACTION_NAMES_RU[v] if p == 25 else ("чёт" if v == 0 else "нечёт"))
    return f"{pos_name(p)} {'=' if eq else '≠'} {val}"


def _entropy(c: np.ndarray) -> np.ndarray:
    s = c.sum(-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(s > 0, c / s, 0.0)
        return -(np.where(p > 0, p * np.log(p), 0.0)).sum(-1)


class SegCortex:
    def __init__(self, min_unique: int = 2, alpha: float = 0.02, floor: float = 1e-5, max_depth: int = 40,
                 cochange: bool = True, tol: float = 1.0, pooled: bool = True, consolidate: bool = False,
                 half_life: float | None = None):
        self.min_unique, self.alpha, self.floor, self.max_depth = min_unique, alpha, floor, max_depth
        self.half_life = half_life   # простое забывание: вес всего накопленного убывает вдвое за столько тактов
        self.cochange = cochange
        self.pooled, self.consolidate = pooled, consolidate
        self.tol = tol   # <1: из условий почти не хуже лучшего берём ближайшее к центру (бритва Оккама)
        self.reset()

    # ------------------------------------------------------------------ хранилище и узлы
    def reset(self):
        self.X = np.zeros((0, NF), np.uint8)
        self.y = np.zeros(0, np.uint8)
        self.w = np.zeros(0)
        self.chg = np.zeros(0, np.int32)
        self.leaf = np.zeros(0, np.int64)
        self.k = np.zeros(0, np.uint64)
        self.last = np.zeros(0, np.int64)    # «часы опыта» последней встречи каждого окна
        self.clock = 0                       # сколько тактов опыта прожито
        self._index: dict | None = None
        self.feat, self.val, self.yes, self.no, self.depth = [], [], [], [], []
        self.members: dict[int, np.ndarray] = {}
        self.root = np.full(K, -1, np.int64)
        self.stuck: set[int] = set()
        self._arr = None
        self._ver = 0
        self._tab_ver = None   # кэш таблицы листьев недействителен после сброса
        self.rules: list = []
        self.info = {}

    def _new_node(self, depth: int) -> int:
        self.feat.append(-1)
        self.val.append(-1)
        self.yes.append(-1)
        self.no.append(-1)
        self.depth.append(depth)
        self._arr = None
        return len(self.feat) - 1

    def _arrays(self):
        if self._arr is None:
            self._arr = (np.array(self.feat, np.int64), np.array(self.val, np.int64),
                         np.array(self.yes, np.int64), np.array(self.no, np.int64))
        return self._arr

    def _route(self, X: np.ndarray) -> np.ndarray:
        """В какой лист попадает каждое окно."""
        feat, val, yes, no = self._arrays()
        c = X[:, CENTER].astype(np.int64)
        node = self.root[c].copy()
        alive = node >= 0
        idx = np.nonzero(alive)[0]
        while idx.size:
            nd = node[idx]
            internal = feat[nd] >= 0
            idx = idx[internal]
            if not idx.size:
                break
            nd = nd[internal]
            go = X[idx, feat[nd]].astype(np.int64) == val[nd]
            node[idx] = np.where(go, yes[nd], no[nd])
        return node

    # ------------------------------------------------------------------ обучение
    def fit(self, S0, A, PH, S1):
        self.reset()
        return self.partial_fit(S0, A, PH, S1)

    def partial_fit(self, S0, A, PH, S1):
        t = time.time()
        n = len(S0)
        if self.half_life and len(self.w):
            self.w *= 0.5 ** (n / self.half_life)
        self.clock += n
        self._add(Samples.collect(S0, A, PH, S1), self.clock)
        self._grow()
        self._ver += 1
        if self.consolidate:
            self.sleep()
        self.info = {"fit_s": round(time.time() - t, 1), **self.stats()}
        return self

    def _add(self, S: Samples, t_end: int = 0):
        kk = keys(S.X) * np.uint64(K) + S.y.astype(np.uint64)
        n0 = len(self.y)
        if n0 == 0:
            self.X, self.y, self.w, self.chg, self.k = S.X.copy(), S.y.copy(), S.w.copy(), S.chg.copy(), kk
            self.last = np.full(len(S.y), t_end, np.int64)
            new_rows = np.arange(len(S.y))
            changed_rows = np.zeros(0, np.int64)
        else:
            if self._index is None:
                self._index = {int(a): i for i, a in enumerate(self.k)}
            pos = np.fromiter((self._index.get(int(a), -1) for a in kk), np.int64, len(kk))
            old = pos >= 0
            changed_rows = pos[old]
            np.add.at(self.w, changed_rows, S.w[old])
            self.last[changed_rows] = t_end
            fresh = ~old
            new_rows = n0 + np.arange(int(fresh.sum()))
            self.X = np.concatenate([self.X, S.X[fresh]])
            self.y = np.concatenate([self.y, S.y[fresh]])
            self.w = np.concatenate([self.w, S.w[fresh]])
            self.chg = np.concatenate([self.chg, S.chg[fresh]])
            self.k = np.concatenate([self.k, kk[fresh]])
            self.last = np.concatenate([self.last, np.full(int(fresh.sum()), t_end, np.int64)])
        if self._index is not None:
            for i in new_rows:
                self._index[int(self.k[i])] = int(i)
        self.leaf = np.concatenate([self.leaf, np.full(len(new_rows), -1, np.int64)])
        # корни для новых сортов центра
        for c in np.unique(self.X[new_rows, CENTER]):
            if self.root[c] < 0:
                self.root[c] = self._new_node(0)
                self.members[self.root[c]] = np.zeros(0, np.int64)
        leaves = self._route(self.X[new_rows])
        self.leaf[new_rows] = leaves
        self._dirty = set()
        for lf in np.unique(leaves):
            self.members[int(lf)] = np.concatenate([self.members[int(lf)], new_rows[leaves == lf]])
            self._dirty.add(int(lf))
            self.stuck.discard(int(lf))
        for r in changed_rows:
            lf = int(self.leaf[r])
            self._dirty.add(lf)

    def _grow(self):
        dirty = list(self._dirty)
        while dirty:
            node = dirty.pop()
            if node in self.stuck or self.depth[node] >= self.max_depth:
                continue
            ids = self.members[node]
            if ids.size < 2:
                continue
            test = self._best_test(ids)
            if test is None:
                self.stuck.add(node)
                continue
            p, v = test
            go = self.X[ids, p] == v
            a, b = self._new_node(self.depth[node] + 1), self._new_node(self.depth[node] + 1)
            self.feat[node], self.val[node], self.yes[node], self.no[node] = p, v, a, b
            self.members[a], self.members[b] = ids[go], ids[~go]
            self.leaf[ids[go]], self.leaf[ids[~go]] = a, b
            del self.members[node]
            dirty += [a, b]
        self._arr = None
        self._dirty = set()

    def _best_test(self, ids):
        y = self.y[ids].astype(np.int64)
        tot = np.bincount(y, minlength=K).astype(np.float64)
        kstar = int(tot.argmax())
        if tot[kstar] == tot.sum():
            return None
        X = self.X[ids].astype(np.int64)
        m = len(ids)
        flat = (np.arange(NF)[None, :] * 10 + X) * K + y[:, None]
        cy = np.bincount(flat.ravel(), minlength=NF * 10 * K).reshape(NF, 10, K).astype(np.float64)
        cn = tot[None, None, :] - cy
        wy, wn = cy.sum(-1), cn.sum(-1)
        gain = _entropy(tot) - (wy * _entropy(cy) + wn * _entropy(cn)) / m
        minor_y = cy.copy()
        minor_n = cn.copy()
        minor_y[..., kstar] = 0
        minor_n[..., kstar] = 0
        frac_y = np.where(wy > 0, minor_y.sum(-1) / np.maximum(wy, 1), 0)
        frac_n = np.where(wn > 0, minor_n.sum(-1) / np.maximum(wn, 1), 0)
        support = np.where(frac_y >= frac_n, minor_y.max(-1), minor_n.max(-1))
        ok = (wy > 0) & (wn > 0) & (gain > 1e-9) & (support >= self.min_unique)
        if not self.pooled:
            ok[NF_BASE:] = False
        if not ok.any():
            return None
        cc = np.zeros((NF, 10))
        if self.cochange:
            surprise = y != kstar
            for p in range(25):
                sel = surprise & (((self.chg[ids] >> p) & 1) == 1)
                if sel.any():
                    cc[p] = np.bincount(X[sel, p], minlength=10)
        P, V = np.nonzero(ok)
        if self.tol < 1.0:
            near = gain[P, V] >= self.tol * gain[P, V].max()
            P, V = P[near], V[near]
            order = np.lexsort((P, -np.round(gain[P, V], 6), -cc[P, V], DIST[P]))
        else:
            order = np.lexsort((P, DIST[P], -cc[P, V], -np.round(gain[P, V], 6)))
        j = order[0]
        return int(P[j]), int(V[j])

    # ------------------------------------------------------------------ предсказание
    def _leaf_logp(self) -> np.ndarray:
        """Таблица log-вероятностей для каждого листа (узлы × сорт)."""
        n = len(self.feat)
        ok = self.leaf >= 0
        cnt = np.bincount(self.leaf[ok] * K + self.y[ok].astype(np.int64), weights=self.w[ok],
                          minlength=n * K).reshape(n, K)
        C = cnt.sum(1, keepdims=True)
        p = (cnt + self.alpha) / (C + K * self.alpha)
        return np.log(np.maximum(p, self.floor))

    def _window_logp(self, U: np.ndarray):
        """Для уникальных окон: номер листа (−1, если такого сорта центра ещё не было) и log-вероятности."""
        if getattr(self, "_tab_ver", None) != self._ver:
            self._tab = self._leaf_logp()
            self._tab_ver = self._ver
        node = self._route(U)
        out = np.empty((len(U), K), np.float32)
        known = node >= 0
        out[known] = self._tab[node[known]]
        if (~known).any():   # сорт, которого ещё не встречали: «останется как есть»
            c = U[~known, CENTER].astype(np.int64)
            lp = np.full((len(c), K), np.log(0.1 / (K - 1)), np.float32)
            lp[np.arange(len(c)), c] = np.log(0.9)
            out[~known] = lp
        return node, out

    def logp(self, g0, a, ph):
        n, h, w = g0.shape
        X = patches(g0, a, ph).reshape(-1, NF)
        uk, first, inv = np.unique(keys(X), return_index=True, return_inverse=True)
        U = X[first]
        _, out = self._window_logp(U)
        if self.rules:
            self._apply_rules(U, out)
        return out[inv].reshape(n, h, w, K)

    # ------------------------------------------------------------------ сон: перестройка и отставка
    def rebuild(self):
        """Вырастить дерево заново по всему накопленному (качество как при разовом обучении)."""
        t = time.time()
        self.feat, self.val, self.yes, self.no, self.depth = [], [], [], [], []
        self.members = {}
        self.root = np.full(K, -1, np.int64)
        self.stuck = set()
        self._arr = None
        self.rules = []
        self.leaf = np.full(len(self.y), -1, np.int64)
        for c in np.unique(self.X[:, CENTER]):
            r = self._new_node(0)
            self.root[c] = r
            rows = np.nonzero(self.X[:, CENTER] == c)[0]
            self.members[r] = rows
            self.leaf[rows] = r
        self._dirty = set(self.members)
        self._grow()
        self._ver += 1
        self.info["rebuild_s"] = round(time.time() - t, 1)
        self.info.update(self.stats())
        return self

    def retire_rows(self, mask: np.ndarray, factor: float):
        """Отставка: вес выбранных строк хранилища (устаревших свидетельств) умножается на factor."""
        self.w[mask] *= factor
        self._ver += 1

    def _apply_rules(self, U: np.ndarray, out: np.ndarray):
        """Общие правила (после сна) перекрывают дерево там, где подходят."""
        best = np.full(len(U), -1.0)
        who = np.full(len(U), -1)
        for ri, r in enumerate(self.rules):
            m = U[:, CENTER] == r["center"]
            for (p, v, eq) in r["cond"]:
                m &= (U[:, p] == v) if eq else (U[:, p] != v)
            sc = (r["n_pos"] + 1) / (r["n_pos"] + r["n_neg"] + 2)
            upd = m & (sc > best)
            best[upd] = sc
            who[upd] = ri
        rows = np.nonzero(who >= 0)[0]
        if not rows.size:
            return
        lp = np.full((len(rows), K), np.log(self.floor), np.float32)
        lab = np.array([self.rules[i]["label"] for i in who[rows]])
        cen = U[rows, CENTER].astype(np.int64)
        pr = np.minimum(best[rows], 1 - 1e-3)
        lp[np.arange(len(rows)), cen] = np.log(1 - pr)
        lp[np.arange(len(rows)), lab] = np.log(pr)
        out[rows] = lp

    # ------------------------------------------------------------------ сон: общие правила
    def sleep(self, min_pos: int = 2, max_neg_frac: float = 0.01, gain_new: int = 2, min_precision: float = 0.9):
        """Сон-обобщение. Каждый лист-«изменение» превращается в правило, из которого выбрасываются
        условия, не подтверждённые опытом как нужные: условие снимается, только если после этого
        правило подтверждается как минимум `gain_new` НОВЫМИ разными случаями и почти без
        контрпримеров. Если новых случаев нет, условие остаётся (опыт не говорит, что оно лишнее)."""
        t = time.time()
        feat, val, yes, no = self._arrays()
        leaves = []

        def walk(node, conds):
            if feat[node] >= 0:
                p, v = int(feat[node]), int(val[node])
                walk(yes[node], conds + [(p, v, True)])
                walk(no[node], conds + [(p, v, False)])
            else:
                leaves.append((node, conds))

        for c in range(K):
            if self.root[c] >= 0:
                walk(int(self.root[c]), [])
        byc: dict = {}
        found: dict = {}
        for node, conds in leaves:
            mem = self.members.get(node)
            if mem is None or not len(mem):
                continue
            c = int(self.X[mem[0], CENTER])
            cnt = np.bincount(self.y[mem].astype(np.int64), minlength=K)
            lab = int(cnt.argmax())
            if lab == c or cnt[lab] < min_pos or cnt[lab] / cnt.sum() < min_precision:
                continue
            if c not in byc:
                rows = np.nonzero(self.X[:, CENTER] == c)[0]
                byc[c] = (self.X[rows], self.y[rows])
            Xc, yc = byc[c]
            B = [(Xc[:, p] == v) if eq else (Xc[:, p] != v) for (p, v, eq) in conds]
            viol = np.zeros(len(Xc), np.int32)
            for b in B:
                viol += ~b
            is_lab = yc == lab
            keep = list(range(len(conds)))
            cur_pos = int((is_lab & (viol == 0)).sum())
            for j in sorted(range(len(conds)), key=lambda j: (-DIST[conds[j][0]], -j)):
                vj = viol - (~B[j])
                hit = vj == 0
                n_pos, n_neg = int((is_lab & hit).sum()), int((~is_lab & hit).sum())
                if n_neg <= max_neg_frac * (n_pos + n_neg) and n_pos >= cur_pos + gain_new:
                    viol, cur_pos = vj, n_pos
                    keep.remove(j)
            hit = viol == 0
            n_pos, n_neg = int((is_lab & hit).sum()), int((~is_lab & hit).sum())
            key = (c, lab, tuple(sorted(conds[j] for j in keep)))
            if key not in found or found[key][0] < n_pos:
                found[key] = (n_pos, n_neg)
        self.rules = [{"center": c, "label": lab, "cond": cond, "n_pos": a, "n_neg": b}
                      for (c, lab, cond), (a, b) in found.items()]
        self.info["sleep_s"] = round(time.time() - t, 1)
        self.info["rules"] = len(self.rules)
        return self

    # ------------------------------------------------------------------ чтение выученного
    def stats(self) -> dict:
        feat = self._arrays()[0]
        return {"nodes": int(len(feat)), "leaves": int((feat < 0).sum()), "unique": int(len(self.y)),
                "depth": int(max(self.depth)) if self.depth else 0}

    def rule_words(self, r: dict) -> str:
        conds = [cond_words(p, v, eq) for (p, v, eq) in r["cond"]]
        return (f"{NAMES[r['center']]} → {NAMES[r['label']]}  если " + (", ".join(conds) or "всегда") +
                f"   [подтверждено {r['n_pos']} разными случаями, исключений {r['n_neg']}]")

    def laws_text(self, top: int = 40, centers=None) -> str:
        """Общие правила после сна, самые подтверждённые сверху (centers — только для этих сортов центра)."""
        rs = sorted((r for r in self.rules if centers is None or r["center"] in centers), key=lambda r: -r["n_pos"])
        return "\n".join(self.rule_words(r) for r in rs[:top])

    def rules_text(self, top: int = 40, min_weight: float = 3.0) -> str:
        """Выученные «законы» словами: листья, где центр меняется."""
        feat, val, yes, no = self._arrays()
        tab = self._leaf_logp()
        out = []

        def walk(node, path):
            if feat[node] >= 0:
                p, v = int(feat[node]), int(val[node])
                walk(yes[node], path + [cond_words(p, v, True)])
                walk(no[node], path + [cond_words(p, v, False)])
                return
            mem = self.members.get(node)
            if mem is None or not len(mem):
                return
            wts = np.bincount(self.y[mem].astype(np.int64), weights=self.w[mem], minlength=K)
            best = int(wts.argmax())
            c = int(self.X[mem[0], CENTER])
            if best != c and wts[best] >= min_weight:
                out.append((wts[best], f"{NAMES[c]} → {NAMES[best]}  если " + ", ".join(path) +
                            f"   [{wts[best]:.0f} раз, чистота {wts[best]/wts.sum()*100:.0f}%]"))

        for c in range(K):
            if self.root[c] >= 0:
                walk(int(self.root[c]), [])
        out.sort(key=lambda t: -t[0])
        return "\n".join(t for _, t in out[:top])
