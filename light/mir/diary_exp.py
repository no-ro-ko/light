"""Опыты с дневником (docs/MIR.md, разд. 13).

  change — смена закона: мир A (10 000 тактов), затем мир B (метроном читается наоборот), порциями по 100 тактов;
  stable — стабильный мир: один непрерывный опыт с контрольными точками (не вредит ли дневник, когда всё спокойно).

Участники: кора; кора + дневник; кора + дневник + закрепление; кора с простым забыванием (полураспад 500 / 2000
тактов); обычная сеть шириной 64. Все «кора»-участники получают опыт одинаково (разово на старом, затем порциями) и
одинаково «засыпают» (дерево вырастает заново) на контрольных точках.

  python3 -m light.mir.diary_exp change --probes dev --seeds 91 92 93
  python3 -m light.mir.diary_exp stable --probes sealed --seeds 11 12 13
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import numpy as np

from . import life, probes, sealed
from .bench import STREAM_T, append
from .diary import Diary
from .seg import SegCortex

DIR = ["осыпание песка", "растекание воды"]
STAGES = [0, 100, 300, 1000, 3000]
BACK = [100, 300, 1000]          # возврат: сколько тактов прожито после того, как закон вернулся к прежнему
BUDGETS = [1000, 3000, 10000, 30000]
MLP64 = {"lr": 0.01, "pooled": True, "steps": 8000, "weighting": "sqrt", "width": 64}

ARMS = {
    "кора": {"cortex": {}, "diary": None},
    "дневник": {"cortex": {}, "diary": {}},
    "дневник+закрепление": {"cortex": {}, "diary": {"consolidate": True}},
    "забывание200": {"cortex": {"half_life": 200}, "diary": None},
    "забывание500": {"cortex": {"half_life": 500}, "diary": None},
    "забывание2000": {"cortex": {"half_life": 2000}, "diary": None},
}


class Arm:
    def __init__(self, name: str, diary_kw: dict | None = None):
        spec = ARMS[name]
        self.name = name
        self.cortex = SegCortex(tol=0.8, **spec["cortex"])
        dk = spec["diary"]
        self.diary = Diary(self.cortex, **{**dk, **(diary_kw or {})}) if dk is not None else None

    @property
    def model(self):
        return self.diary or self.cortex

    def batch(self, S0, A, PH, S1, warm: int = 1000, chunk: int = 200):
        """Разовое обучение на старом, последние `warm` тактов — порциями (так у дневника есть что помнить)."""
        if self.diary:
            self.diary.fit(S0, A, PH, S1, warm=warm, chunk=chunk)
            return
        T = len(S0)
        w = min(warm, T)
        if T - w > 0:
            self.cortex.fit(S0[:T - w], A[:T - w], PH[:T - w], S1[:T - w])
        else:
            self.cortex.reset()
        for i in range(T - w, T, chunk):
            self.cortex.partial_fit(S0[i:i + chunk], A[i:i + chunk], PH[i:i + chunk], S1[i:i + chunk])

    def feed(self, S0, A, PH, S1):
        self.model.partial_fit(S0, A, PH, S1)

    def sleep(self):
        (self.diary.sleep() if self.diary else self.cortex.rebuild())

    def stats(self) -> dict:
        return self.diary.stats() if self.diary else {}


def _fam_means(r: dict) -> dict:
    rest = [v for k, v in r["families"].items() if k not in DIR]
    return {"dir": float(np.mean([r["families"][f] for f in DIR])), "rest": float(np.mean(rest)), "macro": r["macro"],
            "conf": r["conf"]}


def evaluate_ab(model, mode: str, tag: str):
    """Проверка в мире A (как есть) и в мире B (метроном наоборот)."""
    if mode == "sealed":
        ra = sealed.evaluate_sealed(model, tag + ":A")
        rb = sealed.evaluate_sealed(model, tag + ":B", flip=True)
    else:
        pa = sealed.dev()
        ra, rb = probes.evaluate(model, pa), probes.evaluate(model, replace(pa, PH=1 - pa.PH))
    a, b = _fam_means(ra), _fam_means(rb)
    return {"A": round(a["macro"], 4), "B": round(b["macro"], 4), "A_dir": round(a["dir"], 4), "B_dir": round(b["dir"], 4),
            "A_rest": round(a["rest"], 4), "B_rest": round(b["rest"], 4), "A_conf": round(a["conf"], 4), "B_conf": round(b["conf"], 4)}


def change(arm: str, stream: int, mode: str = "dev", t_a: int = 10000, chunk: int = 100, tag: str = "") -> list[dict]:
    S0, A, PH, S1 = life.wander(STREAM_T, stream)
    out = []
    t0 = time.time()
    if arm == "сеть64":
        from .mlp import PatchMLP
        m = PatchMLP(**MLP64).fit(S0[:t_a], A[:t_a], PH[:t_a], S1[:t_a])
        got = 0
        for k in STAGES:
            if k > got:
                lo, hi = t_a + got, t_a + k
                m.fit(S0[lo:hi], A[lo:hi], 1 - PH[lo:hi], S1[lo:hi])
                got = k
            out.append({"arm": arm, "stream": stream, "b_steps": k, **evaluate_ab(m, mode, f"change:{arm}:s{stream}:k{k}:{tag}")})
        gotb = 0
        for k in BACK:
            lo, hi = t_a + STAGES[-1] + gotb, t_a + STAGES[-1] + k
            m.fit(S0[lo:hi], A[lo:hi], PH[lo:hi], S1[lo:hi])
            gotb = k
            out.append({"arm": arm, "stream": stream, "b_steps": STAGES[-1], "back_steps": k,
                        **evaluate_ab(m, mode, f"change:{arm}:s{stream}:back{k}:{tag}")})
        return out
    a = Arm(arm)
    a.batch(S0[:t_a], A[:t_a], PH[:t_a], S1[:t_a])
    a.sleep()
    got = 0
    for k in STAGES:
        for i in range(t_a + got, t_a + k, chunk):
            a.feed(S0[i:i + chunk], A[i:i + chunk], 1 - PH[i:i + chunk], S1[i:i + chunk])
        if k > got:
            a.sleep()
        got = k
        out.append({"arm": arm, "stream": stream, "b_steps": k, **evaluate_ab(a.model, mode, f"change:{arm}:s{stream}:k{k}:{tag}"),
                    **a.stats(), "elapsed_s": round(time.time() - t0, 1)})
    gotb = 0
    for k in BACK:                                   # закон вернулся: мир A, метроном как раньше
        for i in range(t_a + STAGES[-1] + gotb, t_a + STAGES[-1] + k, chunk):
            a.feed(S0[i:i + chunk], A[i:i + chunk], PH[i:i + chunk], S1[i:i + chunk])
        a.sleep()
        gotb = k
        out.append({"arm": arm, "stream": stream, "b_steps": STAGES[-1], "back_steps": k,
                    **evaluate_ab(a.model, mode, f"change:{arm}:s{stream}:back{k}:{tag}"), **a.stats()})
    return out


def stable(arm: str, stream: int, mode: str = "dev", budgets=BUDGETS, chunk: int = 200, tag: str = "") -> list[dict]:
    S0, A, PH, S1 = life.wander(STREAM_T, stream)
    a = Arm(arm)
    out = []
    t0 = time.time()
    first = budgets[0]
    a.batch(S0[:first], A[:first], PH[:first], S1[:first], warm=first, chunk=chunk)
    pos = first
    for T in budgets:
        for i in range(pos, T, chunk):
            a.feed(S0[i:i + chunk], A[i:i + chunk], PH[i:i + chunk], S1[i:i + chunk])
        pos = T
        a.sleep()
        if mode == "sealed":
            r = sealed.evaluate_sealed(a.model, f"stable:{arm}:s{stream}:T{T}:{tag}")
        else:
            r = probes.evaluate(a.model, sealed.dev())
        out.append({"arm": arm, "stream": stream, "T": T, "macro": round(r["macro"], 4), "conf": round(r["conf"], 4),
                    "families": {k: round(v, 3) for k, v in r["families"].items()},
                    **a.stats(), "elapsed_s": round(time.time() - t0, 1)})
    return out


def done(name: str, **kw) -> bool:
    from .bench import OUT
    p = OUT / f"{name}.jsonl"
    if not p.exists():
        return False
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if all(r.get(k) == v for k, v in kw.items()):
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["change", "stable"])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    ap.add_argument("--probes", choices=["dev", "sealed"], default="dev")
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    name = a.out or f"diary_{a.what}_{a.probes}"
    for s in a.seeds:
        for arm in a.arms:
            first_key = {"arm": arm, "stream": s} | ({"back_steps": BACK[-1]} if a.what == "change" else {"T": BUDGETS[-1]})
            if done(name, **first_key):
                print(f"пропуск (уже есть): {a.what} {arm} поток {s}", flush=True)
                continue
            fn = change if a.what == "change" else stable
            for rec in fn(arm, s, a.probes, tag=a.tag):
                append(name, rec)
                if a.what == "change":
                    lab = f"B+{rec['b_steps']:5d}" + (f" назад+{rec['back_steps']:4d}" if "back_steps" in rec else "")
                    print(f"{arm:22s} поток {s} {lab}: мир A {rec['A']*100:5.1f}  мир B {rec['B']*100:5.1f}  "
                          f"направление A {rec['A_dir']*100:4.0f} B {rec['B_dir']*100:4.0f}  остальное B {rec['B_rest']*100:5.1f}", flush=True)
                else:
                    print(f"{arm:22s} поток {s} T={rec['T']:6d}: {rec['macro']*100:5.1f}%  ({rec['elapsed_s']} с)", flush=True)


if __name__ == "__main__":
    main()
