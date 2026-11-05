"""Смена закона: умеет ли модель ПЕРЕУЧИВАТЬСЯ.

Мир тот же, но метроном читается наоборот (0 ↔ 1): для существа это новый закон — теперь песок
осыпается и вода течёт в другую сторону при том же показании метронома. Модель обучается на
опыте мира A, затем получает порции опыта мира B. Проверяем на пробах мира B (те же кадры, метроном
инвертирован) и на пробах мира A (что осталось от прежних знаний).

  python3 -m light.mir.adapt --kind seg --cfg '{"tol": 0.8}'
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace

import numpy as np

from . import life, probes, sealed
from .bench import STREAM_T, make_model

STAGES = [0, 100, 300, 1000, 3000]          # сколько тактов опыта мира B уже получено
DIR_FAMILIES = ["осыпание песка", "растекание воды"]


def flipped(pr: probes.Probes) -> probes.Probes:
    return replace(pr, PH=1 - pr.PH)


def run(kind: str, cfg: dict, stream: int, t_a: int = 10000, mode: str = "dev") -> list[dict]:
    S0, A, PH, S1 = life.wander(STREAM_T, stream)
    pa = sealed.dev()
    pb = flipped(pa)
    model = make_model(kind, cfg).fit(S0[:t_a], A[:t_a], PH[:t_a], S1[:t_a])
    out = []
    got = 0
    for k in STAGES:
        if k > got:   # новый опыт мира B: следующие такты потока, метроном инвертирован
            lo, hi = t_a + got, t_a + k
            args = (S0[lo:hi], A[lo:hi], 1 - PH[lo:hi], S1[lo:hi])
            if kind == "seg":
                model.partial_fit(*args)
            else:
                model.fit(*args)             # дообучение с текущих весов
            got = k
        ra, rb = probes.evaluate(model, pa), probes.evaluate(model, pb)
        out.append({"kind": kind, "cfg": cfg, "stream": stream, "t_a": t_a, "b_steps": k,
                    "A": round(ra["macro"], 4), "B": round(rb["macro"], 4),
                    "B_direction": round(float(np.mean([rb["families"][f] for f in DIR_FAMILIES])), 4),
                    "A_direction": round(float(np.mean([ra["families"][f] for f in DIR_FAMILIES])), 4)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", default="seg")
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    ap.add_argument("--t-a", type=int, default=10000)
    ap.add_argument("--out", default="adapt")
    a = ap.parse_args()
    from .bench import append
    for s in a.seeds:
        for r in run(a.kind, json.loads(a.cfg), s, a.t_a):
            append(a.out, r)
            print(f"{a.kind} поток {s}: после {r['b_steps']:5d} тактов мира B — мир A {r['A']*100:5.1f}%  мир B {r['B']*100:5.1f}%  "
                  f"(осыпание+растекание: A {r['A_direction']*100:5.1f}%, B {r['B_direction']*100:5.1f}%)", flush=True)


if __name__ == "__main__":
    main()
