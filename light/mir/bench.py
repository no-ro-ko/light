"""Запуск экспериментов: кривые «опыт → результат» для разных моделей.

  python3 -m light.mir.bench tune-mlp                      # подбор настроек обычных сетей (черновой набор)
  python3 -m light.mir.bench curve --kind mlp --cfg '{"width": 64}' --budgets 100 1000 --seeds 11 12 13
  python3 -m light.mir.bench curve --kind seg --probes sealed ...   # итоговые цифры (запечатанный набор)

Результаты дописываются строками в mir_runs/<имя>.jsonl.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

from . import life, probes, sealed

STREAM_T = 100_000
OUT = Path(__file__).resolve().parents[2] / "mir_runs"
BUDGETS = [30, 100, 300, 1000, 3000, 10000, 30000]
DEV_STREAM = 91


def make_model(kind: str, cfg: dict):
    if kind == "mlp":
        from .mlp import PatchMLP
        return PatchMLP(**cfg)
    if kind == "seg":
        from .seg import SegCortex
        return SegCortex(**cfg)
    raise ValueError(kind)


def run_one(kind: str, cfg: dict, T: int, stream: int, mode: str = "dev", tag: str = "") -> dict:
    S0, A, PH, S1 = life.wander(STREAM_T, stream)
    t = time.time()
    m = make_model(kind, cfg).fit(S0[:T], A[:T], PH[:T], S1[:T])
    fit_s = time.time() - t
    rec = {"kind": kind, "cfg": cfg, "T": T, "stream": stream, "fit_s": round(fit_s, 1),
           "info": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in getattr(m, "info", {}).items()}}
    if "dev" in mode or mode == "both":
        r = probes.evaluate(m, sealed.dev())
        rec["dev"] = {"macro": round(r["macro"], 4), "families": {k: round(v, 3) for k, v in r["families"].items()},
                      "levels": [round(x, 3) for x in r["by_level"]]}
    if "sealed" in mode or mode == "both":
        r = sealed.evaluate_sealed(m, f"{kind}:{json.dumps(cfg, sort_keys=True)}:T={T}:stream={stream}:{tag}")
        rec["sealed"] = {"macro": round(r["macro"], 4), "families": {k: round(v, 3) for k, v in r["families"].items()},
                         "levels": [round(x, 3) for x in r["by_level"]]}
    return rec


def append(name: str, rec: dict):
    OUT.mkdir(exist_ok=True)
    with open(OUT / f"{name}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def tune_mlp(args):
    for width, weighting, lr in itertools.product(args.widths, args.weighting, args.lr):
        for T in args.budgets:
            cfg = {"width": width, "weighting": weighting, "lr": lr, "steps": args.steps, "pooled": True}
            rec = run_one("mlp", cfg, T, DEV_STREAM, "dev")
            append("tune_mlp", rec)
            print(f"ширина {width:3d} вес {weighting:5s} lr {lr:.0e} T={T:6d}: черновой {rec['dev']['macro']*100:5.1f}%  "
                  f"val {rec['info'].get('val', 0):.4f}  шагов {rec['info'].get('steps')}  {rec['fit_s']} с", flush=True)


def done_before(name: str, kind: str, cfg: dict, T: int, stream: int, mode: str) -> bool:
    """Уже есть запись для этой тройки (модель, бюджет, поток)? Повторно к запечатанному не обращаемся."""
    p = OUT / f"{name}.jsonl"
    if not p.exists():
        return False
    want = "sealed" if ("sealed" in mode or mode == "both") else "dev"
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if r["kind"] == kind and r["cfg"] == cfg and r["T"] == T and r["stream"] == stream and want in r:
            return True
    return False


def curve(args):
    cfg = json.loads(args.cfg)
    for stream in args.seeds:
        for T in args.budgets:
            if done_before(args.out or f"curve_{args.kind}", args.kind, cfg, T, stream, args.probes):
                print(f"пропуск (уже есть): {args.kind} T={T} поток {stream}", flush=True)
                continue
            rec = run_one(args.kind, cfg, T, stream, args.probes, args.tag)
            append(args.out or f"curve_{args.kind}", rec)
            key = "sealed" if "sealed" in rec else "dev"
            print(f"{args.kind} {cfg} T={T:6d} поток {stream}: {key} {rec[key]['macro']*100:5.1f}%  {rec['fit_s']} с", flush=True)


def live_cmd(args):
    from . import curious
    cfg = json.loads(args.cfg)
    for seed in args.seeds:
        for pol in args.policies:
            for rec in curious.live(pol, cfg, seed, args.tmax, args.probes, tag=args.tag, pol=json.loads(args.pol)):
                append(args.out or "live", rec)
                key = "sealed" if "sealed" in rec else "dev"
                print(f"{pol:8s} поток {seed} T={rec['T']:6d}: {key} {rec[key]['macro']*100:5.1f}%  ({rec['elapsed_s']} с)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tune-mlp")
    t.add_argument("--widths", type=int, nargs="+", default=[16, 64, 256])
    t.add_argument("--weighting", nargs="+", default=["unit", "sqrt", "count"])
    t.add_argument("--lr", type=float, nargs="+", default=[3e-3, 1e-2])
    t.add_argument("--budgets", type=int, nargs="+", default=[1000, 10000])
    t.add_argument("--steps", type=int, default=4000)
    t.set_defaults(fn=tune_mlp)
    c = sub.add_parser("curve")
    c.add_argument("--kind", required=True, choices=["mlp", "seg"])
    c.add_argument("--cfg", default="{}")
    c.add_argument("--budgets", type=int, nargs="+", default=BUDGETS)
    c.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    c.add_argument("--probes", default="dev", choices=["dev", "sealed", "both"])
    c.add_argument("--tag", default="")
    c.add_argument("--out", default="")
    c.set_defaults(fn=curve)
    L = sub.add_parser("live", help="дни существа: блуждание против любопытства")
    L.add_argument("--policies", nargs="+", default=["random", "curious"], choices=["random", "curious", "curious0", "approach"])
    L.add_argument("--cfg", default="{}")
    L.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    L.add_argument("--tmax", type=int, default=30000)
    L.add_argument("--pol", default="{}", help="параметры политики любопытства (JSON)")
    L.add_argument("--probes", default="dev", choices=["dev", "sealed", "both"])
    L.add_argument("--tag", default="")
    L.add_argument("--out", default="")
    L.set_defaults(fn=live_cmd)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
