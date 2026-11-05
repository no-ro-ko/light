"""Сводка результатов и вердикты по условиям успеха, записанным заранее (docs/MIR.md, разд. 5 и 9).

  python3 -m light.mir.report curves      # кривые «опыт → итог» и вердикты H1, H2
  python3 -m light.mir.report live        # блуждание против любопытства: H3, H4
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

OUT = Path(__file__).resolve().parents[2] / "mir_runs"
GRID = [30, 100, 300, 1000, 3000, 10000, 30000]
BAR = 0.90


def load(name: str) -> list[dict]:
    p = OUT / f"{name}.jsonl"
    return [json.loads(x) for x in open(p, encoding="utf-8")] if p.exists() else []


def t90(med: dict[int, float], grid=GRID):
    for T in grid:
        if T in med and med[T] >= BAR:
            return T
    return None


def steps_between(a, b, grid=GRID):
    """Сколько шагов сетки бюджетов между a и b (None = не достигнуто → бесконечность)."""
    big = len(grid) + 5
    ia = grid.index(a) if a in grid else big
    ib = grid.index(b) if b in grid else big
    return ib - ia


def tuned(args):
    """Для каждой ширины — настройка с наибольшим средним черновым итогом на бюджетах 1000 и 10000."""
    rows = [r for r in load("tune_mlp") if r["cfg"].get("pooled")]
    agg = defaultdict(list)
    for r in rows:
        if r["T"] in (1000, 10000):
            c = r["cfg"]
            agg[(c["width"], c["weighting"], c["lr"])].append(r["dev"]["macro"])
    best = {}
    for (w, wt, lr), v in agg.items():
        if len(v) < 2 and w != 256:
            continue                      # настройка должна быть измерена на обоих бюджетах (ширина 256 — только на 10000)
        m = sum(v) / len(v)
        if w not in best or m > best[w][0]:
            best[w] = (m, wt, lr)
    for w in sorted(best):
        m, wt, lr = best[w]
        print(json.dumps({"width": w, "weighting": wt, "lr": lr, "steps": 8000, "pooled": True}, sort_keys=True),
              f"  # среднее черновое {m*100:.1f}% по {len(agg[(w, wt, lr)])} бюдж.")


def curves(args):
    rows = load(args.file)
    key = args.key
    groups = defaultdict(lambda: defaultdict(list))
    fams = defaultdict(lambda: defaultdict(list))
    levels = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if key not in r:
            continue
        name = f"{r['kind']} {json.dumps(r['cfg'], sort_keys=True, ensure_ascii=False)}"
        groups[name][r["T"]].append(r[key]["macro"])
        fams[name][r["T"]].append(r[key]["families"])
        levels[name][r["T"]].append(r[key]["levels"])
    meds = {n: {T: median(v) for T, v in g.items()} for n, g in groups.items()}
    print(f"Итог по {key}-набору, медиана по потокам опыта (в скобках — число потоков)")
    print(f"{'модель':58s}" + "".join(f"{T:>8d}" for T in GRID) + "     T90")
    for n, m in sorted(meds.items()):
        cells = "".join(f"{m[T]*100:8.1f}" if T in m else f"{'—':>8s}" for T in GRID)
        print(f"{n[:58]:58s}{cells}   {t90(m) or '> ' + str(max(m))}   ({len(groups[n][max(m)])})")
    cort = {n: m for n, m in meds.items() if n.startswith("seg") and "consolidate" not in n}
    sleepy = {n: m for n, m in meds.items() if n.startswith("seg") and "consolidate" in n}
    mlps = {n: m for n, m in meds.items() if n.startswith("mlp")}
    if cort and sleepy:
        b, d = next(iter(cort.values())), next(iter(sleepy.values()))
        gains = [(T, (d[T] - b[T]) * 100) for T in GRID if T in b and T in d]
        good = sum(1 for T, g in gains if T in (3000, 10000, 30000) and g >= 2)
        bad = [T for T, g in gains if g < -2]
        print("\nH5 (сон): разность D − B, п.п.: " + ", ".join(f"{T}: {g:+.1f}" for T, g in gains))
        print(f"H5: выше на ≥2 п.п. на {good} из бюджетов {{3000, 10000, 30000}}; ниже на >2 п.п. на бюджетах {bad or 'нет'} → "
              f"{'СОН ПОМОГ' if good >= 2 and not bad else 'СОН НЕ ПОКАЗАЛ ПОЛЬЗЫ'}")
    if not cort or not mlps:
        return
    bc = min(cort, key=lambda n: (t90(cort[n]) or 10 ** 9, -max(cort[n].values())))
    bm = min(mlps, key=lambda n: (t90(mlps[n]) or 10 ** 9, -max(mlps[n].values())))
    tc, tm = t90(cort[bc]), t90(mlps[bm])
    print(f"\nЛучшая кора:  {bc}   T90 = {tc}")
    print(f"Лучшая сеть:  {bm}   T90 = {tm}")
    if tc is None:
        v1 = "НЕ ВЫУЧИЛА: кора не достигла 90% на максимальном бюджете"
    elif tm is None:
        v1 = ("обычная сеть не достигла 90% на максимальном бюджете, а коре хватило "
              f"{tc}: преимущество есть, но «втрое» не измерить (цензура); считаем H1 выполненной, если {tc} ≤ {GRID[-2]}")
    else:
        sb = steps_between(tc, tm)
        v1 = (f"H1 {'ВЫПОЛНЕНА' if sb >= 1 else 'НЕ ВЫПОЛНЕНА'}: кора на {sb} шаг(ов) сетки раньше "
              f"(нужно ≥ 1; шаг ≈ ×3,3)")
    print("H1:", v1)
    Tc = tc or max(cort[bc])
    lv = [median(x[i] for x in levels[bc][Tc]) for i in range(3)]
    ratio = lv[2] / lv[0] if lv[0] else 0
    print(f"H2: на T={Tc} итог по уровням мусора {[round(x*100, 1) for x in lv]}; густой/чистый = {ratio:.2f} → "
          f"{'ВЫПОЛНЕНА' if ratio >= .85 else 'НЕ ВЫПОЛНЕНА'} (порог 0,85)")
    Tm = max(cort[bc])
    fm = {f: median(x[f] for x in fams[bc][Tm]) for f in fams[bc][Tm][0]}
    low = {f: round(v * 100) for f, v in fm.items() if v < .6}
    print(f"Семейства коры ниже 60% на T={Tm}: {low or 'нет'}")


def live(args):
    rows = []
    for f in args.file.split(","):
        rows += load(f)
    key = args.key
    by = defaultdict(lambda: defaultdict(list))
    sec = defaultdict(lambda: defaultdict(list))
    seeds = defaultdict(set)
    for r in rows:
        if key not in r:
            continue
        by[r["policy"]][r["T"]].append(r[key]["macro"])
        seeds[r["policy"]].add(r["seed"])
        for law, t in r["second_witness_at"].items():
            sec[r["policy"]][law].append(t)
    Ts = sorted({T for p in by.values() for T in p})
    print(f"Политики ({key}-набор), медиана по потокам; потоков: " + ", ".join(f"{p} — {len(v)}" for p, v in seeds.items()))
    print(f"{'политика':12s}" + "".join(f"{T:>8d}" for T in Ts))
    med = {}
    for pol, g in by.items():
        med[pol] = {T: median(v) for T, v in g.items()}
        print(f"{pol:12s}" + "".join(f"{med[pol][T]*100:8.1f}" if T in med[pol] else f"{'—':>8s}" for T in Ts))
    if "random" not in med:
        return
    for pol in [p for p in med if p not in ("random", "curious0")]:
        print(f"\n=== {pol} против блуждания ===")
        print("разность, п.п.: " + ", ".join(f"{T}: {(med[pol][T] - med['random'][T]) * 100:+.1f}" for T in Ts))
        win = lose = 0
        for T in (300, 1000, 3000, 10000):
            if T in med["random"] and T in med[pol]:
                d = (med[pol][T] - med["random"][T]) * 100
                win += d >= 3
                lose += d <= -3
        print(f"H3: бюджетов из {{300,1000,3000,10000}}, где выше на ≥3 п.п.: {win}; ниже на ≥3: {lose} → "
              f"{'ВЫПОЛНЕНА' if win >= 3 and lose == 0 else 'НЕ ВЫПОЛНЕНА'} (нужно ≥3 и 0)")
        faster, ratios = 0, []
        for law in sec["random"]:
            mr = median(x if x is not None else 10 ** 6 for x in sec["random"][law])
            mc = median(x if x is not None else 10 ** 6 for x in sec[pol][law])
            if mc < mr:
                faster += 1
                if mr < 10 ** 6:
                    ratios.append(mr / max(mc, 1))
        mratio = median(ratios) if ratios else 0
        print(f"H4: законов, где второе свидетельство раньше: {faster} из {len(sec['random'])}; "
              f"медианное ускорение среди них ×{mratio:.2f} → "
              f"{'ВЫПОЛНЕНА' if faster >= 6 and mratio >= 1.5 else 'НЕ ВЫПОЛНЕНА'} (нужно ≥6 законов и ≥×1,5)")
        if args.verbose:
            print(f"  {'закон':16s} {'блуждание':>10s} {pol:>10s}")
            for law in sec["random"]:
                a = median(x if x is not None else 10 ** 6 for x in sec["random"][law])
                b = median(x if x is not None else 10 ** 6 for x in sec[pol][law])
                print(f"  {law:16s} {'—' if a >= 10**6 else int(a):>10} {'—' if b >= 10**6 else int(b):>10}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tuned")
    t.set_defaults(fn=tuned)
    c = sub.add_parser("curves")
    c.add_argument("--file", default="final_curves")
    c.add_argument("--key", default="sealed")
    c.set_defaults(fn=curves)
    l = sub.add_parser("live")
    l.add_argument("--file", default="final_live", help="файлы через запятую")
    l.add_argument("--key", default="sealed")
    l.add_argument("--verbose", action="store_true")
    l.set_defaults(fn=live)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
