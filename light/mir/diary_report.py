"""Сводка опытов с дневником и вердикты H6–H9 по правилам из docs/MIR.md (разд. 13).

  python3 -m light.mir.diary_report --change diary_change_dev --stable diary_stable_dev
  python3 -m light.mir.diary_report --change diary_change_sealed --stable diary_stable_sealed
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

OUT = Path(__file__).resolve().parents[2] / "mir_runs"
STAGES = [0, 100, 300, 1000, 3000]
BUDGETS = [1000, 3000, 10000, 30000]
BACK = [100, 300, 1000]


def load(n):
    p = OUT / f"{n}.jsonl"
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []


def med(rows, key, **flt):
    v = [r[key] for r in rows if all(r.get(k) == x for k, x in flt.items()) and key in r]
    return median(v) if v else None


def fmt(x, nd=0):
    return "—" if x is None else f"{x*100:.{nd}f}"


def summarize(change_name: str, stable_name: str) -> dict:
    ch_all, st = load(change_name), load(stable_name)
    ch = [r for r in ch_all if "back_steps" not in r]          # этапы мира B
    back_rows = [r for r in ch_all if "back_steps" in r]        # возврат закона
    arms_c = list(dict.fromkeys(r["arm"] for r in ch))
    arms_s = list(dict.fromkeys(r["arm"] for r in st))
    print(f"== Смена закона ({change_name}); потоков: {len({r['stream'] for r in ch})}; медианы, %")
    print("   направление в мире B (осыпание + растекание) | остальные 14 семейств в мире B | уверенность в изменениях, мир B")
    print(f"{'участник':24s}" + "".join(f"{k:>6d}" for k in STAGES) + " |" + "".join(f"{k:>6d}" for k in STAGES[3:]) + " |" + "".join(f"{k:>6d}" for k in STAGES[3:]))
    for a in arms_c:
        print(f"{a:24s}" + "".join(f"{fmt(med(ch, 'B_dir', arm=a, b_steps=k)):>6s}" for k in STAGES) + " |" +
              "".join(f"{fmt(med(ch, 'B_rest', arm=a, b_steps=k)):>6s}" for k in STAGES[3:]) + " |" +
              "".join(f"{fmt(med(ch, 'B_conf', arm=a, b_steps=k)):>6s}" for k in STAGES[3:]))
    print(f"\n== Мир A после смены (что осталось): направление A, итог A, %")
    for a in arms_c:
        print(f"{a:24s}" + "".join(f"{fmt(med(ch, 'A_dir', arm=a, b_steps=k)):>6s}" for k in STAGES) + " |" + "".join(f"{fmt(med(ch, 'A', arm=a, b_steps=k)):>6s}" for k in STAGES))
    if back_rows:
        print("\n== Закон вернулся (мир A снова): направление в мире A после возврата, %  (0 = перед возвратом)")
        print(f"{'участник':24s}{'0':>6s}" + "".join(f"{k:>6d}" for k in BACK))
        for a in arms_c:
            print(f"{a:24s}{fmt(med(ch, 'A_dir', arm=a, b_steps=3000)):>6s}" +
                  "".join(f"{fmt(med([x for x in back_rows if x.get('back_steps') == k], 'A_dir', arm=a)):>6s}" for k in BACK))
    print(f"\n== Стабильный мир ({stable_name}); потоков: {len({r['stream'] for r in st})}; итог / уверенность в изменениях, %")
    print(f"{'участник':24s}" + "".join(f"{T:>12d}" for T in BUDGETS))
    for a in arms_s:
        print(f"{a:24s}" + "".join(f"{fmt(med(st, 'macro', arm=a, T=T), 1):>7s}/{fmt(med(st, 'conf', arm=a, T=T)):<4s}" for T in BUDGETS))
    v = {}
    if "кора" in arms_c and "дневник" in arms_c:
        def speed(a):
            d1, c1 = med(ch, "B_dir", arm=a, b_steps=1000), med(ch, "B_dir", arm="кора", b_steps=1000)
            return d1 is not None and d1 >= 0.55 and d1 >= c1 + 0.20
        def stable_ok(a):
            if a not in arms_s or "кора" not in arms_s:
                return None
            return all(med(st, "macro", arm=a, T=T) >= med(st, "macro", arm="кора", T=T) - 0.01 for T in (3000, 10000, 30000)) and \
                   all(med(st, "conf", arm=a, T=T) >= med(st, "conf", arm="кора", T=T) - 0.02 for T in (3000, 10000, 30000))
        print("\n== Вердикты")
        for a in [x for x in arms_c if x.startswith("дневник")]:
            d1, d3 = med(ch, "B_dir", arm=a, b_steps=1000), med(ch, "B_dir", arm=a, b_steps=3000)
            r3, rc3 = med(ch, "B_rest", arm=a, b_steps=3000), med(ch, "B_rest", arm="кора", b_steps=3000)
            h6a, h6b = speed(a), d3 is not None and d3 >= 0.65
            h7 = r3 is not None and r3 >= rc3 - 0.01
            h8 = stable_ok(a)
            print(f"{a}: H6a (скорость: ≥55% и ≥ кора+20 п.п. на 1000) {'да' if h6a else 'нет'}; H6b (≥65% на 3000) {'да' if h6b else 'нет'}; "
                  f"H7 (остальное не хуже коры −1 п.п.) {'да' if h7 else 'нет'}; H8 (стабильный мир не хуже −1 п.п. и уверенность −2 п.п.) "
                  f"{'да' if h8 else ('нет' if h8 is False else '—')}")
            back = med([x for x in back_rows if x.get("back_steps") == 1000], "A_dir", arm=a)
            back_c = med([x for x in back_rows if x.get("back_steps") == 1000], "A_dir", arm="кора")
            h10 = None if (back is None or back_c is None) else back >= back_c - 0.05
            print(f"   H10 (закон вернулся: направление в мире A за 1000 тактов возврата не ниже коры − 5 п.п.): {fmt(back)}% против {fmt(back_c)}% → "
                  f"{'да' if h10 else ('нет' if h10 is False else '—')}" + ("  (закрепление в H10 не оценивается)" if "закрепление" in a else ""))
            v[a] = {"H6a": h6a, "H6b": h6b, "H7": h7, "H8": h8, "H10": h10}
        for a in [x for x in arms_c if x.startswith("забывание")]:
            sp, so = speed(a), stable_ok(a)
            print(f"{a}: быстро (как H6a) {'да' if sp else 'нет'}; стабильно (как H8) {'да' if so else ('нет' if so is False else '—')}")
            v[a] = {"speed": sp, "stable": so}
        fg = [a for a in v if a.startswith("забывание")]
        if fg:
            both = [a for a in fg if v[a]["speed"] and v[a]["stable"]]
            print(f"H9 (дневник нужен: ни одно забывание не бывает и быстрым, и стабильным): {'ВЫПОЛНЕНА' if not both else 'НЕ ВЫПОЛНЕНА — подходит ' + ', '.join(both)}")
    return v


def markdown(change_name: str, stable_name: str) -> str:
    """Таблицы для документа (русские запятые), без вердиктов."""
    ch_all, st = load(change_name), load(stable_name)
    ch = [r for r in ch_all if "back_steps" not in r]
    back_rows = [r for r in ch_all if "back_steps" in r]
    arms = list(dict.fromkeys(r["arm"] for r in ch_all))
    z = lambda x: fmt(x).replace(".", ",")
    out = ["| участник | " + " | ".join(f"B+{k}" for k in STAGES) + " | остальное B+3000 | возврат +100 / +300 / +1000 |", "|---|" + "---|" * (len(STAGES) + 2)]
    for a in arms:
        back = " / ".join(z(med([x for x in back_rows if x.get("back_steps") == k], "A_dir", arm=a)) for k in BACK)
        out.append(f"| {a} | " + " | ".join(z(med(ch, "B_dir", arm=a, b_steps=k)) for k in STAGES) + f" | {z(med(ch, 'B_rest', arm=a, b_steps=3000))} | {back} |")
    out.append("")
    arms_s = list(dict.fromkeys(r["arm"] for r in st))
    out += ["| участник | " + " | ".join(f"{T:,}".replace(",", " ") for T in BUDGETS) + " |", "|---|" + "---|" * len(BUDGETS)]
    for a in arms_s:
        out.append(f"| {a} | " + " | ".join(f"{fmt(med(st, 'macro', arm=a, T=T), 1).replace('.', ',')} / {z(med(st, 'conf', arm=a, T=T))}" for T in BUDGETS) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--change", default="diary_change_dev")
    ap.add_argument("--stable", default="diary_stable_dev")
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args()
    if a.markdown:
        print(markdown(a.change, a.stable))
    else:
        summarize(a.change, a.stable)


if __name__ == "__main__":
    main()
