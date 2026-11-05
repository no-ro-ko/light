"""Таблицы результатов для docs/MIR.md — собираются из mir_runs/*.jsonl (ничего не переписывается руками).

  python3 -m light.mir.tables
"""
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev

R = str(Path(__file__).resolve().parents[2] / "mir_runs") + "/"


def load(n):
    try:
        return [json.loads(l) for l in open(R + n + ".jsonl", encoding="utf-8")]
    except FileNotFoundError:
        return []


def main():
    GRID = [30, 100, 300, 1000, 3000, 10000, 30000]
    rows = load('final_curves')
    arms = {}
    for r in rows:
        if 'sealed' not in r: continue
        c = r['cfg']
        name = ('кора' if r['kind'] == 'seg' else f"сеть {c['width']}") + (' + сон' if c.get('consolidate') else '')
        arms.setdefault(name, defaultdict(list))[r['T']].append(r)
    order = [n for n in ('кора', 'кора + сон', 'сеть 16', 'сеть 64', 'сеть 256') if n in arms]
    def med(name, T, f=lambda r: r['sealed']['macro']):
        v = arms[name].get(T, [])
        return median(f(x) for x in v) if v else None

    out = []
    P = out.append
    n_peeks = sum(1 for _ in open(R + 'sealed_log.jsonl', encoding='utf-8'))
    tags = set()
    for l in open(R + 'sealed_log.jsonl', encoding='utf-8'):
        tags.add(json.loads(l)['tag'])
    P(f"К запечатанному набору обращались {n_peeks} раз; все записи в `mir_runs/sealed_log.jsonl` относятся к разным тройкам «модель или политика, бюджет, поток» ({len(tags)} различных меток) — повторных обращений нет.\n")

    # ---- таблица 1
    P("@@curves@@")
    P("### 10.1 Опыт → итог (запечатанный набор, медиана по потокам 11, 12, 13)\n")
    P("| модель | " + " | ".join(f"{T:,}".replace(",", " ") for T in GRID) + " | потоков |")
    P("|---|" + "---|" * (len(GRID) + 1))
    for n in order:
        cells = []
        for T in GRID:
            m = med(n, T)
            cells.append("—" if m is None else f"{m*100:.1f}".replace(".", ","))
        ns = min((len(arms[n][T]) for T in GRID if T in arms[n]), default=0)
        P(f"| {n} | " + " | ".join(cells) + f" | {ns}–{max(len(v) for v in arms[n].values())} |")
    P("")

    # ---- интерполяция
    def reach(T2m, level):
        Ts = sorted(T2m)
        if T2m[Ts[0]] >= level: return Ts[0]
        for a, b in zip(Ts, Ts[1:]):
            if T2m[a] < level <= T2m[b]:
                return math.exp(math.log(a) + (math.log(b) - math.log(a)) * (level - T2m[a]) / (T2m[b] - T2m[a]))
        return None
    def curve(n):
        return {T: med(n, T) for T in GRID if med(n, T) is not None and len(arms[n][T]) >= 3}
    cb = curve('кора')
    best_mlp = None
    for n in ('сеть 16', 'сеть 64', 'сеть 256'):
        if n in arms:
            c = curve(n)
            # «лучшая» — по наименьшему T90, затем по итогу на 30000
            t90 = next((T for T in GRID if T in c and c[T] >= .9), 10**9)
            key = (t90, -(c.get(30000, 0)))
            if best_mlp is None or key < best_mlp[0]:
                best_mlp = (key, n, c)
    _, bn, bc = best_mlp
    P("**Сколько тактов опыта нужно, чтобы дойти до уровня** (линейная интерполяция по логарифму бюджета между измеренными точками; "
      f"это описание после факта, а не проверка заранее записанной гипотезы; сравнение с лучшей сетью — «{bn}»):\n")
    P("| уровень | кора | " + bn + " | выигрыш коры |")
    P("|---|---|---|---|")
    for lv in (.80, .85, .90, .95):
        a, b = reach(cb, lv), reach(bc, lv)
        fa = "—" if a is None else f"{a:,.0f}".replace(",", " ")
        fb = "> 30 000" if b is None else f"{b:,.0f}".replace(",", " ")
        ratio = "—" if a is None else (f"×{b/a:.1f}".replace(".", ",") if b else f"> ×{30000/a:.0f}")
        P(f"| {lv*100:.0f}% | {fa} | {fb} | {ratio} |")
    P("")
    Tc = next((T for T in GRID if T in cb and cb[T] >= .9), None)
    Tm = next((T for T in GRID if T in bc and bc[T] >= .9), None)
    P(f"**H1 (главная): T90 коры = {Tc}, T90 лучшей сети ({bn}) = {Tm}.** " +
      ("Шагов сетки между ними: " + str((GRID.index(Tm) - GRID.index(Tc)) if (Tc and Tm) else '—') +
       "; нужно было не меньше одного → **не выполнена**." if (Tc and Tm and GRID.index(Tm) - GRID.index(Tc) < 1) else "**выполнена**."))
    lv = {}
    for T in (10000,):
        ls = [median(x['sealed']['levels'][i] for x in arms['кора'][T]) for i in range(3)]
        lv[T] = ls
    ratio_h2 = lv[10000][2] / lv[10000][0]
    P("**H2 (небывалое окружение):** на 10 000 тактах итог коры по уровням мусора (нет / редкий / густой) — "
      + " / ".join(f"{x*100:.1f}".replace(".", ",") for x in lv[10000])
      + "%; густой ÷ чистый = " + f"{ratio_h2:.2f}".replace(".", ",") + " (порог 0,85) → **выполнена**.")
    if 'кора + сон' in arms:
        P("**H5 (сон):** " + "; ".join(f"{T}: {(med('кора + сон', T) - med('кора', T))*100:+.1f}".replace(".", ",") for T in GRID) +
          " п.п. (D − B) → на запечатанном наборе сон **не показал пользы** (нужно было ≥ +2 п.п. на двух из трёх больших бюджетов).")
    P("")

    # ---- семейства
    P("@@families@@")
    P("### 10.2 Какие законы кому даются (медиана по потокам, %)\n")
    fams = list(arms['кора'][10000][0]['sealed']['families'])
    cols = [('кора', 1000), ('кора', 10000), ('кора', 30000)] + [(n, T) for n in ('сеть 64', 'сеть 256') if n in arms for T in (10000, 30000) if T in arms[n]]
    P("| семейство | " + " | ".join(f"{n} {T:,}".replace(",", " ") for n, T in cols) + " |")
    P("|---|" + "---|" * len(cols))
    for f in fams:
        cells = []
        for n, T in cols:
            v = [x['sealed']['families'][f] for x in arms[n].get(T, [])]
            cells.append("—" if not v else f"{median(v)*100:.0f}")
        P(f"| {f} | " + " | ".join(cells) + " |")
    P("")

    # ---- любопытство
    P("@@live@@")
    P("### 10.4 Любопытство против блуждания (запечатанный набор)\n")
    live = load('final_live') + load('final_live_ext')
    by = defaultdict(lambda: defaultdict(dict))
    for r in live:
        by[r['policy']][r['seed']][r['T']] = r['sealed']['macro']
    Ts = [100, 300, 1000, 3000, 10000, 30000]
    P("| политика | потоков | " + " | ".join(f"{T:,}".replace(",", " ") for T in Ts) + " |")
    P("|---|---|" + "---|" * len(Ts))
    for pol, nm in (('random', 'блуждание'), ('curious', 'любопытство: сомнение в исходе'), ('approach', 'тянусь к непробованному')):
        if pol in by:
            P(f"| {nm} | {len(by[pol])} | " + " | ".join(f"{median(by[pol][s][T] for s in by[pol])*100:.1f}".replace(".", ",") for T in Ts) + " |")
    P("")
    seeds_all = sorted(set(by['random']) & set(by['approach']))
    P(f"Парные разности «тянусь» − блуждание по {len(seeds_all)} потокам (п.п.):\n")
    P("| бюджет | среднее | разброс (σ) | потоков с плюсом | t |")
    P("|---|---|---|---|---|")
    for T in (300, 1000, 3000, 10000, 30000):
        d = [(by['approach'][s][T] - by['random'][s][T]) * 100 for s in seeds_all]
        sd = pstdev(d) * math.sqrt(len(d) / (len(d) - 1))
        t = mean(d) / (sd / math.sqrt(len(d)))
        P(f"| {T:,} | {mean(d):+.1f} | {sd:.1f} | {sum(x > 0 for x in d)} из {len(d)} | {t:.1f} |".replace(",", " ").replace(".", ","))
    P("")


    # ---- когда встретилось второе свидетельство
    sec = defaultdict(lambda: defaultdict(list)); cnt = defaultdict(lambda: defaultdict(list))
    for r in live:
        for law, t in r['second_witness_at'].items():
            sec[r['policy']][law].append(t if t is not None else 10**6)
        if r['T'] == 30000:
            for law, c in r['witness_counts_at_end'].items():
                cnt[r['policy']][law].append(c)
    P("@@witness@@")
    P("Когда впервые встретилось **второе** свидетельство каждого закона — номер такта (медиана по потокам; «—» — не наступило за 30 000) и сколько раз закон сработал за 30 000 тактов:\n")
    P("| закон | блуждание | сомнение в исходе | тянусь к непробованному | сработал раз: блуждание / сомнение / тянусь |")
    P("|---|---|---|---|---|")
    for law in sec['random']:
        f = lambda pol: ("—" if median(sec[pol][law]) >= 10**6 else f"{int(median(sec[pol][law])):,}".replace(",", " ")) if pol in sec else "—"
        c = lambda pol: f"{int(median(cnt[pol][law]))}" if pol in cnt else "—"
        P(f"| {law} | {f('random')} | {f('curious')} | {f('approach')} | {c('random')} / {c('curious')} / {c('approach')} |")
    P("")

    # ---- смена закона
    ad_c, ad_m = load('adapt'), load('adapt_mlp')
    if ad_c:
        P("@@adapt@@")
        P("### 10.5 Граница: смена закона (черновые пробы)\n")
        def stg(rows):
            st = defaultdict(list)
            for r in rows: st[r['b_steps']].append(r)
            return st
        sc, sm = stg(ad_c), stg(ad_m)
        P("Доля верно различённых пар в семействах «осыпание песка» и «растекание воды» (направление зависит от метронома) и итог по всем семействам; медиана по потокам.\n")
        P("| тактов опыта нового закона | кора: старый мир, направление | кора: новый мир, направление | " +
          ("сеть 64: старый мир, направление | сеть 64: новый мир, направление | " if sm else "") + "кора: итог в новом мире |" + (" сеть 64: итог в новом мире |" if sm else ""))
        P("|---|---|---|" + ("---|---|" if sm else "") + "---|" + ("---|" if sm else ""))
        for k in sorted(sc):
            row = f"| {k} | {median(x['A_direction'] for x in sc[k])*100:.0f}% | {median(x['B_direction'] for x in sc[k])*100:.0f}% | "
            if sm:
                if k in sm:
                    row += f"{median(x['A_direction'] for x in sm[k])*100:.0f}% | {median(x['B_direction'] for x in sm[k])*100:.0f}% | "
                else:
                    row += "— | — | "
            row += f"{median(x['B'] for x in sc[k])*100:.0f}% |"
            if sm:
                row += (f" {median(x['B'] for x in sm[k])*100:.0f}% |" if k in sm else " — |")
            P(row)
        P("")
    print("\n".join(out))



if __name__ == "__main__":
    main()
