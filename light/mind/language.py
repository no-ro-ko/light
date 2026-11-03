"""Освоение языка «как ребёнок»: без словаря, без разметки, без градиентного спуска.

Система хранит услышанные фразы вместе с тем, что было видно в этот момент
(эпизодическая память), и периодически («во сне») заново их анализирует:

1. Смысл через совпадения (перекрёстно-ситуативное обучение). Слово привязывается
   к тем категориям, которые почти всегда были на виду, когда оно звучало, и редко —
   в остальное время. Для родовых слов («животное») ищется НАБОР категорий, который
   объясняет все случаи употребления (жадное покрытие).
2. Слова-события: «лает» звучит именно тогда, когда слышно «гав».
3. Служебные слова («смотри», «вот», «это») система находит сама: они ни к чему не
   привязаны и стоят перед словами со смыслом.
4. Существительные — слова после служебных или в начале фразы; сказуемые — слова
   после существительного (или после «не»).
5. Речь «не о том, что видно»: если во фразе есть слово «собака», а собаки в сцене
   нет, фраза — про отсутствующее, и для привязки слов к виденному не используется.
6. Взаимоисключение (как у детей): новое слово скорее относится к тому, у чего
   ещё нет названия.

Врождённое допущение одно: частица «не» означает отрицание.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

NEG = "не"
START = "<s>"


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\w+", text.lower()))


@dataclass(frozen=True)
class Utterance:
    tokens: tuple[str, ...]
    cats: frozenset            # категории, бывшие на виду
    events: frozenset          # события в сцене
    focus: int | None = None   # категория, на которую показали (совместное внимание)
    focus_events: frozenset = frozenset()

    def evidence(self) -> frozenset:
        """Чем могло быть то, о чём речь: показанное — точно, иначе любое из видимого."""
        return frozenset([self.focus]) if self.focus is not None else self.cats


class Lexicon:
    def __init__(self, min_lift: float = 1.6):
        self.utts: list[Utterance] = []
        self.min_lift = min_lift
        self._analyzed_at = -1
        # результаты анализа
        self.ext: dict[str, list[int]] = {}
        self.coverage: dict[str, float] = {}
        self.event_of: dict[str, str] = {}
        self.function: set[str] = set()
        self.nouns: set[str] = set()
        self.predicates: set[str] = set()
        self.name_of: dict[int, str] = {}
        self.displaced: set[int] = set()

    def remap(self, mapping: dict[int, int]) -> None:
        """Категории слились во сне — переписать память о сценах."""
        if not mapping:
            return
        m = lambda c: mapping.get(c, c)  # noqa: E731
        self.utts = [Utterance(u.tokens, frozenset(m(c) for c in u.cats), u.events,
                               None if u.focus is None else m(u.focus), u.focus_events)
                     for u in self.utts]
        self._analyzed_at = -1

    def hear(self, text: str, cats, events, focus: int | None = None,
             focus_events=frozenset()) -> Utterance:
        u = Utterance(tokenize(text), frozenset(cats), frozenset(events), focus, frozenset(focus_events))
        self.utts.append(u)
        return u

    # ---------- статистика ----------
    def _base(self, idx: list[int]) -> tuple[Counter, Counter, Counter, int, int]:
        """Базовые частоты: как часто категория просто бывает на виду / бывает показана."""
        present, shown, ev = Counter(), Counter(), Counter()
        n_amb = n_foc = 0
        for i in idx:
            u = self.utts[i]
            ev.update(u.events)
            if u.focus is None:
                present.update(u.cats)
                n_amb += 1
            else:
                shown[u.focus] += 1
                n_foc += 1
        return present, shown, ev, max(1, n_amb), max(1, n_foc)

    def _extension(self, uses: list[int], base, named: set[int]) -> tuple[list[int], float]:
        """Жадно ищем набор категорий, объясняющий употребления слова.

        Категория добавляется, только если она встречается с этим словом заметно чаще
        случайного (на 3 стандартных отклонения выше ожидаемого)."""
        present, shown, _, n_amb, n_foc = base
        items = [(self.utts[i].evidence(), self.utts[i].focus is not None)
                 for i in uses if self.utts[i].evidence()]
        if not items:
            return [], 0.0
        covered = [False] * len(items)
        ext: list[int] = []
        fresh_word = len(items) <= 3
        while True:
            open_ = [it for it, c in zip(items, covered) if not c]
            if not open_:
                break
            cand = Counter(c for s, _ in open_ for c in s if c not in ext)
            best, best_score = None, 0.0
            for c, cnt in cand.items():
                exp = var = 0.0
                for s, foc in open_:
                    p = (shown[c] / n_foc) if foc else (present[c] / n_amb)
                    p = min(max(p, 1e-3), 0.999)
                    exp += p
                    var += p * (1 - p)
                if fresh_word:      # быстрое запоминание: 1–3 употребления, но однозначно
                    ok = cnt == len(open_) and (len(s) == 1 for s, _ in open_)
                    ok = ok and all(c in s for s, _ in open_) and cnt >= exp * 2
                else:
                    ok = cnt >= 3 and cnt >= exp + 3 * var ** 0.5
                score = (cnt - exp) * (0.3 if fresh_word and c in named else 1.0)
                if ok and score > best_score:
                    best, best_score = c, score
            if best is None:
                break
            before = sum(covered)
            covered = [cv or best in s for (s, _), cv in zip(items, covered)]
            if ext and (sum(covered) - before) < 0.08 * len(items):
                break                     # добавка объясняет слишком мало — это шум
            ext.append(best)
        return ext, sum(covered) / len(items)

    def _event(self, uses: list[int], base) -> tuple[str | None, float, float]:
        be = base[2]
        n_all = base[3] + base[4]
        if not uses:
            return None, 0.0, 0.0
        # если показали — смотрим на события показанного объекта, иначе — всей сцены
        seen = lambda u: u.focus_events if u.focus is not None else u.events  # noqa: E731
        cnt = Counter(e for i in uses for e in seen(self.utts[i]))
        best = max(cnt, key=lambda e: cnt[e] / max(be[e], 1), default=None)
        if best is None:
            return None, 0.0, 0.0
        share = cnt[best] / len(uses)
        return best, share, share / (be[best] / n_all)

    # ---------- анализ («во сне») ----------
    def analyze(self, force: bool = False) -> None:
        if not force and self._analyzed_at == len(self.utts):
            return
        self._analyzed_at = len(self.utts)
        all_idx = list(range(len(self.utts)))
        uses: dict[str, list[int]] = defaultdict(list)
        nxt: dict[str, Counter] = defaultdict(Counter)
        prv: dict[str, Counter] = defaultdict(Counter)
        for i, u in enumerate(self.utts):
            for j, w in enumerate(u.tokens):
                if not uses[w] or uses[w][-1] != i:
                    uses[w].append(i)
                prv[w][u.tokens[j - 1] if j else START] += 1
                if j + 1 < len(u.tokens):
                    nxt[w][u.tokens[j + 1]] += 1

        # проход 1: привязка по всем употреблениям
        base = self._base(all_idx)
        ext1, cov1, grounded = {}, {}, set()
        for w, idx in uses.items():
            if w == NEG:
                continue
            ext1[w], cov1[w] = self._extension(idx, base, set())
            e, share, lift = self._event(idx, base)
            basic = ext1[w] and len(ext1[w]) <= 2 and cov1[w] >= 0.5
            if basic or (e and share >= 0.5 and lift >= 3):
                grounded.add(w)

        # служебные слова: часты, сами ни к чему не привязаны, никогда не заканчивают
        # фразу и стоят перед самыми разными словами («смотри …», «вот …», «это …»)
        self.function = set()
        for w, idx in uses.items():
            if w == NEG or w in grounded or len(idx) < 5:
                continue
            total = sum(prv[w].values())
            final = total - sum(nxt[w].values())          # сколько раз слово было последним
            if final <= 0.1 * total and len(nxt[w]) >= 4 \
                    and max(nxt[w].values()) <= 0.4 * sum(nxt[w].values()):
                self.function.add(w)
        # существительные: после служебного слова или в начале фразы
        self.nouns = {w for w in uses if w not in self.function and w != NEG
                      and sum(c for t, c in prv[w].items() if t in self.function or t == START)
                      >= 0.5 * sum(prv[w].values())}
        # сказуемые: после существительного или после «не»
        self.predicates = {w for w in uses if w not in self.function | self.nouns and w != NEG
                           and sum(c for t, c in prv[w].items() if t in self.nouns or t == NEG)
                           >= 0.5 * sum(prv[w].values())}

        # речь о невидимом: базовое существительное есть во фразе, а его категорий нет в сцене
        basic1 = {w: set(ext1[w]) for w in self.nouns
                  if ext1.get(w) and len(ext1[w]) <= 2 and cov1.get(w, 0) >= 0.5}
        self.displaced = {i for i, u in enumerate(self.utts)
                          if any(w in basic1 and not (basic1[w] & u.cats) for w in u.tokens)}
        here = [i for i in all_idx if i not in self.displaced]

        # проход 2: привязка только по речи о видимом
        base = self._base(here)
        self.ext, self.coverage, self.event_of = {}, {}, {}
        named: set[int] = set()
        order = sorted(self.nouns, key=lambda w: -len(uses[w]))
        for w in order:
            idx = [i for i in uses[w] if i not in self.displaced]
            self.ext[w], self.coverage[w] = self._extension(idx, base, named)
            if self.ext[w] and len(self.ext[w]) == 1:
                named.update(self.ext[w])
        # слова-события: только по речи «здесь и сейчас» — подлежащее на виду
        basic = {w: set(e) for w, e in self.ext.items()
                 if e and len(e) <= 2 and self.coverage.get(w, 0) >= 0.6}
        # как часто событие бывает просто при этой категории (без всяких слов)
        with_c, with_ce = Counter(), Counter()
        for i in here:
            u = self.utts[i]
            if u.focus is not None:
                with_c[u.focus] += 1
                for e in u.focus_events:
                    with_ce[(u.focus, e)] += 1
        for w in self.predicates:
            idx = [i for i in uses[w] if i not in self.displaced
                   and any(t in basic and basic[t] & self.utts[i].cats for t in self.utts[i].tokens)]
            e, share, lift = self._event(idx, base)
            if len(idx) < 5 or not (e and share >= 0.6 and lift >= 3):
                continue
            # слово должно сопровождать событие чаще, чем событие просто бывает у предмета:
            # «человек дышит» звучит при говорящих людях, но «дышит» — не «говорит»
            subj = [c for i in idx for t in self.utts[i].tokens if t in basic
                    for c in basic[t] & self.utts[i].cats]
            usual = [with_ce[(c, e)] / with_c[c] for c in subj if with_c[c] >= 5]
            if usual and share < sum(usual) / len(usual) + 0.25 and share < 0.9:
                continue
            self.event_of[w] = e

        # базовое имя категории: самое узкое слово, которое её покрывает
        self.name_of = {}
        for w, ext in self.ext.items():
            if not ext or self.coverage[w] < 0.6:
                continue
            for c in ext:
                cur = self.name_of.get(c)
                if cur is None or (len(ext), -self.coverage[w]) < (len(self.ext[cur]), -self.coverage[cur]):
                    self.name_of[c] = w

    # ---------- разбор фраз в факты ----------
    def parse(self, tokens: tuple[str, ...]) -> tuple[str, str, str, bool] | None:
        """→ ("isa"|"prop", подлежащее, сказуемое/род, истинно ли) или None."""
        core = [t for t in tokens if t not in self.function]
        if len(core) < 2 or core[0] not in self.nouns:
            return None
        x, rest = core[0], core[1:]
        pol = True
        if rest and rest[0] == NEG:
            pol, rest = False, rest[1:]
        if len(rest) != 1:
            return None
        y = rest[0]
        if y in self.nouns:
            return ("isa", x, y, pol) if y != x else None
        if y in self.predicates:
            return ("prop", x, y, pol)
        return None

    def basic_nouns(self) -> set[str]:
        """Имена конкретных видов: слово покрывает одну-две категории."""
        return {w for w, e in self.ext.items() if e and len(e) <= 2 and self.coverage[w] >= 0.6}
