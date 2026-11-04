"""Повествование: события рассказа из обычных фраз.

Рассказ «Росток» понимает как цепочку событий «кто — что делает — кого/что». Простые
истории («Лиса хвалит ворону. Ворона радуется.») он читал и раньше; здесь — обычная речь:

- «Хитрая лиса подошла к ели и стала хвалить ворону» → лиса подходит; лиса хвалит ворону.
  Признак перед названием — описание героя; «к ели» — место, а не то, над чем действуют;
  из двух действий подряд («стала хвалить», «крепко держала») главное — последнее;
- «Мимо бежала голодная лиса» → лиса бежит: «кто» стоит после действия; узнаётся по
  форме слова — такие окончания были у тех, кто действует, в уже понятых фразах;
- «Она увидела сыр и сразу захотела его» → «она» — тот, кто действовал последним;
  «его» — то, над чем действовали; второе действие — того же героя;
- «Глупая птица громко запела» → это ворона: в рассказе уже есть ворона, а ворона — птица;
- «Она говорила, что …» и прямая речь в «…» — это сказанное, а не событие рассказа.

Где в фразе действие, система узнаёт сама. Во фразах, где «кто» — уже известное название
(или «он/она»), следующее слово — действие; так копится, какие окончания бывают у действий
(«-ла», «-ет», «-ит», …), у тех, кто действует, и у тех, над кем действуют, — и по ним
узнаются действия и роли в остальных фразах. Признаки отличаются от действий тоже по
собственному опыту: у слов, которые стояли признаком перед названием («крупные морские
млекопитающие», «синий кит»), свои окончания («-ые», «-ая», «-ий», …).

Врождённое (честно): он/она/оно/они, его/её/их, предлоги, союзы, «не»; «хочет» — желание.

Рассказ — абзац, где герои меняются ролями: то, над чем действовали, потом действует само
(«держала сыр» → «сыр выпал»). Описание («Киты дышат воздухом…») рассказом не считается.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .syntax import PREPS, SOFT, lex

SUBJ = {"он", "она", "оно", "они"}
OBJ = {"его", "ее", "их"}
KIN = {"он": {"он"}, "она": {"она"}, "оно": {"оно"}, "они": {"они"},      # «его» — он или оно
       "его": {"он", "оно"}, "ее": {"она"}, "их": {"они"}}
LINK = {"и", "а", "но", "да", "или", "зато"}
SEP = {",", ";", "—", "(", ")"}
SAID = {"что", "чтобы", "будто", "который", "которая", "которое", "которые", "которого", "которую",
        "которой", "если", "когда", "потому", "пока", "где", "куда", "как", "ведь"}
VOWELS = set("аеиоуыэюя")


def tails(v: str) -> list[str]:
    """Окончание действия без «-ся/-сь» (по нему видно, о ком речь: «она села», «он сел»)."""
    if v.endswith(("ся", "сь")) and len(v) > 4:
        v = v[:-2]
    return [v[-2:], v[-1:]] if len(v) > 3 else []


def case_key(w: str) -> str:
    """Окончание для «кто / кого»: последняя гласная (или ь/й); у согласной — «#» (нулевое)."""
    return w[-1] if w[-1] in VOWELS or w[-1] in "ьй" else "#"


@dataclass
class Event:
    agent: str                      # понятие (после «кто есть кто»)
    action: str                     # лемма действия
    obj: str | None
    sent: int                       # номер предложения в памяти
    forms: tuple                    # как было написано: (кто, действие, кого)


@dataclass
class Scene:
    """Кто есть кто в рассказе."""
    seen: list[str] = field(default_factory=list)       # кто упоминался — по свежести
    agent: str | None = None                             # кто действовал последним
    core: list[str] = field(default_factory=list)        # участники событий (кто и над кем)

    def mention(self, key: str) -> None:
        if key in self.seen:
            self.seen.remove(key)
        self.seen.append(key)


class Narrator:
    def __init__(self):
        self.verbs: dict[str, set] = defaultdict(set)      # окончание → формы на месте «что делает»
        self.agents: dict[str, set] = defaultdict(set)     # окончание → формы на месте «кто»
        self.patients: dict[str, set] = defaultdict(set)   # окончание → формы на месте «кого/что»
        # род — тоже из опыта: «она села» → «-ла» бывает при «она»; «лиса бежала» → лиса — «она»
        self.pron_tails: dict[str, dict] = defaultdict(lambda: defaultdict(set))   # он/она → окончание → формы
        self.noun_tails: dict[str, dict] = defaultdict(lambda: defaultdict(set))   # лемма → окончание → формы
        self.heroes: set[str] = set()                      # леммы: герои простых историй
        self.after_prep: set[str] = set()                  # формы, стоявшие после предлога («на стол»)
        self.verb_last: dict[str, set] = defaultdict(set)  # последняя буква действий
        self._attr: tuple | None = None
        self._acts: set[str] = set()
        self._place: set[str] = set()
        self._marks: set[str] = set()

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_attr"], d["_acts"], d["_place"], d["_marks"] = None, set(), set(), set()
        d["pron_tails"] = {k: dict(v) for k, v in self.pron_tails.items()}
        d["noun_tails"] = {k: dict(v) for k, v in self.noun_tails.items()}
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        for key, v in (("heroes", set()), ("after_prep", set()), ("_place", set()), ("_marks", set())):
            self.__dict__.setdefault(key, v)
        self.__dict__["verb_last"] = defaultdict(set, d.get("verb_last", {}))
        for key in ("pron_tails", "noun_tails"):
            old = d.get(key, {})
            fresh = defaultdict(lambda: defaultdict(set))
            for k, v in old.items():
                fresh[k].update(v)
            self.__dict__[key] = fresh

    # ---------- род: кто «он», кто «она» ----------
    def kind_of(self, key: str) -> str | None:
        """«она» / «он» / … — по окончаниям действий героя, которые бывают только при одном местоимении;
        если о герое этого не видно — по последней букве его имени, как у знакомых героев."""
        k = self._kind_by_tails(key)
        if k is None and getattr(self, "_names", None):
            c = self._letters.get(self._names(key)[-1:], Counter())
            if c and sum(c.values()) >= 2 and c.most_common(1)[0][1] >= 0.8 * sum(c.values()):
                k = c.most_common(1)[0][0]
        return k

    def _kind_by_tails(self, key: str) -> str | None:
        votes = Counter()
        for t, forms in self.noun_tails.get(key, {}).items():
            owners = [p for p, tt in self.pron_tails.items() if tt.get(t)]
            if len(owners) == 1:                         # окончание говорит о роде («-ла»), а не «-ет»
                votes[owners[0]] += len(forms)
        return votes.most_common(1)[0][0] if votes else None

    def fits(self, key: str | None, pron: str) -> int:
        """2 — подходит по роду, 1 — неизвестно, 0 — не подходит («её» — не про яблоко, если оно «выпало»)."""
        if key is None:
            return 0
        k = self.kind_of(key)
        if k is None:
            return 1
        return 2 if k in KIN.get(pron, {pron}) else 0

    def resolve(self, pron: str, candidates: list[str]) -> str | None:
        best = None
        for c in candidates:
            f = self.fits(c, pron)
            if f == 2:
                return c
            if f == 1 and best is None:
                best = c
        return best

    def _prepare(self, org) -> None:
        self._acts = org.story_actions()
        self._place = {org.lang.lemma(w) for w in self.after_prep}   # «стол»: стояло после «на»
        self._attr = self._attributes(org)
        # род по последней букве — обобщение: у кого из знакомых героев «-а» в конце, те чаще «она»
        self._letters: dict[str, Counter] = defaultdict(Counter)
        for key in self.noun_tails:
            k = self._kind_by_tails(key)
            if k:
                self._letters[org.name(key)[-1:]][k] += 1
        self._names = org.name
        self._marks = self._marked_tails(org)
        # окончания знакомых названий («рыбой», «птицей», «молоком») — чтобы узнать название по форме
        self._noun_ends: dict[str, set] = defaultdict(set)
        for f in org.lang.forms:
            if len(f) > 3 and not org.lang.is_function(f) and self.known_noun(org, f):
                self._noun_ends[f[-2:]].add(f)

    # ---------- что знает о словах ----------
    def known_noun(self, org, w: str) -> bool:
        """Название — если о нём что-то сказано, если это герой простой истории или если слово
        (в какой-то форме) стояло после предлога: «на стол», «по полу», «с молоком»."""
        L = org.lang.lemma(w)
        if L in org.know.predicates and L not in org.know.concepts:
            return False
        return L in org.know.concepts or L in self.heroes or L in self._place \
            or org.lang.hints.get(w) == "noun"

    def _attributes(self, org):
        """Признаки — по собственному опыту: слова, стоявшие перед названием («белые медведи»),
        их окончания, а потом и слова той же семьи форм («сильный, сильные» — как «белый, белые»)."""
        L = org.lang
        mods = {L.lemma(m) for m in org.syntax.mods}
        nouns = {c for c in org.know.concepts if c not in org.know.predicates}
        for _ in range(2):                                 # окончание признака — у трёх разных слов
            by_end: dict[str, set] = defaultdict(set)
            for f in L.forms:
                if len(f) > 3 and L.lemma(f) in mods:
                    by_end[f[-2:]].add(L.lemma(f))
            mods |= {L.lemma(f) for f in L.forms if len(f) > 3 and len(by_end.get(f[-2:], ())) >= 3
                     and L.lemma(f) not in nouns and L.lemma(f) not in self._acts}
        forms = {f for f in L.forms if len(f) > 3 and L.lemma(f) in mods}
        stems = {f[:-2] for f in forms if len(f) >= 6}
        forms |= {f for f in L.forms if len(f) >= 6 and f[:-2] in stems and L.lemma(f) not in nouns
                  and L.lemma(f) not in self._acts}
        ends = Counter(f[-2:] for f in forms)
        return mods | {L.lemma(f) for f in forms}, ends

    def is_attr(self, org, w: str) -> bool:
        mods, ends = self._attr or self._attributes(org)
        return org.lang.lemma(w) in mods or (len(w) > 3 and ends[w[-2:]] >= 2)

    def verb_evidence(self, org, w: str) -> int:
        if org.lang.lemma(w) in self._acts:
            return 5
        if self.is_attr(org, w) or len(w) < 3:
            return 0
        best = 0
        for k in (3, 2):
            if len(w) > k + 1:
                best = max(best, len(self.verbs.get(w[-k:], ())))
        return best

    def _by_ending(self, w: str) -> str | None:
        """Незнакомое слово — по окончанию: чаще так кончаются знакомые названия или действия?"""
        if len(w) <= 3:
            return None
        base = w[:-2] if w.endswith(("ся", "сь")) and len(w) > 5 else w
        n = len(getattr(self, "_noun_ends", {}).get(w[-2:], ()))
        v = len(self.verbs.get(w[-2:], ())) + (len(self.verbs.get(base[-2:], ())) if base != w else 0)
        if n >= 2 and n > 2 * v:
            return "N?"
        if v >= 2 and v > 2 * n:
            return "V?"
        return None

    def agentive(self, w: str) -> float:
        """Насколько форма похожа на «кто» (а не «кого/что») — по понятым фразам."""
        k = case_key(w)
        a, p = len(self.agents.get(k, ())), len(self.patients.get(k, ()))
        return (a + 1) / (p + 1)

    def place_only(self, org, w: str) -> bool:
        """Слово знакомо только как место («на ель», «на стол»), а не как тот, кто действует или
        над кем действуют."""
        L = org.lang.lemma(w)
        return L in self._place and L not in self.heroes and L not in org.know.concepts

    @staticmethod
    def oblique(org, w: str) -> bool:
        """Форма не та, в которой слово обычно начинает фразу («стола» при «стол», «лису» при «лиса»)."""
        shown = org.lang.display.get(org.lang._base(w))
        return bool(shown) and shown != w

    def patient_like(self, w: str) -> bool:
        return w in OBJ or len(self.patients.get(case_key(w), ())) > 0

    # ---------- опоры: «<известное название> <действие> [<кого/что>]» ----------
    def learn(self, org, sentences: list[str]) -> None:
        self._prepare(org)
        for s in sentences:
            for seg in self._segments(self._cut(lex(s))):
                ws = [w for w in seg if w not in SOFT]
                k = next((i for i, w in enumerate(ws[:3]) if w in SUBJ or self.known_noun(org, w)), None)
                # «кто» — в начале части (перед ним — разве что признаки: «Хитрая лиса подошла»);
                # «толкнул кувшин лапой» — не опора: «кувшин» здесь — над кем действуют
                if k is None or k + 1 >= len(ws) or any(not self.is_attr(org, w) for w in ws[:k]):
                    continue
                v = ws[k + 1]
                if v in PREPS or v in OBJ or v in SUBJ or v == "не" or org.lang.is_function(v) \
                        or self.known_noun(org, v) or self.is_attr(org, v) or v[0].isdigit():
                    continue
                for n in (2, 3):
                    if len(v) > n + 1:
                        self.verbs[v[-n:]].add(v)
                self.verb_last[v[-1]].add(v)
                for t in tails(v):
                    if ws[k] in SUBJ:
                        self.pron_tails[ws[k]][t].add(v)
                    else:
                        self.noun_tails[org.lang.lemma(ws[k])][t].add(v)
                if ws[k] not in SUBJ:
                    self.agents[case_key(ws[k])].add(ws[k])
                if k + 2 < len(ws) and self.known_noun(org, ws[k + 2]):
                    self.patients[case_key(ws[k + 2])].add(ws[k + 2])
        for s in sentences:                                # после предлога — название (уже зная действия)
            toks = [w for w in self._cut(lex(s)) if w not in SOFT]
            for n, a in enumerate(toks):
                if a not in PREPS:
                    continue
                run = []
                for w in toks[n + 1:n + 3]:                # «на стол», «на высокую ель», «во рту (держала)»
                    if w in PREPS or w in SEP or w in LINK or w in OBJ or w in SUBJ or org.lang.is_function(w) \
                            or w[0].isdigit() or org.lang.lemma(w) in self._acts:
                        break
                    run.append(w)
                if len(run) == 2 and self.verb_evidence(org, run[1]) >= 2:
                    run = run[:1]                          # «во рту держала»: «держала» — действие
                if run and not (len(run) == 2 and len(toks) > n + 3 and toks[n + 3] not in SEP | LINK | PREPS):
                    self.after_prep.add(run[-1])
        self._place = {org.lang.lemma(w) for w in self.after_prep}
        self._attr = None

    # ---------- фразы → части ----------
    @staticmethod
    def _cut(toks: list[str]) -> list[str]:
        """Сказанное и придаточное («…, что …», «: …») — не события рассказа."""
        for i, t in enumerate(toks):
            if t == ":" or (t == "," and i + 1 < len(toks) and toks[i + 1] in SAID):
                return toks[:i]
        return toks

    @staticmethod
    def _segments(toks: list[str]) -> list[list[str]]:
        out, cur = [], []
        for t in toks:
            if t in SEP or t in LINK:
                if cur:
                    out.append(cur)
                cur = []
            else:
                cur.append(t)
        if cur:
            out.append(cur)
        return out

    def _classify(self, org, ws: list[str]) -> list[tuple[str, str]]:
        out = []
        for i, w in enumerate(ws):
            nxt = ws[i + 1] if i + 1 < len(ws) else ""
            if w in SUBJ:
                c = "S"
            elif w in OBJ:
                c = "O"
            elif w in PREPS:
                c = "P"
            elif w == "не":
                c = "NEG"
            elif org.lang.is_function(w) or w[0].isdigit():
                c = "F"
            elif self.known_noun(org, w):
                c = "N"
            elif self.verb_evidence(org, w) >= 2 or w in self.verbs.get(w[-2:], ()):
                c = "V"
            elif self.is_attr(org, w) and nxt and nxt not in PREPS and not org.lang.is_function(nxt):
                c = "A"
            else:
                c = self._by_ending(w)                     # «лапой» — как «рыбой»; «разлилось» — как «обрадовалась»
                if c is None:
                    wc = org.syntax.word_class(org.lang, org.know, w)
                    c = "N?" if wc == "noun" else "V?" if wc == "pred" else "U"
            out.append((w, c))
        return out

    # ---------- одна часть фразы → событие ----------
    def _clause(self, org, items, scene: Scene, inherited: str | None, i_sent: int):
        # места («к ели», «в клюве») — не участники; но кто в них — запоминается
        core, k = [], 0
        while k < len(items):
            w, c = items[k]
            if c == "P":                                   # предлог + признаки + одно название
                j = k + 1
                # признаки (и незнакомые слова перед незнакомым же названием: «на высокую ель»);
                # но «в руках яблоко»: перед знакомым названием незнакомое слово — само название места
                while j + 1 < len(items) and (items[j][1] == "A" or (
                        items[j][1] in ("U", "N", "N?") and items[j + 1][1] in ("U", "N?", "N")
                        and self.place_only(org, items[j + 1][0]))):
                    j += 1                                 # «на высокую ель», «на краю стола»: ель, стол — места
                if j < len(items) and items[j][1] in ("N", "N?", "U", "O", "S", "V?"):
                    j += 1
                heads = [x for x, cx in items[k + 1:j] if cx in ("N", "N?")]
                if heads:
                    scene.mention(self.refer(org, org.lang.lemma(heads[-1]), scene))
                k = j
                continue
            core.append((w, c))
            k += 1
        groups = self._groups(core)
        for g in groups:                                   # кто упомянут — тот на сцене
            scene.mention(self.refer(org, org.lang.lemma(core[g[1]][0]), scene))
        verbs = [n for n, (_, c) in enumerate(core) if c in ("V", "V?")]
        if not verbs:                                      # действие есть в каждой части: незнакомое
            loose = [n for n, (_, c) in enumerate(core)    # слово вне групп названий, лучше — рядом с героем
                     if c == "U" and not any(g[0] <= n <= g[2] for g in groups)]
            near = [n for n in loose if any(n + 1 == g[0] or n - 1 == g[2] for g in groups)]
            pool = near or loose                           # из нескольких — то, что кончается, как действия
            pool = sorted(pool, key=lambda n: -len(self.verb_last.get(core[n][0][-1], ())))
            verbs = pool[:1]
        strong = [n for n in verbs if core[n][1] == "V"]
        if not verbs:
            return None
        # кто: местоимение; иначе — по лестнице предпочтений (порядок слов в русском свободный)
        first_v = (strong or verbs)[0]
        subj, subj_form, obj_group = None, None, None
        before = [g for g in groups if g[0] < first_v]
        after = [g for g in groups if g[0] > first_v]
        pron = next((n for n, (w, c) in enumerate(core) if c == "S" and n < first_v), None)

        def head(g):
            return core[g[1]][0]

        def plain(g):                                      # начальная форма: «стол», а не «стола»
            return not self.oblique(org, head(g))

        def known(g):
            return core[g[1]][1] == "N"

        def first(groups_, *tests):
            return next((g for g in groups_ if all(t(g) for t in tests)), None)

        if pron is not None:
            subj_form = core[pron][0]
            if scene.agent and self.fits(scene.agent, subj_form):
                subj = scene.agent                         # «Сорока сидела… Она захотела…»
            else:                                          # сначала те, кто действовал, потом — упомянутые
                subj = self.resolve(subj_form, list(dict.fromkeys(
                    list(reversed(scene.core)) + list(reversed(scene.seen)))))
            obj_group = after[0] if after else (before[0] if before else None)
        elif inherited and not before:
            subj = inherited                               # «…и крепко держала сыр»: тот же герой
            obj_group = after[0] if after else None
        else:
            sg = (first(before, plain, known)              # «Голодный кот давно хотел…»
                  or first(after, plain, known)            # «Мимо бежала лиса», «Рядом стоял мальчик»
                  or first(before, plain)
                  or first(after, plain)                   # «На краю стола стоял кувшин»
                  or (before[0] if before else None))
            if sg is None:
                subj = inherited or scene.agent            # «Стала хвалить ворону» — тот же герой
                obj_group = after[0] if after else None
            else:
                subj_form = head(sg)
                subj = self.refer(org, org.lang.lemma(subj_form), scene)
                rest = [g for g in after if g != sg] + [g for g in before if g != sg]
                obj_group = rest[0] if rest else None
        if subj is None:
            return None
        # что делает: в подряд идущих действиях главное — последнее («стала хвалить»)
        run, k = [first_v], first_v + 1
        while k < len(core):
            if core[k][1] in ("V", "V?", "U") and not any(g[0] <= k <= g[2] for g in groups):
                run.append(k)
            elif not (core[k][1] == "O" and k + 1 < len(core) and core[k + 1][1] in ("V", "V?")):
                break                                      # «стала его хвалить» — идём дальше
            k += 1
        cands = [n for n in run if core[n][1] == "V"] or [n for n in run if core[n][1] == "V?"] or run
        desire = next((n for n in cands if org.name(org.lang.lemma(core[n][0])).startswith("хоч")), None)
        main = desire if desire is not None else cands[-1]
        neg = main > 0 and core[main - 1][1] == "NEG"
        # кого/что
        obj, obj_form = None, None
        o_pron = next((n for n, (w, c) in enumerate(core) if c == "O" and n > first_v - 2), None)
        if o_pron is not None:                             # «захотела его», «стала его хвалить»
            obj_form = core[o_pron][0]                     # сначала участники событий, потом — места
            pool = [e for e in list(reversed(scene.core)) + list(reversed(scene.seen)) if e != subj]
            obj = self.resolve(obj_form, list(dict.fromkeys(pool)))
        elif obj_group is not None:
            form = core[obj_group[1]][0]
            if self.patient_like(form) or self.known_noun(org, form):
                obj_form = form
                obj = self.refer(org, org.lang.lemma(form), scene)
        if neg:
            return None
        return Event(subj, org.lang.lemma(core[main][0]), obj if obj != subj else None, i_sent,
                     (subj_form or org.name(subj), core[main][0], obj_form))

    def _groups(self, core) -> list[tuple[int, int, int]]:
        """Группы названий: (начало, главное слово, конец). Признаки — перед главным; из двух
        названий подряд главное — знакомое («однажды ворона»), иначе первое («кусок сыра»)."""
        out, k = [], 0
        while k < len(core):
            if core[k][1] in ("A", "N", "N?"):
                j = k
                while j < len(core) and core[j][1] in ("A", "N", "N?"):
                    j += 1
                nouns = [n for n in range(k, j) if core[n][1] in ("N", "N?")]
                split = next((n for n in nouns[1:] if self.agentive(core[n][0]) < 1
                              and len(self.patients.get(case_key(core[n][0]), ())) >= 2), None)
                if split is not None:                      # «сырный дух | Лису (остановил)»
                    j = split
                    nouns = [n for n in nouns if n < split]
                known = [n for n in nouns if core[n][1] == "N"]
                if nouns:
                    # в начале фразы незнакомое слово перед знакомым названием — скорее «однажды»,
                    # «вдруг»; иначе из двух названий подряд главное первое («кусок сыра»)
                    head = known[0] if known and core[nouns[0]][1] == "N?" else nouns[0]   # «свой мяч»
                    out.append((k, head, j - 1))
                k = j
            else:
                k += 1
        return out

    @staticmethod
    def refer(org, key: str, scene: Scene) -> str:
        """Кто это: уже знакомый герой — или знакомый под родовым именем («птица» — это ворона)."""
        if key in scene.seen:
            return key
        for e in reversed(scene.seen):
            if org._isa(e, key):
                return e
        return key

    # ---------- рассказ целиком ----------
    def events(self, org, items: list[tuple[int, str]]) -> list[Event]:
        from .causes import CauseField
        self._prepare(org)
        scene, out, depth = Scene(), [], 0
        for i, s in items:
            lead = len(s) - len(s.lstrip("»” "))              # кавычка закрылась в самом начале
            depth = max(0, depth - s[:lead].count("»") - s[:lead].count("”"))
            inside = depth > 0
            depth = max(0, depth + s.count("«") - s[lead:].count("»"))
            if inside or CauseField.split(s):
                continue
            toks = self._cut(lex(s.split("«")[0]))
            inherited = None
            for seg in self._segments(toks):
                ws = [w for w in seg if w not in SOFT]
                if not ws:
                    continue
                ev = self._clause(org, self._classify(org, ws), scene, inherited, i)
                if ev is None:
                    continue
                inherited = ev.agent
                scene.agent = ev.agent
                for x in (ev.agent, ev.obj):
                    if x:
                        scene.mention(x)
                        scene.core.append(x)
                out.append(ev)
        self._attr = None
        return out

    def _marked_tails(self, org) -> set[str]:
        """Окончания, по которым видно, кто действовал: бывают только при одном из местоимений
        («она села» → «-ла»); и их «соседи» по семье форм: есть «хотела» и «хотел» — значит,
        «-л» тоже такое окончание (о другом герое)."""
        own = {t for p, tt in self.pron_tails.items() for t, forms in tt.items() if len(forms) >= 2
               and sum(1 for q in self.pron_tails.values() if q.get(t)) == 1}
        out = set(own)
        forms = set(org.lang.forms)
        for f in forms:
            base = f[:-2] if f.endswith(("ся", "сь")) and len(f) > 5 else f
            if len(base) > 3 and base[-2:] in own:
                for sib in (base[:-1], base[:-1] + "о", base[:-1] + "и"):
                    if sib in forms or sib + "ся" in forms or sib + "сь" in forms:
                        out |= set(tails(sib))
        return out

    def marked(self, form: str) -> bool:
        """Форма действия, по которой видно, кто действовал («села» — она, «сел» — он): так говорят
        о том, что случилось, а не о том, что бывает всегда («киты дышат»)."""
        return any(t in self._marks for t in tails(form))

    def is_story(self, events: list[Event]) -> bool:
        """Рассказ: о случившемся (формы действий — «села», «упал») и герои меняются ролями —
        над кем действовали, тот потом действует (или наоборот). Описание рассказом не считается."""
        if len(events) < 3:
            return False
        if sum(self.marked(e.forms[1]) for e in events) < 0.4 * len(events):
            return False
        for n, e in enumerate(events):
            later = events[n + 1:]
            if e.obj and any(f.agent == e.obj for f in later):
                return True
            if any(f.obj == e.agent for f in later):
                return True
        return False
