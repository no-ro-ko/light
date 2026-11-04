"""Синтаксис «Ростка»: сложная фраза → простые высказывания.

«Киты — крупные морские млекопитающие» → кит — млекопитающее; кит крупный; кит морской.
«Они дышат воздухом и выкармливают детёнышей» → (они = киты) кит дышит воздухом; кит выкармливает…
«Летучая мышь — это млекопитающее, которое умеет летать» → «летучая мышь» (одно понятие из двух
    слов) — млекопитающее; летучая мышь умеет летать.
«Собаки, кошки и коровы — домашние животные» → три подлежащих.
«Некоторые птицы не летают» → НЕ «птицы не летают»: это про некоторых, обобщать нельзя.

Простые высказывания понимаются уже известными (врождёнными и выученными) оборотами.

Как отличить название от действия или признака — из собственных знаний: какие окончания
бывают у слов, которые в понятых фактах стоят на месте «кто» (кит, животное), а какие — на
месте «что делает / какой» (плавает, большой). Плюс устройство русской группы слов:
последнее слово группы — главное, стоящие перед ним — признаки.

Врождённые «склейки» (честно): «и», «а», «но», «который…», «он/она/оно/они», «все / каждый /
любой», «некоторые / многие», предлоги, «самый / очень / тоже».
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

TOK = re.compile(r"—|[,;:()]|\d+(?:[.,]\d+)?|[а-яёa-z]+(?:-[а-яёa-z]+)*", re.I)
COPULA = {"—", "это"}
PRONOUNS = {"он", "она", "оно", "они"}
ALL = {"все", "всё", "каждый", "каждая", "каждое", "любой", "любая", "любое"}
SOME = {"некоторые", "многие", "иные", "некоторый", "некоторая", "некоторое", "часть"}
PREPS = {"в", "во", "на", "у", "с", "со", "из", "к", "ко", "по", "за", "от", "до", "о", "об", "при",
         "для", "через", "над", "под", "между", "среди", "около", "возле", "без", "про"}
SOFT = {"самый", "самая", "самое", "самые", "очень", "тоже", "также", "уже", "ещё", "еще", "почти",
        "обычно", "часто", "иногда", "всегда", "как", "так", "лишь", "только", "быстро", "медленно",
        "отлично", "хорошо", "плохо", "громко", "тихо", "высоко", "низко", "далеко", "зато", "даже"}
CONJ = {"и", "а", "но", "или", "зато"}


def lex(sentence: str) -> list[str]:
    return [t.lower().replace("ё", "е") for t in TOK.findall(sentence)]


@dataclass
class Clause:
    subject: list[str]                         # формы слов подлежащего (одно понятие)
    rest: list[str]                            # формы слов сказуемого
    some: bool = False                         # «некоторые …» — не обобщать
    detail: list[str] = field(default_factory=list)


class Syntax:
    def __init__(self):
        self.noun_end: Counter = Counter()
        self.pred_end: Counter = Counter()
        self.mods: set[str] = set()                     # формы, стоявшие признаком перед названием

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.__dict__.setdefault("mods", set())

    # ---------- классы слов из собственных знаний ----------
    @staticmethod
    def ending(lang, w: str) -> str:
        for k in (3, 2, 1):
            if len(w) - k >= 3 and w[-k:] in lang.endings:
                return w[-k:]
        return ""

    def observe(self, lang, form: str, cls: str) -> None:
        bucket = self.noun_end if cls == "noun" else self.pred_end
        bucket[self.ending(lang, form)] += 1
        for k in (2, 3):                                  # и просто последние буквы («-ся», «-ет»)
            if len(form) > k + 2:
                bucket["~" + form[-k:]] += 1

    def _by_suffix(self, lang, w: str) -> tuple[str | None, float]:
        """Класс по последним буквам: → (класс, сила свидетельства)."""
        for key in (["~" + w[-3:]] if len(w) > 5 else []) + (["~" + w[-2:]] if len(w) > 4 else []) \
                + [self.ending(lang, w)]:
            n, p = self.noun_end[key], self.pred_end[key]
            if n + p >= 3:
                if n >= 2 * p:
                    return "noun", n / (p + 1)
                if p >= 2 * n:
                    return "pred", p / (n + 1)
        return None, 0.0

    def word_class(self, lang, know, w: str) -> str | None:
        if w in PREPS or w in COPULA or w in {"не", "и", "а", "но"}:
            return None
        if w in lang.frame_words or w in ("является", "являются", "умеет", "умеют", "может", "могут"):
            return None                                   # связка — не название и не признак
        if lang.role(w):                                  # сама видела это слово в такой роли
            return lang.role(w)
        lem = lang.lemma(w)
        by_lemma = None
        if lem in know.concepts and lem not in know.predicates:
            by_lemma = "noun"
        elif lem in know.predicates and lem not in know.concepts:
            by_lemma = "pred"
        if lang.is_function(w) and by_lemma is None:
            return None
        by_suffix, strength = self._by_suffix(lang, w)
        if by_lemma and by_suffix and by_suffix != by_lemma and strength >= 5:
            return by_suffix                              # окончание уверенно спорит с догадкой по основе
        return by_lemma or by_suffix

    # ---------- разбор ----------
    def clauses(self, sentence: str, lang, know, prev: list[str] | None) -> tuple[list[Clause], list[str] | None]:
        toks = lex(sentence)
        if not toks or any(t[0].isdigit() for t in toks):
            return [], prev
        main, rels = self._relatives(toks)
        out: list[Clause] = []
        chunks = self._clause_chunks(main, lang, know)
        subject = prev
        for n, ch in enumerate(chunks):
            # подлежащее можно не повторять только внутри одного предложения («…, но плавают»)
            got, subject = self._chunk(ch, lang, know, subject, ellipsis=n > 0)
            out += got
        # придаточное «…, который/которая … » — про того, к кому относится
        for attach, rel in rels:
            target = self._attach(attach, main, out)
            if target and rel:
                for part in self._split_predicates(rel, lang, know):
                    out.append(Clause(target, part))
        return out, (subject or prev)

    @staticmethod
    def _split(toks: list[str], seps: set[str]) -> list[list[str]]:
        out, cur = [], []
        for t in toks:
            if t in seps:
                if cur:
                    out.append(cur)
                cur = []
            else:
                cur.append(t)
        if cur:
            out.append(cur)
        return out

    def _clause_chunks(self, toks: list[str], lang, know) -> list[list[str]]:
        """Границы простых предложений: «;», «:», «, а …», «, но …» и «, <название> <действие>»."""
        out, cur = [], []
        for i, t in enumerate(toks):
            nxt = toks[i + 1] if i + 1 < len(toks) else ""
            after = toks[i + 2] if i + 2 < len(toks) else ""
            boundary = t in {";", ":", "(", ")"}
            if t == "," and (nxt in {"а", "но", "зато"} or
                             (self.word_class(lang, know, nxt) == "noun"
                              and self.word_class(lang, know, after) == "pred")):
                boundary = True
            if boundary:
                if cur:
                    out.append(cur)
                cur = []
                continue
            if not cur and t in CONJ:
                continue                                # «но отлично плавают» → без «но»
            cur.append(t)
        if cur:
            out.append(cur)
        return out

    @staticmethod
    def _relatives(toks: list[str]) -> tuple[list[str], list[tuple[int, list[str]]]]:
        main, rels, i = [], [], 0
        while i < len(toks):
            if toks[i] == "," and i + 1 < len(toks) and toks[i + 1].startswith("котор"):
                j = i + 2
                rel = []
                while j < len(toks) and toks[j] not in {",", ";", ":"}:
                    rel.append(toks[j])
                    j += 1
                rels.append((len(main) - 1, rel))
                i = j
                continue
            main.append(toks[i])
            i += 1
        return main, rels

    def _attach(self, idx: int, main: list[str], clauses: list[Clause]) -> list[str] | None:
        if idx < 0:
            return None
        word = main[idx]
        # «X — это млекопитающее, которое умеет летать»: придаточное описывает X
        for c in clauses:
            if c.rest and c.rest[0] in COPULA and c.rest[-1] == word:
                return c.subject
        return [word]

    def _chunk(self, ch: list[str], lang, know, prev, ellipsis: bool = False):
        ch = [t for t in ch if t not in SOFT]
        if not ch:
            return [], prev
        some = ch[0] in SOME
        if ch[0] in SOME or ch[0] in ALL:
            ch = ch[1:]
        k = next((i for i, t in enumerate(ch) if t in COPULA), None)
        if k is not None and k > 0:                               # «X — (это) Y»
            subjects = self._subjects(ch[:k], lang, know, prev)
            right = [t for t in ch[k + 1:] if t not in COPULA]
            neg = bool(right) and right[0] == "не"
            if neg:
                right = right[1:]
            cut = next((i for i, t in enumerate(right) if t in PREPS), len(right))
            np_, detail = [t for t in right[:cut] if t not in {",", "и"}], right[cut:]
            if not np_ or not subjects:
                return [], prev
            head, mods = np_[-1], np_[:-1]
            out = []
            for s in subjects:
                out.append(Clause(s, (["не"] if neg else []) + ["это", head], some, detail))
                for m in mods:
                    if not neg and self.word_class(lang, know, m) != "noun":
                        out.append(Clause(s, [m], some))
                        self.mods.add(m)
            return out, subjects[0]
        # «X делает …»: подлежащее — до первого действия/признака
        if ch[0] in PRONOUNS:
            if not prev:
                return [], prev
            subjects, rest = [prev], ch[1:]
        else:
            # подлежащее: признаки* название (названия через «,»/«и»); признак — только ДО названия
            j, seen_noun = 0, False
            while j < len(ch):
                cls = self.word_class(lang, know, ch[j])
                nxt = self.word_class(lang, know, ch[j + 1]) if j + 1 < len(ch) else None
                if ch[j] in {",", "и"}:
                    seen_noun = False
                elif cls == "noun":
                    seen_noun = True
                elif cls == "pred" and not seen_noun and nxt == "noun" and ch[j] != "не":
                    pass                                   # «белые (медведи)» — признак перед названием
                else:
                    break
                j += 1
            if j == 0 and ellipsis and prev and (ch[0] == "не" or self.word_class(lang, know, ch[0]) != "noun"):
                subjects, rest = [prev], ch                # «…, но отлично плавают» — подлежащее то же
            elif j == 0 or j >= len(ch):
                return [], prev
            else:
                subjects, rest = self._subjects(ch[:j], lang, know, prev), ch[j:]
        out = []
        for part in self._split_predicates(rest, lang, know):
            for s in subjects:
                out.append(Clause(s, part, some))
        return out, (subjects[0] if subjects else prev)

    def _subjects(self, words: list[str], lang, know, prev) -> list[list[str]]:
        parts = self._split(words, {",", "и"})
        out = []
        for p in parts:
            p = [w for w in p if w not in SOFT and w not in SOME and w not in ALL]
            if not p:
                continue
            if p[0] in PRONOUNS:
                if prev:
                    out.append(prev)
                continue
            # «детёныши китов»: два названия подряд — главное первое, второе — чьё
            if len(p) >= 2 and all(self.word_class(lang, know, w) == "noun" for w in p[-2:]) \
                    and self.word_class(lang, know, p[0]) == "noun":
                out.append(p)
            else:
                out.append(p)                              # признаки + главное (последнее) слово
        return out

    def _split_predicates(self, rest: list[str], lang, know) -> list[list[str]]:
        """«не летают и быстро бегают» → два сказуемых; союзы и запятые в сказуемое не попадают."""
        parts, cur = [], []
        rest = [w for w in rest if w not in SOFT]
        for i, t in enumerate(rest):
            if t in CONJ or t == ",":
                nxt = rest[i + 1] if i + 1 < len(rest) else ""
                if cur and nxt and (nxt == "не" or self.word_class(lang, know, nxt) != "noun"):
                    parts.append(cur)
                    cur = []
                continue                                 # союз — не часть сказуемого
            cur.append(t)
        if cur:
            parts.append(cur)
        return parts
