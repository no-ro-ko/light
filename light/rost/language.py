"""Язык, который растёт из прочитанных текстов.

Врождённое (честно перечислено — без этого из одних текстов не начать):
- «не» — отрицание; «это» и «—» — связка «есть разновидность»; «ли» — вопрос;
- небольшой набор самых частых служебных слов (предлоги, союзы, местоимения).

Всё остальное ВЫРАСТАЕТ из текстов:
- формы слов (каждая новая форма — новый узел);
- окончания: окончание становится узлом, когда встречается у многих основ, у которых
  есть и другие окончания («собак-а/-и/-у», «лета-ет/-ют/-ть»); формы с общей основой
  и выученными окончаниями сводятся в одно слово (лемму);
- обороты речи («X является Y», «X умеет Y»): если в предложении стоят два понятия,
  связь между которыми уже известна, слова между ними — кандидат в оборот для этой
  связи; повторился на разных известных парах и ни разу не противоречил — оборот
  принят и теперь сам добывает НОВЫЕ факты из других предложений (самодостраивающееся
  чтение: знания → обороты → новые знания).
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

TOKEN = re.compile(r"—|\d+(?:[.,]\d+)?|[а-яёa-z]+(?:-[а-яёa-z]+)*", re.I)
NEG, QUEST = "не", "ли"
COPULA = {"это", "—"}
INNATE_FUNCTION = {
    "не", "это", "—", "ли", "и", "а", "но", "или", "все", "всё", "весь", "в", "во", "на", "у", "с",
    "со", "из", "к", "ко", "по", "за", "от", "до", "о", "об", "при", "для", "что", "как", "так",
    "же", "бы", "то", "он", "она", "оно", "они", "его", "её", "их", "если", "чтобы", "когда",
    "уже", "ещё", "очень", "тоже", "там", "тут", "вот", "был", "была", "было", "были", "есть",
}


def tokens(sentence: str) -> list[str]:
    return [t.lower().replace("ё", "е") for t in TOKEN.findall(sentence)]


def split_sentences(text: str) -> list[tuple[str, bool]]:
    """→ [(предложение, вопрос ли)]; абзацы (пустые строки) отмечаются пустой строкой."""
    out = []
    for para in re.split(r"\n\s*\n", text):
        for m in re.finditer(r"(?:[^.!?…\n]|(?<=\d)\.(?=\d))+[.!?…]?", para):
            s = m.group(0).strip()
            if len(tokens(s)) >= 1:
                out.append((s.rstrip(".!…").strip(), s.endswith("?")))
        out.append(("", False))                       # граница абзаца
    return out


class TextLanguage:
    def __init__(self, min_stems: int = 3):
        self.min_stems = min_stems
        self.forms: Counter = Counter()               # узлы-формы (растут при первом появлении)
        self.endings: set[str] = set()                # выученные окончания
        self.lemma_of: dict[str, str] = {}
        self.display: dict[str, str] = {}             # лемма → самая частая форма
        self.frames: dict[tuple, dict] = {            # обороты: врождённые + выученные
            ("X", "это", "Y"): {"rel": "isa", "pol": True, "innate": True},
            ("X", "—", "Y"): {"rel": "isa", "pol": True, "innate": True},
            ("X", "не", "Y"): {"rel": "?", "pol": False, "innate": True},
            ("X", "Y"): {"rel": "?", "pol": True, "innate": True},
            ("все", "X", "Y"): {"rel": "?", "pol": True, "innate": True},
            ("X", "это", "не", "Y"): {"rel": "isa", "pol": False, "innate": True},
        }
        self.frame_words: set[str] = set()
        self.journal: list[str] = []
        self._families: set[str] = set()
        self._singles: set[str] = set()
        self._guess: dict[str, str] = {}
        self.first: Counter = Counter()               # сколько раз форма открывала фразу
        # подсказки из знаний: форма стояла на месте «кто» (название) или «какой/что делает»;
        # слова разной роли в одно не сливаются («синица» ≠ «синий», «белка» ≠ «белые»)
        self.hints: dict[str, str] = {}                # сильные: из надёжно понятых простых фраз
        self.weak_hints: dict[str, str] = {}           # слабые: из разбора сложных фраз, пересчитываются
        self.weak_next: dict[str, str] = {}            # копятся при перечитывании, заменяют старые в конце
        self.known_concepts: set[str] = set()

    # ---------- формы и окончания ----------
    def see(self, toks: list[str]) -> None:
        self.forms.update(t for t in toks if not t[0].isdigit())
        if toks and not toks[0][0].isdigit():
            self.first[toks[0]] += 1

    def grow_morphology(self) -> list[str]:
        """Выучить окончания по «парадигмам»: основа встречается с разными окончаниями."""
        words = [w for w in self.forms if w not in INNATE_FUNCTION and len(w) >= 3]
        by_stem: dict[str, set[str]] = defaultdict(set)
        for w in words:
            for k in range(0, 4):
                stem, end = (w, "") if k == 0 else (w[:-k], w[-k:])
                if len(stem) >= 3:
                    by_stem[stem].add(end)
        support: Counter = Counter()
        for stem, ends in by_stem.items():
            if len(ends) >= 2:
                for e in ends:
                    support[e] += 1
        new = {e for e, n in support.items() if n >= self.min_stems and len(e) <= 3}
        added = sorted(new - self.endings)
        self.endings |= new
        # лемма: основа с самой большой «семьёй» форм (с выученными окончаниями);
        # при равенстве — более длинная основа (осторожнее)
        forms = set(self.forms)
        self.lemma_of = {}
        for w in words:
            best, best_size = w, 1
            for k in range(0, 4):
                stem, end = (w, "") if k == 0 else (w[:-k], w[-k:])
                if len(stem) < 3 or end not in self.endings:
                    continue
                mine = self.role(w)
                family = {e for e in self.endings if stem + e in forms
                          and not (mine and self.role(stem + e) and self.role(stem + e) != mine)}
                if len(family) > best_size or (len(family) == best_size and best_size >= 2 and len(stem) > len(best)):
                    best, best_size = stem, len(family)
            self.lemma_of[w] = best if best_size >= 2 else w
        fam = Counter(self.lemma_of.values())
        self._families = {L for L, k in fam.items() if k >= 2 and len(L) >= 4}
        # слова, встреченные пока в одной форме, — тоже основа для узнавания новых форм,
        # но только если «хвост» — уже выученное окончание («фигур-а» → «фигур-ой»)
        self._singles = {w for w in self.forms if len(w) >= 4 and w not in INNATE_FUNCTION
                         and self.lemma_of.get(w, w) == w} - self._families
        self._guess = {}
        # как показывать слово: форма, в которой оно чаще стоит первым во фразе
        # (именительный: «млекопитающее», а не «млекопитающим»), а у действий — личная
        # форма («дышит», а не «дышать»)
        self.display = {}
        best: dict[str, float] = {}
        for w, n in self.forms.items():
            lem = self.lemma_of.get(w, w)
            verb_sing = w.endswith(("ет", "ит", "ется", "ится"))       # «дышит», «охотится»
            verb_plur = w.endswith(("ут", "ют", "ат", "ят", "утся", "ются", "атся", "ятся"))
            score = n + 5 * self.first.get(w, 0) + (6 * n if verb_sing else 2 * n if verb_plur else 0)
            if score > best.get(lem, -1):
                self.display[lem], best[lem] = w, score
        if added:
            self.journal.append(f"выучила окончания: {', '.join('-' + e if e else '(нулевое)' for e in added)}")
        return added

    def role(self, w: str) -> str | None:
        return self.hints.get(w) or self.weak_hints.get(w)

    def lemma(self, w: str) -> str:
        lem = self.lemma_of.get(w)
        if lem is not None and lem != w:
            return lem
        if w in self._guess:
            return self._guess[w]
        # запасной путь: незнакомая форма похожа на известное слово с другим окончанием
        # («птицей» ~ «птиц-а», «дышит» ~ «дыша-т»): основа почти совпадает, хвост ≤ 3 букв
        best = w
        if len(w) >= 4 and not self.is_function(w):
            for L in self._singles:
                if L == w:
                    continue
                n = 0
                while n < min(len(L), len(w)) and L[n] == w[n]:
                    n += 1
                if self.role(w) and self.role(L) and self.role(w) != self.role(L):
                    continue
                both_endings = w[n:] in self.endings and L[n:] in self.endings | {""}
                if both_endings and n >= 4 and (n >= len(L) - 1 or n >= 0.6 * max(len(L), len(w))) \
                        and (best == w or len(L) > len(best)):
                    best = self.lemma_of.get(L, L)
            for L in self._families:
                n = 0
                while n < min(len(L), len(w)) and L[n] == w[n]:
                    n += 1
                tail = w[n:]
                fam_hints = {self.role(f) for f, l in self.lemma_of.items() if l == L} - {None}
                if self.role(w) and fam_hints and self.role(w) not in fam_hints:
                    continue
                close = n >= 4 or (n >= 3 and tail in self.endings)   # «дыш-ит» ~ «дыша-т»
                if close and n >= len(L) - 1 and len(tail) <= 3 and (best == w or len(L) > len(best)):
                    best = L
        self._guess[w] = best
        return best

    def name(self, lemma: str) -> str:
        return self.display.get(lemma, lemma)

    def is_function(self, w: str) -> bool:
        return w in INNATE_FUNCTION or w in self.frame_words

    # ---------- обороты ----------
    def shape(self, toks: list[str]) -> tuple[tuple, list[str]]:
        """Предложение → (оборот с местами X, Y, …; леммы понятий на этих местах)."""
        frame, slots = [], []
        for t in toks:
            if t[0].isdigit():
                return (), []
            if self.is_function(t):
                frame.append(t)
            else:
                frame.append("XYZW"[len(slots)] if len(slots) < 4 else "?")
                slots.append(self.lemma(t))
        return tuple(frame), slots

    def learn_frames(self, sentences: list[list[str]], known) -> list[str]:
        """Самодостраивающееся чтение: найти новые обороты по уже известным фактам.
        known(x, y) → ("isa"|"prop", истинно) или None."""
        cand: dict[tuple, Counter] = defaultdict(Counter)
        for toks in sentences:
            lem = [self.lemma(t) for t in toks]
            content = [i for i, t in enumerate(toks) if not self.is_function(t) and not t[0].isdigit()]
            for a in range(len(content)):
                for b in range(a + 1, len(content)):
                    i, j = content[a], content[b]
                    rel = known(lem[i], lem[j])
                    if rel is None:
                        continue
                    frame = tuple("X" if k == i else "Y" if k == j else toks[k] for k in range(len(toks)))
                    if len(frame) > 6 or frame in self.frames or frame[-1] != "Y":
                        continue                           # оборот — это связь: [кто] [связка] [что]
                    literals = [w for w in frame if w not in ("X", "Y")]
                    if not literals or any(lem[k] in self.known_concepts for k in range(len(toks))
                                           if frame[k] not in ("X", "Y") and k not in (i, j)):
                        continue                           # названия в обороте — это не связка
                    cand[frame][rel] += 1
        added = []
        for frame, rels in cand.items():
            (rel, pol), n = rels.most_common(1)[0]
            if n >= 2 and n >= 0.85 * sum(rels.values()):    # подавляющее большинство подтверждений
                self.frames[frame] = {"rel": rel, "pol": pol, "innate": False, "support": n}
                for t in frame:
                    if t not in ("X", "Y") and t not in INNATE_FUNCTION:
                        self.frame_words.add(t)
                note = f"выучила оборот «{' '.join(frame)}» = {'род' if rel == 'isa' else 'свойство'}" \
                       f"{'' if pol else ' (отрицание)'}: подтвердился на {n} известных парах"
                self.journal.append(note)
                added.append(note)
        return added

    def parse(self, toks: list[str], is_category) -> tuple | None:
        """Предложение → ("isa"|"prop", x, y, истинно) по известному обороту, иначе None."""
        frame, slots = self.shape(toks)
        info = self.frames.get(frame)
        if info is None or len(slots) != 2:
            return None
        x, y = slots
        rel = info["rel"]
        if rel == "?":
            rel = "isa" if is_category(y) else "prop"
        return (rel, x, y, info["pol"])
