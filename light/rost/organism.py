"""«Росток»-организм: одна растущая система, которая учится только на текстах.

При рождении пусто: ни слов, ни понятий, ни правил, ни законов (врождённое ядро языка
перечислено в language.py). Всё вырастает из прочитанного:
- язык: формы слов, окончания, новые обороты речи (по уже известным фактам);
- знания: понятия, роды и свойства — с источником каждого;
- правила историй («X толкает Y, а Y шатается → Y падает») — с самопроверкой;
- законы из записей измерений;
- значения слов-ощущений по определениям («зелёный — это свет …»).

Во сне — самоулучшение: выучить новые обороты, перечитать ВСЮ память с ними (и понять
больше, чем при первом чтении), проверить правила, найти законы. Каждое изменение — в
журнал с числами.

Ответы — с объяснением; сомнения — при противоречиях между текстами; собственные
вопросы — о том, что непонятно и что важнее всего узнать.
"""
from __future__ import annotations

import re
from collections import Counter

from ..mind.grounding import DEF, Grounder, stem
from .knowledge import Knowledge
from .language import COPULA, NEG, TextLanguage, split_sentences, tokens
from .measures import MeasureField, extract
from .rules import Ev, RuleField, moral, unknown_word, what_if_not, why_did, why_happened
from .syntax import COPULA as SYN_COPULA, PREPS, SOFT, Syntax
from .causes import CauseField, Sit
from .narrative import OBJ as OBJ_PRONOUNS, Narrator

def _cap(text: str) -> str:
    """С большой буквы — только первая буква (остальные не трогать: «… А началось всё»)."""
    return text[:1].upper() + text[1:]


# «Запела — значит поёт», «Лисица — то же, что лиса»: два слова об одном
SAME = re.compile(r"^«?([а-я-]+)»?\s*(?:—|-|,)?\s*(?:значит|это то же,? что(?: и)?|то же,? что(?: и)?)"
                  r"\s*,?\s+«?([а-я-]+)»?$")


class Organism:
    def __init__(self):
        self.lang = TextLanguage()
        self.know = Knowledge()
        self.know.name = self.lang.name
        self.rules = RuleField()
        self.measures = MeasureField()
        self.colors = Grounder()
        self.definitions: dict[str, str] = {}
        self.memory: list[tuple[str, str]] = []           # (источник, предложение)
        self.understood: set[int] = set()
        self.stories: list[tuple[str, list[Ev]]] = []
        self.incoming_questions: list[tuple[str, str]] = []
        self.files: dict[str, dict] = {}
        self.journal: list[str] = []
        self.last_motive = None
        self.events: set[int] = set()
        self._subject_words = None
        self.syntax = Syntax()
        self.causes = CauseField()
        self.compounds: dict[str, tuple[str, list[str]]] = {}   # «синий кит» → (кит, [синий])
        self.some: dict[tuple, list[str]] = {}                   # «некоторые птицы не летают»
        self.narr = Narrator()                                   # чтение рассказов обычной речью
        self.paragraphs: list[tuple[str, list[int]]] = []        # (источник, номера предложений)
        self.tellings: list[list[tuple[int, str]]] = []          # к каждой истории: (фраза, как названо действие)
        self.episodic: set[int] = set()                          # фразы рассказов — события, а не общие сведения
        self.know.name = self.name

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.__dict__.setdefault("narr", Narrator())
        self.__dict__.setdefault("episodic", set())
        self.__dict__.setdefault("tellings", [[] for _ in self.stories])
        if "paragraphs" not in d:                                # память старой версии: абзац = файл
            by_src: dict[str, list[int]] = {}
            for i, (src, _) in enumerate(self.memory):
                by_src.setdefault(src, []).append(i)
            self.paragraphs = list(by_src.items())
        self.know.name = self.name

    def name(self, key: str) -> str:
        """Как показывать понятие: составное — так, как впервые прочитала («синий кит»)."""
        shown = getattr(self, "compound_names", {}).get(key)
        return shown or " ".join(self.lang.name(p) for p in key.split(" "))

    # ================= чтение =================
    def read(self, name: str, text: str) -> dict:
        start = len(self.memory)
        paragraphs, para, n_q = [], [], 0
        for s, is_q in split_sentences(text):
            if not s:
                if para:
                    paragraphs.append(para)
                para = []
                continue
            if is_q:
                self.incoming_questions.append((name, s))
                n_q += 1
                continue
            self.memory.append((name, s))
            self.lang.see(tokens(s))
            self.lang.spell(s)
            para.append(len(self.memory) - 1)
        if para:
            paragraphs.append(para)
        self.lang.grow_morphology()
        self.paragraphs += [(name, p) for p in paragraphs]
        facts = self._digest(range(start, len(self.memory)))
        n_rules_before = len(self.rules.rules)
        self._heroes(paragraphs)
        self.narr.learn(self, [self.memory[i][1] for i in range(start, len(self.memory))])
        for p in paragraphs:
            got = self._story(p)
            if got:
                story, telling = got
                self.rules.read_events(story)
                self._keep_story(name, story, telling)
        own = range(start, len(self.memory))
        stats = {"предложений": len(own), "понято": sum(1 for i in own if i in self.understood),
                 "новых фактов": facts, "новых правил": len(self.rules.rules) - n_rules_before,
                 "вопросов в файле": n_q}
        self.files[name] = stats
        return stats

    # ---------- истории ----------
    def _story(self, p: list[int]):
        """Абзац → история: [(событие)], [(номер фразы, как названо действие)] — или None."""
        if all(self._eventlike(self.memory[i][1]) for i in p):
            # простые фразы («Лиса хвалит ворону. Ворона радуется.»), как в первых историях;
            # «пингвин плавает. пингвины плавают» — повтор, а не событие и следствие
            kept, prev = [], None
            for i in p:
                key = tuple(self.lang.lemma(w) for w in tokens(self.memory[i][1])[:2])
                if key != prev:
                    kept.append(i)
                prev = key
            if len(kept) < 2:
                return None
            pairs = [(self.rules.parse(self.memory[i][1]), i) for i in kept]
            pairs = [(e, i) for e, i in pairs if e]
            self.__dict__.setdefault("simple_verbs", set()).update(e.action for e, _ in pairs)
            return [e for e, _ in pairs], [(i, tokens(self.memory[i][1])[1]) for _, i in pairs]
        evs = self.narr.events(self, [(i, self.memory[i][1]) for i in p])
        if not self.narr.is_story(evs):
            return None
        story, telling = [], []
        for e in evs:
            who, act, whom = e.forms
            direct = whom and whom not in OBJ_PRONOUNS and self.lang.lemma(whom) == e.obj
            story.append(self.rules.event(self.name(e.agent), self.name(e.action),
                                          self.name(e.obj) if e.obj else None, whom if direct else None))
            telling.append((e.sent, act))
            # рассказ сам подсказывает, где названия, а где действия (пригодится при перечитывании)
            self.lang.weak_next.setdefault(act, "pred")
            for w in (who, whom):
                if w and w not in OBJ_PRONOUNS:
                    self.lang.weak_next.setdefault(w, "noun")
        self.episodic.update(p)
        return story, telling

    def _heroes(self, paragraphs) -> None:
        """Герои простых историй («Кот хочет рыбу») — кто действует и над кем: это названия."""
        for p in paragraphs:
            if len(p) >= 2 and all(self._eventlike(self.memory[i][1]) for i in p):
                for i in p:
                    t = tokens(self.memory[i][1])
                    self.narr.heroes.update(self.lang.lemma(w) for w in (t[0], t[2] if len(t) == 3 else None) if w)

    def _keep_story(self, name: str, story, telling) -> None:
        self.stories.append((name, story))
        self.tellings.append(telling)
        sents = {i for i, _ in telling}
        self.events.update(sents)
        self.understood.update(sents)

    def _retell(self) -> None:
        """Перечитать рассказы заново — с выросшим языком и знаниями (и перестроить опыт правил)."""
        self.narr = Narrator()                          # опыт чтения — заново, без ошибок первого чтения
        self.rules.entities, self.rules.verbs = [], []  # и как названы герои и действия — тоже
        self.rules.as_agent, self.rules.as_object = {}, {}
        self.simple_verbs = set()
        self._heroes([p for _, p in self.paragraphs])
        self.narr.learn(self, [s for _, s in self.memory])
        self.stories, self.tellings, self.events, self.episodic = [], [], set(), set()
        for name, p in self.paragraphs:
            got = self._story(p)
            if got:
                self._keep_story(name, *got)
        self.rules.rebuild([st for _, st in self.stories])

    def story_actions(self) -> set[str]:
        """Действия, в которых система уверена: из простых историй и из выросших правил. (Не из
        разбора обычной речи — иначе ошибка разбора закрепляла бы сама себя.)"""
        words = set(getattr(self, "simple_verbs", set()))
        for r in self.rules.rules:
            words |= {r.trigger, r.effect} | ({r.cond[0]} if r.cond else set())
        return {self.lang.lemma(v.replace("ё", "е")) for v in words}

    @staticmethod
    def _eventlike(s: str) -> bool:
        t = tokens(s)
        return 2 <= len(t) <= 3 and not (set(t) & (COPULA | {NEG})) and not any(x[0].isdigit() for x in t)

    def _digest(self, indices, complex_ok: bool = True) -> int:
        """Понять предложения: факт, измерение, определение, сложная фраза. → новых фактов."""
        new = 0
        self._complex_ok = complex_ok
        last_source = None
        for i in indices:
            if self.memory[i][0] != last_source:
                self._prev, last_source = None, self.memory[i][0]
            if i in self.understood:
                continue
            source, s = self.memory[i]
            toks = tokens(s)
            if DEF.match(s.lower().replace(",", "")):
                self.colors.read(s)
                self.understood.add(i)
                continue
            if sum(1 for t in toks if t[0].isdigit()) >= 2:
                row = extract(toks, self.lang.lemma, self.lang.is_function)
                if row:
                    order = [self.lang.lemma(t) for t in toks if not t[0].isdigit()
                             and self.lang.lemma(t) in row]
                    self.measures.add(row, order, source)
                    self.understood.add(i)
                continue
            m = SAME.match(s.lower().replace("ё", "е"))
            if m:                                       # «Запела — значит поёт»
                note = self.lang.equate(m.group(1), m.group(2))
                if note:
                    self.journal.append(note)
                self.understood.add(i)
                continue
            pair = CauseField.split(s) if self._complex_ok else None
            if pair:                                    # «если …, то …», «… потому что …»
                cause = self.situation(pair[0])
                effect = self.situation(pair[1], prev=self._subject_words)
                self.causes.learn(cause, effect, source)
                self.understood.add(i)
                continue
            f = self.lang.parse(toks, self.know.is_category)
            if f and i in self.episodic and f[0] == "prop":
                continue                                # событие рассказа, а не общее знание о героях
            if f:
                kind, x, y, pol = f
                if self.know.tell(kind, x, y, pol, source) == "new":
                    new += 1
                self.understood.add(i)
                forms = [w for w in toks if not self.lang.is_function(w)]
                if len(forms) == 2:                    # какие окончания у «кто» и у «что делает / какой»
                    self.syntax.observe(self.lang, forms[0], "noun")
                    self.syntax.observe(self.lang, forms[1], "noun" if kind == "isa" else "pred")
                    self.lang.hints.setdefault(forms[0], "noun")
                    self.lang.hints.setdefault(forms[1], "noun" if kind == "isa" else "pred")
                if len(toks) == 2:
                    self.colors.read(s)                # «листья зелёные» — пример цвета
                self._prev = [toks[0]] if not self.lang.is_function(toks[0]) else self._prev
                continue
            m = re.match(r"^(\w+)\s*(?:—|-)?\s*это\s+(.{6,})$", s.lower())
            if m and len(m.group(2).split()) >= 2 and not any(w in m.group(2) for w in (",", "котор")):
                self.definitions[self.lang.lemma(m.group(1))] = m.group(2)
                self.understood.add(i)
                continue
            if i in self.episodic:
                continue                                # «лиса схватила сыр» — про этот рассказ, не про всех лис
            got = self._complex(s, source) if self._complex_ok else 0   # сложная → простые
            if got:
                new += got
                self.understood.add(i)
                continue
            for w in toks:                              # незнакомые слова-ощущения («в зелёном саду»)
                if len(w) > 4 and stem(w) in self.colors.concepts:
                    pass
            self.colors.read(s)
        return new

    # ---------- сложные фразы ----------
    def concept(self, words: list[str]) -> str:
        if len(words) == 1:
            return self.lang.lemma(words[0])
        lem = [self.lang.lemma(w) for w in words]
        key = " ".join(lem)
        # «синий кит»: последнее слово главное; «детёныши китов»: главное первое
        nouns = [self.syntax.word_class(self.lang, self.know, w) == "noun" for w in words]
        head = lem[0] if all(nouns[:2]) else lem[-1]
        self.compounds[key] = (head, [x for x in lem if x != head])
        if not hasattr(self, "compound_names"):
            self.compound_names = {}
        self.compound_names.setdefault(key, " ".join(words))
        if head == lem[-1]:                             # «синий кит»: «синий» — признак, «кит» — название
            for w in words[:-1]:
                self.lang.weak_next.setdefault(w, "pred")
                self.syntax.mods.add(w)
            self.lang.weak_next.setdefault(words[-1], "noun")
        return key

    def _complex(self, s: str, source: str) -> int:
        clauses, self._prev = self.syntax.clauses(s, self.lang, self.know, getattr(self, "_prev", None))
        told = 0
        for c in clauses:
            subj = self.concept(c.subject)
            for kind, x, y, pol in self._interpret(subj, c):
                where = f"{source} (из сложной фразы)"
                if c.some:
                    self.some.setdefault((x, y, pol), []).append(where)
                    told += 1
                    continue
                if self.know.tell(kind, x, y, pol, where) == "new":
                    told += 1
        return told

    def situation(self, text: str, prev: list[str] | None = None) -> Sit:
        """Положение дел: кто и что с ним; если не разобрать — набор слов."""
        toks = tokens(text)
        bag = frozenset(self.lang.lemma(w) for w in toks if not self.lang.is_function(w) and w != "не")
        neg = "не" in toks
        clauses, _ = self.syntax.clauses(text, self.lang, self.know, prev)
        self._subject_words = clauses[0].subject if clauses else prev
        if toks and toks[0] in ("он", "она", "оно", "они") and prev:
            text = " ".join(prev + toks[1:])               # «она падает» → «ваза падает»
        for c in clauses:
            subj = self.concept(c.subject)
            for kind, x, y, pol in self._interpret(subj, c):
                if kind == "prop" and " " not in y or kind == "prop":
                    return Sit(x, y, not pol, bag, text)
        return Sit(None, None, neg, bag, text)

    def _isa(self, a: str, b: str) -> bool:
        return b in {c for lvl in self.know.ancestors(a) for c, _ in lvl}

    def _interpret(self, subj: str, c) -> list[tuple]:
        toks = ["<x>"] + c.rest
        f = self.lang.parse(toks, self.know.is_category)
        if f and f[1] == "<x>":
            kind, _, y, pol = f
            return [(kind, subj, y, pol)]
        rest, pol = list(c.rest), True
        if rest and rest[0] == "не":
            pol, rest = False, rest[1:]
        if not rest or rest[0] in SYN_COPULA or rest[0] in PREPS or rest[0] in self.lang.frame_words \
                or rest[0] in ("умеет", "умеют", "может", "могут", "является", "являются"):
            return []                                    # связка без понятого оборота — не гадаю
        if self.syntax.word_class(self.lang, self.know, rest[0]) == "noun":
            return []                                    # название без связки — не понимаю, что сказано
        pred = self.lang.lemma(rest[0])
        self.lang.weak_next.setdefault(rest[0], "pred")     # слабая подсказка: это слово — действие
        for w in c.subject[-1:]:
            self.lang.weak_next.setdefault(w, "noun")
        out = [("prop", subj, pred, pol)]
        comp = []
        for w in rest[1:]:
            if w in {",", "и", "а", "но", "или"}:
                break
            if w not in SOFT:
                comp.append(w)
        comp = comp[:3]
        if comp and pol:
            out.append(("prop", subj, pred + " " + " ".join(self.lang.lemma(w) for w in comp), True))
        return out

    def coverage(self) -> float:
        return len(self.understood) / max(len(self.memory), 1)

    # ================= сон: самоулучшение =================
    def _facts(self) -> int:
        return sum(1 for v in self.know.isa.values() if v != [0, 0]) + \
            sum(1 for v in self.know.prop.values() if v != [0, 0])

    def reread(self) -> None:
        """Перечитать всю память заново — уже с тем языком, который вырос с тех пор."""
        self.know = Knowledge()
        self.measures = MeasureField()
        self.colors = Grounder()
        self.definitions = {}
        self.syntax = Syntax()
        self.causes = CauseField()
        self.compounds, self.some, self._prev = {}, {}, None
        self.lang.weak_next = {}                        # слабые подсказки — заново, с выросшим языком
        self.know.name = self.name
        self.understood = set()
        # сначала простые фразы (по ним видно, какие слова — названия, а какие — действия),
        # потом сложные — уже с этим знанием
        self._digest(range(len(self.memory)), complex_ok=False)
        self._digest(range(len(self.memory)), complex_ok=True)
        self._retell()
        self.understood |= self.events
        self.lang.weak_hints, self.lang.weak_next = self.lang.weak_next, {}
        self.lang._guess = {}

    def sleep(self) -> list[str]:
        notes, before = [], self.coverage()
        n_facts = self._facts()
        for _ in range(2):                             # язык → перечитать → подсказки → язык точнее
            notes += [f"выучила окончания: {x}" for x in self.lang.grow_morphology()]
            self.reread()
        for _ in range(4):
            self.lang.known_concepts = set(self.know.concepts)
            added = self.lang.learn_frames([tokens(s) for _, s in self.memory], self.know.known)
            if not added:
                break
            notes += added
            self.reread()                              # перечитать всё с новыми оборотами
        notes += self.rules.sleep()
        notes += self.measures.fit(self.lang.name)
        for name, st in self.files.items():            # сколько понято в каждом файле — теперь
            st["понято"] = sum(1 for i, (src, _) in enumerate(self.memory) if src == name and i in self.understood)
        after, n_after = self.coverage(), self._facts()
        if after > before + 1e-9 or n_after != n_facts:
            notes.append(f"перечитала всю память с выросшим языком: понимаю {after:.0%} предложений "
                         f"вместо {before:.0%}, фактов {n_facts} → {n_after}")
        self.journal += notes
        return notes

    # ================= ответы =================
    def answer(self, q: str) -> str:
        n, L = self.lang.name, self.lang.lemma
        t = q.lower().replace("ё", "е").strip(" ?!.")
        t = " ".join(re.sub(r"[,:;]", " ", t).split())
        m = re.match(r"что ты знаешь (?:о|об|про)\s+(.+)$", t)
        if m:
            return self._about(m.group(1).split())
        if re.search(r"что ты знаешь|чему ты научилась", t):
            return self.report()
        if re.search(r"сомнева", t):
            return ("Я сомневаюсь: " + " | ".join(self.know.doubts[-4:])) if self.know.doubts else \
                "Сейчас противоречий в прочитанном я не вижу."
        if re.search(r"ошибал|ошиблась", t):
            return ("Я ошибалась: " + " | ".join(self.know.corrections[-4:])) if self.know.corrections else \
                "Пока прочитанное не опровергало моих выводов."
        if re.search(r"хочешь узнать|твои вопросы|что тебе непонятно", t):
            qs = self.my_questions()
            return ("Больше всего я хочу узнать: " + " | ".join(qs[:5])) if qs else "Вопросов у меня сейчас нет."
        if re.search(r"как ты (выросла|изменилась)", t):
            return "Во мне изменилось: " + " | ".join(self.journal[-8:]) if self.journal else "Я ещё не спала."
        m = re.match(r"все ли\s+(\w+)\s+(.+)$", t)
        if m:
            return self._all(m.group(1), m.group(2).split())
        m = re.match(r"(?:чем отличаются|чем отличается)\s+(\w+)\s+(?:и|от)\s+(\w+)", t)
        if m:
            return self._compare(m.group(1), m.group(2), differ=True)
        m = re.match(r"чем похожи\s+(\w+)\s+и\s+(\w+)", t)
        if m:
            return self._compare(m.group(1), m.group(2), differ=False)
        m = re.match(r"где (?:живет|живут|обитает|обитают)\s+(.+)$", t)
        if m:
            return self._where(m.group(1).split())
        m = re.match(r"(?:что (?:умеет|умеют|делает|делают)|что ты знаешь (?:о|об|про))\s+(.+)$", t)
        if m:
            return self._about(m.group(1).split())
        m = re.match(r"^(?:кто|какие\s+(\w+))\s+(.+)$", t)
        if m and not t.startswith(("кто такой", "кто такая", "кто такие")):
            return self._who(m.group(2).split(), m.group(1))
        m = re.match(r"(?:что значит|что такое|кто такой|кто такая)\s+(\w+)", t)
        if m:
            return self._meaning(m.group(1))
        if re.search(r"перескаж|как ты поняла (?:рассказ|историю|басню)", t):
            return self._retelling()
        m = re.match(r"зачем\s+(.+)$", t)
        if m:
            for agent, action in self._pairs(m.group(1).split()):
                for src, st in reversed(self.stories):
                    text, motive = why_did(self.rules, st, agent, action)
                    if motive:
                        self.last_motive = motive
                        return _cap(text) + f". (из истории в «{src}»)"
            return "Я не нашла истории, где бы это было, или не понимаю, к чему это вело."
        m = re.match(r"почему\s+(.+)$", t)
        if m:
            for agent, action in self._pairs(m.group(1).split()):
                for src, st in reversed(self.stories):
                    ans = why_happened(self.rules, st, agent, action)
                    if not ans.startswith(("в этой истории", "не вижу")):
                        return _cap(ans) + "."
            chain = self.causes.why(self.situation(m.group(1)), self._isa, self.name)
            if chain:
                return "Потому что: " + "; а ".join(chain) + "."
            return "Не вижу причины ни в историях, ни в том, что я читала."
        m = re.match(r"что (?:будет|случится|произойдет|станет)(?: с \w+)?\s+если\s+(.+)$", t)
        if m:
            effects = self.causes.what_if(self.situation(m.group(1)), self._isa, self.name)
            if effects:
                return "Тогда: " + " → ".join(effects) + "."
            return "Не знаю, к чему это приводит: о таком я не читала."
        m = re.match(r"что было бы если бы\s+(\w+)\s+не\s+(\w+)", t)
        if m:
            agent, action = self._said(m.group(1)), self._said(m.group(2))
            unk = unknown_word(self.rules, agent, action)
            if unk:
                unk = m.group(1) if unk == agent else m.group(2)
                return (f"Я не знаю слова «{unk}»: в прочитанных историях его не было "
                        f"(может быть, это другая форма знакомого слова — подскажите)")
            for src, st in reversed(self.stories):
                ans = what_if_not(self.rules, st, agent, action)
                if ans != "такого и не было":
                    return _cap(ans) + "."
            return "Такого не было в известных мне историях."
        if "мораль" in t:
            return _cap(moral(self.last_motive, {n(k): v for k, v in self.definitions.items()})) + "."
        m = re.match(r"(?:чему|сколько)\s+\w*\s*(?:равен|равна|равно|будет)?\s*(\w+)\s+если\s+(.+)", t)
        if m:
            return self._predict(m.group(1), m.group(2))
        m = re.match(r"(?:какой|какая|каким|какое|какую)\s+(?:должна быть|должен быть|должно быть|нужна|нужен|нужно)?\s*(\w+)\s+чтобы\s+(.+)", t)
        if m:
            return self._solve(m.group(1), m.group(2))
        m = re.match(r"является ли\s+(.+)$", t)
        if m:
            words = m.group(1).split()
            x, rest = self._split_subject(words)
            if x and rest:
                return self._say(self._judge_isa(x, self._key(rest)))
        m = re.match(r"(\w+)\s+(?:—\s*)?это\s+(\w+)$", t)
        if m:
            return self._say(self._judge_isa(L(m.group(1)), L(m.group(2))))
        m = re.match(r"(\w+)\s+ли\s+(.+)$", t)
        if m:
            x, rest = self._split_subject(m.group(2).split())
            y = L(m.group(1))
            if self.know.is_category(y) and not rest:
                return self._say(self._judge_isa(x, y))
            if rest:                                     # «откладывает ли пингвин яйца»
                key = self._pred_key([m.group(1)] + rest)
                if key in self.know.predicates:
                    y = key
            return self._say(self.know.judge(x, y))
        return ("Я не поняла вопрос. Я умею отвечать на вопросы вида: «летает ли пингвин?», «является ли "
                "кит животным?», «что значит зелёный?», «зачем лиса хвалит ворону?», «почему сыр падает?», "
                "«что было бы, если бы ворона не пела?», «чему равен период, если длина 4?»")

    # ---------- истории: слова вопроса — словами системы ----------
    def _said(self, w: str) -> str:
        """«хвалила» → «хвалит», «плутовка» → «лиса» (если этому научили): как это названо в историях."""
        return self.name(self.lang.lemma(w))

    def _pairs(self, words: list[str]):
        """«(хитрая) лиса хвалила ворону» → (лиса, хвалит): соседние слова — кто и что делает."""
        words = [w for w in words if w not in SOFT and w != "не"]
        for a, b in zip(words, words[1:]):
            yield self._said(a), self._said(b)

    def _retelling(self) -> str:
        """Как поняла последний рассказ: цепочка событий словами системы."""
        from .rules import say
        for (src, st), telling in zip(reversed(self.stories), reversed(self.tellings)):
            if len(st) >= 3:
                n_sent = len({i for i, _ in telling})
                total = sum(1 for s_, _ in self.memory if s_ == src)
                return (f"Я поняла рассказ «{src}» так: " + " → ".join(say(e, self.rules) for e in st)
                        + f". События нашла в {n_sent} фразах из {total}.")
        return "Я ещё не читала рассказов."

    # ---------- вопросы о многих понятиях сразу ----------
    def _key(self, words: list[str]) -> str:
        """«летучая мышь», «синий кит» → составное понятие, если оно мне известно."""
        words = [w for w in words if w not in SOFT]
        lem = [self.lang.lemma(w) for w in words]
        for k in range(len(lem), 0, -1):
            key = " ".join(lem[:k])
            if key in self.know.concepts or key in self.compounds:
                return key
        return lem[-1] if lem else ""

    def _pred_key(self, words: list[str]) -> str | None:
        words = [w for w in words if w not in SOFT]
        if words and words[0] in ("умеет", "умеют", "может", "могут") and len(words) > 1:
            words = words[1:]
        lem = [self.lang.lemma(w) for w in words]
        for k in range(len(lem), 0, -1):
            key = " ".join(lem[:k])
            if key in self.know.predicates:
                return key
        return lem[0] if lem else None

    def _split_subject(self, words: list[str]) -> tuple[str, list[str]]:
        """Самое длинное известное понятие в начале: «синий кит животным» → («синий кит», [животным])."""
        for k in range(len(words), 0, -1):
            key = " ".join(self.lang.lemma(w) for w in words[:k])
            if key in self.know.concepts or key in self.compounds:
                return key, words[k:]
        return (self.lang.lemma(words[0]), words[1:]) if words else ("", [])

    def _judge_isa(self, x: str, y: str):
        v = self.know.judge_isa(x, y)
        if v.value is None or v.source == "по умолчанию":
            head = self.compounds.get(x, (None,))[0]
            if head and head != x and (head == y or self.know.judge_isa(head, y).value):
                from ..mind.mind import Verdict
                return Verdict(True, 0.6, [f"«{self.name(x)}» — по названию разновидность «{self.name(head)}»",
                                           f"наверное, {self.name(x)} — это {self.name(y)} (но названия "
                                           f"обманывают: летучая мышь — не мышь)"], "гипотеза")
        return v

    def _yes(self, x: str, p: str):
        v = self.know.judge(x, p)
        return v if v.value and v.conf >= 0.75 else None

    def _who(self, words: list[str], among: str | None) -> str:
        neg = bool(words) and words[0] == "не"
        if neg:
            words = words[1:]
        p = self._pred_key(words)
        if not p or p not in self.know.predicates:
            return f"Я не знаю никого, о ком было бы сказано «{' '.join(words)}»."
        pool = sorted(self.know.concepts)
        if among:
            group = self._key([among])
            pool = sorted(self.know.descendants(group))
        read, derived = [], []
        for x in pool:
            v = self.know.judge(x, p)
            if v.value is (not neg) and v.conf >= 0.75:
                (read if v.source == "прочитала" else derived).append(self.name(x))
        what = ("не " if neg else "") + self.name(p)
        if not read and not derived:
            return f"Не знаю никого, кто {what}."
        out = f"{_cap(what)}: " + ", ".join(read) if read else ""
        if derived:
            out += ("; " if out else "") + "а по выводу ещё: " + ", ".join(derived)
        return out + "."

    def _about(self, words: list[str]) -> str:
        x = self._key(words)
        own, inherited = [], []
        yes = [p for p in sorted(self.know.predicates)
               if (v := self.know.judge(x, p)).value and v.conf >= 0.75]
        for p in yes:
            if any(q.startswith(p + " ") for q in yes):
                continue                                  # «живёт» уже сказано в «живёт в море»
            v = self.know.judge(x, p)
            (own if v.source == "прочитала" else inherited).append(self.name(p))
        parents = self.know.parents(x)
        if not own and not inherited and not parents:
            return f"Я ничего не знаю о «{' '.join(words)}»."
        parts = []
        if parents:
            parts.append(f"{self.name(x)} — это {', '.join(self.name(y) for y in parents)}")
        if own:
            parts.append("прочитала: " + ", ".join(own))
        if inherited:
            parts.append("а ещё, по роду: " + ", ".join(inherited))
        return _cap("; ".join(parts) + ".")

    def _where(self, words: list[str]) -> str:
        x = self._key(words)
        roots = ("жив", "обита", "наход", "водит")       # «живёт / живут / обитает / водится»
        found = []
        for p in sorted(self.know.predicates):
            parts = p.split(" ")
            if len(parts) >= 3 and parts[0][:3] in {r[:3] for r in roots} and parts[1] in PREPS \
                    and parts[0] != self.lang.lemma("живое"):
                v = self.know.judge(x, p)
                if v.value and v.conf >= 0.75:
                    found.append((self.name(p).split(" ", 1)[1], v))
        if not found:
            return f"Не знаю, где живёт {self.name(x)}."
        place, v = found[0]
        return f"{_cap(place)}. " + " → ".join(v.chain)

    def _compare(self, a: str, b: str, differ: bool) -> str:
        x, y = self._key([a]), self._key([b])
        diffs, same = [], []
        for p in sorted(self.know.predicates):
            va, vb = self.know.judge(x, p), self.know.judge(y, p)
            if va.conf >= 0.75 and vb.conf >= 0.75 and va.value is not None and vb.value is not None:
                if va.value != vb.value:
                    diffs.append(f"{self.name(x)} {'' if va.value else 'не '}{self.name(p)}, "
                                 f"а {self.name(y)} {'' if vb.value else 'не '}{self.name(p)}")
                elif va.value:
                    same.append(self.name(p))
        # одно и то же действие по-разному: «кит дышит лёгкими, а акула дышит жабрами»
        def ways(z):
            out = {}
            for (s, q), (pos, neg) in self.know.prop.items():
                if s == z and pos > neg and " " in q:
                    out.setdefault(q.split(" ")[0], []).append(q)
            return out
        wx, wy = ways(x), ways(y)
        for base in sorted(set(wx) & set(wy)):
            if set(wx[base]) != set(wy[base]):
                diffs.insert(0, f"{self.name(x)} {', '.join(self.name(q) for q in wx[base])}, "
                                f"а {self.name(y)} {', '.join(self.name(q) for q in wy[base])}")
        pa, pb = set(self.know.parents(x)), set(self.know.parents(y))
        if differ:
            kinds = []
            if pa != pb and pa and pb:
                kinds.append(f"{self.name(x)} — это {', '.join(self.name(v) for v in pa)}, "
                             f"а {self.name(y)} — {', '.join(self.name(v) for v in pb)}")
            items = kinds + diffs
            return _cap("; ".join(items[:6]) + ".") if items else "Различий я не знаю."
        common = [self.name(v) for v in self.know.ancestors and
                  {c for lvl in self.know.ancestors(x) for c, _ in lvl} & {c for lvl in self.know.ancestors(y) for c, _ in lvl}]
        items = ([f"оба — это {', '.join(common)}"] if common else []) + ([f"оба: {', '.join(same[:6])}"] if same else [])
        return _cap("; ".join(items) + ".") if items else "Сходства я не знаю."

    def _all(self, group: str, words: list[str]) -> str:
        x, p = self._key([group]), self._pred_key(words)
        neg = words and words[0] == "не"
        if neg:
            p = self._pred_key(words[1:])
        some = self.some.get((x, p, neg)) or self.some.get((x, p, not neg))
        exceptions = [self.name(k) for k in sorted(self.know.descendants(x))
                      if (v := self.know.judge(k, p)).value is (neg) and v.conf >= 0.75]
        if some or exceptions:
            text = "Нет"
            if some:
                text += f": в «{some[0]}» сказано, что это верно лишь для некоторых"
            if exceptions:
                text += f"; например: {', '.join(exceptions[:5])}"
            return text + "."
        return self._say(self.know.judge(x, p))

    @staticmethod
    def _say(v) -> str:
        return _cap(v.say()) + (". " + " → ".join(v.chain) if v.chain else ".")

    def _meaning(self, w: str) -> str:
        n, lem = self.lang.name, self.lang.lemma(w)
        if lem != self.lang._base(w):                    # «запела — значит поёт»
            return f"«{w}» — то же, что «{n(lem)}» (так мне объяснили)."
        c = self.colors.concepts.get(stem(w))
        if c and c.lo is not None:
            ex = f"; например: {', '.join(sorted(c.examples))}" if c.examples else ""
            return f"«{c.word}» — это свет с длиной волны {c.lo:.0f}–{c.hi:.0f} нм{ex}."
        parts = []
        if lem in self.definitions:
            parts.append(f"{n(lem)} — это {self.definitions[lem]}")
        parents = self.know.parents(lem)
        if parents:
            parts.append(f"{n(lem)} — это {', '.join(n(p) for p in parents)}")
        props = [f"{'' if v[0] > v[1] else 'не '}{n(p)}" for (x, p), v in self.know.prop.items()
                 if x == lem and v[0] != v[1]]
        if props:
            parts.append("о нём известно: " + ", ".join(props))
        kids = self.know.children(lem)
        if kids:
            parts.append("к нему относятся: " + ", ".join(n(k) for k in kids))
        return _cap(". ".join(parts) + ".") if parts else f"Я не знаю, что значит «{w}»."

    def _values(self, text: str) -> dict[str, float]:
        toks = tokens(text)
        out, name = {}, None
        for t in toks:
            if t[0].isdigit():
                if name:
                    out[name] = float(t.replace(",", "."))
                    name = None
            elif not self.lang.is_function(t):
                name = self.lang.lemma(t)
        return out

    def _predict(self, target: str, rest: str) -> str:
        n, tgt = self.lang.name, self.lang.lemma(target)
        vals = self._values(rest)
        law = self.measures.law_for(set(vals) | {tgt})
        if not law or law.target != tgt:
            return f"Я не знаю закона, по которому можно найти «{target}»."
        x = {k: vals.get(k, 1.0) for k, a in law.exps.items()}
        missing = [n(k) for k, a in law.exps.items() if a and k not in vals]
        if missing:
            return f"Чтобы ответить, мне нужно знать ещё: {', '.join(missing)}."
        y = law.predict(x)
        return (f"{_cap(n(tgt))} ≈ {y:.3g}. По закону, который я нашла в записях измерений: "
                f"{law.text(n)}.")

    def _solve(self, var: str, rest: str) -> str:
        n, v = self.lang.name, self.lang.lemma(var)
        vals = self._values(rest)
        if not vals:
            return "Не поняла, чего нужно добиться."
        for key, law in self.measures.laws.items():
            if law.target in vals and v in law.exps:
                y = vals[law.target]
                x = {k: vals.get(k, 1.0) for k in law.exps}
                r = law.solve(v, y, x)
                if r is None:
                    return f"«{n(v)}» на «{n(law.target)}» не влияет — его менять бесполезно."
                return (f"{_cap(n(v))} ≈ {r:.3g}: решила обратную задачу по закону "
                        f"{law.text(n)}.")
        return "Я не знаю закона, который связывает эти величины."

    # ================= свои вопросы =================
    def my_questions(self) -> list[str]:
        n, out = self.lang.name, []
        out += [f"{d}" for d in self.know.doubts]
        out += [f"Что значит «{c.word}»? (встретилось в тексте, а определения я не знаю)"
                for c in self.colors.concepts.values() if c.lo is None]
        cnt = Counter()
        where = {}
        for i, (src, s) in enumerate(self.memory):
            if i in self.understood:
                continue
            for w in tokens(s):
                lem = self.lang.lemma(w)
                if not self.lang.is_function(w) and not w[0].isdigit() and len(w) > 3 \
                        and lem not in self.know.concepts and lem not in self.know.predicates:
                    cnt[lem] += 1
                    where.setdefault(lem, s)
        for lem, k in cnt.most_common(5):
            if k >= 2:
                out.append(f"Что значит «{n(lem)}»? Встречалось {k} раз, например: «{where[lem][:60]}»")
        tops = sorted((c for c in self.know.concepts if not self.know.parents(c)
                       and (self.know.children(c) or any(x == c for (x, _) in self.know.prop))),
                      key=lambda c: -len(self.know.descendants(c)))
        for c in tops[:3]:
            out.append(f"«{n(c)}» — это кто или что? Я не знаю, к какому роду это относится")
        out += self._story_words()
        out += self.measures.questions(n)
        return out

    def _story_words(self) -> list[str]:
        """Действия в рассказах, которых нет ни в других историях, ни в знаниях: без них не понять,
        к чему ведёт событие."""
        out, asked = [], set()
        familiar = self.story_actions()                  # из простых историй и правил
        for (src, st), telling in zip(self.stories, self.tellings):
            for e, (i, form) in zip(st, telling):
                lem = self.lang.lemma(form)
                if lem in familiar or lem in self.know.predicates or form in asked:
                    continue
                asked.add(form)
                out.append(f"Что значит «{self.lang.spelling.get(form, form)}»? В рассказе «{src}»: "
                           f"«{self.memory[i][1][:70]}»")
        return out

    # ================= отчёт =================
    def report(self) -> str:
        k = self.know
        facts = sum(1 for v in k.isa.values() if v != [0, 0]) + sum(1 for v in k.prop.values() if v != [0, 0])
        learned_frames = [" ".join(f) for f, i in self.lang.frames.items() if not i.get("innate")]
        lines = [
            f"Я родилась пустой и прочитала {len(self.files)} файлов ({len(self.memory)} предложений). "
            f"Понимаю {self.coverage():.0%} из них.",
            f"Во мне выросло: форм слов {len(self.lang.forms)}, окончаний {len(self.lang.endings)}, "
            f"выученных оборотов речи {len(learned_frames)}"
            + (f" ({'; '.join(learned_frames[:4])})" if learned_frames else "")
            + f", понятий {len(k.concepts)}, фактов {facts}, правил {len(self.rules.rules)}, "
              f"законов {len(self.measures.laws)}, цветов {sum(1 for c in self.colors.concepts.values() if c.lo)}.",
        ]
        if self.measures.laws:
            lines.append("Законы, которые я нашла: " + "; ".join(l.text(self.lang.name) for l in self.measures.laws.values()))
        if self.rules.rules:
            lines.append("Правила из историй: " + "; ".join(r.text() for r in self.rules.rules))
        for name, st in self.files.items():
            lines.append(f"  «{name}»: понято {st['понято']} из {st['предложений']} предложений")
        return "\n".join(lines)
