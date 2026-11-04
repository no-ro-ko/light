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

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.know.name = self.lang.name

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
            para.append(len(self.memory) - 1)
        if para:
            paragraphs.append(para)
        self.lang.grow_morphology()
        facts = self._digest(range(start, len(self.memory)))
        n_rules_before = len(self.rules.rules)
        for p in paragraphs:
            events = [i for i in p if self._eventlike(self.memory[i][1])]
            # «пингвин плавает. пингвины плавают» — повтор, а не событие и следствие
            kept, prev = [], None
            for i in events:
                key = tuple(self.lang.lemma(w) for w in tokens(self.memory[i][1])[:2])
                if key != prev:
                    kept.append(i)
                prev = key
            if len(kept) >= 2:
                story = self.rules.read([self.memory[i][1] for i in kept])
                self.stories.append((name, story))
                self.events.update(kept)
                self.understood.update(kept)
        own = range(start, len(self.memory))
        stats = {"предложений": len(own), "понято": sum(1 for i in own if i in self.understood),
                 "новых фактов": facts, "новых правил": len(self.rules.rules) - n_rules_before,
                 "вопросов в файле": n_q}
        self.files[name] = stats
        return stats

    @staticmethod
    def _eventlike(s: str) -> bool:
        t = tokens(s)
        return 2 <= len(t) <= 3 and not (set(t) & (COPULA | {NEG})) and not any(x[0].isdigit() for x in t)

    def _digest(self, indices) -> int:
        """Понять предложения: факт, измерение, определение. → сколько новых фактов."""
        new = 0
        for i in indices:
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
            f = self.lang.parse(toks, self.know.is_category)
            if f:
                kind, x, y, pol = f
                if self.know.tell(kind, x, y, pol, source) == "new":
                    new += 1
                self.understood.add(i)
                if len(toks) == 2:
                    self.colors.read(s)                # «листья зелёные» — пример цвета
                continue
            m = re.match(r"^(\w+)\s*(?:—|-)?\s*это\s+(.{6,})$", s.lower())
            if m and len(m.group(2).split()) >= 2:
                self.definitions[self.lang.lemma(m.group(1))] = m.group(2)
                self.understood.add(i)
                continue
            for w in toks:                              # незнакомые слова-ощущения («в зелёном саду»)
                if len(w) > 4 and stem(w) in self.colors.concepts:
                    pass
            self.colors.read(s)
        return new

    def coverage(self) -> float:
        return len(self.understood) / max(len(self.memory), 1)

    # ================= сон: самоулучшение =================
    def _facts(self) -> int:
        return sum(1 for v in self.know.isa.values() if v != [0, 0]) + \
            sum(1 for v in self.know.prop.values() if v != [0, 0])

    def reread(self) -> None:
        """Перечитать всю память заново — уже с тем языком, который вырос с тех пор."""
        self.know = Knowledge()
        self.know.name = self.lang.name
        self.measures = MeasureField()
        self.colors = Grounder()
        self.definitions = {}
        self.understood = set()
        self._digest(range(len(self.memory)))         # событие сказки может быть и фактом
        self.understood |= self.events

    def sleep(self) -> list[str]:
        notes, before = [], self.coverage()
        n_facts = self._facts()
        notes += [f"выучила окончания: {x}" for x in self.lang.grow_morphology()]
        self.reread()
        for _ in range(4):
            added = self.lang.learn_frames([tokens(s) for _, s in self.memory], self.know.known)
            if not added:
                break
            notes += added
            self.reread()                              # перечитать всё с новыми оборотами
        notes += self.rules.sleep()
        notes += self.measures.fit(self.lang.name)
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
        m = re.match(r"(?:что значит|что такое|кто такой|кто такая)\s+(\w+)", t)
        if m:
            return self._meaning(m.group(1))
        m = re.match(r"зачем\s+(\w+)\s+(\w+)", t)
        if m:
            for src, st in reversed(self.stories):
                text, motive = why_did(self.rules, st, m.group(1), m.group(2))
                if motive:
                    self.last_motive = motive
                    return text.capitalize() + f". (из истории в «{src}»)"
            return "Я не нашла истории, где бы это было, или не понимаю, к чему это вело."
        m = re.match(r"почему\s+(\w+)\s+(\w+)", t)
        if m:
            for src, st in reversed(self.stories):
                ans = why_happened(self.rules, st, m.group(1), m.group(2))
                if not ans.startswith(("в этой истории", "не вижу")):
                    return ans.capitalize() + "."
            return "Не вижу причины в известных мне историях."
        m = re.match(r"что было бы если бы\s+(\w+)\s+не\s+(\w+)", t)
        if m:
            unk = unknown_word(self.rules, m.group(1), m.group(2))
            if unk:
                return (f"Я не знаю слова «{unk}»: в прочитанных историях его не было "
                        f"(может быть, это другая форма знакомого слова — подскажите)")
            for src, st in reversed(self.stories):
                ans = what_if_not(self.rules, st, m.group(1), m.group(2))
                if ans != "такого и не было":
                    return ans.capitalize() + "."
            return "Такого не было в известных мне историях."
        if "мораль" in t:
            return moral(self.last_motive, {n(k): v for k, v in self.definitions.items()}).capitalize() + "."
        m = re.match(r"(?:чему|сколько)\s+\w*\s*(?:равен|равна|равно|будет)?\s*(\w+)\s+если\s+(.+)", t)
        if m:
            return self._predict(m.group(1), m.group(2))
        m = re.match(r"(?:какой|какая|каким|какое|какую)\s+(?:должна быть|должен быть|должно быть|нужна|нужен|нужно)?\s*(\w+)\s+чтобы\s+(.+)", t)
        if m:
            return self._solve(m.group(1), m.group(2))
        m = re.match(r"является ли\s+(\w+)\s+(\w+)", t) or re.match(r"(\w+)\s+(?:—\s*)?это\s+(\w+)$", t)
        if m:
            v = self.know.judge_isa(L(m.group(1)), L(m.group(2)))
            return self._say(v)
        m = re.match(r"(\w+)\s+ли\s+(\w+)$", t)
        if m:
            x, y = L(m.group(2)), L(m.group(1))
            v = self.know.judge_isa(x, y) if self.know.is_category(y) else self.know.judge(x, y)
            return self._say(v)
        return ("Я не поняла вопрос. Я умею отвечать на вопросы вида: «летает ли пингвин?», «является ли "
                "кит животным?», «что значит зелёный?», «зачем лиса хвалит ворону?», «почему сыр падает?», "
                "«что было бы, если бы ворона не пела?», «чему равен период, если длина 4?»")

    @staticmethod
    def _say(v) -> str:
        return v.say().capitalize() + (". " + " → ".join(v.chain) if v.chain else ".")

    def _meaning(self, w: str) -> str:
        n, lem = self.lang.name, self.lang.lemma(w)
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
        return (". ".join(parts) + ".").capitalize() if parts else f"Я не знаю, что значит «{w}»."

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
        return (f"{n(tgt).capitalize()} ≈ {y:.3g}. По закону, который я нашла в записях измерений: "
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
                return (f"{n(v).capitalize()} ≈ {r:.3g}: решила обратную задачу по закону "
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
        out += self.measures.questions(n)
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
