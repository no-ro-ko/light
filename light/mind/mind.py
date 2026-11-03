"""Разум: восприятие + язык + знания + рассуждение + новые идеи.

Знания не вводятся вручную — они извлекаются из опыта:
- «мне говорили» — из разобранных фраз родителя;
- «я видела» — из событий, замеченных у объектов категории (собака лает, пингвин
  в воде, но ни разу не в воздухе);
- «мне показали» — родитель указал на рыбу и сказал «живое»;
- «по смыслу слов» — всё, что называют «рыба», входит в то, что называют «живое».

Рассуждение — поиск по этим знаниям с объяснением каждого шага:
1. Прямое знание о самом предмете (сказанное или увиденное) — сильнее всего.
2. Наследование: ближайший род, о котором что-то известно (пингвин → птица → животное
   → дышит). Ближайшее важнее дальнего — так работают исключения.
3. Индукция — новая идея: «все известные мне животные дышат → наверное, все дышат»;
   «он такой же, как его соседи по роду».
4. Закрытый мир: «никаких причин считать, что мяч дышит, нет» — осторожное «нет».
5. Иначе — честное «не знаю».

Каждый вывод хранит цепочку шагов — система может объяснить, как к нему пришла.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from .language import NEG, Lexicon, Utterance, tokenize
from .perception import CategoryMemory


def _pair() -> list[int]:
    return [0, 0]


@dataclass
class Episode:
    """След пережитого: что было видно, что происходило, что сказали, на что показали."""
    looks: list
    events: list
    text: str
    focus: int | None


@dataclass
class Verdict:
    value: bool | None
    conf: float
    chain: list[str] = field(default_factory=list)
    source: str = "не знаю"          # слышала | видела | вывод | гипотеза | по умолчанию | противоречие
    conflict: str | None = None

    def answer(self, threshold: float = 0.5) -> bool | None:
        return self.value if self.conf >= threshold else None

    def say(self) -> str:
        if self.value is None or self.conf < 0.5:
            return "не знаю"
        return ("да" if self.value else "нет") + ("" if self.conf >= 0.75 else ", наверное")


def _neg(v: bool) -> str:
    return "" if v else "не "


class Mind:
    def __init__(self, dim: int, sleep_every: int = 200):
        self.perception = CategoryMemory(dim)
        self.lexicon = Lexicon()
        self.cat_obs: Counter = Counter()
        self.cat_events: dict[int, Counter] = defaultdict(Counter)
        self.told_prop: dict[tuple[str, str], list[int]] = defaultdict(_pair)
        self.told_isa: dict[tuple[str, str], list[int]] = defaultdict(_pair)
        self.shown_isa: Counter = Counter()
        self.sleep_every = sleep_every
        self.experiences = 0
        self.ideas: list[str] = []        # новые выводы и гипотезы, найденные в размышлениях
        self.insights: list[str] = []     # обнаруженные исключения и противоречия
        self.growth: list[str] = []       # как росла структура
        self.episodes: list[Episode] = []  # эпизодическая память (гиппокамп)
        self.corrections: list[str] = []   # где я ошибалась (по ответам учителя)
        self.doubts: list[str] = []        # чему не поверила на слово
        self._disputed: set = set()
        self._version = 0                 # меняется при любом изменении знаний
        self._cache: dict = {}
        self._noted: set = set()

    # ================= опыт =================
    def experience(self, scene: list[tuple[np.ndarray, set]], text: str = "",
                   focus: int | None = None) -> list[int]:
        """Увидеть сцену (облик и события объектов) и, возможно, услышать фразу."""
        cats = []
        for look, ev in scene:
            c, new = self.perception.categorize(look)
            if new:
                self.growth.append(f"опыт {self.experiences}: новая категория #{c}")
            cats.append(c)
            self.cat_obs[c] += 1
            for e in ev:
                self.cat_events[c][e] += 1
        self.episodes.append(Episode([lk for lk, _ in scene], [frozenset(ev) for _, ev in scene],
                                     text, focus))
        if text:
            u = self._utterance(self.episodes[-1], cats)
            self.lexicon.utts.append(u)
            if not scene:
                self._learn_now(u.tokens)
            elif len(scene) == 1 and focus == 0:
                self._fast_map(u.tokens, cats[0])
        self.experiences += 1
        if self.sleep_every and self.experiences % self.sleep_every == 0:
            self.sleep()
        return cats

    @staticmethod
    def _utterance(ep: "Episode", cats: list[int]) -> Utterance:
        events = frozenset().union(*ep.events) if ep.events else frozenset()
        fc = cats[ep.focus] if ep.focus is not None else None
        fe = ep.events[ep.focus] if ep.focus is not None else frozenset()
        return Utterance(tokenize(ep.text), frozenset(cats), events, fc, fe)

    def _learn_now(self, tokens: tuple[str, ...]) -> tuple | None:
        """Сказанное усваивается сразу, не дожидаясь сна. Если это расходится с тем, что я
        думала, — отмечаю свою ошибку (осознание собственных выводов)."""
        f = self.lexicon.parse(tokens) or self._bootstrap(tokens)
        if not f:
            return None
        kind, x, y, pol = f
        say = lambda v: (f"{x} — это {y}" if v else f"{x} не {y}") if kind == "isa" \
            else f"{x} {'' if v else 'не '}{y}"  # noqa: E731
        before = self.judge(x, y) if kind == "prop" else self.judge_isa(x, y)
        if before.value is not None and before.value != pol and before.source == "видела" \
                and before.conf >= 0.85 and (x, y, pol) not in self._disputed:
            # сказанное противоречит надёжному собственному наблюдению — не верю на слово
            self._disputed.add((x, y, pol))
            self.doubts.append(f"мне сказали «{say(pol)}», но {before.chain[0]}. "
                               f"Это расходится с моими наблюдениями — подтвердите, пожалуйста")
            return ("doubt",) + f[1:]
        if before.value is not None and before.conf >= 0.5 and before.value != pol \
                and before.source != "слышала":
            note = " (приняла, хотя это противоречит тому, что я видела)" if before.source == "видела" else ""
            self.corrections.append(f"я думала: {say(before.value)} ({before.source}: "
                                    f"{'; '.join(before.chain[:2])}), а оказалось: {say(pol)}{note}")
        (self.told_prop if kind == "prop" else self.told_isa)[(x, y)][0 if pol else 1] += 1
        self.touch()
        return f

    def _bootstrap(self, tokens: tuple[str, ...]) -> tuple | None:
        """Новые слова по устройству фразы (как дети): «кит это животное» — раз после «это»
        стоит знакомое существительное, то «кит» — тоже существительное."""
        lx = self.lexicon
        core = [t for t in tokens if t not in lx.function or t == "это"]
        has_eto = "это" in core
        core = [t for t in core if t != "это"]
        pol = True
        if len(core) == 3 and core[1] == NEG:
            pol, core = False, [core[0], core[2]]
        if len(core) != 2:
            return None
        x, y = core
        if has_eto or y in lx.nouns:
            for w in (x, y):
                if w not in lx.nouns:
                    lx.nouns.add(w)
                    self.growth.append(f"новое слово из текста: «{w}» — название чего-то")
            return ("isa", x, y, pol) if x != y else None
        if x in lx.nouns or y in lx.predicates:
            if x not in lx.nouns:
                lx.nouns.add(x)
                self.growth.append(f"новое слово из текста: «{x}» — название чего-то")
            if y not in lx.predicates:
                lx.predicates.add(y)
                self.growth.append(f"новое слово из текста: «{y}» — то, что делают или чем являются")
            return ("prop", x, y, pol)
        return None

    def learn_text(self, text: str) -> list[str]:
        """Прочитать текст (без сцены): по предложению; вернуть, что понято."""
        out = []
        for sent in re.split(r"[.!\n;]+", text):
            toks = tokenize(sent)
            if not toks:
                continue
            self.episodes.append(Episode([], [], sent, None))
            self.lexicon.utts.append(Utterance(toks, frozenset(), frozenset(), None, frozenset()))
            f = self._learn_now(toks)
            if f and f[0] == "doubt":
                out.append("СОМНЕВАЮСЬ: " + self.doubts[-1])
            elif f:
                kind, x, y, pol = f
                out.append(f"{x} {'' if pol else 'не '}{'— это ' if kind == 'isa' else ''}{y}")
        return out

    def self_report(self) -> dict:
        """Что я знаю, откуда, насколько уверена; где ошибалась; чего не знаю."""
        known = []
        for (x, y), (pos, neg) in sorted(self.told_prop.items()):
            if pos != neg:
                known.append(f"{x} {'' if pos > neg else 'не '}{y} — мне говорили")
        for (x, y), (pos, neg) in sorted(self.told_isa.items()):
            if pos > neg:
                known.append(f"{x} — это {y} — мне говорили")
        inferred = []
        for x in sorted(self.lexicon.basic_nouns()):
            for p in sorted(self.predicates()):
                v = self.judge(x, p)
                if v.source in ("вывод", "видела") and v.value and v.conf >= 0.75:
                    inferred.append(f"{x} {p} — {v.source}, уверенность {v.conf:.2f}")
        return {"знаю": known, "вывела": inferred, "гипотезы": list(self.ideas),
                "исключения": list(self.insights), "ошибалась": list(self.corrections),
                "сомневаюсь": list(self.doubts),
                "рост": self.growth[-5:],
                "категорий": len(self.perception.ids()), "слов": len(self.lexicon.nouns | self.lexicon.predicates)}

    def touch(self) -> None:
        """Знания изменились — забыть закэшированную структуру понятий."""
        self._version += 1
        self._cache.clear()

    @contextmanager
    def supposing(self, kind: str, x: str, y: str, value: bool):
        """Мысленный эксперимент: «а что, если бы мне сказали, что …»."""
        table = self.told_prop if kind == "prop" else self.told_isa
        old = list(table[(x, y)]) if (x, y) in table else None
        table[(x, y)] = [1, 0] if value else [0, 1]
        self.touch()
        try:
            yield
        finally:
            if old is None:
                del table[(x, y)]
            else:
                table[(x, y)] = old
            self.touch()

    def _fast_map(self, tokens: tuple[str, ...], c: int) -> None:
        """Быстрое запоминание: один раз показали безымянную вещь и назвали — запомнила."""
        lx = self.lexicon
        core = [t for t in tokens if t not in lx.function]
        if len(core) == 1 and c not in lx.name_of and core[0] not in lx.function:
            w = core[0]
            lx.nouns.add(w)
            if lx.ext.get(w):                       # знакомое слово: значение расширяется
                lx.ext[w] = lx.ext[w] + [c]
            else:
                lx.ext[w], lx.coverage[w] = [c], 1.0
            lx.name_of[c] = w
            self.perception.pinned.add(c)
            self.growth.append(f"опыт {self.experiences}: новое слово «{w}» с одного раза")
            self.touch()

    def _replay(self) -> None:
        """Прокрутить всю эпизодическую память заново — уже с нынешними понятиями."""
        self.cat_obs, self.cat_events = Counter(), defaultdict(Counter)
        utts = []
        for ep in self.episodes:
            cats = [self.perception.categorize(lk, learn=False)[0] for lk in ep.looks]
            for c, ev in zip(cats, ep.events):
                self.cat_obs[c] += 1
                for e in ev:
                    self.cat_events[c][e] += 1
            if ep.text:
                utts.append(self._utterance(ep, cats))
        self.lexicon.utts = utts
        self.lexicon.analyze(force=True)

    def _split_by_words(self) -> bool:
        """Язык учит различать: если одну мою категорию взрослые упорно называют двумя
        разными словами («мяч» и «яблоко»), значит, это два разных сорта — делю."""
        lx = self.lexicon
        basic = lx.basic_nouns()
        by_cat: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        utts = iter(lx.utts)
        for ep in self.episodes:
            if not ep.text:
                continue
            u = next(utts)
            if u.focus is None:
                continue
            core = [t for t in u.tokens if t not in lx.function]
            if len(core) == 1 and core[0] in basic:
                by_cat[u.focus][core[0]].append(ep.looks[ep.focus])
        changed = False
        for c, words in by_cat.items():
            total = sum(len(v) for v in words.values())
            # делим, только если оба слова — конкретные имена именно этой категории
            # (а не родовое слово вроде «растение», которым зовут и деревья, и цветы)
            big = {w: v for w, v in words.items() if len(v) >= 4 and len(v) >= 0.15 * total
                   and lx.ext.get(w) == [c]}
            if len(big) < 2:
                continue
            order = sorted(big, key=lambda w: -len(big[w]))
            self.perception.protos[c] = np.mean(big[order[0]], 0)
            self.perception.pinned.add(c)
            for w in order[1:]:
                n = self.perception.add(np.mean(big[w], 0), len(big[w]))
                self.perception.pinned.add(n)
            self.growth.append(f"опыт {self.experiences}: категория #{c} разделена — "
                               f"её называют по-разному: {', '.join(order)}")
            changed = True
        return changed

    def sleep(self) -> list[str]:
        """Сон: слить случайные категории, прокрутить пережитое, разделить то, что язык
        называет по-разному, заново осмыслить услышанное и поразмышлять."""
        for a, b in self.perception.consolidate():
            self.growth.append(f"опыт {self.experiences}: категория #{a} слита с #{b}")
        self._replay()
        if self._split_by_words():
            self._replay()
        self._rebuild_knowledge()
        return self.reflect()

    def _rebuild_knowledge(self) -> None:
        lx = self.lexicon
        self.told_prop.clear()
        self.told_isa.clear()
        self.shown_isa.clear()
        self.touch()
        basic = lx.basic_nouns()
        for u in lx.utts:
            f = lx.parse(u.tokens)
            if f:
                kind, x, y, pol = f
                (self.told_prop if kind == "prop" else self.told_isa)[(x, y)][0 if pol else 1] += 1
                continue
            core = [t for t in u.tokens if t not in lx.function]
            if u.focus is not None and len(core) == 1 and core[0] in lx.nouns and core[0] not in basic:
                b = lx.name_of.get(u.focus)
                if b and b != core[0]:
                    self.shown_isa[(b, core[0])] += 1   # показали на рыбу и сказали «живое»

    # ================= структура понятий =================
    def nouns(self) -> set[str]:
        return set(self.lexicon.nouns)

    def predicates(self) -> set[str]:
        return set(self.lexicon.predicates)

    def parents(self, x: str) -> list[tuple[str, str, float]]:
        """(род, откуда я это знаю, уверенность)."""
        key = ("parents", x)
        if key not in self._cache:
            self._cache[key] = self._parents(x)
        return self._cache[key]

    def _parents(self, x: str) -> list[tuple[str, str, float]]:
        out: dict[str, tuple[str, float]] = {}
        for (a, b), (pos, neg) in self.told_isa.items():
            if a == x and pos > neg:
                out[b] = ("мне говорили", 0.97)
        ex = set(self.lexicon.ext.get(x, []))
        for (a, b), n in self.shown_isa.items():
            if a != x or b in out or self.told_isa.get((a, b), [0, 0])[1] > 0:
                continue
            # показали на это и назвали родовым словом: дважды — или раз, если и по смыслу
            # слов то, что зовут «x», входит в то, что зовут «b»
            inside = ex and ex <= set(self.lexicon.ext.get(b, [])) and len(self.lexicon.ext[b]) > len(ex)
            if n >= 2 or (n >= 1 and inside):
                out[b] = ("мне показывали", 0.85)
        return [(y, how, c) for y, (how, c) in out.items()]

    def children(self, y: str) -> list[str]:
        if "children" not in self._cache:
            kids: dict[str, list[str]] = defaultdict(list)
            for x in sorted(self.nouns()):
                for p, _, _ in self.parents(x):
                    kids[p].append(x)
            self._cache["children"] = kids
        return self._cache["children"].get(y, [])

    def descendants(self, y: str) -> set[str]:
        out, todo = set(), [y]
        while todo:
            for k in self.children(todo.pop()):
                if k not in out:
                    out.add(k)
                    todo.append(k)
        return out

    def ancestors(self, x: str) -> list[tuple[str, list]]:
        """Предки по уровням: [(род, путь)], путь — [(от, к, как, уверенность)]."""
        out, seen, level = [], {x}, [(x, [])]
        while level:
            nxt = []
            for node, path in level:
                for y, how, c in self.parents(node):
                    if y not in seen:
                        seen.add(y)
                        nxt.append((y, path + [(node, y, how, c)]))
            out.append(nxt)
            level = nxt
        return [lvl for lvl in out if lvl]

    # ================= прямое знание =================
    def told(self, x: str, p: str) -> Verdict | None:
        pos, neg = self.told_prop.get((x, p), (0, 0))
        if pos == neg:
            return None
        v = pos > neg
        return Verdict(v, 0.95, [f"мне говорили: {x} {_neg(v)}{p}"], "слышала")

    def seen(self, x: str, p: str) -> Verdict | None:
        """Своими глазами — только про конкретный вид, а не про род («живое», «птица»)."""
        e = self.lexicon.event_of.get(p)
        # только категории, которые носят именно это имя (а не те, с которыми их путают)
        cats = [c for c in self.lexicon.ext.get(x, []) if self.lexicon.name_of.get(c) == x]
        if not e or not cats or x not in self.lexicon.basic_nouns() or self.children(x):
            return None
        obs = sum(self.cat_obs[c] for c in cats)
        hits = sum(self.cat_events[c][e] for c in cats)
        if hits >= 3 and hits / max(obs, 1) >= 0.05:
            return Verdict(True, 0.9, [f"я видела, как {x} {p} ({hits} раз из {obs})"], "видела")
        if hits == 0 and obs >= 30:
            return Verdict(False, 0.85, [f"я видела {x} {obs} раз и ни разу — чтобы {x} {p}"], "видела")
        return None

    def direct(self, x: str, p: str) -> Verdict | None:
        t, s = self.told(x, p), self.seen(x, p)
        if t and s and t.value != s.value:
            t.conflict = f"мне говорили одно, а видела я другое: {s.chain[0]}"
        return t or s

    # ================= рассуждение =================
    def judge(self, x: str, p: str) -> Verdict:
        """Ответ на «{p} ли {x}?» с объяснением."""
        own = self.direct(x, p)
        if own:
            return own
        inherited = self.inherit(x, p)
        if inherited and inherited.value is not None:
            return inherited
        hyp = self.induce(x, p)
        if hyp:
            if inherited and inherited.conflict:
                hyp.conflict = inherited.conflict
            return hyp
        if inherited:                       # противоречие между родами
            return inherited
        if self.well_known(x) and self.holders(p):
            return Verdict(False, 0.6, [f"я не знаю ни одной причины, почему {x} {p}: мне этого не "
                                        f"говорили, я этого не видела, и у рода {x} этого нет"],
                           "по умолчанию")
        return Verdict(None, 0.0, [f"я ничего не знаю о том, {p} ли {x}"], "не знаю")

    def inherit(self, x: str, p: str) -> Verdict | None:
        for level in self.ancestors(x):
            found = [(y, path, self.direct(y, p)) for y, path in level]
            found = [f for f in found if f[2]]
            if not found:
                continue
            values = {f[2].value for f in found}
            if len(values) > 1:
                why = "; ".join(f"{y}: {_neg(v.value)}{p}" for y, _, v in found)
                return Verdict(None, 0.5, [f"роды {x} говорят разное ({why})"], "противоречие",
                               conflict=f"{x}: {why}")
            y, path, v = max(found, key=lambda f: f[2].conf)
            conf = v.conf * float(np.prod([c for *_, c in path])) * 0.98 ** len(path)
            chain = [f"{a} — это {b} ({how})" for a, b, how, _ in path] + v.chain
            chain.append(f"значит, {x} {_neg(v.value)}{p}")
            return Verdict(v.value, conf, chain, "вывод")
        return None

    def induce(self, x: str, p: str) -> Verdict | None:
        """Новая идея: обобщение по известным примерам."""
        all_kids = self.children(x)
        kids = [(k, self.direct(k, p)) for k in all_kids]
        kids = [(k, v) for k, v in kids if v]
        if len(kids) >= 2 and len(kids) >= 0.5 * len(all_kids) and len({v.value for _, v in kids}) == 1:
            v = kids[0][1].value
            names = ", ".join(k for k, _ in kids)
            return Verdict(v, min(0.8, 0.5 + 0.1 * len(kids)),
                           [f"все известные мне «{x}» ({names}) — «{_neg(v)}{p}»",
                            f"наверное, все «{x}» — «{_neg(v)}{p}»"], "гипотеза")
        if len(kids) >= 3:
            # «в основном»: обычно живые не плавают — кроме рыб и пингвинов
            yes = [k for k, v in kids if v.value]
            no = [k for k, v in kids if not v.value]
            major, minor = (yes, no) if len(yes) > len(no) else (no, yes)
            share = len(major) / len(kids)
            if share >= 0.75:
                v = bool(major is yes)
                return Verdict(v, 0.5 + 0.3 * share,
                               [f"из известных мне «{x}» {len(major)} из {len(kids)} — «{_neg(v)}{p}» "
                                f"(кроме: {', '.join(minor)})",
                                f"обычно «{x}» — «{_neg(v)}{p}»"], "гипотеза")
        for y, _, _ in self.parents(x):
            all_sibs = [k for k in self.children(y) if k != x]
            sibs = [(k, self.direct(k, p)) for k in all_sibs]
            sibs = [(k, v) for k, v in sibs if v]
            # аналогия с соседями по роду — только если про большинство соседей известно
            if len(sibs) >= 2 and len(sibs) >= 0.5 * len(all_sibs) and len({v.value for _, v in sibs}) == 1:
                v = sibs[0][1].value
                names = ", ".join(k for k, _ in sibs)
                return Verdict(v, 0.6, [f"{x} — это {y}, как и {names}",
                                        f"про них известно: «{_neg(v)}{p}»",
                                        f"наверное, и {x} {_neg(v)}{p}"], "гипотеза")
        return None

    def well_known(self, x: str) -> bool:
        return bool(self.parents(x)) or sum(self.cat_obs[c] for c in self.lexicon.ext.get(x, [])) >= 20

    def holders(self, p: str) -> bool:
        return any(k[1] == p and v[0] > v[1] for k, v in self.told_prop.items()) \
            or p in self.lexicon.event_of

    def judge_isa(self, x: str, y: str) -> Verdict:
        pos, neg = self.told_isa.get((x, y), (0, 0))
        if neg > pos:
            return Verdict(False, 0.95, [f"мне говорили: {x} не {y}"], "слышала")
        for level in self.ancestors(x):
            for node, path in level:
                if node == y:
                    conf = float(np.prod([c for *_, c in path])) * 0.98 ** len(path)
                    chain = [f"{a} — это {b} ({how})" for a, b, how, _ in path]
                    return Verdict(True, conf, chain + [f"значит, {x} — это {y}"],
                                   "слышала" if len(path) == 1 and path[0][2] == "мне говорили" else "вывод")
        members = [k for k in self.children(x) if self.reaches(k, y)]
        if members:
            # догадка о строении мира: «все известные мне птицы — животные»
            others = [k for k in self.children(x) if k not in members and self.well_known(k)]
            if len(members) >= len(others):
                return Verdict(True, 0.6 + 0.05 * min(len(members), 4),
                               [f"известные мне «{x}» ({', '.join(members)}) — это «{y}»",
                                f"наверное, и «{x}» — это «{y}»"], "гипотеза")
        if self.well_known(x) and self.children(y):
            return Verdict(False, 0.6, [f"я не знаю, чтобы {x} был(а) {y}: среди {y} я знаю "
                                        f"{', '.join(sorted(self.children(y))[:4])}"], "по умолчанию")
        return Verdict(None, 0.0, [f"я не знаю, {y} ли {x}"], "не знаю")

    def reaches(self, x: str, y: str) -> bool:
        return any(node == y for level in self.ancestors(x) for node, _ in level)

    def ask(self, x: str, y: str) -> Verdict:
        """Универсальный вопрос: y — род (существительное) или свойство (сказуемое)."""
        return self.judge_isa(x, y) if y in self.nouns() else self.judge(x, y)

    # ================= узнавание =================
    def identify(self, look: np.ndarray) -> tuple[str | None, str | None, float]:
        """→ (имя, если знаю; на что похоже, если не знаю; насколько похоже).

        Похожесть «на что-то знакомое» мягче, чем порог «это оно самое»."""
        c, d = self.perception.nearest(look)
        if c is None:
            return None, None, 0.0
        sure = d <= self.perception.radius or \
            self.perception.partial_distance(look, self.perception.prototype(c)) <= self.perception.radius
        if not sure and d <= 1.6 * self.perception.radius and self.lexicon.name_of.get(c):
            return self.lexicon.name_of[c], None, float(np.exp(-d))     # «наверное, это …»
        if sure:
            name = self.lexicon.name_of.get(c)
            if name:
                return name, None, self.perception.similarity(d)
            c2, d2 = self.nearest_named(self.perception.prototype(c), exclude={c})
            return None, self.lexicon.name_of.get(c2), float(np.exp(-d2))
        c2, d2 = self.nearest_named(look)
        return None, self.lexicon.name_of.get(c2), float(np.exp(-d2))

    def nearest_named(self, x: np.ndarray, exclude=frozenset()) -> tuple[int | None, float]:
        named = {c for c in self.lexicon.name_of if c not in exclude and self.perception.alive[c]}
        d = {c: float(np.linalg.norm(self.perception.prototype(c) - x)) for c in named}
        if not d:
            return None, float("inf")
        c = min(d, key=d.get)
        return c, d[c]

    def judge_thing(self, look: np.ndarray, p: str) -> Verdict:
        """«{p} ли вот это?» — о конкретном увиденном объекте."""
        name, like, sim = self.identify(look)
        if name:
            v = self.judge(name, p) if p not in self.nouns() else self.judge_isa(name, p)
            if sim >= 0.4:
                v.chain = [f"это {name} (узнала по виду)"] + v.chain
            else:
                v.chain = [f"наверное, это {name} (похоже, но не совсем)"] + v.chain
                v.conf *= 0.8
                v.source = "гипотеза"
            return v
        if like:
            # не знаю, что это, но похоже на like: переносим то, что верно для его рода
            for y, how, c in self.parents(like):
                v = self.judge(y, p) if p not in self.nouns() else self.judge_isa(y, p)
                if v.value is not None:
                    return Verdict(v.value, min(v.conf, 0.75) * (0.7 + 0.3 * sim),
                                   [f"я не знаю, что это, но оно похоже на {like} (сходство {sim:.2f})",
                                    f"{like} — это {y}; наверное, и это — {y}"] + v.chain, "гипотеза")
        return Verdict(None, 0.0, ["я не знаю, что это"], "не знаю")

    # ================= размышление (новые идеи) =================
    def reflect(self) -> list[str]:
        """Поискать новые обобщения и исключения в собственных знаниях.

        Идеи каждый раз пересматриваются: гипотеза, которую опровергли новые факты,
        исчезает. Возвращает то, что появилось впервые."""
        ideas, insights = {}, {}
        for y in sorted(self.nouns()):
            for p in sorted(self.predicates()):
                if self.direct(y, p) or self.inherit(y, p):
                    continue                       # и так известно — не новость
                h = self.induce(y, p)
                if h and h.value and h.chain[0].startswith("все известные"):
                    ideas[(y, p)] = f"{h.chain[-1]} — потому что {h.chain[0]}"
        for x in sorted(self.lexicon.basic_nouns()):
            for p in sorted(self.predicates()):
                own = self.direct(x, p)
                inh = self.inherit(x, p)
                if own and inh and inh.value is not None and own.value != inh.value:
                    gen = inh.chain[-2] if len(inh.chain) >= 2 else inh.chain[0]
                    insights[(x, p)] = f"{x} {_neg(own.value)}{p}, хотя {gen} ({own.chain[0]})"
        new = ["новая идея: " + v for k, v in ideas.items() if ("idea",) + k not in self._noted]
        new += ["исключение: " + v for k, v in insights.items() if ("exc",) + k not in self._noted]
        self._noted |= {("idea",) + k for k in ideas} | {("exc",) + k for k in insights}
        self.ideas, self.insights = list(ideas.values()), list(insights.values())
        return new

    # ================= для сравнения: «зубрила» =================
    def literal(self, x: str, y: str) -> bool | None:
        """Ответ только по дословно услышанному — без всякого рассуждения."""
        for u in self.lexicon.utts:
            core = [t for t in u.tokens if t not in self.lexicon.function]
            if core == [x, y]:
                return True
            if core == [x, NEG, y]:
                return False
        return None


def tokens_of(text: str) -> tuple[str, ...]:
    return tokenize(text)
