"""Смысл сюжета: кто что сделал, ЗАЧЕМ, ПОЧЕМУ так вышло, ЧТО БЫЛО БЫ, ЕСЛИ…, и мораль.

Каждое событие хранится в нейронной памяти как код «агент⊗A + действие⊗D + объект⊗O».

1. Причинные связи — из других историй (с другими героями). Для каждого события
   сеть ищет, откуда взялись его участники: из предыдущего события (повод) или из
   более раннего (условие). Устойчивая схема, повторившаяся в разных историях, — это
   правило с ПЕРЕМЕННЫМИ: «X хвалит Y → Y радуется», «Y поёт, а Y держит Z → Z падает».
   Конкретные герои при этом отбрасываются — остаётся смысл связи.
2. «Зачем X сделал A?» — понимание намерения: у X есть желание («лиса хочет сыр»);
   сеть мысленно разворачивает последствия A по своим правилам и проверяет, ведут
   ли они к исполнению желания. Если да — A было средством.
3. «Почему случилось E?» — обратная цепочка причин до первого поступка.
4. «Что было бы, если бы E не случилось?» — пересчёт последствий без E.
5. Мораль — обобщение понятого намерения: конкретных героев заменяем ролями.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from .core import EventMemory, Space


@dataclass(frozen=True)
class Ev:
    agent: str
    action: str
    obj: str | None = None

    def say(self) -> str:
        return " ".join(w for w in (self.agent, self.action, self.obj) if w)


def _common(a: str, b: str) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


class StoryMind:
    def __init__(self, dim: int = 2048, seed: int = 0):
        self.space = Space(dim, seed)
        self.mem = EventMemory(self.space)
        self.entities: list[str] = []
        self.verbs: list[str] = []
        self.support: Counter = Counter()
        self.examples: dict = defaultdict(list)
        self.definitions: dict[str, str] = {}
        self.forms: dict[tuple[str, str], str] = {}
        self.trigger_count: Counter = Counter()
        self.train_stories: list[list[Ev]] = []
        self._rel: dict = {}
        self._applicable: dict = {}

    # ---------- чтение ----------
    def _norm(self, w: str, pool: list[str], role: str | None = None) -> str:
        """Свести формы слова: «ворону» → «ворона», «рыбу» → «рыба», «хвалила» → «хвалит».
        Падеж меняет окончание — общая часть должна быть почти всем словом."""
        for c in pool:
            n = _common(w, c)
            if w == c or (n >= 3 and n >= min(len(w), len(c)) - 1 and n >= 0.6 * max(len(w), len(c))):
                if role:
                    self.forms[(c, role)] = w
                return c
        pool.append(w)
        if role:
            self.forms[(w, role)] = w
        return w

    def parse(self, sentence: str) -> Ev | None:
        t = re.findall(r"\w+", sentence.lower())
        if len(t) not in (2, 3):
            return None
        agent = self._norm(t[0], self.entities, "агент")
        action = self._norm(t[1], self.verbs)
        obj = self._norm(t[2], self.entities, "объект") if len(t) == 3 else None
        return Ev(agent, action, obj)

    def say(self, e: Ev) -> str:
        """Событие словами — с нужными падежами («хвалит ворону»)."""
        obj = self.forms.get((e.obj, "объект"), e.obj) if e.obj else None
        return " ".join(w for w in (e.agent, e.action, obj) if w)

    def read(self, sentences: list[str]) -> list[Ev]:
        """Прочитать историю: каждое событие — в нейронную память; вернуть, что понято."""
        story = []
        for s in sentences:
            e = self.parse(s)
            if e:
                self.mem.store(self.mem.event(e.agent, e.action, e.obj))
                story.append(self.recall(e))
        return story

    def recall(self, e: Ev) -> Ev:
        """Проверка, что смысл события действительно в памяти: найти след по намёку
        «кто и что делал» и развязать его по ролям."""
        probe = self.space.bundle([self.space.bind(self.space.role("агент"), self.space.code(e.agent)),
                                   self.space.bind(self.space.role("действие"), self.space.code(e.action))])
        i, _ = self.mem.recall(probe, k=1)[0]
        d = self.mem.decode(self.mem.items[i])
        return Ev(d["агент"], d["действие"], d["объект"])

    # ---------- причинные связи из других историй ----------
    @staticmethod
    def _roles(e: Ev) -> dict[str, str]:
        return {"агент": e.agent, "объект": e.obj} if e.obj else {"агент": e.agent}

    def learn(self, stories: list[list[str]]) -> list[str]:
        for st in stories:
            evs = self.read(st)
            self.train_stories.append(evs)
            for t in range(len(evs)):
                self.trigger_count[evs[t].action] += 1      # сколько раз вообще бывал такой повод
            for t in range(1, len(evs)):
                sig = self._signature(evs[:t], evs[t])
                if sig:
                    self.support[sig] += 1
                    self.examples[sig].append((evs[t - 1], evs[t]))
        self._rel.clear()
        return self.rules_text()

    def reliability(self, sig) -> float:
        """«После — не значит вследствие»: во всех историях, где повод был И условие
        выполнялось, — как часто следствие действительно наступало."""
        if sig in self._rel:
            return self._rel[sig]
        applicable = success = 0
        for evs in self.train_stories:
            for t in range(len(evs)):
                pred = self._fire(sig, evs[t], set(evs[:t]))
                if pred is not None:
                    applicable += 1
                    success += t + 1 < len(evs) and evs[t + 1] == pred
        self._rel[sig] = success / max(applicable, 1)
        self._applicable[sig] = applicable
        return self._rel[sig]

    def _signature(self, before: list[Ev], effect: Ev):
        trigger = before[-1]
        tr = self._roles(trigger)
        maps, cond = [], None
        for role, filler in self._roles(effect).items():
            src = next((("повод", r) for r, f in tr.items() if f == filler), None)
            if src is None:
                for e in reversed(before[:-1]):
                    er = self._roles(e)
                    r = next((r for r, f in er.items() if f == filler), None)
                    if r:
                        link = next((r2 for r2, f2 in er.items() if f2 in tr.values() and r2 != r), None)
                        link_t = next((rt for rt, ft in tr.items() if link and ft == er[link]), None)
                        cond = (e.action, link, link_t)
                        src = ("условие", r)
                        break
            if src is None:
                return None                     # участник взялся ниоткуда — это не следствие
            maps.append((role, src))
        return (trigger.action, cond, effect.action, tuple(sorted(maps)))

    def rules(self, min_support: int = 2, min_reliability: float = 0.6) -> list[tuple]:
        return [s for s, n in self.support.items() if n >= min_support and self.reliability(s) >= min_reliability]

    def rejected(self) -> list[str]:
        """Совпадения, которые я НЕ считаю причинами (бывало и по-другому)."""
        return [f"{s[0]} → {s[2]}: так было {self.support[s]} раз из {self._applicable[s]} — "
                f"это совпадение, а не причина" for s, n in self.support.items()
                if n >= 2 and self.reliability(s) < 0.6]

    def rules_text(self) -> list[str]:
        """Правило словами с переменными: героев первого примера заменяю буквами."""
        out = []
        for sig in self.rules():
            ex = None
            for evs in self.train_stories:
                for i in range(len(evs) - 1):
                    if self._fire(sig, evs[i], set(evs[:i])) == evs[i + 1]:
                        cond = next((e for e in evs[:i] if sig[1] and e.action == sig[1][0]), None)
                        ex = (evs[i], cond, evs[i + 1])
                        break
                if ex:
                    break
            if not ex:
                continue
            letters: dict[str, str] = {}
            name = lambda w: letters.setdefault(w, "XYZW"[len(letters)])  # noqa: E731
            show = lambda e: " ".join([name(e.agent), e.action] + ([name(e.obj)] if e.obj else []))  # noqa: E731
            lhs = show(ex[0]) + (f", а {show(ex[1])}" if ex[1] else "")
            out.append(f"{lhs} → {show(ex[2])}  (сбылось {self.support[sig]} из {self._applicable[sig]} раз)")
        return out

    # ---------- применение правил ----------
    def _fire(self, rule, trigger: Ev, state: set[Ev]) -> Ev | None:
        trig, cond, eff, maps = rule
        if trigger.action != trig:
            return None
        tr = self._roles(trigger)
        cond_ev = None
        if cond:
            caction, clink, tlink = cond
            for e in state:
                if e.action == caction and (clink is None or self._roles(e).get(clink) == tr.get(tlink)):
                    cond_ev = e
                    break
            if cond_ev is None:
                return None
        filled = {}
        for role, (where, r) in maps:
            src = tr if where == "повод" else self._roles(cond_ev)
            if r not in src:
                return None
            filled[role] = src[r]
        if "агент" not in filled:
            return None
        return Ev(filled["агент"], eff, filled.get("объект"))

    def consequences(self, start: Ev, state: set[Ev], blocked: set[Ev] = frozenset(),
                     depth: int = 8) -> list[tuple[Ev, Ev | None]]:
        """Мысленно развернуть последствия: [(событие, при каком условии)]."""
        chain, cur, state = [], start, set(state) | {start}
        for _ in range(depth):
            step = None
            for rule in self.rules():
                nxt = self._fire(rule, cur, state)
                if nxt and nxt not in blocked and nxt not in state:
                    cond = None
                    if rule[1]:
                        cond = next((e for e in state if e.action == rule[1][0]), None)
                    step = (nxt, cond)
                    break
            if not step:
                break
            chain.append(step)
            state.add(step[0])
            cur = step[0]
        return chain

    # ---------- вопросы о смысле ----------
    def why_did(self, story: list[Ev], agent: str, action: str, obj: str | None = None) -> str:
        """Зачем агент сделал это? — найти желание, к которому ведёт поступок."""
        agent, action = self._norm(agent, self.entities), self._norm(action, self.verbs)
        obj = self._norm(obj, self.entities) if obj else None
        act = next((e for e in story if e.agent == agent and e.action == action
                    and (obj is None or e.obj == obj)), None)
        if not act:
            return "в истории такого не было"
        desires = [e for e in story if e.agent == agent and e.action.startswith("хоч")]
        state = {e for e in story if story.index(e) < story.index(act)}
        chain = self.consequences(act, state)
        for d in desires:
            for i, (e, _) in enumerate(chain):
                if e.agent == agent and e.obj == d.obj:
                    steps = [self.say(act)] + [self.say(ev) + (f" (ведь {self.say(c)})" if c else "")
                                               for ev, c in chain[:i + 1]]
                    self.last_motive = (act, d, chain[:i + 1])
                    return (f"чтобы получить {d.obj}: {agent} хочет {d.obj}, и «{self.say(act)}» ведёт "
                            f"к этому: " + " → ".join(steps))
        return f"не понимаю, зачем: не вижу, к какому желанию «{agent}» это ведёт"

    def why_happened(self, story: list[Ev], agent: str, action: str) -> str:
        """Почему случилось? — обратная цепочка причин до первого поступка."""
        target = next((e for e in story if e.agent == self._norm(agent, self.entities)
                       and e.action == self._norm(action, self.verbs)), None)
        if not target:
            return "такого не было"
        causes = []
        cur = target
        for _ in range(8):
            before = story[:story.index(cur)]
            found = None
            for prev in reversed(before):
                for rule in self.rules():
                    if self._fire(rule, prev, set(before)) == cur:
                        cond = next((e for e in before if rule[1] and e.action == rule[1][0]), None)
                        found = (prev, cond)
                        break
                if found:
                    break
            if not found:
                break
            causes.append(f"{self.say(cur)} — потому что {self.say(found[0])}"
                          + (f", а {self.say(found[1])}" if found[1] else ""))
            cur = found[0]
        if not causes:
            return "не вижу причины"
        return "; ".join(causes) + f". А началось всё с того, что {self.say(cur)}"

    def what_if_not(self, story: list[Ev], agent: str, action: str) -> str:
        """Что было бы, если бы этого не случилось? — пересчитать последствия без него."""
        removed = next((e for e in story if e.agent == self._norm(agent, self.entities)
                        and e.action == self._norm(action, self.verbs)), None)
        if not removed:
            return "такого и не было"
        # исходные события — те, что не следуют из других по правилам
        derived = set()
        for i, e in enumerate(story[1:], 1):
            if any(self._fire(r, story[i - 1], set(story[:i])) == e for r in self.rules()):
                derived.add(e)
        roots = [e for e in story if e not in derived]
        world = set(roots)
        for r in roots:
            for e, _ in self.consequences(r, world, blocked={removed}):
                world.add(e)
        lost = [self.say(e) for e in story if e not in world and e != removed]
        return ("ничего бы не изменилось" if not lost else
                f"тогда не было бы: {', '.join(lost)}")

    def moral(self) -> str:
        """Обобщить понятое намерение: героев — в роли."""
        if not getattr(self, "last_motive", None):
            return "я ещё не поняла ничьих намерений"
        act, desire, chain = self.last_motive
        holder = next((c.agent for _, c in chain if c and c.obj == desire.obj), "другой")
        roles = {act.agent: "кто-то", holder: "другого", desire.obj: "то, что есть у другого"}
        schema = f"{roles[act.agent]} {act.action} {roles.get(act.obj, act.obj)}, чтобы получить {roles[desire.obj]}"
        text = f"обобщаю: {schema}"
        for word, meaning in self.definitions.items():
            if act.action[:4] in meaning and ("выгод" in meaning or "получ" in meaning):
                text += f". По определению это «{word}» ({meaning})"
        return text + ". Значит: похвала может быть средством, а не правдой — не верь тому, кто хвалит ради своей выгоды"


TRAINING = [
    # держат — и делают разное: «держит, потом поёт» — не причина, а совпадение
    ["папа держит чашку", "папа читает"],
    ["кошка держит мышь", "кошка бежит"],
    ["дедушка держит газету", "дедушка спит"],
    # хотят — но ничего не падает
    ["мальчик хочет сок", "мальчик просит сок"],
    ["девочка хочет куклу", "девочка плачет"],
    ["мама хвалит сына", "сын радуется"],
    ["учитель хвалит ученика", "ученик радуется", "ученик поёт"],
    ["бабушка хвалит внучку", "внучка радуется", "внучка поёт"],
    ["птица радуется", "птица поёт"],
    ["девочка держит конфету", "девочка поёт", "конфета падает"],
    ["мальчик держит ложку", "мальчик поёт", "ложка падает"],
    ["кот хочет рыбу", "рыба падает", "кот берёт рыбу"],
    ["собака хочет кость", "кость падает", "собака берёт кость"],
]
FABLE = ["лиса хочет сыр", "ворона держит сыр", "лиса хвалит ворону", "ворона радуется",
         "ворона поёт", "сыр падает", "лиса берёт сыр"]
