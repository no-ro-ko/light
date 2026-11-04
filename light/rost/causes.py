"""Причины из обычного текста: «если …, то …», «когда …, …», «… потому что …», «… поэтому …».

Каждая такая фраза — связь «положение дел → следствие». Положение дел понимается так же, как
любая фраза: кто (понятие) и что с ним (сказуемое, может быть с «не»). Если разобрать не удалось
(«идёт дождь» — сказуемое раньше подлежащего), положение запоминается как набор слов.

Вопросы:
- «Почему земля мокрая?» → находится связь, где это — следствие; дальше — причина причины;
- «Что будет, если идёт дождь?» → следствия и следствия следствий;
- общее применяется к частному через роды: «если животное не ест, оно слабеет» + «кошка —
  животное» → «если кошка не ест, она, наверное, ослабеет».
"""
from __future__ import annotations

import re
from dataclasses import dataclass

PATTERNS = [
    (re.compile(r"^если\s+(.+?),?\s+то\s+(.+)$"), "cause_effect"),
    (re.compile(r"^если\s+(.+?),\s+(.+)$"), "cause_effect"),
    (re.compile(r"^когда\s+(.+?),\s+(.+)$"), "cause_effect"),
    (re.compile(r"^(.+?),?\s+потому что\s+(.+)$"), "effect_cause"),
    (re.compile(r"^(.+?),?\s+так как\s+(.+)$"), "effect_cause"),
    (re.compile(r"^(.+?),?\s+поэтому\s+(.+)$"), "cause_effect"),
]


@dataclass(frozen=True)
class Sit:
    subject: str | None
    pred: str | None
    neg: bool
    bag: frozenset
    text: str


@dataclass
class Link:
    cause: Sit
    effect: Sit
    source: str


class CauseField:
    def __init__(self):
        self.links: list[Link] = []
        self.journal: list[str] = []

    @staticmethod
    def split(sentence: str) -> tuple[str, str] | None:
        s = sentence.lower().replace("ё", "е").strip(" .!")
        for rx, order in PATTERNS:
            m = rx.match(s)
            if m:
                a, b = m.group(1).strip(" ,"), m.group(2).strip(" ,")
                return (a, b) if order == "cause_effect" else (b, a)
        return None

    def learn(self, cause: Sit, effect: Sit, source: str) -> str:
        self.links.append(Link(cause, effect, source))
        note = f"причина → следствие: «{cause.text}» → «{effect.text}»"
        self.journal.append(note)
        return note

    # ---------- сопоставление положений дел ----------
    @staticmethod
    def matches(q: Sit, known: Sit, isa) -> tuple[bool, bool]:
        """→ (подходит ли, через род ли — тогда «наверное»)."""
        if q.pred and known.pred and q.neg == known.neg and (
                q.pred == known.pred or q.pred.startswith(known.pred + " ") or known.pred.startswith(q.pred + " ")):
            if q.subject == known.subject or known.subject is None:
                return True, False
            if q.subject and known.subject and isa(q.subject, known.subject):
                return True, True
        if q.bag and known.bag and min(len(q.bag), len(known.bag)) >= 2:   # одно общее слово — мало
            common = len(q.bag & known.bag) / min(len(q.bag), len(known.bag))
            if common >= 0.75 and q.neg == known.neg:
                return True, False
        return False, False

    def substitute(self, known: Sit, q: Sit, name) -> str:
        """Перенести общее на частное: «животное слабеет» → «кошка слабеет»."""
        if q.subject and known.subject and q.subject != known.subject and known.text:
            return f"{name(q.subject)} {'не ' if known.neg else ''}{name(known.pred) if known.pred else known.text}"
        return known.text

    def why(self, q: Sit, isa, name, depth: int = 4) -> list[str]:
        chain, cur, seen = [], q, set()
        for _ in range(depth):
            hit = None
            for link in self.links:
                ok, by_kind = self.matches(cur, link.effect, isa)
                if ok and id(link) not in seen:
                    hit = (link, by_kind)
                    break
            if not hit:
                break
            link, by_kind = hit
            seen.add(id(link))
            chain.append(f"«{link.effect.text}», потому что «{link.cause.text}» (в «{link.source}»)"
                         + (" — так сказано о роде, к которому это относится" if by_kind else ""))
            cur = link.cause
        return chain

    def what_if(self, q: Sit, isa, name, depth: int = 4) -> list[str]:
        out, frontier, seen = [], [(q, False)], set()
        for _ in range(depth):
            nxt = []
            for cur, guessed in frontier:
                for link in self.links:
                    ok, by_kind = self.matches(cur, link.cause, isa)
                    if ok and id(link) not in seen:
                        seen.add(id(link))
                        text = self.substitute(link.effect, cur, name) if by_kind else link.effect.text
                        maybe = by_kind or guessed
                        out.append(("наверное, " if maybe else "") + f"{text} (в «{link.source}»: "
                                   f"если «{link.cause.text}», то «{link.effect.text}»)")
                        eff = link.effect
                        if by_kind and cur.subject:
                            eff = Sit(cur.subject, eff.pred, eff.neg, eff.bag, text)
                        nxt.append((eff, maybe))
            if not nxt:
                break
            frontier = nxt
        return out
