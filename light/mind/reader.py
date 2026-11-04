"""Чтение связного текста (в том числе классической литературы) — честно.

Что система МОЖЕТ сделать своими силами (без подключённой большой языковой модели):
- найти действующих лиц (имена с заглавной буквы в середине предложения, сведённые по
  общему корню: «Ворона», «Вороне», «Ворону» — одно лицо);
- понять, кто с кем встречается (в одних предложениях), в каком порядке появляются;
- найти частые предметы, вокруг которых идёт действие;
- найти общие высказывания без действующих лиц (в басне это часто мораль);
- выучить то, что сказано простыми утверждениями («кит — это животное»).

Чего она НЕ может (и сама об этом говорит): понять смысл художественной речи — мотивы,
иронию, мораль как вывод из сюжета. Сколько предложений она смогла разобрать, а сколько
нет, — она сообщает сама. Для настоящего понимания литературы в архитектуре предусмотрено
подключение языковой модели к узлам (light.adapters.HTTPChatAdapter); здесь это не проверялось.
"""
from __future__ import annotations

import re
from collections import Counter
from itertools import combinations

WORD = re.compile(r"[А-ЯЁA-Z][а-яёa-z]+|[а-яёa-z]+")


def _root(w: str) -> str:
    w = w.lower().replace("ё", "е")
    return w[:max(4, len(w) - 3)] if len(w) > 5 else w[:4]


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?;…])\s+|\n{2,}", text)
    return [p.strip() for p in parts if len(WORD.findall(p)) >= 2]


FUNCTION = {"что", "как", "так", "или", "это", "все", "всё", "его", "она", "они", "оно", "где",
            "уже", "только", "ещё", "был", "была", "было", "быть", "при", "для", "под", "над", "без",
            "тот", "той", "чем", "кто", "мне", "тебе", "нас", "вас", "ты", "мы", "вы", "ну", "бы"}


def _cluster(names: Counter) -> dict[str, str]:
    """Свести формы одного имени: «Лиса», «Лису», «Лисица», «Лисицыны» → одно лицо."""
    root_of: dict[str, str] = {}
    roots: list[str] = []
    for w, _ in names.most_common():
        lw = w.lower().replace("ё", "е")
        for r in roots:
            n = 0
            while n < min(len(r), len(lw)) and r[n] == lw[n]:
                n += 1
            if n >= 3 and n >= 0.5 * min(len(r), len(lw)):
                root_of[w] = r
                break
        else:
            roots.append(lw)
            root_of[w] = lw
    return root_of


def _inner_capitals(s: str) -> list[str]:
    """Слова с заглавной буквы не в начале предложения и не в начале строки (в стихах
    каждая строка начинается с заглавной — это не имя)."""
    out = []
    for line in s.split("\n"):
        for i, w in enumerate(WORD.findall(line)):
            if i > 0 and w[0].isupper() and len(w) > 2:
                out.append(w)
    return out


def analyze(text: str, mind=None, top: int = 6) -> dict:
    sents = sentences(text)
    raw: Counter = Counter()
    for s in sents:
        for w in _inner_capitals(s):
            raw[w] += 1
    root_of = _cluster(raw)
    names: Counter = Counter()
    forms: dict[str, Counter] = {}
    per_sent: list[set[str]] = []
    for s in sents:
        here = set()
        for i, w in enumerate(WORD.findall(s)):
            r = root_of.get(w)
            if r is None and w[0].isupper():
                # первое слово предложения — имя, если похоже на уже известное имя
                for known in set(root_of.values()):
                    if w.lower().startswith(known[:3]) and len(known) >= 3:
                        r = known
                        break
            if r is not None:
                names[r] += 1
                forms.setdefault(r, Counter())[w] += 1
                here.add(r)
        per_sent.append(here)
    people = [r for r, n in names.most_common() if n >= 2][:top]
    label = {r: forms[r].most_common(1)[0][0] for r in people}
    # кто с кем встречается
    together = Counter()
    for here in per_sent:
        for a, b in combinations(sorted(here & set(people)), 2):
            together[(label[a], label[b])] += 1
    order = []
    for here in per_sent:
        for r in people:
            if r in here and label[r] not in order:
                order.append(label[r])
    # частые предметы (не имена, не короткие служебные слова)
    common = Counter()
    skip = FUNCTION | (set(mind.lexicon.function) if mind is not None else set())
    for s in sents:
        for w in WORD.findall(s):
            lw = w.lower()
            if len(lw) >= 3 and lw not in skip and root_of.get(w) is None \
                    and not any(lw.startswith(r[:3]) for r in people):
                common[_root(lw)] += 1
    things = [r for r, n in common.most_common(40) if n >= 3][:top]
    generic = [s for s, here in zip(sents, per_sent) if not here and len(WORD.findall(s)) >= 5]
    learned, parsed = [], 0
    if mind is not None:
        for s in sents:
            got = mind.learn_text(s)
            if got:
                parsed += 1
                learned += got
    understood = parsed / max(len(sents), 1)
    report = {
        "предложений": len(sents),
        "действующие лица": [label[r] for r in people],
        "порядок появления": order,
        "кто с кем": [f"{a} и {b}: вместе в {n} предложениях" for (a, b), n in together.most_common(5)],
        "вокруг чего действие": things,
        "общие высказывания (возможно, мораль)": generic[:2],
        "выучила": learned,
        "разобрала": f"{parsed} из {len(sents)} предложений ({understood:.0%})",
        "честно": ("я нашла, КТО действует и С КЕМ, но не поняла, ЧТО произошло и ЗАЧЕМ: "
                   "художественные предложения слишком сложны для моего языка. "
                   "Мораль я угадываю по форме (общая фраза без героев), а не по смыслу."),
    }
    return report
