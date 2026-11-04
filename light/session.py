"""Автономный сеанс: система работает сама, без Claude и без внешних моделей.

Вы даёте ей фразы, тексты, вопросы; она учится, отвечает с объяснениями, сомневается,
когда сказанное противоречит её наблюдениям, сама задаёт вопросы, ставит опыты в
«Мастерской», решает задачи — и сохраняет всё на диск, чтобы продолжить в следующий раз.

    from light.session import Session
    s = Session("моя_light.pkl")      # загрузит, если есть, иначе проживёт «детство»
    s.say("кит это животное")
    s.say("дышит ли кит?")
    s.idle()                           # сама задаст вопрос
    s.save()
"""
from __future__ import annotations

import os
import pickle

from .brain import Brain
from .lab.laws import LawFinder, problem_book, related_constants
from .lab.scientist import Scientist
from .lab.world import LOOK as LAB_LOOK
from .lab.world import Workshop
from .mind.curiosity import Curiosity
from .mind.grounding import Grounder
from .mind.mind import Mind
from .mind.node import MindNode
from .mind.reader import analyze
from .world import LOOK, Nursery


class Session:
    def __init__(self, path: str | None = None, childhood: int = 2000, seed: int = 0, log=None):
        self.path = path
        self.log = log or (lambda *_: None)
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                st = pickle.load(f)
            self.mind, self.curiosity, self.grounder = st["mind"], st["curiosity"], st["grounder"]
            self.science, self.refuted, self.lived = st["science"], st["refuted"], st["lived"]
            self.seed = st["seed"]
            self.loaded = True
        else:
            self.mind = Mind(len(LOOK))
            self.curiosity = Curiosity(self.mind, seed)
            self.grounder = Grounder()
            self.science, self.refuted, self.lived, self.seed = [], [], 0, seed
            self.loaded = False
            if childhood:
                self.live(childhood)
        self.brain = Brain(log=self.log)
        self.brain.answer_threshold = 2.0          # говорит только разум и я-модель, не заглушки
        self.node = self.brain.attach(MindNode(self.mind, self.curiosity, grounder=self.grounder))
        self.node.science, self.node.refuted = self.science, self.refuted
        self.node.verbal_only = True              # в разговоре словами нет общей картинки

    # ---------- опыт ----------
    def live(self, n: int) -> str:
        """Прожить n эпизодов в «Детской» (видеть, слышать, учиться)."""
        w = Nursery(self.seed + self.lived)
        for _ in range(n):
            scene, text, foc = w.episode()
            cats = self.mind.experience([(o.look, o.events) for o in scene], text, foc)
            self.curiosity.saw(cats, [o.look for o in scene])
        self.mind.sleep()
        self.lived += n
        return f"прожито эпизодов: {self.lived}; категорий: {len(self.mind.perception.ids())}"

    # ---------- разговор ----------
    def _run(self, ticks: int = 15, until_answer: bool = False) -> list[str]:
        n = len(self.brain.outputs)
        quiet = 0
        for _ in range(ticks):
            self.brain.tick()
            new = self.brain.outputs[n:]
            if until_answer and any(o.data.get("answer") is not None or o.data.get("introspective")
                                    for o in new):
                break                              # на вопрос — сразу ответ
            quiet = quiet + 1 if len(self.brain.outputs) == n else 0
            if new and quiet >= 2:
                break
        return [o.text() for o in self.brain.outputs[n:]
                if o.source in ("разум", "self", "motor") and not o.text().startswith("motor:")]

    def say(self, text: str) -> list[str]:
        question = text.strip().endswith("?")
        self.brain.hear(text, goal=question)
        if not question:
            self.brain.inbox[-1].data["statement"] = True
        return self._run(until_answer=question)

    def idle(self, ticks: int = 6) -> list[str]:
        """Тишина: система сама думает и, если есть что спросить, спрашивает."""
        self.brain.goals.clear()
        self.brain.open_questions.clear()
        n = len(self.brain.outputs)
        for _ in range(ticks):
            self.brain.tick()
            if any(o.data.get("ask") for o in self.brain.outputs[n:]):
                break
        return [o.text() for o in self.brain.outputs[n:] if o.data.get("ask")]

    def read(self, text: str) -> dict:
        """Прочитать связный текст: разобрать, что получится, и честно сказать, что нет."""
        return analyze(text, self.mind)

    # ---------- наука ----------
    def lab(self, experiments: int = 15, seed: int | None = None) -> list[str]:
        """Самостоятельные опыты в «Мастерской»: что тонет, что плавает и почему."""
        w = Workshop(self.seed if seed is None else seed)
        s = Scientist(len(LAB_LOOK))
        s.set_cup_feature(LAB_LOOK.index("чаша"))
        shelf = w.shelf(60)
        obs = [s.look_at(t.look, w.label(t), w.weigh(t), w.measure_volume(t)) for t in shelf]
        for _ in range(experiments):
            i = s.choose(obs)
            s.record(obs.pop(i), w.drop(shelf.pop(i)))
        self.refuted += [j for j in s.journal if j.startswith("ОПРОВЕРЖЕНИЕ")]
        if s.law:
            self.science.append(f"плавание: {s.law.text()} (по {len(s.data)} моим опытам)")
        self.scientist = s
        return s.journal

    def problems(self) -> list[str]:
        """Задачи по физике и геометрии: сама планирует замеры и ищет законы."""
        found, out = {}, []
        for p in problem_book():
            f = LawFinder(p, seed=self.seed)
            law = f.discover()
            if law:
                found[p.name] = law.const
                note = f.recognize_constant(found)
                self.science.append(f"{p.name}: {law.text()}" + (f" (постоянная — {note})" if note else ""))
                out.append(f"{p.name}: {law.text()} — {len(f.asked)} вопросов к миру")
            else:
                out.append(f"{p.name}: закона не нашла")
        rel = related_constants(found)
        self.science += rel
        return out + rel

    # ---------- память ----------
    def save(self, path: str | None = None) -> str:
        path = path or self.path or "light_state.pkl"
        with open(path, "wb") as f:
            pickle.dump({"mind": self.mind, "curiosity": self.curiosity, "grounder": self.grounder,
                         "science": self.science, "refuted": self.refuted, "lived": self.lived,
                         "seed": self.seed}, f)
        return path
