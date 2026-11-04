"""Проверки «детского» разума: обучение без меток, рассуждение, любопытство, встреча с новым.

Экзамен составлен из фактов, которые родитель НИКОГДА не произносит, — поэтому
ответить можно только рассуждением; «зубрила» (ответы по дословно услышанному)
получает на нём 0.
"""
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from light import Brain  # noqa: E402
from light.mind.curiosity import Curiosity  # noqa: E402
from light.mind.mind import Mind  # noqa: E402
from light.mind.node import MindNode  # noqa: E402
from light.world import EVERYDAY, LOOK, Nursery, exam  # noqa: E402

SPARSE = {("рыба", "животное"), ("птица", "животное"), ("животное", "живое")}
LIVES: dict = {}


def life(seed: int, episodes: int = 2000, sparse: bool = False):
    """Прожить «детство»: эпизоды из мира, никаких меток. Кэшируется."""
    key = (seed, episodes, sparse)
    if key not in LIVES:
        withhold = {(x, p) for x, p, _ in exam()} | ({("рыба", "животное")} | (SPARSE if sparse else set()))
        w = Nursery(seed, withhold=withhold)
        m = Mind(len(LOOK))
        cu = Curiosity(m, seed)
        truth = defaultdict(Counter)
        for _ in range(episodes):
            scene, text, foc = w.episode()
            cats = m.experience([(o.look, o.events) for o in scene], text, foc)
            cu.saw(cats, [o.look for o in scene])
            for c, o in zip(cats, scene):
                truth[c][o.kind] += 1
        m.sleep()
        LIVES[key] = (w, m, cu, truth)
    return LIVES[key]


def score(m: Mind) -> int:
    return sum(m.ask(x, y).answer() == t for x, y, t in exam())


class PerceptionAndLanguage(unittest.TestCase):
    def test_categories_found_without_labels(self):
        _, m, _, truth = life(1)
        big = [c for c in m.perception.ids() if m.perception.counts[c] > 50]
        kinds = {truth[c].most_common(1)[0][0] for c in big}
        purity = sum(truth[c].most_common(1)[0][1] for c in big) / sum(sum(truth[c].values()) for c in big)
        print(f"\n  категорий: {len(big)}, видов узнано: {len(kinds)}/12, чистота {purity:.3f}")
        self.assertEqual(kinds, set(EVERYDAY))
        self.assertGreater(purity, 0.95)

    def test_word_classes_discovered(self):
        _, m, _, _ = life(1)
        lx = m.lexicon
        self.assertEqual(lx.function, {"вот", "какой", "ой", "смотри", "это"})
        self.assertTrue({"лает", "дышит", "летает", "растёт", "плавает"} <= lx.predicates)
        self.assertTrue({"собака", "животное", "птица", "живое"} <= lx.nouns)
        self.assertEqual(lx.event_of.get("лает"), "звук_гав")

    def test_names_ground_in_perception(self):
        _, m, _, truth = life(1)
        for c, w in m.lexicon.name_of.items():
            if m.perception.counts[c] > 50:
                self.assertEqual(truth[c].most_common(1)[0][0], w)


class Reasoning(unittest.TestCase):
    def test_exam_of_never_stated_facts(self):
        scores, literal = [], []
        for seed in (1, 2, 3):
            _, m, _, _ = life(seed)
            scores.append(score(m))
            literal.append(sum(m.literal(x, y) == t for x, y, t in exam()))
        print(f"\n  экзамен (25 вопросов): разум {scores}, зубрила {literal}")
        self.assertTrue(all(s >= 21 for s in scores))
        self.assertEqual(sum(literal), 0)

    def test_explained_chain(self):
        _, m, _, _ = life(1)
        v = m.judge("пингвин", "дышит")
        self.assertTrue(v.value)
        self.assertTrue(any("птица" in s for s in v.chain))

    def test_exception_found_by_itself(self):
        """Родитель НИКОГДА не говорит, что пингвин не летает."""
        _, m, _, _ = life(1)
        self.assertFalse(m.judge("пингвин", "летает").value)
        self.assertTrue(any(s.startswith("пингвин не летает") for s in m.insights))

    def test_honest_uncertainty(self):
        _, m, _, _ = life(1)
        v = m.judge_isa("рыба", "животное")      # этого ей неоткуда было узнать
        self.assertLess(v.conf, 0.75)


class CuriosityTest(unittest.TestCase):
    def session(self, seed, mode, k=6):
        w, m0, _, _ = life(seed, 1500, sparse=True)
        import copy
        m = copy.deepcopy(m0)
        cu = Curiosity(m, seed)
        cu.last_look = dict(LIVES[(seed, 1500, True)][2].last_look)
        for _ in range(k):
            q = cu.best() if mode == "curious" else cu.random()
            if q is None:
                break
            cu.mark(q)
            scene, ans = w.answer(q.for_parent())
            m.experience([(o.look, o.events) for o in scene], ans, 0 if scene else None)
        return score(m)

    def test_right_questions_teach_more(self):
        seeds = (1, 2, 3)
        passive = [score(life(s, 1500, sparse=True)[1]) for s in seeds]
        rnd = [self.session(s, "random") for s in seeds]
        cur = [self.session(s, "curious") for s in seeds]
        print(f"\n  после 6 вопросов: без вопросов {passive}, случайные {rnd}, любопытство {cur}")
        self.assertGreater(np.mean(cur), np.mean(rnd) + 1)
        self.assertGreater(np.mean(cur), np.mean(passive) + 1)


class Novelty(unittest.TestCase):
    def test_meeting_an_owl(self):
        import copy
        w, m0, cu0, _ = life(2)
        m = copy.deepcopy(m0)
        cu = Curiosity(m, 2)
        cu.last_look = dict(cu0.last_look)
        before = len(m.perception.protos)
        for _ in range(6):
            owl, other = w.instance("сова"), w.instance(w.rng.choice(EVERYDAY))
            cats = m.experience([(owl.look, owl.events), (other.look, other.events)], f"смотри {other.kind}", 1)
            cu.saw(cats, [owl.look, other.look])
        self.assertGreater(len(m.perception.protos), before)          # структура выросла
        guess = m.judge_thing(w.instance("сова").look, "летает")
        self.assertTrue(guess.value)                                   # догадка до всяких слов
        self.assertEqual(guess.source, "гипотеза")
        q1 = cu.best()
        self.assertEqual(q1.kind, "name")                              # сначала — «как это называется?»
        cu.mark(q1)
        scene, ans = w.answer(q1.for_parent())
        m.experience([(o.look, o.events) for o in scene], ans, 0)
        self.assertEqual(ans, "это сова")
        q2 = cu.best()
        self.assertEqual((q2.kind, q2.x), ("parent", "сова"))         # потом — «сова — это кто?»
        cu.mark(q2)
        _, ans = w.answer(q2.for_parent())
        m.experience([], ans)
        v = m.judge("сова", "дышит")
        self.assertTrue(v.value)
        self.assertIn("сова — это птица (мне говорили)", v.chain)


class InTheBrain(unittest.TestCase):
    def test_stepwise_reasoning_and_why(self):
        _, m, _, _ = life(1)
        b = Brain(log=lambda *_: None)
        b.attach(MindNode(m, curious=False))
        b.hear("дышит ли пингвин?")
        for _ in range(12):
            b.tick()
            if b.outputs and b.outputs[-1].data.get("answer") is not None:
                break
        out = b.outputs[-1]
        self.assertTrue(out.data["answer"])
        steps = [s for s in b.workspace.stream if s.source == "разум" and s.kind == "thought"]
        self.assertGreaterEqual(len(steps), 3)                         # шаги прошли через сцену
        n = len(b.outputs)
        b.hear("почему ты так думаешь?")
        for _ in range(6):
            b.tick()
            own = [o for o in b.outputs[n:] if o.data.get("introspective")]
            if own:
                break
        self.assertEqual(own[0].data["chain"], out.data["chain"])     # отчёт = реальная цепочка


if __name__ == "__main__":
    unittest.main()
