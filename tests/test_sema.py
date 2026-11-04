"""СЕМА — нейросеть смыслов: рассуждение нейронными операциями, устойчивость к
повреждению, смысл сюжета (зачем / почему / что было бы, если / мораль), закон в весах."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from light.lab.laws import problem_book  # noqa: E402
from light.lab.world import Workshop  # noqa: E402
from light.sema.neuron import LinearNeuron, LogisticNeuron  # noqa: E402
from light.sema.reasoner import NeuralReasoner  # noqa: E402
from light.sema.story import FABLE, TRAINING, StoryMind  # noqa: E402
from light.world import exam  # noqa: E402
from test_mind import life  # noqa: E402


def neural_score(nr, mind) -> int:
    ok = 0
    for x, y, truth in exam():
        v, conf, _ = nr.judge(x, y, isa=y in mind.nouns())
        ok += (v if conf >= 0.5 else None) == truth
    return ok


class NeuralReasoning(unittest.TestCase):
    def test_same_exam_as_algorithmic_mind(self):
        _, m, _, _ = life(1)
        nr = NeuralReasoner(2048, seed=1)
        nr.load_from(m)
        neural = neural_score(nr, m)
        algorithmic = sum(m.ask(x, y).answer() == t for x, y, t in exam())
        print(f"\n  экзамен: нейросеть {neural}/25, алгоритмический разум {algorithmic}/25")
        self.assertGreaterEqual(neural, algorithmic - 1)
        v, _, chain = nr.judge("пингвин", "дышит")
        self.assertTrue(v)
        self.assertTrue(any("птица" in s for s in chain))           # объяснение — путь фокуса

    def test_graceful_degradation(self):
        _, m, _, _ = life(1)
        nr = NeuralReasoner(512, seed=2)
        nr.load_from(m)
        curve = []
        for frac in (0.0, 0.5, 0.8, 0.95, 0.99):
            nr.space.lesion(frac, seed=3)
            curve.append(neural_score(nr, m))
        print(f"\n  убито нейронов 0/50/80/95/99% → {curve}")
        self.assertGreaterEqual(curve[1], curve[0] - 1)               # половина — почти без потерь
        self.assertLess(curve[-1], curve[0])                           # но не бесконечно
        self.assertTrue(all(a >= b - 2 for a, b in zip(curve, curve[1:])))  # без обрывов


class StoryMeaning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sm = StoryMind(2048, seed=1)
        cls.rules = cls.sm.learn(TRAINING)
        cls.story = cls.sm.read(FABLE)          # эту историю сеть раньше не видела

    def test_causal_rules_with_variables(self):
        text = " | ".join(self.rules)
        self.assertIn("X хвалит Y → Y радуется", text)
        self.assertIn("X поёт, а X держит Y → Y падает", text)
        rejected = " | ".join(self.sm.rejected())
        self.assertIn("держит → поёт", rejected)                      # совпадение, не причина

    def test_why_did_fox_praise(self):
        a = self.sm.why_did(self.story, "лиса", "хвалила", "ворону")
        self.assertTrue(a.startswith("чтобы получить сыр"))
        for step in ("ворона радуется", "ворона поёт", "сыр падает", "лиса берёт сыр"):
            self.assertIn(step, a)

    def test_why_cheese_fell(self):
        a = self.sm.why_happened(self.story, "сыр", "падает")
        self.assertIn("потому что ворона поёт", a)
        self.assertIn("началось всё с того, что лиса хвалит ворону", a)

    def test_counterfactual(self):
        a = self.sm.what_if_not(self.story, "ворона", "поёт")
        self.assertIn("сыр падает", a)
        self.assertIn("лиса берёт сыр", a)

    def test_moral(self):
        self.sm.why_did(self.story, "лиса", "хвалит", "ворону")
        self.sm.definitions["лесть"] = "похвала ради выгоды"
        a = self.sm.moral()
        self.assertIn("хвалит другого, чтобы получить то, что есть у другого", a)
        self.assertIn("лесть", a)


class LawInWeights(unittest.TestCase):
    def test_density_emerges_in_neuron(self):
        w = Workshop(3)
        things = w.shelf(40)
        X = np.array([[math.log(w.weigh(t)), math.log(w.measure_volume(t))] for t in things])
        y = np.array([w.drop(t) for t in things], float)
        n = LogisticNeuron(2)
        n.learn(X, y)
        ratio = n.w[1] / n.w[0]
        self.assertLess(abs(ratio + 1), 0.15)                          # веса (w, −w): масса/объём
        tw = Workshop(9)
        test = [tw.make(mat) for mat in ("пенопласт", "алюминий", "свинец", "янтарь") for _ in range(5)]
        acc = np.mean([(n.out(np.array([math.log(tw.weigh(t)), math.log(tw.measure_volume(t))])) > 0.5)
                       == t.floats() for t in test])
        self.assertGreaterEqual(acc, 0.9)

    def test_pendulum_in_neuron(self):
        p = problem_book()[0]
        rng = np.random.default_rng(0)
        pts = [{"длина": float(np.exp(rng.uniform(np.log(.1), np.log(3)))),
                "масса груза": float(np.exp(rng.uniform(np.log(.1), np.log(5))))} for _ in range(30)]
        X = np.array([[math.log(x["длина"]), math.log(x["масса груза"])] for x in pts])
        Y = np.array([math.log(p.oracle(**x)) for x in pts])
        n = LinearNeuron(2)
        n.learn(X, Y)
        self.assertLess(abs(n.w[0] - 0.5), 0.05)
        self.assertLess(abs(n.w[1]), 0.05)


if __name__ == "__main__":
    unittest.main()
