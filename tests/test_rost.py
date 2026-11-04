"""«Росток»: рождается пустой, растёт по материалу, сам себя улучшает во сне."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from light.rost.fabric import Rostok  # noqa: E402
from light.rost.rules import NEW, STORIES, RuleField  # noqa: E402
from light.world import EVERYDAY, KINDS, Nursery  # noqa: E402

SIX = ["собака", "кошка", "рыба", "дерево", "мяч", "машинка"]
RUNS: dict = {}


def grow(seed: int, kinds: list[str], episodes: int = 1600, improve: bool = True):
    key = (seed, tuple(kinds), episodes, improve)
    if key not in RUNS:
        w = Nursery(seed, kinds=kinds)
        r = Rostok(self_improve=improve)
        for _ in range(episodes):
            scene, text, foc = w.episode()
            r.experience([o.look for o in scene], text, foc)
        r.sleep()
        RUNS[key] = (w, r)
    return RUNS[key]


def naming(w: Nursery, r: Rostok, kinds: list[str], n: int = 10) -> float:
    return float(np.mean([r.name(w.instance(k).look)[0] == k for k in kinds for _ in range(n)]))


class BornEmpty(unittest.TestCase):
    def test_nothing_at_birth(self):
        r = Rostok()
        self.assertEqual((len(r.units), len(r.words), len(r.links)), (0, 0, 0))

    def test_no_randomness_structure_comes_from_material(self):
        """Тот же материал → та же сеть: в ней нет случайных начальных чисел."""
        def life():
            w, r = Nursery(5, kinds=SIX), Rostok()
            for _ in range(300):
                scene, text, foc = w.episode()
                r.experience([o.look for o in scene], text, foc)
            return r
        a, b = life(), life()
        self.assertEqual(len(a.units), len(b.units))
        self.assertTrue(all(np.allclose(x, y) for x, y in zip(a.units.protos, b.units.protos)))
        self.assertEqual(a.journal, b.journal)


class GrowsWithMaterial(unittest.TestCase):
    def test_size_follows_material(self):
        _, small = grow(1, SIX)
        _, big = grow(1, EVERYDAY)
        _, bigger = grow(1, list(KINDS))
        sizes = [r.stable_units() for r in (small, big, bigger)]
        print(f"\n  устойчивых узлов: 6 видов → {sizes[0]}, 12 → {sizes[1]}, 13 → {sizes[2]}")
        self.assertLessEqual(abs(sizes[0] - 6), 1)
        self.assertLessEqual(abs(sizes[1] - 12), 1)
        self.assertLessEqual(abs(sizes[2] - 13), 1)
        self.assertGreater(len(big.words), len(small.words))

    def test_growth_saturates(self):
        _, r = grow(2, EVERYDAY)
        early, late = r.history[3]["устойчивых"], r.history[-1]["устойчивых"]
        self.assertLessEqual(abs(late - early), 2)          # материал освоен — рост остановился


class ImprovesItself(unittest.TestCase):
    def test_self_improvement_beats_growth_only(self):
        seeds = (1, 2, 3)
        plain = [grow(s, EVERYDAY, improve=False) for s in seeds]
        smart = [grow(s, EVERYDAY, improve=True) for s in seeds]
        acc_p = [naming(w, r, EVERYDAY) for w, r in plain]
        acc_s = [naming(w, r, EVERYDAY) for w, r in smart]
        size_p = [len(r.units) for _, r in plain]
        size_s = [len(r.units) for _, r in smart]
        print(f"\n  только рост: узлов {size_p}, верно {np.round(acc_p, 2)}; "
              f"с самоулучшением: узлов {size_s}, верно {np.round(acc_s, 2)}")
        self.assertGreaterEqual(np.mean(acc_s), np.mean(acc_p))
        self.assertLess(np.mean(size_s), np.mean(size_p))
        self.assertGreaterEqual(min(acc_s), 0.94)

    def test_journal_explains_changes(self):
        _, r = grow(2, EVERYDAY)
        self.assertTrue(r.journal)
        self.assertTrue(all("самооценка" in j and "→" in j for j in r.journal))
        self.assertTrue(any("разделила" in j or "слила" in j or "убрала" in j for j in r.journal))


class Rules(unittest.TestCase):
    def evaluate(self, rf: RuleField):
        predicted = correct = 0
        for st in NEW:
            evs = [e for e in (rf.parse(s) for s in st) if e]
            for t in range(1, len(evs)):
                p, _ = rf.predict(evs[:t])
                if p is not None:
                    predicted += 1
                    correct += p == evs[t]
        return correct, predicted

    def test_rules_grow_get_refined_and_generalize(self):
        plain, smart = RuleField(), RuleField()
        for st in STORIES:
            plain.read(st)
            smart.read(st)
        smart.sleep()
        texts = [r.text() for r in smart.rules]
        self.assertIn("X хвалит Y → Y радуется", texts)
        self.assertIn("X толкает Y, а Y шатается → Y падает", texts)     # уточнила сама
        self.assertNotIn("X держит → X поёт", texts)                      # отвергла совпадение
        self.assertTrue(any(j.startswith("уточнила") for j in smart.journal))
        self.assertTrue(any(j.startswith("отказалась") for j in smart.journal))
        c_p, n_p = self.evaluate(plain)
        c_s, n_s = self.evaluate(smart)
        print(f"\n  новые истории: без самопроверки {c_p}/{n_p} верных, с самопроверкой {c_s}/{n_s}")
        self.assertEqual(c_s, n_s)                                         # не ошибается
        self.assertGreater(c_s / n_s, c_p / n_p)
        self.assertGreaterEqual(c_s, 5)


if __name__ == "__main__":
    unittest.main()
