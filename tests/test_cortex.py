import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from light import Brain  # noqa: E402
from light.cortex import Connectome, CorticalNode, CorticalRegion, WordEncoder  # noqa: E402

CORPUS = ["небо голубое потому что свет рассеивается в воздухе",
          "закат красный потому что свет проходит длинный путь",
          "кот спит на тёплом окне"]


def trained() -> CorticalNode:
    node = CorticalNode()
    for _ in range(6):
        for s in CORPUS:
            node.read(s)
    return node


class CortexTest(unittest.TestCase):
    def test_surprise_drops_with_learning(self):
        node = CorticalNode()
        first = np.mean(node.read(CORPUS[0])[1:])
        for _ in range(5):
            last = np.mean(node.read(CORPUS[0])[1:])
        self.assertGreater(first, 0.9)
        self.assertLess(last, 0.1)

    def test_context_is_kept(self):
        """Одно и то же слово в разных фразах — разные ожидания (память высокого порядка)."""
        node = trained()
        node.read("небо голубое потому что свет", learn=False)
        self.assertEqual(node.expect()[0], "рассеивается")
        node.read("закат красный потому что свет", learn=False)
        self.assertEqual(node.expect()[0], "проходит")

    def test_imagination(self):
        node = trained()
        self.assertEqual(node.imagine("кот спит"), ["на", "тёплом", "окне"])

    def test_novel_is_surprising(self):
        node = trained()
        self.assertGreater(np.mean(node.read("кот бежит по красному воздуху", learn=False)[1:]), 0.5)

    def test_sleep_prunes_but_keeps_memory(self):
        node = trained()
        node.sleep([])
        self.assertEqual(node.imagine("кот спит"), ["на", "тёплом", "окне"])

    def test_hierarchy(self):
        enc = WordEncoder()
        net = Connectome()
        low = net.add(CorticalRegion("слова", enc.size, seed=1))
        net.add(CorticalRegion("фразы", low.N, columns=128, seed=2))
        net.connect("слова", "фразы")
        for _ in range(3):
            for w in WordEncoder.tokens(CORPUS[2]):
                s = net.step({"слова": enc.encode(w)})
        self.assertIn("фразы", s)
        self.assertGreater(low.apical.max(), 0)  # верхняя область подсказывает нижней

    def test_brain_integration(self):
        brain = Brain(log=lambda *_: None)
        node = brain.attach(trained())
        brain.hear("кот бежит по красному воздуху")
        brain.run(max_ticks=6)
        kinds = {s.kind for s in brain.workspace.stream if s.source == node.name}
        self.assertIn("surprise", kinds)


if __name__ == "__main__":
    unittest.main()
