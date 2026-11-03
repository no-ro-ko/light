"""Научный подход: открытие законов собственными опытами, ответы на «как»,
понимание незнакомого текста, задачи по физике/геометрии, связь слова с новым чувством."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from light.lab.laws import LawFinder, problem_book, related_constants  # noqa: E402
from light.lab.scientist import Scientist  # noqa: E402
from light.lab.world import LOOK, MATERIALS, NOVEL, TEXTBOOK, WRONG_TEXT, Workshop  # noqa: E402
from light.mind.grounding import COLOR_OF, DEFINITIONS, WAVES, Grounder, camera, spectrum  # noqa: E402


def experiment(seed: int, n: int = 15):
    """Система сама выбирает, что опустить в воду; ей ничего не говорят о законах."""
    w = Workshop(seed)
    s = Scientist(len(LOOK))
    s.set_cup_feature(LOOK.index("чаша"))
    shelf = w.shelf(60)
    obs = [s.look_at(t.look, w.label(t), w.weigh(t), w.measure_volume(t)) for t in shelf]
    for _ in range(n):
        i = s.choose(obs)
        t, o = shelf.pop(i), obs.pop(i)
        s.record(o, w.drop(t))
    return w, s


def exam_objects(seed: int):
    tw = Workshop(seed + 100)
    things = [tw.make(m) for m in list(MATERIALS) + list(NOVEL) for _ in range(5)]
    things += [tw.make("железо", hollow=True, wall=0.08), tw.make("медь", hollow=True, wall=0.05)]
    return tw, things


class Buoyancy(unittest.TestCase):
    def test_discovers_density_law_itself(self):
        for seed in (1, 2, 3):
            _, s = experiment(seed)
            self.assertEqual((s.law.a, s.law.b), (1, -1), s.journal)   # масса / объём
            self.assertTrue(s.law.below_floats)
            self.assertLess(abs(s.law.theta - 1.0), 0.1)
            self.assertTrue(any("ввожу новые величины" in j for j in s.journal))

    def test_refutes_wrong_ideas(self):
        _, s = experiment(1)
        self.assertTrue(any(j.startswith("ОПРОВЕРЖЕНИЕ") for j in s.journal))

    def test_predicts_unseen_materials_and_shapes(self):
        accs, near = [], []
        for seed in (1, 2, 3):
            _, s = experiment(seed)
            tw, things = exam_objects(seed)
            ok = memo = 0
            for t in things:
                o = s.look_at(t.look, tw.label(t), tw.weigh(t), tw.measure_volume(t))
                p, _ = s.predict(o)
                ok += p == t.floats()
                # «зубрила»: как вёл себя самый похожий на вид предмет из опытов
                nn = min(s.data, key=lambda d: np.linalg.norm(d.look - o.look))
                memo += nn.floats == t.floats()
            accs.append(ok / len(things))
            near.append(memo / len(things))
        print(f"\n  новые материалы и формы: закон {np.round(accs, 2)}, «по похожести» {np.round(near, 2)}")
        self.assertGreater(min(accs), 0.95)
        self.assertGreater(np.mean(accs), np.mean(near) + 0.1)

    def test_material_constant_discovered(self):
        _, s = experiment(2, n=25)
        consts = s.material_constants()
        for lab, (val, spread) in consts.items():
            self.assertLess(abs(val - MATERIALS[lab][0]) / MATERIALS[lab][0], 0.05)
            self.assertLess(spread, 0.05)

    def test_how_to_make_stone_float_and_verify(self):
        _, s = experiment(3)
        w = Workshop(77)
        stone = w.make("камень", hollow=False, size=2.0)
        o = s.look_at(stone.look, "камень", w.weigh(stone), w.measure_volume(stone))
        plans, text = s.how_to_float(o, float_density=0.24)
        self.assertIn("разрезать пополам бесполезно", text)
        self.assertEqual(len(plans), 2)
        for pl in plans:
            t2 = w.attach_float(stone, pl["volume"]) if pl["action"] == "поплавок" else w.hollow_out(stone, pl["keep"])
            self.assertTrue(w.drop(t2), pl)                     # план проверен опытом
        half = w.cut(stone)
        p, _ = s.predict(s.look_at(half.look, "камень", w.weigh(half), w.measure_volume(half)))
        self.assertEqual(p, w.drop(half))

    def test_reads_unfamiliar_word_and_checks_text(self):
        _, s = experiment(3, n=25)
        r = s.read(TEXTBOOK)
        self.assertEqual((r["word"], r["meaning"]), ("плотность", "масса на единицу объёма"))
        self.assertLess(abs(r["water"] - 1.0), 0.1)            # «плотность воды» ≈ её граница
        bad = s.read(TEXTBOOK[:4] + WRONG_TEXT)
        self.assertTrue(any("у пробки большая плотность" in d for d in bad["disagree"]))


class Laws(unittest.TestCase):
    def test_problem_book(self):
        found = {}
        for p in problem_book():
            f = LawFinder(p, seed=3)
            law = f.discover()
            self.assertIsNotNone(law, f"{p.name}: {f.journal}")
            found[p.name] = law.const
            # прямая задача далеко за пределами опытов
            x = {k: hi * 5 for k, (lo, hi) in p.variables.items()}
            self.assertLess(abs(law.predict(x) - p.oracle(**x)) / p.oracle(**x), 0.05, p.name)
        print(f"\n  законов найдено: {len(found)}/{len(problem_book())}")

    def test_irrelevant_variables_and_constants(self):
        book = {p.name: p for p in problem_book()}
        pend = LawFinder(book["маятник"], seed=4).discover()
        self.assertEqual(pend.exps["масса груза"], 0)
        self.assertEqual(pend.exps["длина"], 0.5)
        circle = LawFinder(book["круг"], seed=4)
        circle.discover()
        self.assertEqual(circle.recognize_constant({}), "π")
        hyp = LawFinder(book["прямоугольный треугольник"], seed=4).discover()
        self.assertEqual((hyp.form, hyp.power), ("сумма степеней", 2))   # Пифагор

    def test_inverse_problem(self):
        """Какой длины нужен маятник, чтобы период был ровно 3 секунды?"""
        p = {q.name: q for q in problem_book()}["маятник"]
        law = LawFinder(p, seed=5).discover()
        L = law.solve("длина", 3.0, {"длина": 1.0, "масса груза": 1.0})
        self.assertLess(abs(p.oracle(длина=L, **{"масса груза": 1.0}) - 3.0), 0.05)
        self.assertIsNone(law.solve("масса груза", 3.0, {"длина": 1.0, "масса груза": 1.0}))

    def test_cross_problem_hypothesis(self):
        found = {}
        for p in problem_book():
            law = LawFinder(p, seed=6).discover()
            found[p.name] = law.const
        rel = related_constants(found)
        self.assertTrue(any("маятник" in r and "падение" in r for r in rel))


class Grounding(unittest.TestCase):
    def test_what_does_green_mean(self):
        rng = np.random.default_rng(1)
        g = Grounder(known_words={"дом", "стоит", "саду"})
        self.assertEqual(g.read("Дом стоит в зелёном саду."), ["Что значит «зелёный»?"])
        g.read(DEFINITIONS["зелёный"] + ". листья зелёные. трава зелёная")
        g.read(DEFINITIONS["синий"] + ". небо синее")
        self.assertTrue(any("листья" in p and "500" in p for p in g.predictions()))
        # камера: смысл каналов неизвестен — связываю через пример
        g.attach("камера")
        r = g.perceive("камера", camera(spectrum(COLOR_OF["лист"], rng), rng), "лист")
        self.assertEqual(r["color"], "зелёный")
        r = g.perceive("камера", camera(spectrum(COLOR_OF["огурец"], rng), rng), "огурец")
        self.assertEqual(r["color"], "зелёный")                  # вывод, а не услышанное
        self.assertIn("проверить", r["question"])
        # спектрометр: смысл каналов известен — узнаю по определению
        g.attach("спектрометр", list(WAVES))
        r = g.perceive("спектрометр", spectrum(COLOR_OF["трава"], rng), "трава")
        self.assertEqual(r["color"], "зелёный")
        self.assertTrue(any("подразумевалось" in n for n in r["notes"]))
        self.assertTrue(any("подтвердилось" in n for n in r["notes"]))
        r = g.perceive("спектрометр", spectrum(COLOR_OF["банан"], rng), "банан")
        self.assertIsNone(r["color"])
        self.assertIn("ещё не называли", r["question"])


if __name__ == "__main__":
    unittest.main()
