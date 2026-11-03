"""Автономная работа без Claude: разговор, сомнение, вопросы, наука, память между запусками,
чтение классического текста (с честной оценкой понимания)."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from light.mind.reader import analyze  # noqa: E402
from light.session import Session  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class Autonomy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.path = os.path.join(cls.dir, "state.pkl")
        cls.s = Session(cls.path, childhood=2000, seed=1)

    def test_01_learns_from_text_and_reasons(self):
        out = self.s.say("кит это животное")
        self.assertTrue(any("кит — это животное" in o for o in out))
        ans = self.s.say("дышит ли кит?")
        self.assertTrue(ans[-1].startswith("Да"))
        self.assertIn("кит — это животное", ans[-1])
        why = self.s.say("почему ты так думаешь?")
        self.assertIn("животное дышит", why[-1])

    def test_02_doubts_what_contradicts_observation(self):
        out = self.s.say("пингвин летает")
        self.assertTrue(any(o.startswith("Сомневаюсь") for o in out))

    def test_03_asks_about_unknown_words(self):
        out = self.s.say("Дом стоит в зелёном саду.")
        self.assertTrue(any("Что значит «зелёный»?" in o for o in out))
        out = self.s.say("зелёный — это свет с длиной волны от 500 до 565 нанометров")
        self.assertTrue(any("500–565" in o for o in out))

    def test_04_science_and_self_report(self):
        self.s.lab()
        self.s.problems()
        self.assertTrue(any("масса на единицу объёма" in x for x in self.s.science))
        know = self.s.say("что ты знаешь?")[-1]
        self.assertIn("открытые мной законы", know)
        err = self.s.say("в чём ты ошибалась?")[-1]
        self.assertIn("ОПРОВЕРЖЕНИЕ", err)

    def test_05_memory_survives_restart(self):
        self.s.save()
        again = Session(self.path, childhood=0)
        self.assertTrue(again.loaded)
        ans = again.say("дышит ли кит?")
        self.assertTrue(ans[-1].startswith("Да"))          # помнит кита из прошлого сеанса
        self.assertTrue(any("масса на единицу объёма" in x for x in again.science))


class Literature(unittest.TestCase):
    def test_fable_structure_and_honesty(self):
        text = (ROOT / "examples" / "texts" / "krylov_crow_fox.txt").read_text(encoding="utf-8")
        r = analyze(text)
        self.assertEqual(set(r["действующие лица"]), {"Ворона", "Лиса"})
        self.assertEqual(r["порядок появления"][0], "Ворона")
        self.assertIn("сыр", r["вокруг чего действие"])
        self.assertTrue(any("лесть" in m for m in r["общие высказывания (возможно, мораль)"]))
        self.assertIn("не поняла", r["честно"])


if __name__ == "__main__":
    unittest.main()
