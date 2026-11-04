"""Причины из обычного текста: «если …, то …», «потому что», «поэтому»."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from light.rost.organism import Organism  # noqa: E402

CUR = Path(__file__).resolve().parents[1] / "curriculum"


class CausesFromText(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.o = Organism()
        for p in sorted(CUR.glob("*.txt")):
            if not p.name.startswith(("08", "11")):
                cls.o.read(p.name, p.read_text(encoding="utf-8"))
        cls.o.sleep()

    def test_links_read(self):
        self.assertEqual(self.o.files["12_причины.txt"]["понято"], self.o.files["12_причины.txt"]["предложений"])
        self.assertGreaterEqual(len(self.o.causes.links), 8)

    def test_why_chain(self):
        a = self.o.answer("Почему трава растёт?")
        self.assertIn("земля мокрая", a)
        self.assertIn("дождь", a)                      # причина причины

    def test_what_if_chain(self):
        a = self.o.answer("Что будет, если вазу толкнуть?")
        self.assertIn("падает", a)
        self.assertIn("разбивается", a)
        self.assertIn("ваза", a)                       # местоимение заменено

    def test_general_to_specific(self):
        a = self.o.answer("Что будет, если кошка не ест?")
        self.assertIn("кошка слабеет", a)
        self.assertIn("наверное", a)                   # честно: вывод через род

    def test_other_patterns(self):
        self.assertIn("солнце светит", self.o.answer("Почему днём светло?"))    # «…, поэтому …»
        self.assertIn("нагревают", self.o.answer("Почему лёд тает?"))           # «… потому что …»
        self.assertIn("замерзает", self.o.answer("Что будет, если наступает зима?"))  # «когда …, …»
        self.assertIn("ворона поёт", self.o.answer("Почему сыр падает?"))        # истории — тоже


if __name__ == "__main__":
    unittest.main()
