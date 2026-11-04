"""Чтение обычного текста: сложные фразы → простые утверждения; точность важнее охвата."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from light.rost.organism import Organism  # noqa: E402

CUR = Path(__file__).resolve().parents[1] / "curriculum"
ENC = "10_энциклопедия.txt"


class ReadingNaturalText(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.o = Organism()
        for p in sorted(CUR.glob("*.txt")):
            if not p.name.startswith(("08", "11")):
                cls.o.read(p.name, p.read_text(encoding="utf-8"))
        cls.o.sleep()
        cls.facts = [(k, x, y, pol) for (k, x, y, pol), src in cls.o.know.src.items()
                     if any(ENC in s for s in src)]

    def test_coverage(self):
        st = self.o.files[ENC]
        print(f"\n  энциклопедия: понято {st['понято']} из {st['предложений']}, фактов {len(self.facts)}")
        self.assertGreaterEqual(st["понято"] / st["предложений"], 0.8)
        self.assertGreaterEqual(len(self.facts), 30)

    def test_precision_no_garbage(self):
        name = self.o.name
        for kind, x, y, pol in self.facts:
            text = self.o.know._say(kind, x, y, pol)
            for junk in (" и ", " а ", " но ", " зато "):
                self.assertNotIn(junk, f" {name(y)} ", text)          # союз не прилип к утверждению
            if kind == "prop":
                self.assertFalse(self.o.know.is_category(y), text)    # название — не свойство
        self.assertNotIn(("prop", self.o.lang.lemma("птицы"), self.o.lang.lemma("летают"), False),
                         self.facts)                                    # «некоторые» не обобщены

    def test_complex_sentences(self):
        a = self.o.answer
        self.assertTrue(a("Является ли синий кит животным?").startswith("Да"))
        self.assertTrue(a("Дышит ли летучая мышь?").startswith("Да"))            # понятие из двух слов
        self.assertIn("млекопитающ", a("Дышит ли летучая мышь?"))
        self.assertTrue(a("Откладывает ли пингвин яйца?").startswith("Да"))      # «все птицы» → пингвин
        self.assertIn("антарктид", a("Где живёт пингвин?").lower())
        self.assertIn("жабрами", a("Чем отличаются кит и акула?"))                # «…, а киты дышат лёгкими»
        self.assertIn("лёгкими", a("Чем отличаются кит и акула?"))                 # пишет с «ё», как прочитала
        self.assertTrue(a("Все ли птицы летают?").startswith("Нет"))
        self.assertIn("пингвин", a("Какие птицы не летают?"))
        self.assertIn("плавает", a("Что умеет дельфин?"))

    def test_who_can_fly(self):
        ans = self.o.answer("Кто умеет летать?")
        for x in ("летучая мышь", "птица", "пчел"):
            self.assertIn(x, ans)
        for x in ("пингвин", "страус", "курица"):
            self.assertNotIn(x, ans)

    def test_pronouns_and_ellipsis(self):
        # «Они дышат воздухом …» (они = киты); «Пингвины не летают, но отлично плавают»
        self.assertTrue(self.o.answer("Дышит ли кит?").startswith("Да"))
        self.assertTrue(self.o.answer("Плавает ли пингвин?").startswith("Да"))
        self.assertTrue(self.o.answer("Летает ли пингвин?").startswith("Нет"))


if __name__ == "__main__":
    unittest.main()
