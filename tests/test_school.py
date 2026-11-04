"""Школа «Ростка»: с нуля — только по текстовым файлам в папке — до рассуждений,
смысла историй, законов, своих вопросов и сомнений."""
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from light.rost.organism import Organism  # noqa: E402
from light.rost.school import CURRICULUM, School  # noqa: E402

# вопросы, ответы на которые НЕ написаны ни в одном тексте — только вывести
EXAM = [("Дышит ли пингвин?", "Да"), ("Дышит ли ворона?", "Да"), ("Растёт ли кошка?", "Да"),
        ("Летает ли сова?", "Да"), ("Летает ли пингвин?", "Нет"), ("Летает ли корова?", "Нет"),
        ("Является ли пингвин животным?", "Да"), ("Является ли кит животным?", "Да"),
        ("Дышит ли кит?", "Да"), ("Является ли мяч животным?", "Нет"), ("Дышит ли муравей?", "Да"),
        ("Растёт ли лягушка?", "Да")]


def number(text: str) -> float:
    return float(re.search(r"≈ (\d+(?:\.\d+)?)", text).group(1))


class FromScratch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp())
        cls.school = School(cls.dir)
        cls.school.start()
        cls.log = cls.school.learn()
        cls.org = cls.school.org

    def test_00_born_empty(self):
        o = Organism()
        self.assertEqual((len(o.lang.forms), len(o.know.concepts), len(o.rules.rules), len(o.measures.laws)),
                         (0, 0, 0, 0))

    def test_files_and_outputs(self):
        self.assertEqual(list((self.dir / "входящие").iterdir()), [])
        self.assertEqual(len(list((self.dir / "изучено").iterdir())), len(list(CURRICULUM.glob("*.txt"))))
        for f in ("ответы.txt", "мои_вопросы.txt", "отчёт.txt", "журнал.txt"):
            self.assertTrue((self.dir / "ответы" / f).exists(), f)

    def test_language_grew_from_text(self):
        L = self.org.lang
        frames = {" ".join(f) for f, i in L.frames.items() if not i.get("innate")}
        self.assertIn("X является Y", frames)                      # выучила сама по известным фактам
        self.assertIn("X умеет Y", frames)
        self.assertEqual(L.lemma("животным"), L.lemma("животное"))  # формы одного слова
        self.assertEqual(L.lemma("летают"), L.lemma("летать"))
        self.assertGreaterEqual(len(L.endings), 10)

    def test_exam_by_reasoning(self):
        texts = " ".join(p.read_text(encoding="utf-8").lower() for p in CURRICULUM.glob("*.txt"))
        self.assertNotIn("пингвин дышит", texts)                   # ответов в текстах нет
        self.assertNotIn("кит является животным", texts)
        ok = 0
        for q, expected in EXAM:
            a = self.org.answer(q)
            ok += a.startswith(expected)
            if not a.startswith(expected):
                print("\n  ✗", q, "→", a[:150])
        print(f"\n  экзамен по невыписанным фактам: {ok}/{len(EXAM)}")
        self.assertGreaterEqual(ok, len(EXAM) - 1)
        self.assertIn("млекопитающ", self.org.answer("Дышит ли дельфин?"))   # цепочка через новый оборот

    def test_story_meaning(self):
        o = self.org
        self.assertTrue(o.answer("Зачем лиса хвалит ворону?").startswith("Чтобы получить сыр"))
        self.assertIn("потому что ворона поёт", o.answer("Почему сыр падает?"))
        what_if = o.answer("Что было бы, если бы лиса не хвалила ворону?")
        self.assertIn("сыр падает", what_if)
        self.assertIn("лиса берёт сыр", what_if)
        self.assertIn("лесть", o.answer("Какая мораль?"))
        self.assertIn("сыр падает", o.answer("Что было бы, если бы ворона не пела?"))   # «пела» — объяснили (14)
        self.assertIn("не знаю слова «плясала»", o.answer("Что было бы, если бы ворона не плясала?"))

    def test_laws_from_records(self):
        o = self.org
        self.assertGreaterEqual(len(o.measures.laws), 5)
        self.assertAlmostEqual(number(o.answer("Чему равен период, если длина 4?")), 4.01, delta=0.1)
        self.assertAlmostEqual(number(o.answer("Какой должна быть длина, чтобы период был 3?")), 2.24, delta=0.06)
        self.assertAlmostEqual(number(o.answer("Чему равен ток, если напряжение 12, сопротивление 6?")), 2.0, delta=0.05)
        self.assertAlmostEqual(number(o.answer("Чему равна диагональ, если ширина 6, высота 8?")), 10.0, delta=0.15)
        self.assertIn("не влияет: масса", o.answer("Чему равен период, если длина 1?"))

    def test_honest_about_classic_text(self):
        st = self.org.files["05_басня_крылов.txt"]
        self.assertLessEqual(st["понято"] / st["предложений"], 0.6)    # события — лишь в части строк
        self.assertIn("05_басня_крылов.txt", self.org.report())

    def test_own_questions(self):
        qs = (self.dir / "ответы" / "мои_вопросы.txt").read_text(encoding="utf-8")
        self.assertIn("Что значит", qs)
        self.assertIn("Хочу проверить закон", qs)

    def test_zz_doubt_answers_restart_growth(self):
        inbox = self.dir / "входящие"
        (inbox / "другая_книга.txt").write_text("Пингвин летает.", encoding="utf-8")
        (inbox / "мои_ответы.txt").write_text("Прямоугольник это фигура. Квадрат это прямоугольник.",
                                              encoding="utf-8")
        self.school.learn()
        o = self.school.org
        self.assertTrue(any("другая_книга.txt" in d for d in o.know.doubts))          # противоречие → сомнение
        self.assertIn("Как правильно", o.answer("В чём ты сомневаешься?"))
        self.assertTrue(o.answer("Является ли квадрат фигурой?").startswith("Да"))    # вывод из ответов
        again = School(self.dir)                                                       # перезапуск
        self.assertTrue(again.org.answer("Дышит ли кит?").startswith("Да"))
        (inbox / "повтор.txt").write_text((CURRICULUM / "01_букварь.txt").read_text(encoding="utf-8"),
                                          encoding="utf-8")
        self.assertTrue(any("уже читала" in line for line in again.learn()))


class GrowsWithMaterial(unittest.TestCase):
    def test_network_grows_file_by_file(self):
        d = Path(tempfile.mkdtemp())
        s = School(d)
        sizes = []
        for p in sorted(CURRICULUM.glob("*.txt")):
            shutil.copy(p, s.inbox / p.name)
            s.learn()
            o = s.org
            sizes.append((len(o.lang.forms), len(o.know.concepts), len(o.rules.rules), len(o.measures.laws)))
        print(f"\n  рост по файлам (формы, понятия, правила, законы): {sizes}")
        self.assertTrue(all(b[0] >= a[0] for a, b in zip(sizes, sizes[1:])))
        self.assertEqual(sizes[0][2:], (0, 0))                     # из букваря ни правил, ни законов
        self.assertGreater(sizes[-1][1], sizes[0][1])
        self.assertGreaterEqual(sizes[-1][3], 5)


if __name__ == "__main__":
    unittest.main()
