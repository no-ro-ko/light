"""Рассказы обычной речью: события из литературного текста, кто есть кто, учёба на ответах.

Литературный пересказ басни читается БЕЗ упрощённого пересказа (04) — и сначала без словаря (14):
система должна разобрать рассказ, честно не понять «зачем» и спросить о незнакомых словах; после
ответов (14_слова.txt) — понять. Плюс два рассказа с другими героями и словами (на них разбор
проверялся и исправлялся, так что это проверка механизмов, а не слепой тест — см. docs/SCHOOL.md).
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from light.rost.organism import Organism  # noqa: E402

CUR = Path(__file__).resolve().parents[1] / "curriculum"

STORY_APPLE = ("Вечером девочка сидела у окна и держала в руках яблоко. Рядом стоял хитрый мальчик. "
               "Он очень хотел яблоко. Мальчик подошёл к девочке и начал хвалить её. Девочка обрадовалась "
               "и запела. Яблоко выпало у неё из рук. Мальчик поднял яблоко и съел его.")
STORY_CANDY = ("Утром по дороге шёл маленький мальчик. В руке он держал сладкую конфету. На заборе сидела "
               "хитрая сорока. Она сразу захотела конфету. Сорока подлетела к мальчику и стала его хвалить. "
               "Мальчик обрадовался и громко запел. Конфета упала на дорогу. Сорока схватила конфету и улетела.")


def grow(skip=(), extra=()):
    o = Organism()
    for p in sorted(CUR.glob("*.txt")):
        if not p.name.startswith(tuple(skip)):
            o.read(p.name, p.read_text(encoding="utf-8"))
    o.sleep()
    for name, text in extra:
        o.read(name, text)
        o.sleep()
    return o


class LiteraryFable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.o = grow(skip=("04", "14"))
        a = cls.o.answer
        cls.before = {"tell": a("Перескажи рассказ"), "why": a("Зачем лиса хвалила ворону?"),
                      "questions": " ".join(cls.o.my_questions())}
        p = CUR / "14_слова.txt"                                   # ответы на её вопросы
        cls.o.read(p.name, p.read_text(encoding="utf-8"))
        cls.o.sleep()

    def test_events_from_ordinary_prose(self):
        tell = self.before["tell"]
        print("\n  до ответов:", tell)
        self.assertIn("13_басня_литературно", tell)
        for ev in ("ворона держит сыр",      # «держала» ~ «держит»: чередование, выучено само
                   "лиса бежит",             # «Мимо бежала голодная лиса»: кто — после действия
                   "лиса увидела сыр",       # «Она увидела…»: она — лиса
                   "лиса хвалит ворону",     # «стала хвалить ворону»: главное действие — второе
                   "ворона запела"):         # «Глупая птица…»: птица — это ворона
            self.assertIn(ev, tell)
        self.assertNotIn("чудесные", tell)   # прямая речь «…» — сказанное, не событие

    def test_asks_then_learns_from_answers(self):
        self.assertFalse(self.before["why"].startswith("Чтобы"))    # слов «захотела», «запела» не знает
        for w in ("захотела", "обрадовалась", "запела", "выпал", "схватила"):
            self.assertIn(f"«{w}»", self.before["questions"])        # и спрашивает о них
        a = self.o.answer("Зачем лиса хвалила ворону?")
        print("  после ответов:", a)
        self.assertTrue(a.startswith("Чтобы получить сыр"), a)
        self.assertIn("13_басня_литературно", a)
        self.assertIn("потому что ворона поёт", self.o.answer("Почему сыр упал?"))
        self.assertIn("лиса берёт сыр", self.o.answer("Что было бы, если бы лиса не хвалила ворону?"))
        self.assertIn("лесть", self.o.answer("Какая мораль?"))
        self.assertIn("поёт", self.o.answer("Что значит запела?"))
        self.assertNotIn("«захотела»", " ".join(self.o.my_questions()))

    def test_story_is_not_general_knowledge(self):
        about = self.o.answer("Что ты знаешь о лисе?")
        self.assertIn("животное", about)
        self.assertNotIn("схватил", about)           # «лиса схватила сыр» — про этот рассказ, не про всех лис
        stories = {src for src, _ in self.o.stories}
        self.assertNotIn("10_энциклопедия.txt", stories)   # описание («киты дышат…») — не рассказ
        self.assertIn("13_басня_литературно.txt", stories)

    def test_rules_stay_reliable(self):
        texts = {r.text(): info.get("надёжность", 1.0) for r, info in self.o.rules.rules.items()}
        self.assertGreaterEqual(texts.get("X хвалит Y → Y радуется", 0), 0.8)


class NewStories(unittest.TestCase):
    """Другие герои и слова; мотив — из правил, выросших на прежних историях."""
    @classmethod
    def setUpClass(cls):
        cls.o = grow(extra=[("про_яблоко.txt", STORY_APPLE), ("про_конфету.txt", STORY_CANDY)])

    def test_apple(self):
        a = self.o.answer("Зачем мальчик хвалил девочку?")
        print("\n ", a)
        self.assertTrue(a.startswith("Чтобы получить яблоко"), a)   # «её» — девочка (род из опыта)
        self.assertIn("про_яблоко", a)
        self.assertIn("«поднял»", " ".join(self.o.my_questions()))   # незнакомое — спрашивает

    def test_candy(self):
        a = self.o.answer("Зачем сорока хвалила мальчика?")
        print(" ", a)
        self.assertTrue(a.startswith("Чтобы получить конфету"), a)  # «стала его хвалить»; «она» — сорока
        self.assertIn("мальчик держит конфету", self.o.answer("Перескажи рассказ"))


# эталоны записаны ДО первого запуска; «щенок и кошка» после этого не подгонялся (слепая проверка)
STORY_JUG = ("Вечером на краю стола стоял кувшин с молоком. Он сильно шатался. Голодный кот давно хотел "
             "молока. Кот запрыгнул на стол и толкнул кувшин лапой. Кувшин упал и разбился. Молоко "
             "разлилось по полу. Довольный кот быстро вылакал всё молоко.")
GOLD_JUG = ["кувшин стоял", "кувшин шатался", "кот хотел молоко", "кот запрыгнул", "кот толкнул кувшин",
            "кувшин упал", "кувшин разбился", "молоко разлилось", "кот вылакал молоко"]
STORY_PUPPY = ("Жил-был маленький щенок. Однажды он нашёл во дворе старый мяч. Щенок толкнул мяч носом, и мяч "
               "покатился. Рядом гуляла кошка. Она увидела мяч и побежала за ним. Кошка догнала мяч и спрятала "
               "его под скамейкой. Щенок долго искал свой мяч.")
GOLD_PUPPY = ["щенок жил", "щенок нашёл мяч", "щенок толкнул мяч", "мяч покатился", "кошка гуляла",
              "кошка увидела мяч", "кошка побежала", "кошка догнала мяч", "кошка спрятала мяч", "щенок искал мяч"]


def score(o, name, gold):
    """Сколько событий эталона найдено верно: кто, что делает (как написано), над кем."""
    items = [(i, s) for i, (src, s) in enumerate(o.memory) if src == name]
    norm = lambda w: (w or "").replace("ё", "е")      # noqa: E731
    got = {(norm(o.name(e.agent)), norm(e.forms[1]), norm(o.name(e.obj)) if e.obj else "")
           for e in o.narr.events(o, items)}
    want = [tuple(norm(x) for x in (g.split() + [""])[:3]) for g in gold]
    return sum(1 for w in want if w in got), len(want), sorted(got)


class UnseenStories(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.o = grow(extra=[("про_кувшин.txt", STORY_JUG), ("про_щенка.txt", STORY_PUPPY)])

    def test_jug_after_fixes(self):
        ok, n, got = score(self.o, "про_кувшин.txt", GOLD_JUG)
        print(f"\n  «кот и кувшин» (на нём разбор исправлялся): верно {ok} из {n}")
        self.assertGreaterEqual(ok, 8)

    def test_puppy_blind(self):
        ok, n, got = score(self.o, "про_щенка.txt", GOLD_PUPPY)
        print(f"\n  «щенок и кошка» (слепой рассказ): верно {ok} из {n}; понято: {got}")
        self.assertGreaterEqual(ok, 6)


class KrylovHonestly(unittest.TestCase):
    def test_original_is_not_understood(self):
        o = grow(skip=("04", "13"))                  # только оригинал Крылова (и словарь)
        self.assertFalse(o.answer("Зачем лиса хвалила ворону?").startswith("Чтобы"))
        st = o.files["05_басня_крылов.txt"]
        print(f"\n  Крылов: события нашла в {st['понято']} строках из {st['предложений']}")
        self.assertLess(st["понято"], st["предложений"])


if __name__ == "__main__":
    unittest.main()
