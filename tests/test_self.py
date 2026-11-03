"""Проверки я-модели. Главный приём — «повреждение» (lesion): та же система, но я-модель
видит шум вместо реальных состояний. Если в норме самоотчёты совпадают с реальностью,
а при повреждении — нет, значит, они действительно опираются на внутренние состояния,
а не на заученные фразы."""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from light import Brain  # noqa: E402
from light.cortex import CorticalNode  # noqa: E402
from light.nodes import Node  # noqa: E402
from light.signals import Signal  # noqa: E402

KNOWN = ["кот спит на тёплом окне", "собака бежит по зелёной траве",
         "птица поёт на высокой ветке", "рыба плывёт в холодной реке"]
NEW_WORDS = "жираф летит над синим морем облако читает старую книгу лампа поёт тихую песню".split()


def make_brain(lesioned=False, seed=0):
    b = Brain(log=lambda *_: None)
    cx = b.attach(CorticalNode("кора"))
    for _ in range(6):
        for s in KNOWN:
            cx.read(s)
    b.self_model.lesioned = lesioned
    b.self_model.rng.seed(seed)
    return b


def ask(b, q):
    """Задать вопрос и вернуть именно ответ на него (а не запоздалый ответ на прошлое)."""
    n = len(b.outputs)
    b.hear(q)
    for _ in range(10):
        b.tick()
        own = [o for o in b.outputs[n:] if o.data.get("introspective")]
        if own:
            return own[0]
    raise AssertionError(f"нет ответа на {q!r}")


def feel_accuracy(lesioned, trials=16, seed=1):
    rng = random.Random(seed)
    b, ok = make_brain(lesioned, seed), 0
    for i in range(trials):
        novel = i % 2 == 0
        b.hear(" ".join(rng.sample(NEW_WORDS, 5)) if novel else rng.choice(KNOWN))
        b.run(8, stop_on_output=True)
        said = "удивление" in ask(b, "что ты сейчас чувствуешь?").data["feelings"]
        ok += said == novel
    return ok / trials


def think_accuracy(lesioned, trials=8, seed=2):
    rng = random.Random(seed)
    b, ok = make_brain(lesioned, seed), 0
    for _ in range(trials):
        s = rng.choice(KNOWN + [" ".join(rng.sample(NEW_WORDS, 5))])
        b.hear(s)
        b.run(8, stop_on_output=True)
        topics = set(ask(b, "о чём ты сейчас думаешь?").data["topics"])
        ok += bool(topics & set(s.split()))
    return ok / trials


def auc(pos, neg):
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def fok_auc(lesioned, seed=3):
    """Чувство знания до попытки вспомнить: различает ли система, что она знает."""
    rng = random.Random(seed)
    b = make_brain(lesioned, seed)
    sm = b.self_model
    known = [" ".join(s.split()[:2]) for s in KNOWN]
    unknown = [" ".join(rng.sample(NEW_WORDS + "кот собака птица рыба".split(), 2)) for _ in range(8)]
    return auc([sm.feeling_of_knowing(b, s) for s in known], [sm.feeling_of_knowing(b, s) for s in unknown])


class Worry(Node):
    """Навязчивая мысль: на любое событие снова предлагает одно и то же с высокой важностью."""
    name = "worry"

    def __init__(self):
        super().__init__()
        self.rng = random.Random(0)

    def on_broadcast(self, focus, brain):
        # каждый раз другими словами, но об одном и том же — дословное привыкание не спасает
        words = "а вдруг утюг остался дома включённым".split()
        self.rng.shuffle(words)
        return [Signal("thought", " ".join(words), self.name, salience=0.9)]


def worry_share(lesioned, ticks=40):
    """Какую долю тактов навязчивая мысль занимает сцену внимания."""
    b = make_brain(lesioned)
    for s in KNOWN:
        b.hear(s)
        b.run(8, stop_on_output=True)
    b.attach(Worry())
    hits = 0
    for _ in range(ticks):
        hits += any(w.source == "worry" for w in b.tick())
    return hits / ticks, b.self_model.switches


class SelfModelTest(unittest.TestCase):
    def test_feelings_are_grounded(self):
        intact, lesion = feel_accuracy(False), feel_accuracy(True)
        print(f"\n  отчёт об удивлении: норма {intact:.2f}, повреждение {lesion:.2f}")
        self.assertGreaterEqual(intact, 0.9)
        self.assertLess(lesion, 0.75)

    def test_thought_reports_are_grounded(self):
        intact, lesion = think_accuracy(False), think_accuracy(True)
        print(f"\n  отчёт «о чём думаю»: норма {intact:.2f}, повреждение {lesion:.2f}")
        self.assertGreaterEqual(intact, 0.85)
        self.assertLess(lesion, intact - 0.3)

    def test_intervention_changes_report(self):
        """Меняем внутреннее состояние напрямую, вход тот же → отчёт меняется."""
        b = make_brain()
        b.hear(KNOWN[0])
        b.run(8, stop_on_output=True)
        calm = ask(b, "что ты сейчас чувствуешь?").data["feelings"]
        cx = b.node("кора")
        cx.last_surprise, cx.region.nm.noradrenaline = 0.0, 0.9
        b.tick()
        alert = ask(b, "что ты сейчас чувствуешь?").data["feelings"]
        self.assertNotIn("настороженность", calm)
        self.assertIn("настороженность", alert)

    def test_feeling_of_knowing(self):
        intact, lesion = fok_auc(False), fok_auc(True)
        print(f"\n  чувство знания (AUC): норма {intact:.2f}, повреждение {lesion:.2f}")
        self.assertGreater(intact, 0.85)
        self.assertLess(lesion, 0.75)

    def test_attention_schema_breaks_rumination(self):
        intact, sw = worry_share(False)
        lesion, _ = worry_share(True)
        print(f"\n  навязчивая мысль на сцене: норма {intact:.2f} тактов (переключений {sw}), "
              f"повреждение {lesion:.2f}")
        self.assertGreater(sw, 0)
        self.assertLess(intact, lesion - 0.3)

    def test_self_knowledge_tracks_structure(self):
        b = make_brain()
        before = ask(b, "кто ты?").data["nodes"]
        b.attach(CorticalNode("зрительная_кора"))
        after = ask(b, "кто ты?").data["nodes"]
        self.assertNotIn("зрительная_кора", before)
        self.assertIn("зрительная_кора", after)

    def test_no_unfounded_claims(self):
        text = ask(make_brain(), "у тебя есть сознание?").text()
        self.assertIn("не могу", text)
        self.assertNotIn("я сознательна", text.lower())


if __name__ == "__main__":
    unittest.main()
