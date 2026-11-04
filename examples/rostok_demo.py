"""«Росток»: сеть рождается пустой и растёт по материалу; во сне сама себя улучшает.

python3 examples/rostok_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from light.rost.fabric import Rostok  # noqa: E402
from light.rost.rules import NEW, STORIES, RuleField  # noqa: E402
from light.world import EVERYDAY, Nursery  # noqa: E402

print("=== 1. Рождение и рост: образы и слова ===")
world = Nursery(2, kinds=EVERYDAY)
net = Rostok()
print(f"при рождении: узлов {len(net.units)}, слов {len(net.words)}, связей {len(net.links)}")
for step in range(2000):
    scene, text, point = world.episode()
    net.experience([o.look for o in scene], text, point)
    if step + 1 in (10, 100, 500, 2000):
        print(f"после {step + 1:4d} моментов: устойчивых узлов {net.stable_units():2d} (всего {len(net.units):2d}), "
              f"слов {len(net.words):2d}, связей {sum(len(v) for v in net.links.values()):3d}")
print("\nчто я в себе изменила во сне (первые записи):")
for note in net.journal[:6]:
    print("  ", note)
print("\n" + net.report())
for kind in ("собака", "пингвин", "яблоко"):
    print(f"  вижу новую особь «{kind}» → называю: {net.name(world.instance(kind).look)[0]}")

print("\n=== 2. Правила вырастают из историй и уточняются сами ===")
rules = RuleField()
for story in STORIES:
    rules.read(story)
rules.sleep()
for note in rules.journal:
    print("  ", note)
print("\nновая история (герои незнакомы): предсказываю, что будет дальше")
for story in NEW[:2]:
    evs = [e for e in (rules.parse(s) for s in story) if e]
    for t in range(1, len(evs)):
        p, rule = rules.predict(evs[:t])
        if p:
            mark = "✓" if p == evs[t] else "✗"
            print(f"   после «{' '.join(w for w in (evs[t - 1].agent, evs[t - 1].action, evs[t - 1].obj) if w)}» "
                  f"жду «{' '.join(w for w in (p.agent, p.action, p.obj) if w)}» {mark}  (по правилу: {rule.text()})")
