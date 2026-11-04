"""Работа с системой без Claude: python -m light.cli

Пишите фразы («кит это животное»), вопросы («дышит ли кит?», «почему ты так думаешь?»,
«что ты знаешь?», «в чём ты ошибалась?», «что ты хочешь узнать?») или команды:

  (пустая строка) / спроси   — пусть сама задаст вопрос
  /читать ФАЙЛ               — прочитать текст из файла (и честно сказать, что поняла)
  /лаборатория               — самостоятельные опыты: что тонет и почему
  /задачи                    — физика и геометрия: сама ищет законы
  /прожить N                 — ещё N эпизодов «детства»
  /сохранить, /выход
"""
from __future__ import annotations

import argparse
import sys

from .session import Session


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Light: автономный разум")
    ap.add_argument("--state", default="light_state.pkl", help="файл памяти")
    ap.add_argument("--childhood", type=int, default=2000, help="эпизодов «детства» при первом запуске")
    ap.add_argument("--script", help="файл с репликами вместо ввода с клавиатуры")
    args = ap.parse_args(argv)
    s = Session(args.state, childhood=args.childhood)
    print(f"light> {'Я помню прошлые сеансы' if s.loaded else 'Я родилась и прожила детство'}: "
          f"{len(s.mind.perception.ids())} понятий-образов, "
          f"{len(s.mind.lexicon.nouns | s.mind.lexicon.predicates)} слов.")
    lines = open(args.script, encoding="utf-8").read().splitlines() if args.script else None
    while True:
        if lines is not None:
            if not lines:
                break
            line = lines.pop(0)
            print(f"вы> {line}")
        else:
            try:
                line = input("вы> ")
            except EOFError:
                break
        line = line.strip()
        if line in ("/выход", "/exit"):
            break
        if line in ("", "/спроси"):
            out = s.idle() or ["Сейчас мне нечего спросить."]
        elif line.startswith("/читать "):
            r = s.read(open(line.split(" ", 1)[1], encoding="utf-8").read())
            out = [f"{k}: {v}" for k, v in r.items()]
        elif line == "/лаборатория":
            out = s.lab()
        elif line == "/задачи":
            out = s.problems()
        elif line.startswith("/прожить"):
            out = [s.live(int(line.split()[1]) if len(line.split()) > 1 else 500)]
        elif line == "/сохранить":
            out = [f"сохранено в {s.save()}"]
        else:
            out = s.say(line) or ["…"]
        for o in out:
            print(f"light> {o}")
    print(f"light> память сохранена в {s.save()}")


if __name__ == "__main__":
    main(sys.argv[1:])
