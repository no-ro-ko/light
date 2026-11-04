"""Школа «Ростка»: кладёте текстовые файлы в папку — он на них учится.

Папки (создаются сами):
  данные/входящие/   ← сюда кладёте .txt (любые тексты; строки с «?» — вопросы к системе)
  данные/изучено/    ← сюда система переносит прочитанное
  данные/ответы/     → ответы.txt (ответы на вопросы из файлов, с объяснениями)
                     → мои_вопросы.txt (что ей непонятно и что важнее всего узнать)
                     → отчёт.txt (как она выросла, сколько поняла в каждом файле)
                     → журнал.txt (что она изменила в себе во сне)
  данные/росток.pkl  — память между запусками

Команды:
  python3 -m light.rost.school начать          # положить стартовый набор текстов во «входящие»
  python3 -m light.rost.school учиться         # прочитать всё новое, поспать, написать ответы
  python3 -m light.rost.school следить         # учиться само, как только появляются файлы
  python3 -m light.rost.school спросить "Дышит ли кит?"
  python3 -m light.rost.school беседа          # разговор: вопросы, а утверждения — как новые знания
  python3 -m light.rost.school отчёт
Ответить на её вопросы можно обычным текстовым файлом во «входящие».
"""
from __future__ import annotations

import argparse
import hashlib
import pickle
import shutil
import sys
import time
from pathlib import Path

from .organism import Organism

CURRICULUM = Path(__file__).resolve().parents[2] / "curriculum"


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1251", "koi8-r"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


class School:
    def __init__(self, root: str | Path = "данные"):
        self.root = Path(root)
        self.inbox, self.done, self.out = self.root / "входящие", self.root / "изучено", self.root / "ответы"
        self.state = self.root / "росток.pkl"
        for d in (self.inbox, self.done, self.out):
            d.mkdir(parents=True, exist_ok=True)
        if self.state.exists():
            with open(self.state, "rb") as f:
                st = pickle.load(f)
            self.org, self.seen, self.answered = st["organism"], st["seen"], st["answered"]
        else:
            self.org, self.seen, self.answered = Organism(), set(), set()

    def save(self) -> None:
        tmp = self.state.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"organism": self.org, "seen": self.seen, "answered": self.answered}, f)
        tmp.replace(self.state)

    # ---------- учёба ----------
    def learn(self) -> list[str]:
        log = []
        files = sorted(p for p in self.inbox.iterdir() if p.is_file() and p.suffix.lower() in (".txt", ".md"))
        for path in files:
            text = _read_text(path)
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
            if digest in self.seen:
                log.append(f"«{path.name}»: уже читала — пропускаю")
            else:
                st = self.org.read(path.name, text)
                self.seen.add(digest)
                log.append(f"«{path.name}»: предложений {st['предложений']}, понято {st['понято']}, "
                           f"новых фактов {st['новых фактов']}, новых правил {st['новых правил']}, "
                           f"вопросов {st['вопросов в файле']}")
            target = self.done / path.name
            if target.exists():
                target = self.done / f"{path.stem}_{int(time.time())}{path.suffix}"
            shutil.move(str(path), target)
        if not files:
            return log
        log += ["СОН: " + n for n in self.org.sleep()]
        self._write()
        self.save()
        return log

    def _write(self) -> None:
        new = []
        for src, q in self.org.incoming_questions:
            key = (src, q)
            if key in self.answered:
                continue
            self.answered.add(key)
            new.append(f"? {q}   (из «{src}»)\n→ {self.org.answer(q)}\n")
        if new:
            with open(self.out / "ответы.txt", "a", encoding="utf-8") as f:
                f.write("\n".join(new) + "\n")
        (self.out / "мои_вопросы.txt").write_text(
            "Вопросы, которые у меня есть (ответить можно обычным текстовым файлом во «входящие»):\n\n"
            + "\n".join(f"- {q}" for q in self.org.my_questions()) + "\n", encoding="utf-8")
        (self.out / "отчёт.txt").write_text(self.org.report() + "\n", encoding="utf-8")
        (self.out / "журнал.txt").write_text(
            "Что я изменила в себе (во сне):\n\n" + "\n".join(f"- {j}" for j in self.org.journal)
            + "\n\nОкончания и обороты, которые я выучила:\n\n"
            + "\n".join(f"- {j}" for j in self.org.lang.journal) + "\n", encoding="utf-8")

    def watch(self, every: float = 5.0, rounds: int | None = None) -> None:
        print(f"слежу за папкой «{self.inbox}» (Ctrl+C — остановить)")
        k = 0
        while rounds is None or k < rounds:
            for line in self.learn():
                print(line)
            k += 1
            time.sleep(every)

    def start(self) -> list[str]:
        """Положить стартовый набор текстов во «входящие»."""
        copied = []
        for p in sorted(CURRICULUM.glob("*.txt")):
            shutil.copy(p, self.inbox / p.name)
            copied.append(p.name)
        return copied


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Школа «Ростка»: учится на текстовых файлах")
    ap.add_argument("команда", choices=["начать", "учиться", "следить", "спросить", "беседа", "отчёт"])
    ap.add_argument("текст", nargs="?", default="")
    ap.add_argument("--папка", default="данные")
    ap.add_argument("--раз-в", type=float, default=5.0, dest="every")
    a = ap.parse_args(argv)
    s = School(a.папка)
    if a.команда == "начать":
        print("положила во «входящие»:", ", ".join(s.start()))
        print("теперь: python3 -m light.rost.school учиться")
    elif a.команда == "учиться":
        for line in s.learn() or ["новых файлов нет"]:
            print(line)
        print(f"ответы, мои вопросы и отчёт — в «{s.out}»")
    elif a.команда == "следить":
        s.watch(a.every)
    elif a.команда == "спросить":
        print(s.org.answer(a.текст))
    elif a.команда == "отчёт":
        print(s.org.report())
    else:
        print("Пишите вопросы (с «?») или новые знания. Пустая строка — мои вопросы. «выход» — конец.")
        while True:
            try:
                line = input("вы> ").strip()
            except EOFError:
                break
            if line in ("выход", "/выход", "exit"):
                break
            if not line:
                print("росток>", "\n         ".join(s.org.my_questions()[:5]) or "вопросов нет")
            elif line.endswith("?"):
                print("росток>", s.org.answer(line))
            else:
                st = s.org.read(f"беседа {time.strftime('%Y-%m-%d %H:%M')}", line)
                notes = s.org.sleep()
                print("росток>", f"запомнила (понято {st['понято']} из {st['предложений']})"
                      + ("; " + "; ".join(notes[-3:]) if notes else ""))
        s._write()
        s.save()


if __name__ == "__main__":
    main(sys.argv[1:])
