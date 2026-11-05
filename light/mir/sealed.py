"""Запечатанный набор проверок и журнал обращений к нему.

Правило: подбирать, чинить и настраивать можно только по «черновому» набору (DEV).
К запечатанному (SEALED) обращаются, когда настройка закончена; каждое обращение
дописывается в mir_runs/sealed_log.jsonl — чтобы число «подглядываний» было видно.
Это правило процедурное, а не криптографическое: проверки и модель пишет один автор.
Чтобы перепроверить независимо, достаточно построить набор с другим зерном:
    python3 -m light.mir.bench --sealed-seed 12345
"""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path

from . import probes

DEV_SEED = 1_000_003
SEALED_SEED = 7_000_001
DEV_PER_MODE = 20
SEALED_PER_MODE = 25
LOG = Path(__file__).resolve().parents[2] / "mir_runs" / "sealed_log.jsonl"

_cache: dict = {}


def dev() -> probes.Probes:
    if "dev" not in _cache:
        _cache["dev"] = probes.build(DEV_SEED, DEV_PER_MODE)
    return _cache["dev"]


def sealed(seed: int = SEALED_SEED) -> probes.Probes:
    if ("sealed", seed) not in _cache:
        _cache[("sealed", seed)] = probes.build(seed, SEALED_PER_MODE)
    return _cache[("sealed", seed)]


def _head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=LOG.parent, timeout=10).stdout.strip()
    except Exception:
        return "?"


def evaluate_sealed(model, tag: str, seed: int = SEALED_SEED, log: bool = True, flip: bool = False) -> dict:
    """flip=True: те же кадры, но метроном читается наоборот («мир B» в опытах со сменой закона)."""
    pr = sealed(seed)
    if flip:
        pr = replace(pr, PH=1 - pr.PH)
    res = probes.evaluate(model, pr)
    if log and seed == SEALED_SEED:
        LOG.parent.mkdir(exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "tag": tag, "git": _head(),
                                "macro": round(res["macro"], 4)}, ensure_ascii=False) + "\n")
    return res


def peeks() -> int:
    return sum(1 for _ in open(LOG, encoding="utf-8")) if LOG.exists() else 0
