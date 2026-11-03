"""Кодировщик: слово → разреженный код (SDR), как сетчатка/улитка кодируют сигнал
небольшим числом активных волокон из многих.

Похожие по написанию слова получают частично общие биты (через триграммы букв),
поэтому «небо» и «неба» — соседи, а «небо» и «кот» — нет.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np


def _bits(token: str, n: int, size: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "little")
    return np.random.default_rng(seed).choice(size, n, replace=False)


class WordEncoder:
    def __init__(self, size: int = 1024, word_bits: int = 14, trigram_bits: int = 2):
        self.size = size
        self.word_bits = word_bits
        self.trigram_bits = trigram_bits

    @staticmethod
    def tokens(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def encode(self, word: str) -> np.ndarray:
        sdr = np.zeros(self.size, dtype=bool)
        sdr[_bits("w:" + word, self.word_bits, self.size)] = True
        padded = f"^{word}$"
        for i in range(len(padded) - 2):
            sdr[_bits("t:" + padded[i:i + 3], self.trigram_bits, self.size)] = True
        return sdr
