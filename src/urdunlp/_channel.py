"""The noisy-channel model behind Roman -> Urdu transliteration of open vocabulary.

Private: the public interface is `transliterate_with_confidence` and `roman_key`.

Given a Roman word r, pick the Urdu word u that maximises

    P(u) * P(r | u)

where P(u) is the word's frequency in Urdu Wikipedia and P(r | u) is the probability
that someone romanising u would type r. That second factor is a letter-by-letter
emission model: each Urdu letter emits a short Roman string (possibly empty, since ع
and ھ often vanish; possibly with a vowel attached, since Urdu does not write short
vowels), conditioned on the letter's position and on whether the next letter is a
vowel letter. It was trained with EM on the 106,260 attested (Urdu, Roman) pairs of
Dakshina's Urdu training lexicon - see scripts/build_translit_model.py.

Candidates are retrieved by a coarse consonant key that both scripts map to: `h` is
dropped entirely (Roman writers drop and insert it freely), the letters Urdu
distinguishes but Roman cannot (س ص ث ش, ت ط ٹ, ز ذ ض ظ ج, ...) share a class, and
vowel letters are not part of the key. A Roman word can have several keys, because
some Roman letters are ambiguous (`d` is د or ڑ, `z` is ز or س in English loans).

Everything is plain Python over a small bundled table; there is no model download.
"""

from __future__ import annotations

import functools
import gzip
import itertools
import json
import math
from importlib import resources

from .normalize import normalize

MAX_EMIT = 4
_VOWEL_LETTERS = frozenset("اآعءئؤویےں")

# Urdu letter -> key class. Letters absent here (vowel letters, h-letters, ع) are
# not part of the key.
_URDU_CLASS: dict[str, str] = {}
for _letters, _cls in (
    ("بپف", "P"),
    ("تٹط", "T"),
    ("سصشث", "S"),
    ("زذضظژج", "Z"),
    ("چ", "C"),
    ("دڈرڑ", "R"),
    ("کقخگغ", "K"),
    ("ل", "L"),
    ("م", "M"),
    ("ن", "N"),
):
    for _ch in _letters:
        _URDU_CLASS[_ch] = _cls

# Roman letter -> the classes it can stand for. The first is the usual one. Each
# second class was kept only if removing it cost accuracy on the dev sentences:
# g as ج (germany) +0.15 points, s as ز +0.08, c as س +0.21, m as ن +0.01. Two
# more were tried and dropped because they changed nothing - z as س, d as ت.
_ROMAN_CLASSES: dict[str, tuple[str, ...]] = {
    "b": ("P",),
    "p": ("P",),
    "f": ("P",),
    "t": ("T",),
    "s": ("S", "Z"),
    "z": ("Z",),
    "j": ("Z",),
    "d": ("R",),
    "r": ("R",),
    "k": ("K",),
    "q": ("K",),
    "g": ("K", "Z"),
    "c": ("K", "S"),
    "l": ("L",),
    "m": ("M", "N"),
    "n": ("N",),
    "x": ("KS",),
}

_MAX_KEYS = 32


def urdu_key(word: str) -> str:
    """The coarse consonant key of an Urdu word."""
    out: list[str] = []
    for ch in word:
        cls = _URDU_CLASS.get(ch)
        if cls and (not out or out[-1] != cls):
            out.append(cls)
    return "".join(out)


def roman_keys(word: str) -> frozenset[str]:
    """Every coarse key a Roman word could have, most likely first in construction."""
    w = word.lower()
    slots: list[tuple[str, ...]] = []
    i = 0
    while i < len(w):
        if w.startswith("tion", i) or w.startswith("sion", i):
            slots.append(("S",))  # English -tion/-sion is "shan": اسٹیشن
            i += 2
            continue
        if w.startswith("ch", i):
            slots.append(("C",) if w.startswith("chh", i) else ("C", "K"))
            i += 2
            continue
        ch = w[i]
        alts = _ROMAN_CLASSES.get(ch)
        if alts:
            if ch == "n" and i == len(w) - 1 and i > 0 and w[i - 1] in "aeiouy":
                alts = (*alts, "")  # final nasal: ں is a vowel mark, not a consonant
            elif ch == "d" and w.startswith("dh", i):
                alts = ("R",)
            slots.append(alts)
        i += 1
    keys: set[str] = set()
    for combo in itertools.islice(itertools.product(*slots), _MAX_KEYS):
        out: list[str] = []
        for part in combo:
            for cls in part:
                if not out or out[-1] != cls:
                    out.append(cls)
        keys.add("".join(out))
    return frozenset(keys)


def _context(word: str, i: int) -> tuple[str, str, str]:
    n = len(word)
    position = "I" if i == 0 else ("F" if i == n - 1 else "M")
    following = "E" if i == n - 1 else ("V" if word[i + 1] in _VOWEL_LETTERS else "C")
    return word[i], position, following


class Channel:
    """Emission tables, vocabulary and the candidate index, loaded once."""

    def __init__(self, data: dict) -> None:
        self.emit: dict[tuple[str, str, str], dict[str, float]] = {
            tuple(k.split("|")): v for k, v in data["emit"].items()
        }
        self.emit_mid: dict[tuple[str, str], dict[str, float]] = {
            tuple(k.split("|")): v for k, v in data["emit_mid"].items()
        }
        self.emit_letter: dict[str, dict[str, float]] = data["emit_letter"]
        self.prior_weight: float = data["prior_weight"]
        counts: dict[str, int] = dict(data["vocabulary"])
        total = sum(counts.values())
        self.log_prior = {w: math.log(c / total) for w, c in counts.items()}
        self.index: dict[str, list[str]] = {}
        for w in counts:
            self.index.setdefault(urdu_key(w), []).append(w)

    def table(self, key: tuple[str, str, str]) -> dict[str, float]:
        found = self.emit.get(key)
        if found is None:
            found = self.emit_mid.get(key[:2]) or self.emit_letter.get(key[0], {})
        return found

    def log_emission(self, urdu: str, roman: str) -> float:
        """log P(roman | urdu), summed over every monotone letter alignment."""
        n, m = len(urdu), len(roman)
        forward: list[dict[int, float]] = [{} for _ in range(n + 1)]
        forward[0][0] = 1.0
        for i in range(n):
            table = self.table(_context(urdu, i))
            row = forward[i + 1]
            for j, value in forward[i].items():
                for length in range(min(MAX_EMIT, m - j) + 1):
                    p = table.get(roman[j : j + length])
                    if p:
                        row[j + length] = row.get(j + length, 0.0) + value * p
        z = forward[n].get(m, 0.0)
        return math.log(z) if z > 0 else -math.inf

    @functools.lru_cache(maxsize=65536)  # noqa: B019 - one Channel per process
    def best(self, roman: str) -> str | None:
        """The most probable vocabulary word for a lowercase Roman token, or None."""
        best_word, best_score = None, -math.inf
        seen: set[str] = set()
        for key in roman_keys(roman):
            for word in self.index.get(key, ()):
                if word in seen:
                    continue
                seen.add(word)
                # A word more than twice as long as the Roman string, or four times
                # shorter, cannot align under MAX_EMIT anyway; skip the DP.
                if len(word) > 2 * len(roman) + 1 or MAX_EMIT * len(word) < len(roman):
                    continue
                score = self.log_emission(word, roman)
                if score == -math.inf:
                    continue
                score += self.prior_weight * self.log_prior[word]
                if score > best_score:
                    best_word, best_score = word, score
        return best_word


@functools.lru_cache(maxsize=1)
def channel() -> Channel:
    blob = resources.files("urdunlp").joinpath("data", "translit.json.gz").read_bytes()
    return Channel(json.loads(gzip.decompress(blob)))


def resolve(roman: str) -> str | None:
    """Most probable Urdu vocabulary word for a Roman token, normalised, or None."""
    word = channel().best(roman.lower())
    return normalize(word) if word else None
