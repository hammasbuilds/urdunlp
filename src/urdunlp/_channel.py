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

Two refinements, each measured on Dakshina's dev sentences:

  * **A word's own spellings.** For the ~24,000 words in the training lexicon, the
    spellings annotators actually wrote are mixed into P(r | u). The letter model
    alone rates `ke` as a spelling of کہ at log-probability -6.6, because a final ہ
    is rarely typed as `e`; people type کہ as `ke` all the time.
  * **The word before.** `decode` picks the whole sentence at once, by Viterbi over
    each word's top candidates, with a word-bigram model (344,258 bigrams, absolute
    discounting) in place of the plain frequency prior. That is what separates کہ
    (*that*) from کے (*of*): after کہا (*said*) it is almost always کہ.

Word by word, held-out test accuracy is 88.4%; decoding the sentence, 90.7%.

Everything is plain Python over a bundled table; there is no model download.
"""

from __future__ import annotations

import functools
import gzip
import itertools
import json
import math
from importlib import resources
from typing import Any

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

    def __init__(self, data: dict[str, Any]) -> None:
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

        # A word's own attested spellings, mixed with the letter model.
        self.kappa: float = data["kappa"]
        self.attested: dict[str, dict[str, int]] = data["attested"]
        self.attested_total = {u: sum(r.values()) for u, r in self.attested.items()}
        self.by_spelling: dict[str, list[str]] = {}
        for urdu, spellings in self.attested.items():
            if urdu in counts:
                for roman in spellings:
                    self.by_spelling.setdefault(roman, []).append(urdu)

        # The word-bigram model: interpolated absolute discounting, backing off to
        # the word's add-half smoothed frequency.
        self.lm_weight: float = data["lm_weight"]
        self.lexicon_bonus: float = data["lexicon_bonus"]
        self.discount: float = data["discount"]
        self.k: int = data["candidates"]
        self.ids: dict[str, int] = {w: i for i, w in enumerate(data["words"])}
        self.context = {int(k): v for k, v in data["context"].items()}
        self.bigrams: dict[int, dict[int, int]] = {
            int(k): dict(zip(flat[0::2], flat[1::2], strict=True))
            for k, flat in data["bigrams"].items()
        }
        denominator = total + 0.5 * len(counts)
        self.unigram = {w: (c + 0.5) / denominator for w, c in counts.items()}
        self.unigram_floor = 0.5 / denominator

    def table(self, key: tuple[str, str, str]) -> dict[str, float]:
        found = self.emit.get(key)
        if found is None:
            found = self.emit_mid.get(key[:2]) or self.emit_letter.get(key[0], {})
        return found

    def log_emission(self, urdu: str, roman: str) -> float:
        """log P(roman | urdu) under the letter model, over every monotone alignment."""
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

    def log_spelling(self, urdu: str, roman: str) -> float:
        """log P(roman | urdu): the word's own attested spellings, backed off to letters.

        The letter model has learnt that a final ہ is rarely typed as `e`. True of
        the letter, false of کہ, which annotators wrote as `ke`. With its own
        spellings mixed in, کہ can be reached from `ke` - and the bigram model then
        decides between it and کے.
        """
        letters = self.log_emission(urdu, roman)
        seen = self.attested_total.get(urdu)
        if not seen:
            return letters
        p_letters = math.exp(letters) if letters > -700 else 0.0
        p = (self.attested[urdu].get(roman, 0) + self.kappa * p_letters) / (seen + self.kappa)
        return math.log(p) if p > 0 else -math.inf

    @functools.lru_cache(maxsize=65536)  # noqa: B019 - one Channel per process
    def candidates(self, roman: str) -> tuple[tuple[str, float], ...]:
        """The k likeliest words for a lowercase Roman token, with log P(roman | word).

        Ranked by P(roman | word) * P(word), best first.
        """
        attested = set(self.by_spelling.get(roman, ()))
        pool = set(attested)
        for key in roman_keys(roman):
            pool.update(self.index.get(key, ()))
        scored = []
        for word in pool:
            # A word more than twice as long as the Roman string, or four times
            # shorter, cannot align under MAX_EMIT; skip the DP unless attested.
            too_far = len(word) > 2 * len(roman) + 1 or MAX_EMIT * len(word) < len(roman)
            if too_far and word not in attested:
                continue
            emission = self.log_spelling(word, roman)
            if emission == -math.inf:
                continue
            scored.append((emission + self.prior_weight * self.log_prior[word], emission, word))
        scored.sort(reverse=True)
        return tuple((w, e) for _, e, w in scored[: self.k])

    def best(self, roman: str) -> str | None:
        """The most probable vocabulary word for a lowercase Roman token, or None."""
        found = self.candidates(roman)
        return found[0][0] if found else None

    def log_bigram(self, word: str, previous: str) -> float:
        """log P(word | previous word)."""
        p_word = self.unigram.get(word, self.unigram_floor)
        v = self.ids.get(previous)
        row = self.context.get(v) if v is not None else None
        if v is None or not row:
            return math.log(p_word)
        total, types = row
        w = self.ids.get(word)
        count = self.bigrams.get(v, {}).get(w, 0) if w is not None else 0
        p = max(count - self.discount, 0) / total + self.discount * types / total * p_word
        return math.log(p)

    def decode(self, romans: list[str], lexicon: dict[str, str]) -> list[tuple[str, str]]:
        """The most probable Urdu for a run of lowercase Roman words, by Viterbi.

        Each word offers its curated-lexicon entry, if it has one, and its top
        candidates; the path maximises the sum of log P(roman | word) and a weighted
        log P(word | previous word). Returns (urdu, source) per Roman word; source is
        `lexicon`, `vocabulary`, or `rules` when there was nothing to choose from.
        """
        from .translit import _apply_rules  # deferred: translit imports this module

        # One layer per word: last Urdu word on the path -> (score, previous key,
        # choice). Back-pointers, not paths: carrying the whole path in every state
        # copied it at every step, and 50,000 words without a full stop took 36 s.
        layers: list[dict[str, tuple[float, str, tuple[str, str]]]] = []
        states: dict[str, float] = {"<s>": 0.0}
        for roman in romans:
            options = [(w, e, "vocabulary") for w, e in self.candidates(roman)]
            if roman in lexicon:
                entry = normalize(lexicon[roman])
                best_emission = max((e for _, e, _ in options), default=0.0)
                options = [(entry, best_emission + self.lexicon_bonus, "lexicon")] + [
                    option for option in options if option[0] != entry
                ]
            if not options:
                options = [(normalize(_apply_rules(roman)), 0.0, "rules")]
            layer: dict[str, tuple[float, str, tuple[str, str]]] = {}
            for urdu, emission, source in options:
                parts = urdu.split() or [urdu]
                best_score, best_previous = -math.inf, "<s>"
                for previous, score in states.items():
                    total, last = score + emission, previous
                    for part in parts:
                        total += self.lm_weight * self.log_bigram(part, last)
                        last = part
                    if total > best_score:
                        best_score, best_previous = total, previous
                key = parts[-1]
                if key not in layer or best_score > layer[key][0]:
                    layer[key] = (best_score, best_previous, (urdu, source))
            layers.append(layer)
            states = {key: entry[0] for key, entry in layer.items()}
        if not layers:
            return []
        key = max(states, key=states.__getitem__)
        path: list[tuple[str, str]] = []
        for layer in reversed(layers):
            _, key_before, choice = layer[key]
            path.append(choice)
            key = key_before
        return path[::-1]


@functools.lru_cache(maxsize=1)
def channel() -> Channel:
    blob = resources.files("urdunlp").joinpath("data").joinpath("translit.json.gz").read_bytes()
    return Channel(json.loads(gzip.decompress(blob)))


def resolve(roman: str) -> str | None:
    """Most probable Urdu vocabulary word for a Roman token on its own, or None."""
    word = channel().best(roman.lower())
    return normalize(word) if word else None
