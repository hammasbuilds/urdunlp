"""Which language is this? Two questions Urdu text raises that script cannot answer.

**Perso-Arabic script is not Urdu.** `is_urdu` checks the script, and eleven
Wikipedias are written in it - five of them by people in Pakistan. Punjabi
(Shahmukhi), Saraiki, Sindhi, Pashto and Kashmiri all pass a script check, and so do
Persian, Arabic, Central Kurdish, Uyghur and South Azerbaijani. `identify_language`
tells them apart with a character n-gram model trained on Wikipedia in each of them,
and reports the letters that only some of them use (Sindhi ڪ ٻ, Pashto ښ ځ, Arabic ة)
as evidence a reader can check.

**Roman Urdu is full of English.** "kal meeting cancel ho gayi" is two languages in
one alphabet, and nothing in the spelling marks the switch. `tag_roman_tokens` labels
each word `ur` or `en` with two models per language - a word-frequency table and a
character model for unseen words - smoothed over the sentence by a two-state Viterbi
pass, because the most frequent words are the ambiguous ones: `is`, `to`, `me`, `the`
and `hai` are all real words in both.

Both run on the raw text. `normalize` rewrites Arabic ي and ك as their Urdu forms,
which is right for Urdu and destroys exactly the evidence that says a text is Arabic.
Accuracy, measured on held-out data, is in docs/CORPUS.md.
"""

from __future__ import annotations

import functools
import gzip
import json
import math
import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

from .normalize import _require_str

LANGUAGES: dict[str, str] = {
    "ur": "Urdu",
    "pnb": "Punjabi (Shahmukhi)",
    "skr": "Saraiki",
    "sd": "Sindhi",
    "ps": "Pashto",
    "ks": "Kashmiri",
    "fa": "Persian",
    "ar": "Arabic",
    "ckb": "Central Kurdish",
    "ug": "Uyghur",
    "azb": "South Azerbaijani",
}

_ARABIC_RUNS = re.compile("[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿‌]+")
_ROMAN_WORD = re.compile(r"[A-Za-z]+")


def _load(name: str) -> dict[str, Any]:
    blob = resources.files("urdunlp").joinpath("data").joinpath(name).read_bytes()
    data: dict[str, Any] = json.loads(gzip.decompress(blob))
    return data


def _grams(text: str, n_max: int) -> Iterator[str]:
    padded = f" {text} "
    for n in range(1, n_max + 1):
        for i in range(len(padded) - n + 1):
            gram = padded[i : i + n]
            if n == 1 or gram.strip():
                yield gram


class _ScriptModel:
    def __init__(self, data: dict[str, Any]) -> None:
        self.n_max: int = data["n_max"]
        self.alpha: float = data["alpha"]
        self.vocabulary_size: int = data["vocabulary_size"]
        self.counts: dict[str, dict[str, int]] = data["counts"]
        self.totals = {lang: sum(c.values()) for lang, c in self.counts.items()}
        self.distinctive: dict[str, list[str]] = data["distinctive"]

    def log_likelihoods(self, text: str) -> dict[str, float]:
        grams: dict[str, int] = {}
        for gram in _grams(text, self.n_max):
            grams[gram] = grams.get(gram, 0) + 1
        out = {}
        for lang, counts in self.counts.items():
            denominator = math.log(self.totals[lang] + self.alpha * self.vocabulary_size)
            out[lang] = sum(
                k * (math.log(counts.get(g, 0) + self.alpha) - denominator)
                for g, k in grams.items()
            )
        return out


@functools.lru_cache(maxsize=1)
def _script_model() -> _ScriptModel:
    return _ScriptModel(_load("langid.json.gz"))


@dataclass(frozen=True)
class LanguageGuess:
    """The result of `identify_language`.

    `language` is an ISO 639 code from `LANGUAGES`, or `None` when the text holds no
    Perso-Arabic letters at all. `margin` is how much more likely, per character,
    the winner is than the runner-up, in nats. **Do not read it as confidence.** On
    held-out 20-character windows, 768 of 771 guesses had a margin above 0.1 - and
    so did 33 of the 35 wrong ones: naive Bayes is confidently wrong when Urdu,
    Punjabi and Saraiki share every word in a short window. What predicts an error
    is length, not margin; see the accuracy by length below. `evidence` lists
    letters in the text that only a few of the eleven languages use, keyed by those
    languages.
    """

    language: str | None
    name: str | None
    margin: float
    ranking: list[tuple[str, float]] = field(repr=False)
    evidence: dict[str, list[str]] = field(default_factory=dict)


def identify_language(text: str) -> LanguageGuess:
    """Which of eleven Perso-Arabic-script languages a text is written in.

    >>> identify_language("یہ کتاب میری ہے اور میں اسے پڑھ رہا ہوں۔").language
    'ur'
    >>> identify_language("هذا الكتاب لي وأنا أقرأه الآن").language
    'ar'

    Only the Perso-Arabic letters in the text are used; Latin words, digits and
    punctuation are ignored. Accuracy falls with length, and falls fastest between
    the three closest languages - Urdu, Punjabi and Saraiki share most of their
    letters and much of their vocabulary. On held-out Wikipedia paragraphs it is
    right 99.5% of the time; on 50 characters 99.1%, on 20 95.5%, on ten 88.3%.
    """
    _require_str(text, "identify_language")
    runs = " ".join(_ARABIC_RUNS.findall(unicodedata.normalize("NFC", text)))
    if not runs.strip():
        return LanguageGuess(None, None, 0.0, [])
    model = _script_model()
    scores = model.log_likelihoods(runs)
    ranking = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, runner_up = ranking[0], ranking[1]
    margin = (best[1] - runner_up[1]) / max(1, len(runs))
    letters = set(runs)
    evidence = {
        lang: sorted(letters & set(chars))
        for lang, chars in model.distinctive.items()
        if letters & set(chars)
    }
    return LanguageGuess(
        language=best[0],
        name=LANGUAGES[best[0]],
        margin=round(margin, 4),
        ranking=ranking,
        evidence=evidence,
    )


# --- Roman script: Urdu or English, word by word ----------------------------------


class _WordModel:
    """P(word | language): a unigram table interpolated with a character model."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.unigram: dict[str, float] = data["unigram"]
        total = sum(self.unigram.values())
        self.unigram = {w: c / total for w, c in self.unigram.items()}
        self.order: int = data["order"]
        self.chars: dict[str, dict[str, int]] = data["chars"]
        self.char_totals = {ctx: sum(c.values()) for ctx, c in self.chars.items()}
        self.weight: float = data["unigram_weight"]

    def char_log_prob(self, word: str) -> float:
        padded = "^" * (self.order - 1) + word + "$"
        total = 0.0
        for j in range(self.order - 1, len(padded)):
            p = 1 / 27
            for o in range(self.order):
                ctx = padded[j - o : j]
                counts = self.chars.get(ctx)
                if not counts:
                    continue
                n, types = self.char_totals[ctx], len(counts)
                keep = n / (n + types)  # Witten-Bell
                p = keep * counts.get(padded[j], 0) / n + (1 - keep) * p
            total += math.log(p)
        return total

    def log_prob(self, word: str) -> float:
        return math.log(
            self.weight * self.unigram.get(word, 0.0)
            + (1 - self.weight) * math.exp(self.char_log_prob(word))
        )


class _Tagger:
    def __init__(self, data: dict[str, Any]) -> None:
        self.urdu = _WordModel(data["ur"])
        self.english = _WordModel(data["en"])
        self.stay = math.log(data["stay"])
        self.switch = math.log(1 - data["stay"])
        self.urdu_bias: float = data["urdu_bias"]

    @functools.lru_cache(maxsize=65536)  # noqa: B019 - one tagger per process
    def emissions(self, word: str) -> tuple[float, float]:
        return self.urdu.log_prob(word) + self.urdu_bias, self.english.log_prob(word)

    def tag(self, words: list[str]) -> list[str]:
        if not words:
            return []
        emissions = [self.emissions(w.lower()) for w in words]
        score = list(emissions[0])
        back: list[tuple[int, int]] = []
        for urdu, english in emissions[1:]:
            step = []
            new = []
            for state, emission in ((0, urdu), (1, english)):
                stay = score[state] + self.stay
                move = score[1 - state] + self.switch
                step.append(state if stay >= move else 1 - state)
                new.append(max(stay, move) + emission)
            back.append((step[0], step[1]))
            score = new
        state = 0 if score[0] >= score[1] else 1
        path = [state]
        for pointers in reversed(back):
            state = pointers[state]
            path.append(state)
        return ["ur" if s == 0 else "en" for s in reversed(path)]


@functools.lru_cache(maxsize=1)
def _tagger() -> _Tagger:
    return _Tagger(_load("roman_tagger.json.gz"))


def tag_roman_tokens(text: str) -> list[tuple[str, str]]:
    """Label each Latin-script word in Roman Urdu text as `ur` or `en`.

    >>> tag_roman_tokens("kal meeting cancel ho gayi")
    [('kal', 'ur'), ('meeting', 'en'), ('cancel', 'en'), ('ho', 'ur'), ('gayi', 'ur')]

    Returns one pair per run of Latin letters, in order; everything else is skipped.
    Names are the weak spot: a name spelled the English way (`Robert`, `Illinois`) is
    labelled `en`, and a name spelled the Urdu way (`Muttahida`) sometimes is too.
    """
    _require_str(text, "tag_roman_tokens")
    words = _ROMAN_WORD.findall(text)
    return list(zip(words, _tagger().tag(words), strict=True))
