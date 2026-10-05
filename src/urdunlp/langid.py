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
import itertools
import math
import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from .normalize import _PRESENTATION_FORMS, IDENTIFIER, _expand_presentation_forms, _require_str

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

# Below this many Perso-Arabic letters a guess is flagged `short`. Held-out accuracy
# by window length: 97.9% on a paragraph, 96.6% on 50 characters, 91.0% on 20, 81.5%
# on 10 (docs/CORPUS.md, section 10).
SHORT_TEXT = 20

_ARABIC_RUNS = re.compile("[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿‌]+")

# A letter typed three or more times in a row (`nahiii`, `bohttt`), and the runs of
# a repeated letter in a word, for `_Tagger.collapse`.
_STRETCHED = re.compile(r"([a-z])\1\1")
_STRETCH_RUN = re.compile(r"([a-z])\1+")

# Words of the curated Roman Urdu lexicon that are also everyday English, and so may
# be tagged either way. Every other lexicon word - hai, nahi, pe, ko, kya - is Urdu
# wherever it appears.
ENGLISH_HOMOGRAPHS = frozenset({
    "a", "ab", "ana", "beta", "chai", "din", "dr", "h", "hay", "he", "hi", "hum", "karen",
    "log", "main", "mat", "me", "mr", "mrs", "o", "or", "par", "prof", "sham", "school",
    "sorry", "tab", "the", "to", "university", "ya",
    # chat English the lexicon spells in Urdu script
    "msg", "ok", "okay", "pls", "plz", "thanks", "thx", "wow",
})  # fmt: skip

# One token of Roman Urdu text, for `tag_roman_tokens`: a URL, email, @mention or
# #hashtag; a code mixing letters and digits (5th, mp3); a number (2.5, 3:30, 10%);
# a Latin word; a run of letters in another script; or one other character.
_ROMAN_TOKEN = re.compile(
    rf"""
      (?P<id>{IDENTIFIER})
    | (?P<code>[A-Za-z0-9]*(?:[A-Za-z][0-9]|[0-9][A-Za-z])[A-Za-z0-9]*)
    | (?P<num>\d+(?:[.,:/-]\d+)*%?)
    | (?P<word>[A-Za-z]+(?![^\W\d_]))  # not the start of café
    | (?P<letters>[^\W\d_]+)
    | (?P<other>\S)
    """,
    re.VERBOSE,
)


def _load(name: str) -> dict[str, Any]:
    # Imported here, not at the top: `import urdunlp` should not pay for them.
    import gzip
    import json
    from importlib import resources

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
    held-out 20-character windows, 1,239 of 1,254 guesses had a margin above 0.1 -
    and so did 102 of the 113 wrong ones: naive Bayes is confidently wrong when Urdu,
    Punjabi and Saraiki share every word in a short window. What predicts an error
    is length, not margin; see the accuracy by length below. `evidence` lists
    letters in the text that only a few of the eleven languages use, keyed by those
    languages.

    `short` is True when the text had fewer than 20 Perso-Arabic letters - one or
    two words. Held-out accuracy there is 91% at best and 81.5% at ten characters,
    and a single word is often a word several of the languages share, so treat a
    short guess as a guess. It is the flag to check; `margin` is not.
    """

    language: str | None
    name: str | None
    margin: float
    ranking: list[tuple[str, float]] = field(repr=False)
    evidence: dict[str, list[str]] = field(default_factory=dict)
    short: bool = False


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
    right 97.9% of the time; on 50 characters 96.6%, on 20 91.0%, on ten 81.5%.
    Urdu itself: 98.7%, 98.0%, 93.3% and 84.0%. Under 20 letters the result has
    `short=True`: on one word, expect a neighbouring language about as often as not.
    """
    _require_str(text, "identify_language")
    # Presentation forms (text copied out of a PDF) become the letters they are
    # shapes of: that changes no letter's identity, so no evidence is lost.
    text = _PRESENTATION_FORMS.sub(_expand_presentation_forms, unicodedata.normalize("NFC", text))
    runs = " ".join(_ARABIC_RUNS.findall(text))
    if not runs.strip():
        return LanguageGuess(None, None, 0.0, [], short=True)
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
        short=sum(c.isalpha() for c in runs) < SHORT_TEXT,
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
        """log P(word), mixing the unigram table with the character model.

        Done in log space on purpose. Computing it as
        `log(w * unigram + (1 - w) * exp(char_log_prob))` raised
        `ValueError: expected a positive input, got 0.0` on any long out-of-vocabulary
        token: `exp` underflowed to 0.0, the unigram term was already 0.0 for an unseen
        word, and `log(0.0)` is undefined. A 136-character base64 run was enough, even
        inside an ordinary sentence - and base64, JWTs, hashes and tracking URLs are
        routine in the code-mixed text this model is for.

        `char_log_prob` is a sum of logs of strictly positive probabilities, so it is
        always finite however long the word is; only exponentiating it was lossy. The
        log-sum-exp below is the same quantity as the old expression wherever the old
        one did not underflow.
        """
        char_part = math.log(1.0 - self.weight) + self.char_log_prob(word)
        unigram = self.unigram.get(word, 0.0)
        if unigram <= 0.0:
            return char_part
        word_part = math.log(self.weight * unigram)
        high, low = max(word_part, char_part), min(word_part, char_part)
        return high + math.log1p(math.exp(low - high))


class _Tagger:
    def __init__(self, data: dict[str, Any]) -> None:
        self.urdu = _WordModel(data["ur"])
        self.english = _WordModel(data["en"])
        self.stay = math.log(data["stay"])
        self.switch = math.log(1 - data["stay"])
        self.urdu_bias: float = data["urdu_bias"]

    @functools.lru_cache(maxsize=65536)  # noqa: B019 - one tagger per process
    def emissions(self, word: str) -> tuple[float, float]:
        if word in self.anchored:
            # A curated Urdu function word that is not also an English word: never
            # English, whatever surrounds it. Without this `WhatsApp pe msg kr do`
            # kept `pe` in Latin, carried along by the English words either side.
            return self.urdu.log_prob(word) + self.urdu_bias, -math.inf
        return self.urdu.log_prob(word) + self.urdu_bias, self.english.log_prob(word)

    @functools.cached_property
    def anchored(self) -> frozenset[str]:
        from .translit import LEXICON  # deferred: translit imports this module lazily

        return frozenset(LEXICON) - ENGLISH_HOMOGRAPHS

    @functools.lru_cache(maxsize=65536)  # noqa: B019 - one tagger per process
    def collapse(self, word: str) -> str:
        """`nahiii` -> `nahi`, `bohttt` -> `boht`: letters stretched for emphasis.

        Chat repeats a letter three or more times for emphasis, and no dictionary
        holds the stretched form, so `nahiii` came out نہی and `bohttt` بہتات
        (*abundance*). Each run of three or more is shortened to two or to one;
        the first form the curated lexicon knows wins, then the form either word
        table has seen most often, then the run shortened to two. A word with no
        such run is returned unchanged, so this never touches ordinary spelling.
        """
        lowered = word.lower()
        if not _STRETCHED.search(lowered):
            return lowered
        from .translit import LEXICON  # deferred: translit imports this module lazily

        # Every run of a repeated letter, with the text between runs: a run of three
        # or more may become two letters or one, a shorter run stays as typed.
        texts, options, end = [], [], 0
        for run in _STRETCH_RUN.finditer(lowered):
            texts.append(lowered[end : run.start()])
            letters = run.group()
            options.append((letters[0] * 2, letters[0]) if len(letters) >= 3 else (letters,))
            end = run.end()
        tail = lowered[end:]
        variants = [
            "".join(t + r for t, r in zip(texts, choice, strict=True)) + tail
            for choice in itertools.islice(itertools.product(*options), 64)
        ]
        for variant in variants:
            if variant in LEXICON:
                return variant
        seen = max(
            variants,
            key=lambda v: max(self.urdu.unigram.get(v, 0.0), self.english.unigram.get(v, 0.0)),
        )
        if self.urdu.unigram.get(seen) or self.english.unigram.get(seen):
            return seen
        return variants[0]

    def tag(self, words: list[str]) -> list[str]:
        if not words:
            return []
        emissions = [self.emissions(self.collapse(w)) for w in words]
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
    """Label every token of Roman Urdu text: `ur` or `en` for a Latin word, and a
    kind for everything else, so the result lines up with the text.

    >>> tag_roman_tokens("kal meeting cancel ho gayi")
    [('kal', 'ur'), ('meeting', 'en'), ('cancel', 'en'), ('ho', 'ur'), ('gayi', 'ur')]
    >>> tag_roman_tokens("kal 3 baje, ok?")
    [('kal', 'ur'), ('3', 'num'), ('baje', 'ur'), (',', 'punct'), ('ok', 'en'), ('?', 'punct')]

    Every character that is not whitespace belongs to exactly one token, in order.
    Latin words are `ur` or `en`; numbers (`3`, `2.5`, `10%`) are `num`; a URL,
    email, @mention or #hashtag is `id`; a code mixing letters and digits (`5th`,
    `mp3`) is `code`; punctuation is `punct`; anything else - emoji, symbols, words
    in another script - is `other`. Only the Latin words are tagged by the model,
    and they are tagged as one sequence, so a number or a comma between two words
    does not break the context. The first version returned the words alone and
    silently dropped `3` and `,`, so its output could not be matched back to the
    text it came from.

    Names are the weak spot: a name spelled the English way (`Robert`, `Illinois`)
    is labelled `en`, and a name spelled the Urdu way (`Muttahida`) sometimes is too.
    """
    _require_str(text, "tag_roman_tokens")
    tokens: list[tuple[str, str]] = []
    for match in _ROMAN_TOKEN.finditer(text):
        kind = match.lastgroup or "other"
        token = match.group()
        if kind == "letters":
            kind = "other"
        elif kind == "other":
            kind = "punct" if unicodedata.category(token).startswith("P") else "other"
        tokens.append((token, kind))
    positions = [i for i, (_, kind) in enumerate(tokens) if kind == "word"]
    tags = _tagger().tag([tokens[i][0] for i in positions])
    for i, tag in zip(positions, tags, strict=True):
        tokens[i] = (tokens[i][0], tag)
    return tokens
