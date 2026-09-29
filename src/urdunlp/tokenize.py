"""Urdu tokenisation.

Two things make this harder than splitting on spaces.

**Sentence boundaries.** Urdu ends a sentence with ۔ (U+06D4), not a full stop. Text
that has been through a Latin-centric pipeline often contains both, sometimes in the
same document, so both are handled.

**Space is unreliable.** Urdu is written in a cursive script where the visual word
boundary comes from letter joining, not from a space character. Non-joining letters
(ا د ڈ ذ ر ڑ ز ژ و ے) break the connection on their own, so a typist can omit a space
after them and the text still looks correct - "میں نے" typed as "میںنے" is rare, but
"کر دیا" typed as "کردیا" is extremely common. The reverse also happens: a space
inserted inside one word, because the typist paused mid-word.

Neither is solvable in general without a lexicon and a language model. What is
solvable, and what actually helps downstream, is handling the frequent closed-class
cases - which is what `fix_spacing` does, from an explicit and auditable list.
"""

from __future__ import annotations

import re

from .normalize import IDENTIFIER, _require_str, normalize

# Urdu sentence terminators, plus their Latin equivalents for mixed text.
SENTENCE_END = "۔؟!?."

# Letters that never join to the following letter. A word boundary after one of these
# is invisible, which is why spaces get dropped after them.
NON_JOINING = set("ادڈذرڑزژوے")

_SENTENCE_SPLIT = re.compile(rf"(?<=[{re.escape(SENTENCE_END)}])\s+")

# Urdu punctuation lives *inside* the Arabic block, interleaved with the letters, so
# a range like ؀-ۿ silently swallows it. The letter ranges below skip each
# punctuation codepoint individually: U+060C comma, U+061B semicolon, U+061F question
# mark, U+066A-U+066D signs, U+06D4 full stop.
_URDU_LETTERS = (
    "ؠ-ي"  # letters
    "ً-ْ"  # diacritics (normalise usually removes these)
    "٠-٩"  # Arabic-Indic digits
    "ٮ-ۓ"  # more letters
    "ە-ۿ"  # letters, Urdu digits, marks
    "ݐ-ݿﭐ-﷿ﹰ-﻿"  # supplements and presentation forms
    "‌"  # ZWNJ is word-internal in Urdu compounds
)
_WORD = re.compile(rf"[\w{_URDU_LETTERS}]+")
_PUNCT = re.compile(rf"[^\w\s{_URDU_LETTERS}]")

# URLs, emails, @mentions and #hashtags (IDENTIFIER, shared with the transliterator
# and remove_urls_and_mentions), and numbers written with separators (2.5,
# 12,34,567, 3:30, ۱۲٫۵) are one token whatever punctuation they contain. `words`
# used to split `test@x.com` into test, x, com and 2.5 into 2 and 5, while the
# transliterator kept both whole - two parts of one library disagreeing about what
# a token is.
NUMBER = r"\d+(?:[.,:/٫٬-]\d+)+"
_TOKEN = re.compile(rf"{IDENTIFIER}|{NUMBER}|{_WORD.pattern}")
_TOKEN_OR_PUNCT = re.compile(rf"{IDENTIFIER}|{NUMBER}|{_WORD.pattern}|{_PUNCT.pattern}")

# Very common compounds written both ways. Split them, because the two-token form is
# what a tagger, a stemmer and an embedding model all expect.
#
# Kept deliberately short and explicit rather than derived from a frequency list: an
# aggressive splitter does more damage than an incomplete one, and every entry here
# can be checked by a reader who knows Urdu.
COMMON_MERGES = {
    "کردیا": "کر دیا",
    "کردی": "کر دی",
    "کردیں": "کر دیں",
    "کرلیا": "کر لیا",
    "کرلی": "کر لی",
    "ہوگیا": "ہو گیا",
    "ہوگئی": "ہو گئی",
    "ہوگئے": "ہو گئے",
    "دےدیا": "دے دیا",
    "لےلیا": "لے لیا",
    "آگیا": "آ گیا",
    "چلاگیا": "چلا گیا",
    "جارہا": "جا رہا",
    "جارہی": "جا رہی",
    "آرہا": "آ رہا",
    "آرہی": "آ رہی",
    "کررہا": "کر رہا",
    "کررہی": "کر رہی",
    "کررہے": "کر رہے",
    "کرلیں": "کر لیں",
    "ہوگئیں": "ہو گئیں",
    "جارہے": "جا رہے",
    "آرہے": "آ رہے",
    "ہورہا": "ہو رہا",
    "ہورہی": "ہو رہی",
    "ہورہے": "ہو رہے",
    "دیاگیا": "دیا گیا",
    "کیاگیا": "کیا گیا",
    "لیاگیا": "لیا گیا",
    # the reporting verb and its complementiser, and the future written as one word:
    # `اس نےکہاکہ وہ آئےگا` came back unchanged
    "نےکہا": "نے کہا",
    "کہاکہ": "کہا کہ",
    "نےکہاکہ": "نے کہا کہ",
    "میںنے": "میں نے",
    "آئےگا": "آئے گا",
    "آئےگی": "آئے گی",
    "آئیںگے": "آئیں گے",
    "جائےگا": "جائے گا",
    "جائےگی": "جائے گی",
    "جائیںگے": "جائیں گے",
    "کرےگا": "کرے گا",
    "کرےگی": "کرے گی",
    "کریںگے": "کریں گے",
}


def sentences(text: str) -> list[str]:
    """Split into sentences on Urdu and Latin terminators."""
    _require_str(text, "sentences")
    text = normalize(text)
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def words(text: str, *, keep_punctuation: bool = False) -> list[str]:
    """Split into word tokens.

    >>> words("رابطہ: test@x.com یا 0300-1234567")
    ['رابطہ', 'test@x.com', 'یا', '0300-1234567']

    URLs, emails, @mentions, #hashtags and numbers with separators (2.5, 1,500,
    3:30, ۱۲٫۵) are one token each, as they are to the transliterator. ZWNJ is kept
    inside tokens, because in Urdu it marks a real internal boundary in compounds
    rather than separating two words.
    """
    _require_str(text, "words")
    text = normalize(text)
    if not text:
        return []
    if keep_punctuation:
        return _TOKEN_OR_PUNCT.findall(text)
    return _TOKEN.findall(text)


def fix_spacing(text: str) -> str:
    """Insert the missing space in common merged compounds.

    A merge only fires on a whole word, never on a substring of a longer one -
    "کردیا" splits, "کردیانہ" does not.

    Matching is on word spans rather than on whitespace-separated chunks. Splitting
    on whitespace attaches any adjacent punctuation to the token, so "کردیا" was
    found and "کردیا۔" was not - and these are perfective auxiliaries, so the end
    of a sentence is exactly where they like to sit. Every occurrence followed by
    ۔ ، ؟ or a quote was missed, which on XL-Sum Urdu is most of them.

    Working on spans also leaves the text alone between the words it rewrites.
    `" ".join(text.split())` reflowed the whole input - newlines, indentation and
    runs of spaces all collapsed to one space - which is a surprising thing for a
    function that claims to insert a space to do.
    """
    _require_str(text, "fix_spacing")
    return _WORD.sub(lambda m: COMMON_MERGES.get(m.group(0), m.group(0)), text)


def character_ngrams(text: str, n: int = 3, *, pad: bool = True) -> list[str]:
    """Character n-grams over a word.

    Urdu is morphologically rich and has no reliable stemmer, so character n-grams
    are often the strongest cheap feature available for classification and retrieval.
    """
    _require_str(text, "character_ngrams")
    if n < 1:
        raise ValueError("n must be >= 1")
    token = f"<{text}>" if pad else text
    if len(token) < n:
        return [token] if token else []
    return [token[i : i + n] for i in range(len(token) - n + 1)]
