"""Roman Urdu spelling variants.

Roman Urdu has no standard spelling, so one word arrives in many forms. Dakshina's
annotators, romanising the same Urdu words independently, wrote آئینی as `aaeeni`,
`aaini`, `aayinee`, `aayini`, `aayiny`, `ainey` and `aini` - seven spellings of one
word from a handful of people. Counting, deduplicating or searching Roman Urdu
without collapsing those is counting spellings, not words.

`roman_key` collapses them by asking which *Urdu* word each spelling stands for, and
using that as the key: `nahi`, `nahin`, `nhi` and `naheen` all key to نہیں. That is a
stronger notion of "same word" than any rule over the Roman letters, because the
Urdu word is the thing the spellings are variants *of*. It uses the same three
stages as `transliterate_with_confidence` - curated lexicon, then the vocabulary
search, then rules - so a key is also a readable transliteration.

Scripts/measure_roman_key.py scores this against exact matching and against a
consonant skeleton on Dakshina's held-out lexicon; docs/CORPUS.md has the result.
"""

from __future__ import annotations

import functools
from collections import defaultdict

from . import _channel
from .normalize import _require_str, _require_words, normalize
from .translit import LEXICON, _apply_rules


def roman_key(word: str) -> str:
    """The Urdu word a Roman spelling most likely stands for, as a grouping key.

    >>> roman_key("nahi") == roman_key("nahin") == roman_key("nhi") == "نہیں"
    True

    One word in, one key out; call it per token. The key is normalised Urdu script,
    so it can be compared directly with keys from Urdu-script text - and a word
    already in Urdu script is its own key, normalised.
    """
    _require_str(word, "roman_key")
    latin = "".join(c for c in word.lower() if "a" <= c <= "z")
    if not latin:
        # Nothing Roman to resolve: Urdu script, digits or punctuation is its own key.
        return normalize(word)
    # Punctuation and digits around the letters are not part of the spelling:
    # `nahi!` and `nahi` are the same word.
    return _roman_key(latin)


@functools.lru_cache(maxsize=65536)
def _roman_key(lowered: str) -> str:
    if not lowered:
        # The empty string has the empty consonant key, which is also the key of
        # every consonant-free word - so it resolved to ء until this line.
        return ""
    if lowered in LEXICON:
        return normalize(LEXICON[lowered])
    found = _channel.resolve(lowered)
    if found:
        return found
    return normalize(_apply_rules(lowered))


def group_roman_variants(words: list[str]) -> dict[str, list[str]]:
    """Group Roman spellings by `roman_key`, preserving first-seen order.

    >>> group_roman_variants(["nahi", "acha", "nhi", "accha"])
    {'نہیں': ['nahi', 'nhi'], 'اچھا': ['acha', 'accha']}

    Each distinct spelling appears once, in the group of the word it stands for.
    """
    _require_words(words, "group_roman_variants")
    groups: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for word in words:
        if word in seen:
            continue
        seen.add(word)
        groups[roman_key(word)].append(word)
    return dict(groups)
