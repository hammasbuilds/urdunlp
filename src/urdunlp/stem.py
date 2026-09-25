"""A light, rule-based Urdu stemmer.

Urdu inflects nouns, adjectives and verbs by changing the end of the word:

    لڑکا  لڑکے  لڑکوں        boy, boys / oblique, oblique plural
    لڑکی  لڑکیاں  لڑکیوں     girl, girls, oblique plural
    کتاب  کتابیں  کتابوں     book, books, oblique plural
    کرتا  کرتی  کرتے  کرنا   does (m), does (f), do (pl), to do

A word-level index treats every one of those as a different term, so a query for
کتاب does not match a document that only says کتابوں. Stripping the ending conflates
them. This module does that with an explicit suffix list and nothing else - no
lexicon, no model - so every decision can be read and checked.

It is a *stemmer*, not a lemmatiser: the output is a retrieval key, not a word.
لڑکا, لڑکی and لڑکے all become لڑک, which is not an Urdu word and is not meant to be
shown to anyone. That is the right trade for search and the wrong one for display.

**The defaults were chosen by measurement, and the gain is small.** `stem` strips
inflectional endings, verbal endings (تا تی تے نا نی نے گا گی گے) and the Arabic
plural ات, and never leaves fewer than three letters; `light=True` strips the
inflectional endings only. Both levels and both minimum lengths were scored on
title-to-body and lead-sentence retrieval over 5,016 Urdu Wikipedia articles
(scripts/measure_stemmer.py), and this setting won on the validation queries of
both tasks. On the held-out test queries it adds +0.008 recall@10 to title
retrieval (sign test p = 0.019) and +0.005 to lead-sentence retrieval, which is
not significant (p = 0.32). A stemmer is cheap and does no harm here; it is not
the large win it is in English folklore, and docs/CORPUS.md says why.
"""

from __future__ import annotations

from .normalize import _require_str, _require_words, normalize

# Longest first, so یاں is tried before اں and ں.
LIGHT_SUFFIXES: tuple[str, ...] = (
    "ئیاں",  # fem. plural after a vowel: دوائیاں
    "یاں",  # fem. plural: لڑکیاں
    "یوں",  # fem. oblique plural: لڑکیوں
    "ئیں",  # plural after a vowel: دعائیں
    "ؤں",  # oblique plural after a vowel: گاؤں is a word, see _PROTECTED
    "وں",  # oblique plural: کتابوں, لڑکوں
    "یں",  # fem. plural: کتابیں
    "ئے",  # masc. plural/oblique after a vowel: گئے
    "ے",  # masc. plural/oblique: لڑکے
    "ی",  # feminine: لڑکی, اچھی
    "ا",  # masculine: لڑکا, اچھا
)

SUFFIXES: tuple[str, ...] = (
    "والے",
    "والی",
    "والا",
    "یاں",
    "یوں",
    "ئیاں",
    "ئیں",
    "ؤں",
    "وں",
    "یں",
    "گا",  # future, when written joined: کریگا
    "گی",
    "گے",
    "تا",  # imperfective participle: کرتا
    "تی",
    "تے",
    "نا",  # infinitive: کرنا
    "نی",
    "نے",
    "ات",  # Arabic plural: خیالات, معلومات
    "ئے",
    "ے",
    "ی",
    "ا",
)

# Frequent words whose ending only looks like an inflection. Stripping them would
# conflate function words with unrelated content words (سے -> س, تھا -> تھ).
_PROTECTED_WORDS = """
سے کے کی کا کو نے ہے ہی بھی یہ وہ جو تو نہ یا اور تھا تھی تھے گا گی گے
گاؤں پاؤں ہوا ہوئی ہوئے کیا کیے کئی گیا گئی گئے دیا دی دیے لیا لی لیے
ایسا ایسی ایسے ویسا جیسا جیسی جیسے کیسا کیسی کیسے اپنا اپنی اپنے
میری میرا میرے تیرا تیری تیرے ہماری ہمارا ہمارے تمہاری تمہارا تمہارے
"""
_PROTECTED = frozenset(normalize(w) for w in _PROTECTED_WORDS.split())


def stem(word: str, *, light: bool = False, min_stem: int = 3) -> str:
    """Strip one suffix from an Urdu word.

    At most one suffix is removed, and never one that would leave fewer than
    `min_stem` letters: بات is not reduced to ب. Words in a short protected list of
    function words are returned unchanged.

    >>> stem("کتابوں")
    'کتاب'
    >>> stem("لڑکیاں")
    'لڑک'
    """
    _require_str(word, "stem")
    word = normalize(word)
    if word in _PROTECTED:
        return word
    for suffix in LIGHT_SUFFIXES if light else SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= min_stem:
            return word[: -len(suffix)]
    return word


def stem_tokens(tokens: list[str], *, light: bool = False, min_stem: int = 3) -> list[str]:
    """`stem` over a token list."""
    _require_words(tokens, "stem_tokens")
    return [stem(t, light=light, min_stem=min_stem) for t in tokens]
