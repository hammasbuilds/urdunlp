"""urdunlp - Urdu and Roman Urdu text processing.

Pure Python, no dependencies, no model downloads.

    >>> from urdunlp import normalize, transliterate_to_urdu
    >>> normalize("كتاب")
    'کتاب'
    >>> transliterate_to_urdu("main theek hoon")
    'میں ٹھیک ہوں'
"""

from .langid import LANGUAGES, LanguageGuess, identify_language, tag_roman_tokens
from .normalize import is_urdu, normalize, remove_urls_and_mentions, resolve_arabic_heh
from .numbers import NumberSpan, find_numbers, format_number, number_to_words, parse_number
from .roman import group_roman_variants, roman_key
from .stem import stem, stem_tokens
from .stopwords import NEGATION, STOPWORDS, is_stopword, remove_stopwords
from .tokenize import character_ngrams, fix_spacing, sentences, words
from .translit import (
    Transliteration,
    transliterate_to_roman,
    transliterate_to_urdu,
    transliterate_with_confidence,
)

__version__ = "0.2.0"

__all__ = [
    "LANGUAGES",
    "NEGATION",
    "STOPWORDS",
    "LanguageGuess",
    "NumberSpan",
    "Transliteration",
    "character_ngrams",
    "find_numbers",
    "fix_spacing",
    "format_number",
    "group_roman_variants",
    "identify_language",
    "is_stopword",
    "is_urdu",
    "normalize",
    "number_to_words",
    "parse_number",
    "remove_stopwords",
    "remove_urls_and_mentions",
    "resolve_arabic_heh",
    "roman_key",
    "sentences",
    "stem",
    "stem_tokens",
    "tag_roman_tokens",
    "transliterate_to_roman",
    "transliterate_to_urdu",
    "transliterate_with_confidence",
    "words",
]
