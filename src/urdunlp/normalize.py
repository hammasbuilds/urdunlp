"""Urdu text normalisation.

Urdu is written in a Perso-Arabic script, and the same word routinely appears in
several byte sequences that look identical on screen. Text scraped from the web mixes
Arabic codepoints with Urdu ones, because most keyboards and many fonts do not
distinguish them:

    ي  U+064A  ARABIC YEH        vs  ی  U+06CC  FARSI YEH
    ك  U+0643  ARABIC KAF        vs  ک  U+06A9  KEHEH
    ه  U+0647  ARABIC HEH        vs  ہ  U+06C1  HEH GOAL  *or*  ھ  U+06BE

Without normalisation these are different strings, so exact match fails, vocabularies
fragment, and every downstream model silently learns three versions of the same word.
This module is the first thing any Urdu pipeline needs and the piece most often
missing.

Every transformation here is reversible in meaning, not in bytes: normalisation is
lossy on purpose, because the distinctions it removes carry no information in Urdu.
"""

from __future__ import annotations

import re
import unicodedata

# --- Character-level equivalences ------------------------------------------------
# Arabic codepoint -> the Urdu one that is actually correct.
ARABIC_TO_URDU = {
    "ي": "ی",  # ARABIC YEH        -> FARSI YEH
    "ى": "ی",  # ALEF MAKSURA      -> FARSI YEH
    "ك": "ک",  # ARABIC KAF        -> KEHEH
    "ګ": "گ",  # GAF WITH RING     -> GAF
    "ة": "ۃ",  # TEH MARBUTA       -> TEH MARBUTA GOAL
    "أ": "ا",  # ALEF WITH HAMZA ABOVE -> ALEF
    "إ": "ا",  # ALEF WITH HAMZA BELOW -> ALEF
    "آ": "آ",  # ALEF WITH MADDA is a real Urdu letter; kept
    "ؤ": "ؤ",  # WAW WITH HAMZA is real; kept
    "ۀ": "ۂ",  # HEH WITH YEH ABOVE -> HEH GOAL WITH HAMZA ABOVE
}

# ARABIC HEH is deliberately NOT in the table above, because it has no single Urdu
# counterpart. Urdu splits the job across two letters:
#
#     ہ  U+06C1  HEH GOAL       an ordinary h      نہ, اللہ
#     ھ  U+06BE  DOACHASHMEE    aspiration         بھی, تھا, کھانا
#
# Which one a stray ه stands for depends on what precedes it. Measured over the
# 977 corpus occurrences that the vocabulary can adjudicate - a token containing
# no ه is correctly spelled by definition, so the corpus itself says which
# candidate is a real word:
#
#     always HEH GOAL, as this module's docstring used to promise    9.0% correct
#     DOACHASHMEE after an aspirable consonant, else HEH GOAL       96.9% correct
#
# The intuitive rule is wrong more than nine times in ten: the substitution shows
# up overwhelmingly in aspirated consonants, because a keyboard without ھ is a
# keyboard without bh, ph, th, kh or gh.
ARABIC_HEH = "ه"
DOACHASHMEE_HE = "ھ"
HEH_GOAL = "ہ"

# The consonants that carry aspiration in Urdu. ل م ن ر are excluded on purpose:
# they never aspirate, and including ل would spell الله as اللھ instead of اللہ.
ASPIRABLE = "بپتٹجچدڈکگڑ"

_HEH_AFTER_ASPIRABLE = re.compile(f"([{ASPIRABLE}]){ARABIC_HEH}")

# Digits. Urdu uses Extended Arabic-Indic (U+06F0), Arabic uses U+0660. Both occur.
ARABIC_INDIC_DIGITS = {chr(0x0660 + i): str(i) for i in range(10)}
URDU_DIGITS = {chr(0x06F0 + i): str(i) for i in range(10)}

# Punctuation that has an Urdu-specific form.
URDU_PUNCTUATION = {
    "،": ",",  # ARABIC COMMA
    "؛": ";",  # ARABIC SEMICOLON
    "؟": "?",  # ARABIC QUESTION MARK
    "۔": ".",  # URDU FULL STOP
    "٫": ".",  # ARABIC DECIMAL SEPARATOR
    "٬": ",",  # ARABIC THOUSANDS SEPARATOR
}

# Harakat / diacritics. Optional in Urdu, almost always absent, and their presence in
# some copies of a word but not others fragments the vocabulary for no gain.
DIACRITICS = re.compile("[ً-ْٰٓ-ٕٖ-ٟۖ-ۭ]")

# Tatweel stretches a letter for typographic justification. It is never meaningful.
TATWEEL = "ـ"

# Zero-width joiner/non-joiner. ZWNJ is meaningful in Urdu compound words, so it is
# preserved by default and removed only when explicitly requested.
ZWNJ = "‌"
ZWJ = "‍"
_ZERO_WIDTH_OTHER = re.compile("[​‎‏﻿⁠]")

_SPACES = re.compile(r"[ \t  -   　]+")
_NEWLINES = re.compile(r"\s*\n\s*")

# A URL, an email, an @mention or a #hashtag: spans that are one token and must never
# be transliterated, split or half-removed. Shared by `words`, the transliterator and
# `remove_urls_and_mentions`, which used to have three different ideas of it - the
# last removed `@x` from `test@x.com` as a mention and left `test .com` behind. A
# URL stops before punctuation that ends the sentence around it: in `see
# http://x.co.` the full stop is not part of the address.
IDENTIFIER = (
    r"https?://\S+?(?=[.,;:!?؟،۔)\]}'\"]*(?:\s|$))"
    r"|www\.\S+?(?=[.,;:!?؟،۔)\]}'\"]*(?:\s|$))"
    r"|[\w.+-]+@[\w-]+\.[\w.-]*\w"
    r"|[@#]\w+"
)
_IDENTIFIER = re.compile(IDENTIFIER)

_ARABIC_TRANSLATION = str.maketrans(ARABIC_TO_URDU)
_DIGIT_TRANSLATION = str.maketrans({**ARABIC_INDIC_DIGITS, **URDU_DIGITS})
_PUNCT_TRANSLATION = str.maketrans(URDU_PUNCTUATION)


def _require_str(value: object, function: str) -> None:
    """Raise the same TypeError from every public function given a non-string.

    Before this, the toolkit answered a wrong type five different ways: normalize(None)
    silently returned '', words(None) returned [], roman_key(None) raised
    AttributeError from deep inside, is_urdu(['kal']) returned False, and the rest
    raised whatever the first string operation happened to. A None read from a
    missing DataFrame cell - or a NaN, which is a float - should fail at the call,
    with a message that names the call.
    """
    if not isinstance(value, str):
        raise TypeError(f"{function}() expects a str, got {type(value).__name__}")


def _require_words(value: object, function: str) -> list[str]:
    """Check a list of words and return it as a list, naming the caller on failure.

    A string is iterable, so stem_tokens("کتابوں سے") would quietly stem each
    *character* and return a list of letters - no error, just wrong output. Bytes are
    iterable too, as ints, so remove_stopwords(b"x") used to fail inside the per-word
    function with "is_stopword() expects a str, got int", naming neither the function
    that was called nor the type that was passed.
    """
    if isinstance(value, str):
        raise TypeError(
            f"{function}() expects a list of words, got a str - split it first, "
            "for example with words(text)"
        )
    if isinstance(value, (bytes, bytearray)):
        raise TypeError(
            f"{function}() expects a list of words, got {type(value).__name__} - "
            "decode it and split it first"
        )
    try:
        iterator = iter(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(
            f"{function}() expects a list of words, got {type(value).__name__}"
        ) from None
    items: list[str] = []
    for position, item in enumerate(iterator):
        if not isinstance(item, str):
            kind = type(item).__name__
            raise TypeError(f"{function}() expects a list of str, but item {position} is {kind}")
        items.append(item)
    return items


def resolve_arabic_heh(text: str) -> str:
    """Replace stray ARABIC HEH with the Urdu letter it stands for.

    A heuristic, not a rule: 96.9% correct on the corpus occurrences that could be
    adjudicated, against 9.0% for mapping it to HEH GOAL everywhere. The remaining
    3.1% are words where both spellings are real - بہار (spring) and بھار (weight)
    differ only in this letter - and no amount of context-free rewriting separates
    them.
    """
    _require_str(text, "resolve_arabic_heh")
    text = _HEH_AFTER_ASPIRABLE.sub(r"\1" + DOACHASHMEE_HE, text)
    return text.replace(ARABIC_HEH, HEH_GOAL)


def normalize(
    text: str,
    *,
    unify_characters: bool = True,
    strip_diacritics: bool = True,
    normalize_digits: bool = False,
    normalize_punctuation: bool = False,
    strip_zwnj: bool = False,
    collapse_whitespace: bool = True,
) -> str:
    """Normalise Urdu text.

    The defaults are the ones you almost always want: unify the Arabic/Urdu
    look-alikes and drop diacritics, but keep Urdu digits and Urdu punctuation, since
    converting those changes how the text reads rather than how it is encoded.

    >>> normalize("كتاب") == "کتاب"
    True
    """
    _require_str(text, "normalize")
    if not text:
        return ""

    # NFC first: composed forms make every table below single-codepoint.
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH_OTHER.sub("", text)
    text = text.replace(TATWEEL, "")

    if unify_characters:
        text = text.translate(_ARABIC_TRANSLATION)
        text = resolve_arabic_heh(text)
    if strip_diacritics:
        text = DIACRITICS.sub("", text)
    if normalize_digits:
        text = text.translate(_DIGIT_TRANSLATION)
    if normalize_punctuation:
        text = text.translate(_PUNCT_TRANSLATION)
    if strip_zwnj:
        text = text.replace(ZWNJ, "").replace(ZWJ, "")

    if collapse_whitespace:
        text = _NEWLINES.sub("\n", text)
        text = _SPACES.sub(" ", text)
        text = text.strip()

    return text


# The Unicode blocks that hold Perso-Arabic letters. `is_urdu` counted only the
# first and third of these, while `tokenize._URDU_LETTERS` counted all four - so a
# string could be tokenised as Urdu words and simultaneously reported as not Urdu.
#
# Presentation Forms-B is the block that matters in practice: it is what PDF text
# layers and older systems emit. It is not common in edited prose - 231 characters
# across 83 of the 84,581 corpus articles, too few to change any single article's
# verdict - but two functions in one toolkit disagreeing about what Urdu is, is a
# defect whatever the frequency.
_URDU_BLOCKS = (
    ("؀", "ۿ"),  # Arabic, which holds the Urdu letters
    ("ݐ", "ݿ"),  # Arabic Supplement
    ("ࢠ", "ࣿ"),  # Arabic Extended-A
    ("ﭐ", "﷿"),  # Presentation Forms-A
    ("ﹰ", "﻿"),  # Presentation Forms-B
)


def _is_urdu_letter(char: str) -> bool:
    return any(lo <= char <= hi for lo, hi in _URDU_BLOCKS)


def is_urdu(text: str, *, threshold: float = 0.5) -> bool:
    """Whether the text is predominantly Urdu script.

    Measured over letters only. Counting all characters would make any Urdu sentence
    containing a number or a Latin brand name look less Urdu than it is.
    """
    _require_str(text, "is_urdu")
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    urdu = sum(1 for c in letters if _is_urdu_letter(c))
    return urdu / len(letters) >= threshold


def remove_urls_and_mentions(text: str) -> str:
    """Strip URLs, emails, @mentions and #hashtags. A common first step on scraped text.

    >>> remove_urls_and_mentions("رابطہ test@x.com یا @ali پر")
    'رابطہ یا پر'
    """
    _require_str(text, "remove_urls_and_mentions")
    # Punctuation straight after a removed URL closes up onto the word before it:
    # `dekho http://x.co.` gives `dekho.`, not `dekho .`.
    pieces: list[str] = []
    end = 0
    for match in _IDENTIFIER.finditer(text):
        before = text[end : match.start()]
        after = text[match.end() : match.end() + 1]
        if after and not after.isspace() and not after.isalnum():
            pieces.append(before.rstrip(" 	"))
        elif before[-1:].isspace() or after.isspace():
            pieces.append(before)
        else:
            pieces.append(before + " ")
        end = match.end()
    pieces.append(text[end:])
    return _SPACES.sub(" ", "".join(pieces)).strip()
