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
from collections.abc import Mapping

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

# Arabic-Indic digit -> the Urdu one. They are different codepoints for the same
# digit (٣ U+0663 and ۳ U+06F3 are both 3), and Urdu fonts draw 4, 5, 6 and 7
# differently from Arabic ones, so unifying the letters and not the digits left
# "۲۳" and "٢٣" unequal after normalize(). Done with the letters, by default;
# normalize_digits=True goes further and writes 0-9.
ARABIC_INDIC_TO_URDU_DIGITS = {chr(0x0660 + i): chr(0x06F0 + i) for i in range(10)}

# Arabic Presentation Forms A and B: one codepoint per letter *shape* (initial,
# medial, final, isolated) and for ligatures such as ﷲ. Nobody types them; they are
# what text copied out of a PDF, or produced by an old shaping engine, is made of -
# ﻛﺘﺎﺏ looks like کتاب and shares no codepoint with it. NFKC maps each to the letter
# it is a shape of, Urdu-specific letters included (ﮨ -> ہ, ﮐ -> ک, ﯾ -> ی, ﮮ -> ے),
# and is applied to these two blocks only: over the whole text it would also rewrite
# Latin ligatures, full-width forms and superscripts that are none of its business.
# Three ligatures are kept as they are, because each is a whole phrase used as a
# symbol in Urdu writing, and expanding it would put a sentence where one character
# was: ﷺ (U+FDFA), ﷻ (U+FDFB) and the basmala ﷽ (U+FDFD).
_PRESENTATION_FORMS = re.compile("[\ufb50-\ufdf9\ufdfc\ufdfe-\ufdff\ufe70-\ufefc]+")


def _expand_presentation_forms(match: re.Match[str]) -> str:
    # The isolated forms of the harakat (U+FE70-FE7F) decompose to a space and the
    # mark; the space is not in the text the reader saw, so it is not kept.
    return "".join(unicodedata.normalize("NFKC", char).lstrip(" ") for char in match.group())


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
# The lookbehind is load-bearing, exactly as in IDENTIFIER below. Without it the leading
# \s* matched a whole whitespace run, failed to find the \n, and backtracked one
# character at a time from every start position in that run - quadratic in the run's
# length. Two hundred thousand spaces took 153 seconds, and it is reachable from any
# untrusted string through normalize, words, sentences, stem and roman_key, in a
# zero-dependency library meant to sit underneath other people's services. Refusing to
# start mid-run makes it linear (64k spaces: 15.78 s -> 0.002 s), and the output is
# identical on every case tested, including tabs, NBSP and runs of bare newlines.
_NEWLINES = re.compile(r"(?<!\s)\s*\n\s*")

# A URL, an email, an @mention or a #hashtag: spans that are one token and must never
# be transliterated, split or half-removed. Shared by `words`, the transliterator and
# `remove_urls_and_mentions`, which used to have three different ideas of it - the
# last removed `@x` from `test@x.com` as a mention and left `test .com` behind. A
# URL stops before punctuation that ends the sentence around it: in `see
# http://x.co.` the full stop is not part of the address.
IDENTIFIER = (
    r"https?://\S+?(?=[.,;:!?؟،۔)\]}'\"]*(?:\s|$))"
    r"|www\.\S+?(?=[.,;:!?؟،۔)\]}'\"]*(?:\s|$))"
    # The lookbehind is load-bearing, not tidiness. Without it the engine attempted the
    # local part at every interior position of a long whitespace-free run and backtracked,
    # which is quadratic: one 40 KB base64 data: URI took 12.9 s, and scraped text is the
    # documented use case for this pattern. Refusing to start mid-token makes it linear
    # (40 KB: 8.157 s -> 0.001 s). The bounds are RFC 5321's 64-octet local part and
    # 255-octet domain label, so nothing valid is lost.
    r"|(?<![\w.+-])[\w.+-]{1,64}@[\w-]{1,255}\.[\w.-]*\w"
    r"|[@#]\w+"
)
_IDENTIFIER = re.compile(IDENTIFIER)

_ARABIC_TRANSLATION = str.maketrans({**ARABIC_TO_URDU, **ARABIC_INDIC_TO_URDU_DIGITS})
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
    # A mapping is iterable over its KEYS, so remove_stopwords({"a": 1}) silently
    # returned ["a"] - the values were dropped and no error was raised. Iterating a dict
    # is almost never what the caller meant here, so say so rather than guess.
    if isinstance(value, Mapping):
        raise TypeError(
            f"{function}() expects a list of words, got {type(value).__name__} - "
            "iterating it would use only its keys; pass list(mapping) if that is "
            "what you meant"
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
    >>> normalize("ﻛﺘﺎﺏ") == "کتاب"   # presentation forms, as copied out of a PDF
    True
    >>> normalize("٣") == "۳"         # Arabic-Indic digit -> Urdu digit
    True

    `unify_characters` (on by default) does three things: maps the Arabic letters
    that stand in for Urdu ones (ي ك ه ...) to the Urdu letters, maps Arabic-Indic
    digits (٠-٩) to the Urdu digits (۰-۹), and turns the presentation forms a PDF
    or an old system emits (ﻛﺘﺎﺏ, ﷲ) into ordinary letters. `normalize_digits=True`
    also writes every Urdu and Arabic digit as 0-9.
    """
    _require_str(text, "normalize")
    if not text:
        return ""

    # NFC first: composed forms make every table below single-codepoint.
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH_OTHER.sub("", text)
    text = text.replace(TATWEEL, "")

    if unify_characters:
        text = _PRESENTATION_FORMS.sub(_expand_presentation_forms, text)
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
    # The only argument in the package that was unvalidated: a str leaked a raw
    # comparison TypeError from the last line, nan silently returned False whatever the
    # text, and a negative threshold called English Urdu. Every other bad argument here
    # gets a named message, so this one does too.
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError(f"is_urdu() expects a number for threshold, got {type(threshold).__name__}")
    if not 0.0 <= threshold <= 1.0:  # also rejects nan, which fails both sides
        raise ValueError(f"is_urdu() threshold must be between 0 and 1, got {threshold!r}")
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
