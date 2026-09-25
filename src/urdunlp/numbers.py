"""Urdu numbers: words to values, values to words, and South Asian digit grouping.

Three things make this more than a lookup table.

**Every number below a hundred is its own word.** Urdu has no "twenty-one" built
from "twenty" and "one": اکیس (21), بائیس (22), ... ننانوے (99) are ninety-nine
irregular words, so the table below spells all of them out.

**Fractions attach to the whole group, not to a digit.** ڈیڑھ لاکھ is one and a half
lakh - 150,000 - and سوا دو کروڑ is two and a quarter crore. سوا adds a quarter,
ساڑھے adds a half, پونے takes a quarter away, and ڈیڑھ and ڈھائی are standalone
words for 1.5 and 2.5. None of these have an English counterpart, and a parser that
treats them as unknown words drops most of the value.

**The large units are lakh and crore, not thousand-thousand.** 10,00,000 is one
lakh... ten lakh; the grouping is 3 digits then 2 at a time, so twelve lakh
thirty-four thousand five hundred and sixty-seven is written 12,34,567.
`format_number` does that grouping; `number_to_words` spells it out.

What is deliberately *not* done: several number words are also ordinary words.
اسی is 80 and also "that same", بہتر is 72 and also "better", سو is 100 and also
"so", نو is 9 and also "new". `parse_number` is given a phrase and trusts it;
`find_numbers`, which scans free text, only reads those five as numbers when a unit
or another number word is next to them. On its own, اسی is almost always a pronoun.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

from .normalize import _require_str, normalize

# 0-99, one irregular word each. Where two spellings are both in common use, the
# first is the one `number_to_words` writes and both are accepted by the parser.
_UNITS: dict[int, tuple[str, ...]] = {
    0: ("صفر",),
    1: ("ایک",),
    2: ("دو",),
    3: ("تین",),
    4: ("چار",),
    5: ("پانچ",),
    6: ("چھ", "چھے", "چھہ"),
    7: ("سات",),
    8: ("آٹھ",),
    9: ("نو",),
    10: ("دس",),
    11: ("گیارہ",),
    12: ("بارہ",),
    13: ("تیرہ",),
    14: ("چودہ",),
    15: ("پندرہ",),
    16: ("سولہ",),
    17: ("سترہ",),
    18: ("اٹھارہ",),
    19: ("انیس",),
    20: ("بیس",),
    21: ("اکیس",),
    22: ("بائیس",),
    23: ("تئیس", "تیئس"),
    24: ("چوبیس",),
    25: ("پچیس",),
    26: ("چھبیس",),
    27: ("ستائیس",),
    28: ("اٹھائیس",),
    29: ("انتیس",),
    30: ("تیس",),
    31: ("اکتیس",),
    32: ("بتیس",),
    33: ("تینتیس",),
    34: ("چونتیس",),
    35: ("پینتیس",),
    36: ("چھتیس",),
    37: ("سینتیس",),
    38: ("اڑتیس",),
    39: ("انتالیس",),
    40: ("چالیس",),
    41: ("اکتالیس",),
    42: ("بیالیس",),
    43: ("تینتالیس",),
    44: ("چوالیس",),
    45: ("پینتالیس",),
    46: ("چھیالیس",),
    47: ("سینتالیس",),
    48: ("اڑتالیس",),
    49: ("انچاس",),
    50: ("پچاس",),
    51: ("اکیاون",),
    52: ("باون",),
    53: ("ترپن",),
    54: ("چون",),
    55: ("پچپن",),
    56: ("چھپن",),
    57: ("ستاون",),
    58: ("اٹھاون",),
    59: ("انسٹھ",),
    60: ("ساٹھ",),
    61: ("اکسٹھ",),
    62: ("باسٹھ",),
    63: ("ترسٹھ",),
    64: ("چونسٹھ",),
    65: ("پینسٹھ",),
    66: ("چھیاسٹھ",),
    67: ("سڑسٹھ",),
    68: ("اڑسٹھ",),
    69: ("انہتر",),
    70: ("ستر",),
    71: ("اکہتر",),
    72: ("بہتر",),
    73: ("تہتر",),
    74: ("چوہتر",),
    75: ("پچھتر",),
    76: ("چھہتر",),
    77: ("ستتر",),
    78: ("اٹھہتر",),
    79: ("اناسی",),
    80: ("اسی",),
    81: ("اکیاسی",),
    82: ("بیاسی",),
    83: ("تراسی",),
    84: ("چوراسی",),
    85: ("پچاسی",),
    86: ("چھیاسی",),
    87: ("ستاسی",),
    88: ("اٹھاسی",),
    89: ("نواسی",),
    90: ("نوے",),
    91: ("اکانوے",),
    92: ("بانوے",),
    93: ("ترانوے",),
    94: ("چورانوے",),
    95: ("پچانوے",),
    96: ("چھیانوے",),
    97: ("ستانوے",),
    98: ("اٹھانوے",),
    99: ("ننانوے",),
}

HUNDRED = "سو"

# Multipliers above a hundred, largest first. ارب and کھرب follow the Indian
# system: 100 crore = 1 arab, 100 arab = 1 kharab.
SCALES: tuple[tuple[str, int], ...] = (
    ("کھرب", 10**11),
    ("ارب", 10**9),
    ("کروڑ", 10**7),
    ("لاکھ", 10**5),
    ("ہزار", 10**3),
)

# Standalone fractional numbers.
_FRACTION_WORDS: dict[str, Fraction] = {
    "ڈیڑھ": Fraction(3, 2),
    "ڈھائی": Fraction(5, 2),
    "آدھا": Fraction(1, 2),
    "آدھی": Fraction(1, 2),
    "آدھے": Fraction(1, 2),
    "پاؤ": Fraction(1, 4),
}

# Words that modify the number after them: سوا دو = 2 1/4, ساڑھے تین = 3 1/2,
# پونے چار = 3 3/4. With no number after them they modify an implied one:
# سوا لاکھ = 1 1/4 lakh.
_MODIFIERS: dict[str, Fraction] = {
    "سوا": Fraction(1, 4),
    "ساڑھے": Fraction(1, 2),
    "پونے": Fraction(-1, 4),
    "پون": Fraction(-1, 4),
}

_WORD_VALUE: dict[str, int] = {w: n for n, forms in _UNITS.items() for w in forms}
_SCALE_VALUE = dict(SCALES)

# Number words that are far more often something else. See the module docstring.
AMBIGUOUS = frozenset({"اسی", "بہتر", "سو", "نو", "ستر", "چون", "دو"})

_NUMERAL = re.compile(r"^\d+(?:[.,]\d+)*$")
_DIGITS = str.maketrans(
    {
        **{chr(0x06F0 + i): str(i) for i in range(10)},
        **{chr(0x0660 + i): str(i) for i in range(10)},
        "٫": ".",
        "٬": ",",
    }
)


def _numeral_value(token: str) -> Fraction | None:
    """'15', '2.5', '12,34,567', '۱۵' -> value. Commas are grouping, a dot is decimal."""
    token = token.translate(_DIGITS)
    if not _NUMERAL.match(token):
        return None
    if token.count(".") > 1:
        return None
    return Fraction(token.replace(",", ""))


# Roman spellings of the words that carry most of the value in a Roman Urdu amount.
# Explicit rather than left to `roman_key`, because these are the words an amount
# cannot afford to get wrong, and several of them are also ordinary Roman Urdu
# words (`do` is "two" and "give"). Every other Latin word is resolved with
# `roman_key` and accepted only if the Urdu word it resolves to is a number word.
ROMAN_NUMBER_WORDS: dict[str, str] = {
    **dict.fromkeys(("ek", "aik", "ik"), "ایک"),
    **dict.fromkeys(("do", "doh"), "دو"),
    **dict.fromkeys(("teen", "tin"), "تین"),
    **dict.fromkeys(("char", "chaar"), "چار"),
    **dict.fromkeys(("panch", "paanch", "panj"), "پانچ"),
    **dict.fromkeys(("chay", "chhe", "chhay", "che", "chh"), "چھ"),
    **dict.fromkeys(("saat", "sat"), "سات"),
    **dict.fromkeys(("aath", "ath", "aat"), "آٹھ"),
    **dict.fromkeys(("nau", "no", "naw"), "نو"),
    **dict.fromkeys(("das", "dus"), "دس"),
    **dict.fromkeys(("bees", "bis"), "بیس"),
    **dict.fromkeys(("pachas", "pachaas"), "پچاس"),
    **dict.fromkeys(("sau", "so", "sou", "sao"), "سو"),
    **dict.fromkeys(("hazar", "hazaar", "hajar", "hzar"), "ہزار"),
    **dict.fromkeys(("lakh", "lac", "lakhs", "lacs", "laakh"), "لاکھ"),
    **dict.fromkeys(("crore", "karor", "karod", "crores", "cr"), "کروڑ"),
    **dict.fromkeys(("arab", "arb"), "ارب"),
    **dict.fromkeys(("kharab", "kharb"), "کھرب"),
    **dict.fromkeys(("dedh", "derh", "deedh", "dairh", "dayrh"), "ڈیڑھ"),
    **dict.fromkeys(("dhai", "dhaai", "dhaee", "adhai", "arhai"), "ڈھائی"),
    **dict.fromkeys(("sawa", "sawaa", "swa"), "سوا"),
    **dict.fromkeys(("sadhe", "sarhe", "saadhe", "saarhe", "sade"), "ساڑھے"),
    **dict.fromkeys(("pone", "paune", "pauney", "poney"), "پونے"),
    **dict.fromkeys(("aadha", "adha", "aadhi", "adhi"), "آدھا"),
}


# A numeral (digits in any script, with Latin or Arabic grouping and decimal marks),
# or a run of anything that is not space, digit or punctuation - a word.
_TOKEN = re.compile(r"\d+(?:[.,٫٬]\d+)*|[^\s\d.,،٫٬؟۔!?:;\"'()\[\]]+")


_MAX_PHRASE = 24


def _canonical(token: str) -> str:
    """A token as the evaluator reads it: normalised, digits made ASCII."""
    return normalize(token, normalize_digits=True)


def _tokens(text: str) -> list[str]:
    tokens = [_canonical(t) for t in _TOKEN.findall(text)]
    return [_from_roman(t) if t.isascii() and t.isalpha() else t for t in tokens if t]


def _from_roman(token: str) -> str:
    """The Urdu number word a Roman token spells, or the token unchanged."""
    lowered = token.lower()
    if lowered in ROMAN_NUMBER_WORDS:
        return ROMAN_NUMBER_WORDS[lowered]
    from .roman import roman_key  # deferred: roman imports the transliteration model

    key = roman_key(lowered)
    return key if _is_number_token(key) else token


def _is_number_token(token: str) -> bool:
    return (
        token in _WORD_VALUE
        or token == HUNDRED
        or token in _SCALE_VALUE
        or token in _FRACTION_WORDS
        or token in _MODIFIERS
        or _numeral_value(token) is not None
    )


def _evaluate(tokens: list[str]) -> Fraction:
    """Value of a sequence of number tokens. Raises ValueError if it is not one."""
    total = Fraction(0)
    current: Fraction | None = None
    # A number after سو adds to it rather than colliding with it: پانچ سو تیس is
    # 500 + 30. Without this, "two numbers in a row" rejected every hundred that
    # was not a round one.
    open_hundred = False
    largest_scale = 0
    modifier: Fraction | None = None
    seen = False

    def take_modifier(value: Fraction) -> Fraction:
        nonlocal modifier
        if modifier is not None:
            value += modifier
            modifier = None
        return value

    for token in tokens:
        if token in _MODIFIERS:
            if modifier is not None or current is not None:
                raise ValueError(f"unexpected {token!r}")
            modifier = _MODIFIERS[token]
            continue
        seen = True
        if token in _WORD_VALUE or token in _FRACTION_WORDS or _numeral_value(token) is not None:
            if token in _WORD_VALUE:
                value = Fraction(_WORD_VALUE[token])
            elif token in _FRACTION_WORDS:
                value = _FRACTION_WORDS[token]
            else:
                value = _numeral_value(token)  # type: ignore[assignment]
            if current is not None:
                if not (open_hundred and value < 100 and modifier is None):
                    raise ValueError(f"two numbers in a row: {token!r}")
                current += value
                open_hundred = False
                continue
            current = take_modifier(value)
        elif token == HUNDRED:
            if open_hundred:
                raise ValueError("سو twice")
            base = current if current is not None else take_modifier(Fraction(1))
            current = base * 100
            open_hundred = True
        elif token in _SCALE_VALUE:
            open_hundred = False
            scale = _SCALE_VALUE[token]
            if current is None and modifier is None and total and scale <= largest_scale:
                # کروڑ ہزار: a bare ہزار straight after a finished group has nothing
                # to count. It was read as an implied "one thousand" and added, so
                # "12,34,567 کروڑ ہزار" came out as 12,345,670,001,000.
                raise ValueError(f"{token!r} has no number before it")
            if scale > largest_scale and total:
                # "ایک ہزار کروڑ": the scale applies to everything before it. The
                # implied "one" of a bare scale word must not be added here - it
                # made a thousand crore 10,010,000,000 instead of 10,000,000,000.
                total = (total + (current or 0)) * scale
            else:
                base = current if current is not None else take_modifier(Fraction(1))
                total += base * scale
            largest_scale = max(largest_scale, scale)
            current = None
        else:
            raise ValueError(f"not a number word: {token!r}")

    if modifier is not None:
        raise ValueError("a fraction word with nothing to modify")
    if not seen:
        raise ValueError("no number")
    return total + (current or 0)


def parse_number(text: str) -> int | float:
    """The value of an Urdu number phrase.

    >>> parse_number("ڈیڑھ لاکھ")
    150000
    >>> parse_number("دو ہزار پانچ سو تیس")
    2530
    >>> parse_number("2.5 کروڑ")
    25000000
    >>> parse_number("sawa do crore")
    22500000

    Accepts Urdu words, Roman Urdu number words, digits in any of the three
    scripts, and mixtures of them. Returns an int whenever the value is whole.
    Raises ValueError for text that is not a single number phrase.
    """
    _require_str(text, "parse_number")
    tokens = _tokens(text)
    if not tokens:
        raise ValueError("empty")
    value = _evaluate(tokens)
    return int(value) if value.denominator == 1 else float(value)


@dataclass(frozen=True)
class NumberSpan:
    text: str
    value: int | float
    start: int
    end: int


def find_numbers(text: str) -> list[NumberSpan]:
    """Every number phrase in running text, with its value and character span.

    `start` and `end` index into the text you passed, and `text` is that exact
    slice - diacritics, Arabic letters, Urdu digits and all. Each word is normalised
    on its own to be read, so the offsets never drift.

    A phrase ends at any punctuation: ایک لاکھ، دو ہزار is two numbers, not 102,000.
    A phrase that is a single ambiguous word - اسی, بہتر, سو, نو, ستر, چون, دو -
    is skipped: in running prose those are pronouns, adjectives and verbs far more
    often than numbers. Next to a unit or another number word they are read as
    numbers: دو لاکھ is 200,000. Roman Urdu amounts are found too (`15 lakh`,
    `dedh crore`), but a single Roman word never is.
    """
    _require_str(text, "find_numbers")
    spans: list[NumberSpan] = []
    matches = list(_TOKEN.finditer(text))
    # Latin words are read through the explicit Roman table only - not through
    # roman_key, which would find a nearest number word for almost anything.
    roman = [m.group().isascii() and m.group().isalpha() for m in matches]
    canonical = [
        ROMAN_NUMBER_WORDS.get(m.group().lower(), m.group()) if is_roman else _canonical(m.group())
        for m, is_roman in zip(matches, roman, strict=True)
    ]
    i = 0
    while i < len(matches):
        if not _is_number_token(canonical[i]):
            i += 1
            continue
        # Grow the longest run of number tokens that still evaluates. Only
        # whitespace may separate the words of one phrase; the first version
        # joined across a comma and read اسی،lakh as eighty lakh.
        best = None
        j = i
        # The longest real phrase - کھرب down to units - is about 13 words; a cap
        # keeps a run of 5,000 number words from being re-evaluated at every
        # length from every start, which took 60 s before it was bounded.
        while j < min(len(matches), i + _MAX_PHRASE) and _is_number_token(canonical[j]):
            if j > i and text[matches[j - 1].end() : matches[j].start()].strip():
                break
            try:
                value = _evaluate(canonical[i : j + 1])
                best = (j, value)
            except ValueError:
                pass
            j += 1
        if best is None:
            i += 1
            continue
        j, value = best
        words = canonical[i : j + 1]
        # A lone ambiguous word is not a number: اسی is "that same", and in Roman
        # text so, no and do are English and Urdu words long before they are 100,
        # 9 and 2.
        if len(words) == 1 and (words[0] in AMBIGUOUS or roman[i]):
            i += 1
            continue
        start, end = matches[i].start(), matches[j].end()
        spans.append(
            NumberSpan(
                text=text[start:end],
                value=int(value) if value.denominator == 1 else float(value),
                start=start,
                end=end,
            )
        )
        i = j + 1
    return spans


def format_number(value: int | float, *, urdu_digits: bool = False) -> str:
    """Group digits the South Asian way: 1234567 -> '12,34,567'.

    The last three digits form one group and every group above that has two, so
    a crore is 1,00,00,000. With `urdu_digits` the result is written in Extended
    Arabic-Indic digits and the Arabic separators ٬ and ٫.

    A float keeps exactly the digits Python prints for it - 0.1 is '0.1', and
    1e-05 is '0.00001'. The first version split `repr(value)` on the dot, which
    raised IndexError for any float Python prints in exponent form.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"format_number() expects an int or float, got {type(value).__name__}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"format_number() cannot group {value!r}")
    exact = format(Decimal(repr(value)), "f") if isinstance(value, float) else str(value)
    negative = exact.startswith("-") and exact.strip("-0.") != ""
    digits, _, fraction = exact.lstrip("-").partition(".")
    fraction = fraction.rstrip("0")
    head, tail = digits[:-3], digits[-3:]
    groups: list[str] = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    out = ",".join([*groups, tail])
    if fraction:
        out += "." + fraction
    if negative:
        out = "-" + out
    if urdu_digits:
        out = out.translate(str.maketrans("0123456789,.", "۰۱۲۳۴۵۶۷۸۹٬٫"))
    return out


def number_to_words(value: int) -> str:
    """Spell out a non-negative integer in Urdu, in lakh and crore.

    >>> number_to_words(150000)
    'ایک لاکھ پچاس ہزار'

    Always writes the plain form: 150,000 comes out as ایک لاکھ پچاس ہزار, not as
    ڈیڑھ لاکھ, because the fractional forms are a choice of register that a
    formatter should not make for you. `parse_number` reads both.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("number_to_words takes an int")
    if value < 0:
        raise ValueError("negative numbers are not spelled out")
    if value == 0:
        return _UNITS[0][0]
    if value >= 100 * 10**11:
        raise ValueError("above 99 kharab")

    parts: list[str] = []
    remainder = value
    for word, scale in SCALES:
        count, remainder = divmod(remainder, scale)
        if count:
            parts.append(f"{_below_thousand(count)} {word}")
    if remainder:
        parts.append(_below_thousand(remainder))
    return " ".join(parts)


def _below_thousand(value: int) -> str:
    hundreds, rest = divmod(value, 100)
    parts = []
    if hundreds:
        parts.append(f"{_UNITS[hundreds][0]} {HUNDRED}")
    if rest:
        parts.append(_UNITS[rest][0])
    return " ".join(parts)
