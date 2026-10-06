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

**Ordinals are half irregular.** پہلا دوسرا تیسرا چوتھا چھٹا (1st-4th, 6th) and
یکم (the first of a month) are their own words; every other ordinal is the cardinal
plus واں or ویں. `parse_ordinal` reads them, and `find_numbers` flags them with
`ordinal=True`.

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

# Denominators, which DIVIDE the number before them: تین چوتھائی is three quarters, دو
# تہائی two thirds. Without these the denominator was simply not a number word, and
# `find_numbers` returned the numerator on its own - three quarters of a kilo read as
# THREE, which is a confidently wrong value rather than a refusal. پاؤ above is the
# standalone quarter and keeps working; these are the counted form.
_DENOMINATORS: dict[str, Fraction] = {
    "چوتھائی": Fraction(1, 4),
    "تہائی": Fraction(1, 3),
    "نصف": Fraction(1, 2),
    "دوتہائی": Fraction(2, 3),
}

_WORD_VALUE: dict[str, int] = {w: n for n, forms in _UNITS.items() for w in forms}
_SCALE_VALUE = dict(SCALES)

# Ordinals. 1st-4th and 6th are their own words; every other ordinal is the cardinal
# plus واں (masculine) or ویں (oblique/feminine) - پانچواں, دسویں, بیسویں, ہزارواں -
# and 9th is نواں/نویں. Checked against the corpus: all of these are attested. The
# ending -وی is not accepted: ہزاروی is mostly the surname Hazarvi.
_ORDINAL_WORDS: dict[str, int] = {
    **dict.fromkeys(("پہلا", "پہلی", "پہلے", "یکم"), 1),
    **dict.fromkeys(("دوسرا", "دوسری", "دوسرے"), 2),
    **dict.fromkeys(("تیسرا", "تیسری", "تیسرے"), 3),
    **dict.fromkeys(("چوتھا", "چوتھی", "چوتھے"), 4),
    **dict.fromkeys(("چھٹا", "چھٹی", "چھٹے"), 6),
    **dict.fromkeys(("نواں", "نویں"), 9),
}
_ORDINAL_SUFFIXES = ("واں", "ویں")

# Ordinal words that are far more often something else: پہلے is "before" (11,584
# times in the corpus against a few hundred ordinal uses), دوسرے/دوسری "other".
AMBIGUOUS_ORDINALS = frozenset({"پہلے", "دوسرے", "دوسری", "دوسرا"})


def _show(text: str, limit: int = 80) -> str:
    """repr() for an error message, cut down so a 100 kB input is not echoed whole."""
    if len(text) <= limit:
        return repr(text)
    return f"{text[:limit]!r}... ({len(text):,} characters)"


def _ordinal_as_cardinal(token: str) -> str | None:
    """The cardinal token an ordinal word stands for (پانچواں -> پانچ), or None."""
    if token in _ORDINAL_WORDS:
        return str(_ORDINAL_WORDS[token])
    for suffix in _ORDINAL_SUFFIXES:
        base = token[: -len(suffix)]
        if token.endswith(suffix) and (
            base in _WORD_VALUE or base == HUNDRED or base in _SCALE_VALUE
        ):
            return base
    return None


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


# CPython refuses to convert an integer string longer than 4,300 digits by default.
# The largest amount this module names is 99 kharab (13 digits), so anything remotely
# near this bound is not a quantity anyone wrote; staying under it keeps the limit from
# surfacing as a ValueError from inside the standard library.
_MAX_NUMERAL_DIGITS = 4_000


def _numeral_value(token: str) -> Fraction | None:
    """'15', '2.5', '12,34,567', '۱۵' -> value. Commas are grouping, a dot is decimal.

    A token with more digits than CPython will convert is not a number here. Without
    the guard, `int()` inside `Fraction` raised
    `ValueError: Exceeds the limit (4300 digits) for integer string conversion … use
    sys.set_int_max_str_digits()` straight out of `find_numbers`, `parse_number` and
    `parse_ordinal`. That is CPython's internals reaching a caller, it suggests a fix
    the caller should not have to make, and `parse_number` already raises ValueError
    for "not a number word", so the two were indistinguishable. A comma-separated
    number list pasted without spaces - one CSV row - is enough to trigger it.
    """
    token = token.translate(_DIGITS)
    if not _NUMERAL.match(token):
        return None
    if token.count(".") > 1:
        return None
    digits = token.replace(",", "").replace(".", "").lstrip("0")
    if len(digits) > _MAX_NUMERAL_DIGITS:
        return None
    return Fraction(token.replace(",", ""))


# Roman spellings of every number word, 1-99, and of the units and fractions.
# Explicit rather than left to `roman_key`, because these are the words an amount
# cannot afford to get wrong: `roman_key` picks the nearest Urdu word, and for
# `sattar` (70) that was ستتر (77). Several are also ordinary Roman Urdu words
# (`do` is "two" and "give", `tera` "thirteen" and "your", `saath` "sixty" and
# "with"); `find_numbers` reads those only next to another number word. Every other
# Latin word is resolved by `parse_number` with `roman_key`, and accepted only if the
# Urdu word it resolves to is a number word.
_ROMAN_UNITS: dict[int, tuple[str, ...]] = {
    1: ("ek", "aik", "ik", "ekk"),
    2: ("do", "doh"),
    3: ("teen", "tin"),
    4: ("char", "chaar"),
    5: ("panch", "paanch", "panj"),
    6: ("chay", "chhe", "chhay", "che", "chh", "chhah", "chha", "cheh"),
    7: ("saat", "sat"),
    8: ("aath", "ath", "aat", "aathh"),
    9: ("nau", "no", "naw", "nou"),
    10: ("das", "dus", "dass"),
    11: ("gyarah", "gyara", "gyaara", "gyaarah", "giyara", "giyarah", "gyarha"),
    12: ("barah", "bara", "baara", "baarah", "barha", "baraa"),
    13: ("terah", "tera", "tehra", "teraa", "tehrah"),
    14: ("chaudah", "chauda", "chodah", "choda", "chaudha", "chodha"),
    15: ("pandrah", "pandra", "pundra", "pandara", "pandarah"),
    16: ("solah", "sola", "solha"),
    17: ("satrah", "satra", "sattrah", "sattra", "satarah"),
    18: ("atharah", "athara", "athaara", "attharah", "atthara", "atharha"),
    19: ("unnees", "unees", "unnis", "unis"),
    20: ("bees", "bis"),
    21: ("ikkees", "ikees", "ikkis", "ikis"),
    22: ("baees", "bais", "baais", "baaees", "baies"),
    23: ("teis", "tais", "taees", "teyis", "taeis", "teees"),
    24: ("chaubees", "chobees", "chaubis", "chobis"),
    25: ("pachees", "pachchees", "pachis", "pachchis"),
    26: ("chhabbees", "chabbees", "chabees", "chhabis", "chabbis", "chabis"),
    27: ("sattaees", "sattais", "satais", "sataees", "sattayis"),
    28: ("atthaees", "athaees", "athais", "atthais", "athayis"),
    29: ("untees", "untis", "unatees", "unattis"),
    30: ("tees", "tis"),
    31: ("ikattees", "iktees", "ikatis", "ikattis", "ikatees"),
    32: ("battees", "batees", "batis", "battis"),
    33: ("taintees", "tentees", "taintis", "tentis", "tetees", "tetis"),
    34: ("chauntees", "chontees", "chontis", "chautis", "chautees"),
    35: ("paintees", "pentees", "paintis", "pentis"),
    36: ("chhattees", "chattees", "chhatis", "chattis", "chhattis"),
    37: ("saintees", "sentees", "saintis", "sentis"),
    38: ("artees", "adtees", "artis", "adtis", "arrtees"),
    39: ("untaalees", "untalees", "untalis", "antalees"),
    40: ("chalees", "chaalees", "chalis", "chaalis"),
    41: ("iktaalees", "iktalees", "iktalis", "iktaalis"),
    42: ("bayalees", "bialees", "bayalis", "byalis", "biyalees", "bayaalees"),
    43: ("taintaalees", "tentalees", "taintalis", "tentalis", "tetalees", "taitalis"),
    44: ("chawalees", "chawaalees", "chaualis", "chawalis", "chauvalees"),
    45: ("paintaalees", "pentalees", "paintalis", "pentalis", "paintalees"),
    46: ("chhiyalees", "chiyalees", "chhiyalis", "chiyalis", "chhayalees"),
    47: ("saintaalees", "sentalees", "saintalis", "sentalis", "saintalees"),
    48: ("artaalees", "artalees", "artalis", "adtalis", "adtalees"),
    49: ("unchaas", "unchas", "unanchas"),
    50: ("pachaas", "pachas"),
    51: ("ikyavan", "ikyawan", "ikkyawan", "ekawan", "ikiyawan", "ikyaavan"),
    52: ("baavan", "bawan", "baawan", "bavan"),
    53: ("tirpan", "tirepan", "trepan", "tarpan"),
    54: ("chavvan", "chawan", "chauwan", "chavan"),
    55: ("pachpan", "pachpun"),
    56: ("chhappan", "chappan", "chhapan", "chapan"),
    57: ("sattavan", "sattawan", "satawan", "satavan"),
    58: ("atthavan", "athawan", "atthawan", "athavan"),
    59: ("unsath", "unsaath", "unsatth"),
    60: ("saath", "sath"),
    61: ("iksath", "ikasath", "iksaath", "ikasaath"),
    62: ("baasath", "basath", "baasaath", "basaath"),
    63: ("tresath", "tirsath", "tresaath", "tirsaath"),
    64: ("chaunsath", "chonsath", "chaunsaath", "chonsaath"),
    65: ("painsath", "pensath", "painsaath", "pensaath"),
    66: ("chhiyasath", "chiyasath", "chhiyasaath", "chiyasaath"),
    67: ("sarsath", "sadsath", "sarsaath", "sadsaath"),
    68: ("arsath", "adsath", "arsaath", "adsaath"),
    69: ("unhattar", "unhatar"),
    70: ("sattar", "satar", "sattur"),
    71: ("ikhattar", "ikahattar", "ikhatar"),
    72: ("bahattar", "behattar", "bahatar"),
    73: ("tihattar", "tehattar", "tihatar"),
    74: ("chauhattar", "chohattar", "chauhatar"),
    75: ("pachhattar", "pachattar", "pachhatar", "pichattar"),
    76: ("chhihattar", "chihattar", "chhihatar"),
    77: ("sathattar", "satattar", "sathatar", "satahattar"),
    78: ("athhattar", "athattar", "athhatar", "athahattar"),
    79: ("unasi", "unaasi", "unnasi", "unasee"),
    80: ("assi", "asi", "assee", "assy"),
    81: ("ikyasi", "ikiyasi", "ikasi", "ikyaasi"),
    82: ("bayasi", "biyasi", "bayaasi"),
    83: ("tirasi", "tiraasi"),
    84: ("chaurasi", "chorasi", "chauraasi"),
    85: ("pachasi", "pachaasi"),
    86: ("chhiyasi", "chiyasi", "chhiyaasi"),
    87: ("sattasi", "satasi", "sattaasi"),
    88: ("atthasi", "athasi", "atthaasi"),
    89: ("navasi", "nawasi", "nawaasi"),
    90: ("nabbe", "navve", "nawway", "nawe", "navay", "nabbay", "nabe", "nave"),
    91: ("ikyanave", "ikanway", "ikyanve", "ikyanway", "ikkyanve"),
    92: ("banave", "banway", "baanway", "baanve", "baanave"),
    93: ("tiranve", "tiranway", "tiranave"),
    94: ("chauranve", "choranway", "chauranway", "chauranave"),
    95: ("pachanve", "pachanway", "pachanvay", "pachanave"),
    96: ("chhiyanve", "chiyanway", "chhiyanway", "chhiyanave"),
    97: ("sattanve", "sattanway", "sattanave"),
    98: ("atthanve", "athanway", "atthanway", "atthanave"),
    99: ("ninnanve", "ninanway", "ninyanve", "ninnanway", "ninnanave"),
}

ROMAN_NUMBER_WORDS: dict[str, str] = {
    **{spelling: _UNITS[n][0] for n, spellings in _ROMAN_UNITS.items() for spelling in spellings},
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


# Roman ordinals. The irregular ones are spelled out; every other ordinal is a Roman
# cardinal plus -wan/-van/-vaan or -wen/-ven/-veen/-vin (paanchwan, dasvi), and an
# English numeral ordinal (5th, 21st) is accepted too.
ROMAN_ORDINAL_WORDS: dict[str, str] = {
    **dict.fromkeys(("pehla", "pehli", "pehle", "pahla", "pahli", "pahle", "pela"), "پہلا"),
    **dict.fromkeys(("doosra", "dusra", "doosri", "dusri", "doosre", "dusre"), "دوسرا"),
    **dict.fromkeys(("teesra", "tisra", "teesri", "tisri", "teesre", "tisre"), "تیسرا"),
    **dict.fromkeys(("chautha", "chotha", "chauthi", "chothi", "chauthe", "chothe"), "چوتھا"),
    **dict.fromkeys(("chhata", "chata", "chhati", "chati", "chhate", "chate"), "چھٹا"),
}
_ROMAN_ORDINAL_SUFFIX = re.compile(
    r"^(?P<base>[a-z]+?)(?:waan|wan|vaan|van|ween|wen|veen|ven|vin|win|vi|wi)$"
)
_ENGLISH_ORDINAL = re.compile(r"^(?P<n>\d+)(?:st|nd|rd|th)$", re.IGNORECASE)

# Roman number words that are read as a number even on their own in running text,
# as ایک is. The rest - do (give), so (sleep), no, sat, tin, char, arab - are ordinary
# words far more often, and count only next to another number word.
UNAMBIGUOUS_ROMAN = frozenset(
    {"ek", "aik", "teen", "chaar", "paanch", "panch", "saat", "aath", "das", "dus",
     "bees", "pachas", "pachaas", "sau", "hazar", "hazaar", "lakh", "laakh", "crore",
     "karor", "dedh", "derh", "dhai", "dhaai", "adhai"}
)  # fmt: skip


def _roman_ordinal(token: str) -> str | None:
    """The Urdu ordinal word, or cardinal numeral, a Roman ordinal stands for."""
    lowered = token.lower()
    if lowered in ROMAN_ORDINAL_WORDS:
        return ROMAN_ORDINAL_WORDS[lowered]
    english = _ENGLISH_ORDINAL.match(lowered)
    if english:
        return english["n"]
    match = _ROMAN_ORDINAL_SUFFIX.match(lowered)
    if match and match["base"] in ROMAN_NUMBER_WORDS:
        base = ROMAN_NUMBER_WORDS[match["base"]]
        if base not in _MODIFIERS and base not in _FRACTION_WORDS and base not in _DENOMINATORS:
            return base
    return None


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
        or token in _DENOMINATORS
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
                raise ValueError(f"unexpected {_show(token)}")
            modifier = _MODIFIERS[token]
            continue
        seen = True
        if token in _DENOMINATORS:
            # Divides what came before it. A bare denominator means one of them: نصف on
            # its own is a half, as آدھا is.
            share = _DENOMINATORS[token]
            if current is None:
                current = take_modifier(Fraction(1)) * share
            else:
                current = take_modifier(current) * share
            open_hundred = False
            continue
        if token in _WORD_VALUE or token in _FRACTION_WORDS or _numeral_value(token) is not None:
            if token in _WORD_VALUE:
                value = Fraction(_WORD_VALUE[token])
            elif token in _FRACTION_WORDS:
                value = _FRACTION_WORDS[token]
            else:
                value = _numeral_value(token)  # type: ignore[assignment]
            if current is not None:
                if not (open_hundred and value < 100 and modifier is None):
                    raise ValueError(f"two numbers in a row: {_show(token)}")
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
                raise ValueError(f"{_show(token)} has no number before it")
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
            raise ValueError(f"not a number word: {_show(token)}")

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


def parse_ordinal(text: str) -> int:
    """The position an Urdu ordinal names.

    >>> parse_ordinal("تیسرا")
    3
    >>> parse_ordinal("ایک سو پانچواں")
    105
    >>> parse_ordinal("pehla"), parse_ordinal("paanchwan"), parse_ordinal("21st")
    (1, 5, 21)

    Accepts the irregular ordinals (پہلا دوسرا تیسرا چوتھا چھٹا, and یکم for the
    first of a month), any cardinal plus واں or ویں, and a numeral followed by
    either ending (5ویں). Roman Urdu works the same way: pehla, doosri, teesra,
    chautha, chhata, any Roman cardinal plus -wan or -vi (paanchwan, dasvi), and
    English 5th / 21st. Only the last word may be ordinal. Raises ValueError for
    anything else - including a plain cardinal, which is not an ordinal.
    """
    _require_str(text, "parse_ordinal")
    raw = text.split()
    last = _roman_ordinal(raw[-1]) if raw else None
    if last is not None:
        # pehla, teesri, paanchwan, ek sau paanchwan, 5th: the Roman last word is
        # read as its Urdu ordinal (or, for -wan and 5th, as the cardinal).
        before = _tokens(" ".join(raw[:-1]))
        position = _evaluate([*before, _ordinal_as_cardinal(last) or last])
        if position.denominator != 1:
            raise ValueError(f"not a whole position: {_show(text)}")
        return int(position)
    tokens = _tokens(text)
    if tokens and tokens[-1] in _ORDINAL_SUFFIXES and len(tokens) >= 2:
        tokens = tokens[:-1]  # 5 ویں: the ending written apart from a numeral
        cardinal = tokens[-1] if _numeral_value(tokens[-1]) is not None else None
    else:
        cardinal = _ordinal_as_cardinal(tokens[-1]) if tokens else None
    if cardinal is None:
        raise ValueError(f"not an ordinal: {_show(text)}")
    value = _evaluate([*tokens[:-1], cardinal])
    if value.denominator != 1:
        raise ValueError(f"not a whole position: {_show(text)}")
    return int(value)


@dataclass(frozen=True)
class NumberSpan:
    """A number phrase found by `find_numbers`.

    `ordinal` is True for a position rather than a quantity - تیسرا (third),
    پانچویں (fifth) - in which case `value` is the position: 3, 5.
    """

    text: str
    value: int | float
    start: int
    end: int
    ordinal: bool = False


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
    `dedh crore`), and so is a single Roman number word that is nothing else -
    `ek`, `teen`, `paanch`, `hazar` - as ایک is; `do`, `so`, `no` and `sat` on their
    own are words, not numbers.
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

    def joined(a: int, b: int) -> bool:
        """Whether tokens a and b belong to one phrase - only whitespace between."""
        return not text[matches[a].end() : matches[b].start()].strip()

    # Straight before a unit, a Latin word the table does not know is read with
    # `roman_key` too, as `parse_number` reads every word: `pandhra lakh` and
    # `barrah hazar` are amounts, and there a number word is what the writer meant.
    for k in range(len(matches) - 1):
        if (
            roman[k]
            and not _is_number_token(canonical[k])
            and (canonical[k + 1] in _SCALE_VALUE or canonical[k + 1] == HUNDRED)
            and joined(k, k + 1)
        ):
            canonical[k] = _from_roman(canonical[k])

    def ordinal_at(k: int) -> str | None:
        """What token k contributes as an ordinal: a cardinal word, "" for a bare
        ویں/واں ending glued to the numeral before it (5ویں), or None."""
        if k >= len(matches) or roman[k]:
            return None
        token = canonical[k]
        if token in _ORDINAL_SUFFIXES:
            glued = k > 0 and matches[k - 1].end() == matches[k].start()
            return "" if glued and _numeral_value(canonical[k - 1]) is not None else None
        return _ordinal_as_cardinal(token)

    i = 0
    while i < len(matches):
        if not _is_number_token(canonical[i]) and ordinal_at(i) in (None, ""):
            i += 1
            continue
        if not _is_number_token(canonical[i]):
            # An ordinal on its own: تیسرا, پانچویں. پہلے ("before") and دوسرے
            # ("other") are ordinals far less often than they are anything else.
            if canonical[i] not in AMBIGUOUS_ORDINALS:
                value = _evaluate([ordinal_at(i) or ""])
                start, end = matches[i].start(), matches[i].end()
                spans.append(NumberSpan(text[start:end], int(value), start, end, ordinal=True))
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
        # A cardinal phrase followed by an ordinal is one ordinal: ایک سو پانچواں is
        # the 105th, and 5ویں the 5th.
        following = ordinal_at(j + 1)
        if following is not None and joined(j, j + 1):
            words = canonical[i : j + 1] + ([following] if following else [])
            try:
                position = _evaluate(words)
            except ValueError:
                position = None
            if position is not None and position.denominator == 1:
                start, end = matches[i].start(), matches[j + 1].end()
                spans.append(NumberSpan(text[start:end], int(position), start, end, True))
                i = j + 2
                continue
        words = canonical[i : j + 1]
        # lakh, crore, arab or kharab with no number before it, straight after a Latin
        # word that is not a number word even by `roman_key`: `sawa baara lakh`, read
        # before `baara` was in the table, gave a bare `lakh` = 100,000 for 1,225,000.
        # The word before is then a number spelled some way nothing here knows, so
        # the amount is left out rather than reported wrong. A bare `sau` or `hazar`
        # after a word is kept: `mere paas sau rupay` is a hundred rupees.
        if (
            words[0] in _SCALE_VALUE
            and words[0] != "ہزار"
            and i > 0
            and roman[i - 1]
            and joined(i - 1, i)
            and not _is_number_token(canonical[i - 1])
        ):
            i = j + 1
            continue
        # A lone ambiguous word is not a number: اسی is "that same", and in Roman
        # text so, no and do are English and Urdu words long before they are 100,
        # 9 and 2.
        # A Roman word is judged by its spelling, not by the Urdu word it stands for:
        # `sau` is only ever a hundred, though سو is also "so".
        if len(words) == 1 and (
            matches[i].group().lower() not in UNAMBIGUOUS_ROMAN
            if roman[i]
            else words[0] in AMBIGUOUS
        ):
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
