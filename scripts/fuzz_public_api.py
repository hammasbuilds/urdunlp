r"""Throw hostile input at every name in `__all__` and check the invariants hold.

    python scripts/fuzz_public_api.py

Not "does it crash" alone - a library that never crashes can still be wrong in ways that
are checkable without knowing the right answer:

  * `normalize` is idempotent.
  * every piece `words` returns appears in the NORMALISED input, in order.
  * `parse_number(format_number(n)) == n`, for integers and fractions, in both digit
    scripts.
  * a parser raises `ValueError` on text that is not what it parses, and `TypeError` on a
    value that is not a string - never `IndexError`, `KeyError`, `RecursionError` or a
    `UnicodeError` from inside.

It found two things on its first run. `format_number(-1234567)` writes `'-12,34,567'` and
`parse_number` raised *"not a number word: '-'"*, so the documented pair was not a pair for
any negative number. And `words` returns normalised tokens, which are not always substrings
of its input - true, reasonable, and nowhere in the docstring until the property check
asked.

**It also reported 80 findings that were its own fault**, which is the more useful lesson:
it walked `dir()` instead of `__all__` and so called the `tokenize` submodule; it fed
strings to `number_to_words`, which takes an int; it fed a list to `remove_stopwords`,
which takes a list; and it counted a documented `ValueError` from a parser as a defect.
A harness that reports its own misuse at twenty lines a time is a harness whose two real
findings nobody reads, so the contract is written down here instead.
"""

from __future__ import annotations

import pathlib
import sys
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import urdunlp  # noqa: E402
from urdunlp import normalize  # noqa: E402

HOSTILE = [
    "",
    " ",
    "\t\n\r ",
    "‌",  # zero-width non-joiner, legal inside an Urdu word
    "‍" * 50,  # zero-width joiner run
    "‮" + "abc" + "‬",  # RTL override
    "ّ",  # a lone shadda: a combining mark with no base
    "ًٌٍ" * 20,  # a pile of harakat
    "﻿الف",  # byte-order mark then letters
    "a" * 5000,  # the input a quadratic path was found on
    "ہ" * 3000,
    "۱۲۳٤٥٦789",  # digits in three scripts at once
    "ـ" * 100,  # tatweel run
    "ایک\x00دو",  # embedded NUL
    "कاا",  # Devanagari beside Urdu
    "\U0001f600ا",  # astral plane beside an Urdu letter
    "-" * 200,
    "ڈیڑھ لاکھ " * 200,
    "ا" + "ّ" * 500,  # one base, five hundred marks
    "﷽",  # a single presentation-form codepoint
]

NOT_STRINGS = [None, 0, 1.5, [], {}, b"bytes", object()]

# What each public callable takes. Written out rather than guessed, because guessing is
# what produced most of this harness's first run.
TAKES_TEXT = (
    "normalize",
    "words",
    "sentences",
    "is_urdu",
    "identify_language",
    "stem",
    "roman_key",
    "fix_spacing",
    "remove_urls_and_mentions",
    "resolve_arabic_heh",
    "transliterate_to_roman",
    "transliterate_to_urdu",
    "transliterate_with_confidence",
    "find_numbers",
    "is_stopword",
    "tag_roman_tokens",
    "character_ngrams",
)
# Parsers: the same, but a ValueError on text that is not a number is the contract.
TAKES_TEXT_MAY_REFUSE = ("parse_number", "parse_ordinal")
TAKES_TOKEN_LIST = ("remove_stopwords", "stem_tokens", "group_roman_variants")
TAKES_NUMBER = ("format_number", "number_to_words")

# Exceptions that are never an answer, only a crash.
BUGS = (IndexError, KeyError, RecursionError, UnicodeError, AttributeError, ZeroDivisionError)

problems: list[str] = []


def note(what: str) -> None:
    problems.append(what)
    print(f"  {what}")


def _where() -> str:
    lines = traceback.format_exc().strip().splitlines()
    return lines[-2].strip() if len(lines) > 1 else ""


def main() -> int:
    missing = [n for n in TAKES_TEXT + TAKES_TEXT_MAY_REFUSE + TAKES_TOKEN_LIST
               + TAKES_NUMBER if n not in urdunlp.__all__]  # fmt: skip
    for name in missing:
        note(f"{name} is in this harness but no longer in urdunlp.__all__")

    callables = [
        n
        for n in urdunlp.__all__
        if callable(getattr(urdunlp, n)) and not isinstance(getattr(urdunlp, n), type)
    ]
    covered = set(TAKES_TEXT + TAKES_TEXT_MAY_REFUSE + TAKES_TOKEN_LIST + TAKES_NUMBER)
    for name in sorted(set(callables) - covered):
        note(f"{name} is public and this harness does not exercise it - classify it")

    print("=== hostile text ===")
    for name in TAKES_TEXT + TAKES_TEXT_MAY_REFUSE:
        fn = getattr(urdunlp, name)
        allowed = (ValueError,) if name in TAKES_TEXT_MAY_REFUSE else ()
        for text in HOSTILE:
            try:
                fn(text)
            except allowed:
                pass
            except BUGS as exc:
                note(f"{name}({text[:16]!r}...) raised {type(exc).__name__}: "
                     f"{str(exc)[:70]}\n      {_where()}")  # fmt: skip
            except Exception as exc:  # noqa: BLE001
                note(f"{name}({text[:16]!r}...) raised {type(exc).__name__}: "
                     f"{str(exc)[:70]}\n      {_where()}")  # fmt: skip

    print("=== hostile token lists ===")
    for name in TAKES_TOKEN_LIST:
        fn = getattr(urdunlp, name)
        for text in HOSTILE:
            for tokens in ([], [text], list(text[:40]), [text, "", text]):
                try:
                    fn(tokens)
                except BUGS as exc:
                    note(f"{name}({str(tokens)[:22]}...) raised {type(exc).__name__}: "
                         f"{str(exc)[:60]}\n      {_where()}")  # fmt: skip
                except Exception as exc:  # noqa: BLE001
                    note(f"{name}({str(tokens)[:22]}...) raised {type(exc).__name__}: "
                         f"{str(exc)[:60]}")  # fmt: skip

    print("=== normalize is idempotent ===")
    for text in HOSTILE:
        once = normalize(text)
        if normalize(once) != once:
            note(f"normalize is not idempotent on {text[:24]!r}")

    print("=== every token appears in the normalised input, in order ===")
    for text in HOSTILE:
        flat = normalize(text)
        at = 0
        for piece in urdunlp.words(text):
            found = flat.find(piece, at)
            if found < 0:
                note(f"words({text[:20]!r}) returned {piece[:20]!r}, which is not in the "
                     "normalised input at or after the previous token")  # fmt: skip
                break
            at = found + len(piece)

    print("=== numbers round-trip ===")
    values: list[int | float] = [0, 1, 9, 99, 1000, 100000, 1234567, 10**9, 2**31,
                                 -1, -5, -100000, -1234567, 0.5, -0.5, 2.5, -2.5]  # fmt: skip
    for value in values:
        for urdu in (False, True):
            text = urdunlp.format_number(value, urdu_digits=urdu)
            try:
                back = urdunlp.parse_number(text)
            except Exception as exc:  # noqa: BLE001
                note(f"parse_number(format_number({value}, urdu_digits={urdu})={text!r}) "
                     f"raised {type(exc).__name__}: {str(exc)[:50]}")  # fmt: skip
                continue
            if back != value:
                note(f"round trip changed {value}: {text!r} parsed back as {back!r}")

    print("=== a non-string must raise TypeError ===")
    for name in TAKES_TEXT + TAKES_TEXT_MAY_REFUSE:
        fn = getattr(urdunlp, name)
        for bad in NOT_STRINGS:
            try:
                fn(bad)
            except TypeError:
                pass
            except Exception as exc:  # noqa: BLE001
                note(f"{name}({bad!r}) raised {type(exc).__name__}, not TypeError")
            else:
                note(f"{name}({bad!r}) returned instead of raising TypeError")

    print()
    if problems:
        print(f"{len(problems)} problem(s)")
        return 1
    print(
        f"nothing found. {len(TAKES_TEXT + TAKES_TEXT_MAY_REFUSE)} text functions and "
        f"{len(TAKES_TOKEN_LIST)} list functions survived {len(HOSTILE)} hostile inputs; "
        "normalize is idempotent on all of them; every token appears in the normalised\n"
        f"input; and {len(values)} values round-trip through format_number and "
        "parse_number in both digit scripts."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
