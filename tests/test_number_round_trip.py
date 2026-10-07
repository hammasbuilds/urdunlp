"""`format_number` and `parse_number` have to be each other's inverse.

Found by fuzzing the public API rather than by a report: `format_number(-1234567)` writes
`'-12,34,567'` and `parse_number` raised *"not a number word: '-'"* on it. Every other
shape already round-tripped - Urdu digits, the Arabic group separator `٬`, the Arabic
decimal separator `٫`, a fractional part - so the minus sign was the single asymmetry in a
pair of functions documented as a pair.

A sign is accepted at the START of a phrase only. `5-10` is a range, not negative five,
and must keep raising; silently returning 5 for it would be worse than the original bug.
"""

from __future__ import annotations

import pytest

from urdunlp import format_number, parse_number

VALUES = [0, 1, 5, 9, 10, 99, 100, 1000, 100000, 1234567, 10**9, 2**31, -1, -5, -100000, -1234567]


@pytest.mark.parametrize("value", VALUES)
@pytest.mark.parametrize("urdu_digits", [False, True])
def test_an_integer_survives_format_then_parse(value: int, urdu_digits: bool) -> None:
    text = format_number(value, urdu_digits=urdu_digits)
    assert parse_number(text) == value, f"{value} -> {text!r} -> {parse_number(text)}"


@pytest.mark.parametrize("value", [0.5, -0.5, 2.5, -2.5, 1.25, -1.25])
@pytest.mark.parametrize("urdu_digits", [False, True])
def test_a_fraction_survives_format_then_parse(value: float, urdu_digits: bool) -> None:
    text = format_number(value, urdu_digits=urdu_digits)
    assert parse_number(text) == value, f"{value} -> {text!r} -> {parse_number(text)}"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("-5", -5),
        ("−5", -5),            # U+2212 MINUS SIGN, from a document
        ("－5", -5),            # U+FF0D FULLWIDTH HYPHEN-MINUS, from a spreadsheet
        ("- 5", -5),                # a space after the sign
        ("-۵", -5),            # an Urdu digit
        ("+5", 5),                  # accepted, and means nothing
        ("＋5", 5),
        ("-ڈیڑھ لاکھ", -150000),  # a negated WORD phrase
    ],
)
def test_the_sign_characters_a_paste_brings(text: str, expected: int) -> None:
    """Each sign is asserted against a value, not against a sign-of-the-value.

    The first version of this test computed the expected value from the text with a
    conditional ending in `or True`, which made the whole expression vacuous - it
    asserted that something was truthy and passed whatever the function returned. That is
    the same defect this file's subject was found by.
    """
    assert parse_number(text) == expected


@pytest.mark.parametrize("text", ["5-10", "۵-۱۰", "-", "- ", "+", "10-"])
def test_a_sign_that_is_not_a_sign_still_raises(text: str) -> None:
    """A range must not parse as a negative number, and a bare sign is not a number."""
    with pytest.raises(ValueError):
        parse_number(text)


def test_a_sign_in_the_middle_is_not_a_sign() -> None:
    """Only a LEADING sign is one. `دو - تین` is not minus anything."""
    with pytest.raises(ValueError):
        parse_number("دو - تین")


def test_tokens_are_normalised_and_the_docstring_says_so() -> None:
    """A token that is not a substring of the input is surprising, so it is documented.

    `words` normalises before splitting, which is what makes every function downstream
    agree with itself - but it means `str.find` cannot map a token back to a source
    offset. Found by a property check ("every piece appears in the input, in order")
    rather than by anything failing, and the fix was to state the contract, not to stop
    normalising.
    """
    from urdunlp import words
    from urdunlp.tokenize import words as words_fn

    assert words("كتاب") == ["کتاب"]
    assert words("٤٥٦") == ["۴۵۶"]
    doc = words_fn.__doc__ or ""
    assert "not always substrings of the input" in doc, (
        "the docstring no longer warns that tokens are normalised"
    )
