"""Urdu numbers: every word below a hundred is irregular, and fractions scale groups."""

from __future__ import annotations

import random

import pytest

import urdunlp as U
from urdunlp import find_numbers, format_number, number_to_words, parse_number, parse_ordinal
from urdunlp.numbers import _UNITS


class TestParse:
    @pytest.mark.parametrize(
        ("text", "value"),
        [
            ("ایک", 1),
            ("اکیس", 21),  # not "twenty-one": its own word
            ("ننانوے", 99),
            ("سو", 100),
            ("ایک سو ایک", 101),
            ("دو ہزار پانچ سو تیس", 2530),
            ("ایک لاکھ", 100_000),
            ("تین کروڑ", 30_000_000),
            ("پانچ ارب", 5_000_000_000),
        ],
    )
    def test_plain_numbers(self, text, value):
        assert parse_number(text) == value

    @pytest.mark.parametrize(
        ("text", "value"),
        [
            ("ڈیڑھ", 1.5),
            ("ڈھائی", 2.5),
            ("ڈیڑھ لاکھ", 150_000),  # the fraction scales the whole group
            ("ڈھائی سو", 250),
            ("سوا دو کروڑ", 22_500_000),  # سوا adds a quarter
            ("ساڑھے تین ہزار", 3_500),  # ساڑھے adds a half
            ("پونے دو سو", 175),  # پونے takes a quarter away
            ("سوا لاکھ", 125_000),  # with nothing after it, سوا modifies an implied one
        ],
    )
    def test_fractions_scale_the_group_they_modify(self, text, value):
        """None of these has an English counterpart. A parser that treats ڈیڑھ as
        an unknown word reads ڈیڑھ لاکھ as 100,000 and loses a third of the value."""
        assert parse_number(text) == value

    @pytest.mark.parametrize(
        ("text", "value"),
        [
            ("15 لاکھ", 1_500_000),
            ("2.5 کروڑ", 25_000_000),
            ("۱۵ لاکھ", 1_500_000),
            ("12,34,567", 1_234_567),
        ],
    )
    def test_digits_in_any_script_combine_with_scale_words(self, text, value):
        assert parse_number(text) == value

    def test_a_larger_scale_after_a_smaller_one_multiplies_everything_before_it(self):
        """ایک ہزار کروڑ is a thousand crore - 10,000,000,000.

        The first version added an implied "one" before کروڑ and returned
        10,010,000,000. The round-trip test could not catch it, because
        number_to_words never writes this form - but people do.
        """
        assert parse_number("ایک ہزار کروڑ") == 10_000_000_000
        assert parse_number("ایک لاکھ کروڑ") == 10**12

    def test_a_number_after_hundred_is_added_not_rejected(self):
        """پانچ سو تیس is 530. The first version saw تیس arriving while 500 was
        still open and raised "two numbers in a row" - rejecting every hundred
        that was not a round one."""
        assert parse_number("پانچ سو تیس") == 530

    @pytest.mark.parametrize(
        ("text", "value"),
        [
            ("dedh lakh", 150_000),
            ("sawa do crore", 22_500_000),
            ("dhai sau", 250),
            ("sadhe teen hazar", 3_500),
            ("15 lakh", 1_500_000),
            ("pone do sau", 175),
        ],
    )
    def test_roman_urdu_amounts(self, text, value):
        assert parse_number(text) == value

    @pytest.mark.parametrize("text", ["دو تین", "سوا", "سو سو", "ہیلو", ""])
    def test_non_numbers_are_rejected_rather_than_guessed(self, text):
        with pytest.raises(ValueError):
            parse_number(text)


class TestWords:
    def test_all_hundred_irregular_words_are_present(self):
        assert sorted(_UNITS) == list(range(100))

    def test_round_trip_every_value_below_two_hundred_thousand(self):
        for value in range(200_001):
            assert parse_number(number_to_words(value)) == value

    def test_round_trip_random_large_values(self):
        rng = random.Random(0)
        for _ in range(20_000):
            value = rng.randrange(10**13)
            assert parse_number(number_to_words(value)) == value

    def test_lakh_and_crore_not_million(self):
        assert number_to_words(150_000) == "ایک لاکھ پچاس ہزار"
        assert number_to_words(10**7) == "ایک کروڑ"

    def test_negative_and_non_int_are_refused(self):
        with pytest.raises(ValueError):
            number_to_words(-1)
        with pytest.raises(TypeError):
            number_to_words(1.5)  # type: ignore[arg-type]


class TestFormat:
    @pytest.mark.parametrize(
        ("value", "text"),
        [
            (999, "999"),
            (1_000, "1,000"),
            (100_000, "1,00,000"),
            (1_234_567, "12,34,567"),
            (10**7, "1,00,00,000"),
            (-100_000, "-1,00,000"),
            (1234.5, "1,234.5"),
        ],
    )
    def test_three_then_two_grouping(self, value, text):
        assert format_number(value) == text

    def test_urdu_digits_and_separators(self):
        assert format_number(1_234_567, urdu_digits=True) == "۱۲٬۳۴٬۵۶۷"


class TestFind:
    def test_finds_amounts_in_running_text(self):
        spans = find_numbers("اس نے ڈیڑھ لاکھ روپے اور دو سو گائیں خریدیں")
        assert [(s.text, s.value) for s in spans] == [("ڈیڑھ لاکھ", 150_000), ("دو سو", 200)]

    def test_a_lone_ambiguous_word_is_not_a_number(self):
        """اسی is 80 and also "that same"; in prose it is almost always the pronoun."""
        assert find_numbers("اسی لیے وہ آیا") == []

    def test_roman_amounts_but_not_lone_roman_words(self):
        """`so`, `no` and `do` are English and Urdu words before they are 100, 9, 2."""
        spans = find_numbers("mera ghar 15 lakh ka hai aur gaari dedh crore ki")
        assert [(s.text, s.value) for s in spans] == [
            ("15 lakh", 1_500_000),
            ("dedh crore", 15_000_000),
        ]
        assert find_numbers("so what do you want, no") == []

    def test_spans_point_into_the_normalised_text(self):
        span = find_numbers("سال 1998 میں")[0]
        assert span.value == 1998 and "سال 1998 میں"[span.start : span.end] == "1998"


class TestOrdinals:
    @pytest.mark.parametrize(
        ("text", "position"),
        [
            ("پہلا", 1),
            ("یکم", 1),  # the first of a month
            ("تیسرا", 3),
            ("چھٹی", 6),  # irregular, like 1st-4th
            ("پانچواں", 5),
            ("پانچویں", 5),
            ("نویں", 9),  # 9th drops the و: نو + یں
            ("بیسویں", 20),
            ("ہزارواں", 1000),
            ("ایک سو پانچواں", 105),  # only the last word is ordinal
            ("5ویں", 5),
        ],
    )
    def test_positions(self, text, position):
        assert U.parse_ordinal(text) == position

    @pytest.mark.parametrize("text", ["دس", "بیسویں صدی", "کتاب", ""])
    def test_what_is_not_an_ordinal_is_refused(self, text):
        with pytest.raises(ValueError):
            U.parse_ordinal(text)

    def test_found_in_running_text_and_flagged(self):
        spans = U.find_numbers("بیسویں صدی میں ایک سو پانچواں دن اور دو ہزار روپے")
        assert [(s.text, s.value, s.ordinal) for s in spans] == [
            ("بیسویں", 20, True),
            ("ایک سو پانچواں", 105, True),
            ("دو ہزار", 2000, False),
        ]

    @pytest.mark.parametrize("text", ["اس سے پہلے وہ آیا", "دوسرے لوگ آئے"])
    def test_words_that_mostly_mean_something_else_are_left_alone(self, text):
        """پہلے is "before" 11,584 times in the corpus; دوسرے is "other"."""
        assert U.find_numbers(text) == []


# --- audit round 2 -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("pehla", 1),  # raised ValueError, though Roman cardinals worked
        ("pehli", 1),
        ("doosri", 2),
        ("teesra", 3),
        ("chautha", 4),
        ("chhata", 6),
        ("paanchwan", 5),
        ("dasvi", 10),
        ("ek sau paanchwan", 105),
        ("5th", 5),
        ("21st", 21),
    ],
)
def test_roman_ordinals(text, value):
    assert parse_ordinal(text) == value


def test_a_roman_cardinal_is_still_not_an_ordinal():
    with pytest.raises(ValueError):
        parse_ordinal("paanch")


def test_a_single_unambiguous_roman_number_word_is_found_as_urdu_is():
    """ایک was found on its own and `ek` was not."""
    found = [(n.text, n.value) for n in find_numbers("ek din main aaya")]
    assert found == [("ek", 1)]
    assert [n.value for n in find_numbers("ایک دن میں آیا")] == [1]
    # do (give), so (sleep) and no are words, not numbers, on their own
    assert find_numbers("do din baad so gaye, no problem") == []
    assert [n.value for n in find_numbers("do lakh")] == [200000]
