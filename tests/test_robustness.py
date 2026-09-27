"""Inputs real users send, and the ways they broke things.

Every test here comes from a probe or a fuzzing run over the public API: 20,000
adversarial inputs across all 23 entry points, plus hand-written messy Roman Urdu -
prices, times, phone numbers, acronyms, line breaks - and very large inputs.
"""

from __future__ import annotations

import time

import pytest

import urdunlp as U

# --- Wrong types ------------------------------------------------------------------

STRING_FUNCTIONS = [
    U.normalize,
    U.is_urdu,
    U.remove_urls_and_mentions,
    U.resolve_arabic_heh,
    U.words,
    U.sentences,
    U.fix_spacing,
    U.character_ngrams,
    U.is_stopword,
    U.stem,
    U.roman_key,
    U.transliterate_to_urdu,
    U.transliterate_with_confidence,
    U.transliterate_to_roman,
    U.identify_language,
    U.tag_roman_tokens,
    U.parse_number,
    U.find_numbers,
]


@pytest.mark.parametrize("function", STRING_FUNCTIONS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("bad", [None, 1.5, 123, b"kal", ["kal"]], ids=repr)
def test_every_function_rejects_a_non_string_the_same_way(function, bad):
    """A None from a missing DataFrame cell, or a NaN (a float), used to get five
    different answers: normalize(None) returned '', words(None) returned [],
    roman_key(None) raised AttributeError from deep inside, is_urdu(['kal'])
    returned False. Now every function fails at the call and names itself."""
    with pytest.raises(TypeError, match=rf"{function.__name__}\(\) expects a str"):
        function(bad)


@pytest.mark.parametrize("function", [U.stem_tokens, U.remove_stopwords, U.group_roman_variants])
def test_a_string_is_not_a_list_of_words(function):
    """A string is iterable, so stem_tokens("کتابوں سے") used to stem each
    character and return a list of letters - no error, just wrong output."""
    with pytest.raises(TypeError, match="expects a list of words, got a str"):
        function("کتابوں سے")


# --- Transliteration keeps what is not a word ---------------------------------------


@pytest.mark.parametrize(
    ("roman", "kept"),
    [
        ("price 2.5 crore hai", "2.5"),
        ("meeting 3:30 baje hai", "3:30"),
        ("1,500 rupay", "1,500"),
        ("0300-1234567 pe call karo", "0300-1234567"),
        ("aaj 25-12-2024 hai", "25-12-2024"),
        ("10% discount", "10%"),
        ("5th class mein", "5th"),
        ("mp3 file bhejo", "mp3"),
    ],
)
def test_numbers_and_codes_survive_intact(roman, kept):
    """The output used to be the converted tokens joined by single spaces, which
    split 2.5 into `2. 5` and 0300-1234567 into `0300- 1234567`, and read 5th as
    the number 5 plus the Roman word `th`."""
    assert kept in U.transliterate_to_urdu(roman)


@pytest.mark.parametrize(
    "roman",
    ["kal\nparso", "sab theek hai :)", "  kal  ", "kal\n\nparso\tphir", "nahi.... pata"],
)
def test_the_input_spacing_is_kept_exactly(roman):
    """Line breaks, tabs, runs of spaces and the space before an emoticon all
    vanished when tokens were joined by single spaces."""
    out = U.transliterate_to_urdu(roman)
    assert [c for c in out if c.isspace()] == [c for c in roman if c.isspace()]


def test_a_line_break_ends_the_sentence_context():
    result = U.transliterate_to_urdu("is ke baad\nwoh gaya")
    assert result == "اس کے بعد\nوہ گیا"


@pytest.mark.parametrize(
    ("roman", "urdu"),
    [
        ("TV", "ٹی وی"),
        ("BBC", "بی بی سی"),
        ("FBI", "ایف بی آئی"),
        ("U.S.A", "یو ایس اے"),
        ("U.N.", "یو این"),
    ],
)
def test_acronyms_are_spelled_by_letter_name(roman, urdu):
    """Dakshina's annotators wrote TV as ٹی وی 25 times out of 25."""
    assert U.transliterate_to_urdu(roman) == urdu
    assert U.transliterate_with_confidence(roman).sources == [(roman, "acronym")]


def test_acronyms_said_as_words_are_not_spelled():
    """FIFA and UNESCO are pronounced as words - two vowels or more."""
    assert U.transliterate_with_confidence("FIFA").sources[0][1] != "acronym"


def test_other_scripts_pass_through_unchanged():
    assert U.transliterate_to_urdu("café hai") == "café ہے"


# --- Numbers --------------------------------------------------------------------


def test_offsets_index_the_text_that_was_passed_in():
    """Spans used to index the *normalised* text. With a double space or a
    diacritic before the number, start and end pointed at the wrong characters of
    the caller's own string."""
    text = "اس نے  کِتاب کے لیے ڈیڑھ  لاکھ روپے"
    (span,) = U.find_numbers(text)
    assert text[span.start : span.end] == span.text == "ڈیڑھ  لاکھ"
    assert span.value == 150_000


def test_punctuation_ends_a_number_phrase():
    """ایک لاکھ، دو ہزار was read as one number, 102,000."""
    spans = U.find_numbers("ایک لاکھ، دو ہزار")
    assert [s.value for s in spans] == [100_000, 2_000]
    # not eighty lakh; `lakh` on its own is a lakh, as لاکھ on its own is
    assert [s.value for s in U.find_numbers("اسی،lakh")] == [100_000]


def test_arabic_decimal_and_thousands_separators():
    assert U.parse_number("۱۲٫۵") == 12.5
    assert U.parse_number("۱٬۰۰٬۰۰۰") == 100_000
    assert U.find_numbers("۱۲٫۵ لاکھ")[0].value == 1_250_000


def test_a_bare_scale_after_a_finished_group_is_not_an_implied_one():
    """کروڑ ہزار was 10,001,000, and "12,34,567 کروڑ ہزار" 12,345,670,001,000."""
    with pytest.raises(ValueError, match="has no number before it"):
        U.parse_number("کروڑ ہزار")
    assert U.parse_number("ہزار لاکھ") == 100_000_000  # a larger scale still multiplies


# --- Edge values ------------------------------------------------------------------


def test_empty_input_gives_empty_output_everywhere():
    assert U.normalize("") == ""
    assert U.words("") == []
    assert U.sentences("") == []
    assert U.transliterate_to_urdu("") == ""
    assert U.transliterate_to_roman("") == ""
    assert U.roman_key("") == ""  # was ء: the empty key is every vowel-only word's key
    assert U.tag_roman_tokens("") == []
    assert U.find_numbers("") == []
    assert U.identify_language("").language is None
    assert U.stem("") == ""


def test_a_word_already_in_urdu_is_its_own_roman_key():
    assert U.roman_key("نہیں") == "نہیں"


# --- Scale ----------------------------------------------------------------------


def test_transliteration_is_linear_in_the_length_of_a_run():
    """The decoder carried each path in every state and copied it at every word:
    50,000 words with no full stop took 36 s. Back-pointers made it linear."""
    words = ["mera", "naam", "ali", "hai", "aur", "main", "lahore", "mein", "rehta", "hoon"]
    short = " ".join(words * 200)  # 2,000 words
    long = " ".join(words * 2000)  # 20,000 words
    from urdunlp import translit

    U.transliterate_to_urdu(short)  # loads the model and warms the per-word caches
    translit._decode.cache_clear()  # a sentence seen before is not decoded again
    start = time.perf_counter()
    U.transliterate_to_urdu(short)
    t_short = time.perf_counter() - start
    translit._decode.cache_clear()
    start = time.perf_counter()
    U.transliterate_to_urdu(long)
    t_long = time.perf_counter() - start
    assert t_long < 30 * max(t_short, 0.01)  # 10x the input; quadratic would be ~100x


def test_a_long_run_of_number_words_is_bounded():
    """Every start was re-evaluated at every length: 5,000 number words took 60 s."""
    text = " ".join(["ایک", "سو", "ہزار", "لاکھ"] * 1250)
    start = time.perf_counter()
    U.find_numbers(text)
    assert time.perf_counter() - start < 10


# --- format_number ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (1e-05, "0.00001"),
        (1.5e20, "15,00,00,00,00,00,00,00,00,000"),
        (0.1, "0.1"),
        (-0.0, "0"),
        (22_500_000.0, "2,25,00,000"),
        (-1_234_567, "-12,34,567"),
    ],
)
def test_format_number_handles_every_float(value, text):
    """It split repr(value) on the dot, so 1e-05 - which Python prints in exponent
    form - raised IndexError."""
    assert U.format_number(value) == text


@pytest.mark.parametrize("bad", [True, "5", None])
def test_format_number_rejects_what_is_not_a_number(bad):
    with pytest.raises(TypeError, match="expects an int or float"):
        U.format_number(bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_format_number_refuses_nan_and_infinity(bad):
    with pytest.raises(ValueError, match="cannot group"):
        U.format_number(bad)


# --- Loading --------------------------------------------------------------------


def _run_fresh(script: str, timeout: int) -> None:
    """Run a script in a new interpreter that imports *this* urdunlp.

    The package's parent directory goes on PYTHONPATH, so it works from a fresh
    clone with nothing installed (src/) as well as from an installed wheel.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    env = dict(os.environ, PYTHONPATH=str(Path(U.__file__).resolve().parent.parent))
    subprocess.run([sys.executable, "-c", script], check=True, timeout=timeout, env=env)


def test_threads_racing_through_a_cold_start_agree_with_one_thread():
    """The models load lazily behind caches; a web server's first requests arrive
    together. Run in a fresh interpreter so the models really are cold."""
    script = """
import urdunlp as U
from concurrent.futures import ThreadPoolExecutor
inputs = ["us ne kaha ke woh kal aayega", "dedh lakh rupay", "یہ کتاب میری ہے"] * 20
def work(s):
    return (U.transliterate_to_urdu(s), U.identify_language(s).language,
            tuple(U.tag_roman_tokens(s)), tuple(x.value for x in U.find_numbers(s)))
with ThreadPoolExecutor(16) as pool:
    together = list(pool.map(work, inputs))
assert together == [work(s) for s in inputs]
"""
    _run_fresh(script, timeout=300)


def test_importing_the_package_loads_no_model():
    """3 MB of tables must wait for the first call that needs them."""
    script = """
import urdunlp
from urdunlp import _channel, langid
assert _channel.channel.cache_info().currsize == 0
assert langid._script_model.cache_info().currsize == 0
assert langid._tagger.cache_info().currsize == 0
"""
    _run_fresh(script, timeout=120)
