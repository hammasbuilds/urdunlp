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


def test_an_error_does_not_echo_a_huge_input_whole():
    with pytest.raises(ValueError) as caught:
        U.parse_number("x" * 100_000)
    message = str(caught.value)
    assert len(message) < 200 and "100,000 characters" in message
    with pytest.raises(ValueError, match="not an ordinal: 'kuch'$"):
        U.parse_ordinal("kuch")


def test_the_quick_check_sample_aligns_as_the_readme_says():
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    if not (root / "scripts/measure_translit.py").exists() or not (root / "eval").exists():
        # The installed-wheel CI job copies only tests/; the sample and the aligner
        # live in the source tree.
        pytest.skip("scripts/ and eval/ are not next to the tests (installed-wheel run)")
    sys.path.insert(0, str(root / "scripts"))
    try:
        from measure_translit import align
    finally:
        sys.path.remove(str(root / "scripts"))
    rows = (root / "eval/dakshina_test_sample.tsv").read_text(encoding="utf-8").splitlines()
    pairs = [align(*row.split("\t")) for row in rows]
    assert len(rows) == 364 and all(p is not None for p in pairs)
    assert sum(len(p) for p in pairs) == 5249


def test_identifier_pattern_is_linear_on_one_long_token() -> None:
    """A long whitespace-free run must not make the identifier pattern backtrack.

    The email branch used to attempt its local part at every interior position of such a
    run, which is quadratic: a single 40 KB base64 `data:` URI took 12.9 seconds, against
    a docstring that advertises scraped text as the use case. The bound here is generous
    - the fixed pattern does 40 KB in about a millisecond - so this fails only if the
    quadratic behaviour comes back, not when CI is slow.
    """
    from urdunlp.normalize import _IDENTIFIER

    blob = "<img src=data:image/png;base64," + "QUJD" * 10_000 + ">"
    start = time.perf_counter()
    _IDENTIFIER.findall(blob)
    assert time.perf_counter() - start < 1.0

    start = time.perf_counter()
    U.remove_urls_and_mentions("a" * 40_000)
    assert time.perf_counter() - start < 1.0


def test_identifier_still_matches_real_addresses() -> None:
    """The speed fix must not narrow what counts as an email, mention or URL."""
    from urdunlp.normalize import _IDENTIFIER

    for text, expected in [
        ("mail me at test@x.com please", ["test@x.com"]),
        ("a.b+c-d@sub.domain.co.uk", ["a.b+c-d@sub.domain.co.uk"]),
        ("x@y.z", ["x@y.z"]),
        ("two a@b.com and c@d.org", ["a@b.com", "c@d.org"]),
        ("trailing test@x.com.", ["test@x.com"]),
        ("UPPER.Case+tag@Example.COM", ["UPPER.Case+tag@Example.COM"]),
        ("no-at-sign here", []),
    ]:
        assert _IDENTIFIER.findall(text) == expected, text


# The first version of these tests built a 40 KB base64 blob and then asserted only on
# the private _IDENTIFIER regex and remove_urls_and_mentions. It never routed that blob
# through the two functions that actually broke on it, so a commit named "fix the
# quadratic identifier regex" shipped with two more super-linear paths and an underflow
# still open. These tests call the public API, with the shapes a user really supplies.

PATHOLOGICAL = {
    "one long whitespace run": " " * 200_000,
    "tabs": "\t" * 40_000,
    "non-breaking spaces": " " * 40_000,
    "one long Urdu token": "ک" * 20_000,
    "merged Urdu, no spaces": "کردیا" * 2_000,
    "base64 data URI": "<img src=data:image/png;base64," + "QUJD" * 10_000 + ">",
    "one long Latin token": "a" * 40_000,
    "hyphenated run": "a-e-" * 10_000,
    "base64 token in a sentence": "mera naam ali hai " + "YWJh" * 500,
    # A digit run and an unspaced comma-number list were missing, and that is exactly
    # how CPython's 4,300-digit int_max_str_digits ValueError escaped find_numbers,
    # parse_number and parse_ordinal: no shape here had enough digits to reach it.
    # One CSV row pasted into text is enough.
    "long ASCII digit run": "9" * 20_000,
    "long Urdu digit run": "۹" * 20_000,
    "unspaced comma number list": "qeemat: " + ",".join(str(i) for i in range(1, 3000)),
    "digits in a sentence": "mujhe " + "9" * 9_000 + " rupay chahiye",
}

# Every public function that takes a single string. The first version of this list held
# 11 of them and left out the number parsers, which is why the shapes above could not
# have caught the digit-limit defect even after they existed. A list of "the functions I
# thought were risky" is not coverage of the public surface.
LINEAR_PUBLIC_FUNCTIONS = [
    "normalize",
    "words",
    "sentences",
    "roman_key",
    "transliterate_to_urdu",
    "transliterate_to_roman",
    "transliterate_with_confidence",
    "identify_language",
    "tag_roman_tokens",
    "remove_urls_and_mentions",
    "resolve_arabic_heh",
    "is_urdu",
    "fix_spacing",
    "stem",
    "find_numbers",
    "parse_number",
    "parse_ordinal",
]

# These two raise a documented ValueError on text that is not a number. That is correct
# behaviour, and distinct from the raw-internals leak the no-raise test looks for.
MAY_RAISE_VALUE_ERROR = {"parse_number", "parse_ordinal"}


@pytest.mark.parametrize("name", LINEAR_PUBLIC_FUNCTIONS)
@pytest.mark.parametrize("shape", sorted(PATHOLOGICAL))
def test_public_functions_stay_fast_on_pathological_input(name: str, shape: str) -> None:
    """No public function may take super-linear time on one unbroken token or run.

    The bound is deliberately loose. Every one of these completes in milliseconds once
    the models are warm; the defects this replaces took 153 seconds on 200k spaces, 28
    minutes on a 5,000-character Urdu token, and 12.9 seconds on one scraped page. A
    slow CI machine will not trip this, a returning quadratic will.
    """
    import time

    function = getattr(U, name)
    text = PATHOLOGICAL[shape]

    # The number parsers reject non-numeric text by design, and a rejection still pays
    # the model load and still has to be fast - refusing slowly is the defect this test
    # exists to catch, so they are timed like everything else rather than skipped.
    def call(value: str) -> None:
        try:
            function(value)
        except ValueError:
            if name not in MAY_RAISE_VALUE_ERROR:
                raise

    call("warm")  # pay the model load outside the clock
    start = time.perf_counter()
    call(text)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"{name} took {elapsed:.1f}s on {shape} ({len(text)} chars)"


@pytest.mark.parametrize("shape", sorted(PATHOLOGICAL))
def test_no_public_function_raises_on_pathological_input(shape: str) -> None:
    """Pathological input may be useless to process, but it must not raise.

    tag_roman_tokens used to raise ValueError("expected a positive input, got 0.0") from
    math.log(0.0) once exp() of the character log-probability underflowed, which a
    136-character base64 token was enough to cause - including in the middle of an
    ordinary sentence.
    """
    text = PATHOLOGICAL[shape]
    for name in LINEAR_PUBLIC_FUNCTIONS:
        try:
            getattr(U, name)(text)
        except ValueError as exc:
            if name in MAY_RAISE_VALUE_ERROR:
                # A documented "not a number word" is correct behaviour. What is not is
                # CPython's own int_max_str_digits error, which escaped these functions
                # telling the caller to go and raise the interpreter's digit limit.
                assert "int_max_str_digits" not in str(exc), (
                    f"{name} leaked a CPython internal error on {shape}: {exc}"
                )
                continue
            raise AssertionError(f"{name} raised ValueError on {shape}: {exc}") from None
        except Exception as exc:  # noqa: BLE001 - that is the assertion
            raise AssertionError(f"{name} raised {type(exc).__name__} on {shape}: {exc}") from None
