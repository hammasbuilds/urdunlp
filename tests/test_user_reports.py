"""What a user-view audit of 0.2 found, each pinned so it stays fixed.

The audit installed the package and used it the way a newcomer would - chat Roman
Urdu, emoji, emails, one-word language checks, the wrong argument type - and wrote
down every answer that would make someone file an issue.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

import urdunlp as U
from urdunlp import _channel
from urdunlp.cli import main

# --- Urdu -> Roman keeps what it cannot spell ---------------------------------------


@pytest.mark.parametrize("method", ["learned", "rules"])
@pytest.mark.parametrize("kept", ["😀", "★", "✓", "©", "™", "₨", "👨‍👩‍👧", "1️⃣", "मैं"])
def test_emoji_symbols_and_other_scripts_survive(method, kept):
    """`میں خوش ہوں 😀` came out `main khush hoon ` - every character without a
    Roman form was deleted, in both methods."""
    out = U.transliterate_to_roman(f"میں خوش ہوں {kept}", method=method)
    assert out.endswith(f" {kept}")


def test_the_audit_examples_exactly():
    assert U.transliterate_to_roman("میں خوش ہوں 😀") == "main khush hoon 😀"
    assert U.transliterate_to_roman("ٹیسٹ ★ ✓ © ™") == "test ★ ✓ © ™"


def test_marks_that_belong_to_an_urdu_word_are_still_dropped():
    # ZWNJ inside a compound and a diacritic have no Roman spelling and are not text.
    out = U.transliterate_to_roman("کتاب‌یں")
    assert "‌" not in out and out.startswith("kitab")
    assert "ٔ" not in U.transliterate_to_roman("شاعرٔ", method="rules")


def test_arabic_percent_sign_is_converted():
    # qimat and qeemat were written equally often; the tie now goes to the spelling
    # the letter model rates higher, not to the alphabetically first
    assert U.transliterate_to_roman("قیمت ۵۰٪") == "qimat 50%"


# --- Roman -> Urdu punctuation ------------------------------------------------------


@pytest.mark.parametrize(
    ("roman", "urdu"),
    [
        ("ye kitab hai?", "یہ کتاب ہے؟"),
        ("haan, theek hai.", "ہاں، ٹھیک ہے۔"),
        ('"sach hai."', '"سچ ہے۔"'),
        ("kya? nahi!", "کیا؟ نہیں!"),
    ],
)
def test_latin_punctuation_after_a_word_is_written_the_urdu_way(roman, urdu):
    """`ye kitab hai?` gave یہ کتاب ہے? - an Urdu sentence with a Latin question mark."""
    assert U.transliterate_to_urdu(roman) == urdu


@pytest.mark.parametrize(
    ("roman", "kept"),
    [
        ("price 2.5 crore hai", "2.5"),
        ("nahi.... pata", "...."),
        ("sab theek hai :)", ":)"),
        ("acha ;)", ";)"),
        ("dekho http://x.co.", "http://x.co."),
        ("mail test@x.com, phir", "test@x.com,"),
    ],
)
def test_punctuation_that_is_not_ending_a_word_is_left_alone(roman, kept):
    assert kept in U.transliterate_to_urdu(roman)


def test_urdu_punctuation_can_be_switched_off():
    assert U.transliterate_to_urdu("ye kitab hai?", urdu_punctuation=False) == "یہ کتاب ہے?"
    result = U.transliterate_with_confidence("hai?")
    assert result.sources[-1] == ("?", "punctuation")


def test_a_comma_still_does_not_end_the_sentence_context():
    """Converting `,` to ، must not change what it means to the decoder."""
    with_comma = U.transliterate_to_urdu("us ne kaha, ke woh aayega")
    assert with_comma.startswith("اس نے کہا، کہ")


# --- Chat Roman Urdu ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("roman", "urdu"),
    [
        ("meri gaari", "میری گاڑی"),  # was میری غار - Dakshina once spelled غار `gaari`
        ("meri gaari kharab hai", "میری گاڑی خراب ہے"),  # was گاری
        ("kya hal h", "کیا حال ہے"),  # `h` for ہے
        ("kse ho", "کیسے ہو"),
        ("mujhe pata hai", "مجھے پتہ ہے"),  # was پاتا (finds)
        ("kal, parso aur phir", "کل، پرسوں اور پھر"),  # was پرشو in any sentence
    ],
)
def test_chat_spellings(roman, urdu):
    assert U.transliterate_to_urdu(roman) == urdu


@pytest.mark.parametrize(
    ("roman", "urdu"),
    [
        ("main karunga", "میں کروں گا"),  # was کرؤنگ
        ("hum dekhenge", "ہم دیکھیں گے"),  # fell through to the rules
        ("ye milega", "یہ ملے گا"),  # the stem alone is `mile` -> مائل
        ("woh ayega", "وہ آئے گا"),  # was ئیگا
        ("kitne paise lagenge", "کتنے پیسے لگیں گے"),
        ("tum karogi", "تم کرو گی"),
    ],
)
def test_the_merged_future_is_written_as_two_words(roman, urdu):
    assert U.transliterate_to_urdu(roman) == urdu


@pytest.mark.parametrize("word", ["karunga", "dekhenge"])
def test_a_merged_future_on_its_own(word):
    """One word takes the word-by-word path, which must split it too."""
    assert " " in U.transliterate_to_urdu(word)
    assert " " in U.roman_key(word)


@pytest.mark.parametrize(
    ("roman", "urdu"),
    [("challenge", "چیلنج"), ("jungi", "جنگی"), ("omega", "اومیگا")],
)
def test_words_that_only_look_like_a_future_are_not_split(roman, urdu):
    """English words (challenge, omega) and words annotators spelled that way
    (jungi جنگی) keep the ordinary lookup."""
    assert U.transliterate_to_urdu(roman) == urdu


# --- Tokens: one idea of what a URL, an email and a number are ----------------------


def test_words_keeps_identifiers_and_numbers_whole():
    """words("... test@x.com") gave test, x, com while the transliterator kept it."""
    assert U.words("رابطہ: test@x.com یا 0300-1234567") == [
        "رابطہ",
        "test@x.com",
        "یا",
        "0300-1234567",
    ]
    assert U.words("قیمت 2.5 کروڑ، ۱۲٫۵ http://x.co/a. @ali #ٹیگ") == [
        "قیمت",
        "2.5",
        "کروڑ",
        "۱۲٫۵",
        "http://x.co/a",
        "@ali",
        "#ٹیگ",
    ]


def test_words_with_punctuation_puts_a_url_full_stop_back():
    assert U.words("دیکھو http://x.co.", keep_punctuation=True) == ["دیکھو", "http://x.co", "."]


@pytest.mark.parametrize(
    ("text", "clean"),
    [
        ("mail test@x.com ya @ali", "mail ya"),  # was `mail test .com ya`
        ("dekho http://x.co. phir", "dekho. phir"),
        ("a\nhttp://x.co\nb", "a\n\nb"),
    ],
)
def test_remove_urls_and_mentions_removes_emails_whole(text, clean):
    assert U.remove_urls_and_mentions(text) == clean


# --- Errors that name the call ------------------------------------------------------


@pytest.mark.parametrize("function", [U.remove_stopwords, U.stem_tokens, U.group_roman_variants])
def test_bytes_are_not_a_list_of_words(function):
    """remove_stopwords(b'x') said "is_stopword() expects a str, got int"."""
    with pytest.raises(TypeError, match=rf"^{function.__name__}\(\) .* got bytes"):
        function(b"x")


@pytest.mark.parametrize("function", [U.remove_stopwords, U.stem_tokens, U.group_roman_variants])
def test_a_bad_item_is_reported_by_position(function):
    with pytest.raises(TypeError, match=rf"^{function.__name__}\(\) .* item 1 is NoneType"):
        function(["کتاب", None])


def test_a_generator_of_words_is_accepted():
    assert U.remove_stopwords(w for w in ["یہ", "اچھا", "نہیں"]) == ["اچھا", "نہیں"]


# --- Language identification on a word --------------------------------------------


def test_a_one_word_guess_is_flagged_short():
    """identify_language on one word answered as confidently as on a paragraph."""
    assert U.identify_language("کتاب").short
    assert U.identify_language("دا کتاب زما دی").short
    assert not U.identify_language("یہ کتاب میری ہے اور میں اسے پڑھتا ہوں").short
    assert U.identify_language("hello").short


# --- The faster key must be the same key ------------------------------------------


@pytest.mark.parametrize("word", ["اللہ", "محمد", "گاﺅں", "مثلاﹰ", "Pکتاب", "", "ا", "پاکستان"])
def test_urdu_key_matches_the_letter_by_letter_definition(word):
    out: list[str] = []
    for ch in word:
        cls = _channel._URDU_CLASS.get(ch)
        if cls and (not out or out[-1] != cls):
            out.append(cls)
    assert _channel.urdu_key(word) == "".join(out)


# --- The command line ---------------------------------------------------------------


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exit_:
        main(["--version"])
    assert exit_.value.code == 0
    assert capsys.readouterr().out.strip() == f"urdunlp {U.__version__}"


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["to-urdu", "mera", "naam", "ali", "hai"], "میرا نام علی ہے"),
        (["to-roman", "میں ٹھیک ہوں"], "main theek hoon"),
        (["normalize", "كتاب"], "کتاب"),
        (["words", "کیا، واقعی؟"], "کیا | واقعی"),
        (["langid", "هي ڪتاب منهنجو آهي"], "sd\tSindhi"),
    ],
)
def test_cli_commands(capsys, argv, expected):
    assert main(argv) == 0
    assert capsys.readouterr().out.startswith(expected)


def test_cli_reads_standard_input_line_by_line(monkeypatch, capsys):
    import io

    monkeypatch.setattr(sys, "stdin", io.StringIO("kal milte hain\nshukriya\n"))
    assert main(["to-urdu"]) == 0
    assert capsys.readouterr().out.splitlines() == ["کل ملتے ہیں", "شکریہ"]


def test_python_dash_m_runs_the_cli():
    import os
    from pathlib import Path

    env = dict(os.environ, PYTHONPATH=str(Path(U.__file__).resolve().parent.parent))
    done = subprocess.run(
        [sys.executable, "-m", "urdunlp", "to-urdu", "ye kitab hai?"],
        capture_output=True,
        env=env,
        timeout=120,
        check=True,
    )
    assert done.stdout.decode("utf-8").strip() == "یہ کتاب ہے؟"


# --- Chat in capitals ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("roman", "urdu"),
    [
        ("KYA HAAL HAI", "کیا حال ہے"),  # KYA was spelled کے وائی اے
        ("main NHI jaunga", "میں نہیں جاؤں گا"),  # a lexicon word is never an acronym
        ("PLZ CALL KRO", "پلیز کال کرو"),
        ("KAL MEETING HAI", "کل میٹنگ ہے"),
    ],
)
def test_shouting_is_not_a_row_of_acronyms(roman, urdu):
    assert U.transliterate_to_urdu(roman) == urdu


@pytest.mark.parametrize(
    ("roman", "urdu"),
    [
        ("BBC TV", "بی بی سی ٹی وی"),
        ("FBI ne kaha", "ایف بی آئی نے کہا"),
        ("C. M. Ali", "سی ایم علی"),
    ],
)
def test_real_acronyms_are_still_spelled(roman, urdu):
    assert U.transliterate_to_urdu(roman) == urdu


def test_the_version_is_the_same_in_every_place_it_is_declared() -> None:
    """__version__, pyproject.toml and the installed metadata must agree.

    The version is written twice - here and in pyproject.toml - and the other tests
    only check that `--version` prints `__version__`, which is true however wrong
    both are. Bump pyproject alone and the wheel says urdunlp {new} while
    `urdunlp --version` says the old one; release.yml compares the tag to
    pyproject, so nothing would have caught it.
    """
    from importlib.metadata import version

    assert version("urdunlp") == U.__version__

    # pyproject.toml is absent wherever only tests/ is shipped, as in the
    # installed-wheel CI job.
    pyproject = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
    if pyproject.exists():
        import tomllib

        declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        assert declared["project"]["version"] == U.__version__
