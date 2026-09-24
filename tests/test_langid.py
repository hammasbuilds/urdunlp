"""Script is not language: eleven languages share Urdu's script."""

from __future__ import annotations

import pytest

from urdunlp import LANGUAGES, identify_language, is_urdu, tag_roman_tokens

# One ordinary sentence per language, written for these tests rather than taken
# from the training sample, so the model has not seen them.
SENTENCES = {
    "ur": "یہ کتاب میری ہے اور میں اسے روز شام کو پڑھتا ہوں۔",
    "ar": "هذا الكتاب لي وأنا أقرأه كل مساء في المكتبة.",
    "fa": "این کتاب مال من است و من هر شب آن را می‌خوانم.",
    "sd": "هي ڪتاب منهنجو آهي ۽ آئون هر شام ان کي پڙهان ٿو.",
    "ps": "دا کتاب زما دی او زه یې هره ماښام لولم.",
    "ug": "بۇ كىتاب مېنىڭ، مەن ئۇنى ھەر كەچتە ئوقۇيمەن.",
    "ckb": "ئەم کتێبە هی منە و هەموو ئێوارەیەک دەیخوێنمەوە.",
}


@pytest.mark.parametrize("code", sorted(SENTENCES))
def test_each_language_is_recognised(code):
    assert identify_language(SENTENCES[code]).language == code


def test_the_script_check_accepts_every_one_of_them():
    """The reason identify_language exists. is_urdu checks the script, and on the
    Wikipedia samples it said yes to 99.5% of Arabic and 99.9% of Persian."""
    assert all(is_urdu(text) for text in SENTENCES.values())


def test_distinctive_letters_are_reported_as_evidence():
    """ڪ is Sindhi's kaf; no other language of the eleven uses it."""
    guess = identify_language(SENTENCES["sd"])
    assert "ڪ" in guess.evidence.get("sd", [])


def test_runs_on_raw_text_not_normalised_text():
    """normalize() rewrites Arabic ي and ك as Urdu letters - right for Urdu, and
    exactly the evidence that says a text is Arabic. identify_language must see
    the text as written."""
    from urdunlp import normalize

    arabic = SENTENCES["ar"]
    assert normalize(arabic) != arabic
    assert identify_language(arabic).language == "ar"


def test_latin_digits_and_punctuation_are_ignored():
    guess = identify_language("BBC 2024: " + SENTENCES["ur"] + " !!!")
    assert guess.language == "ur"


def test_no_perso_arabic_letters_means_no_answer():
    guess = identify_language("hello world 123")
    assert guess.language is None and guess.name is None


def test_every_code_has_a_name():
    guess = identify_language(SENTENCES["ps"])
    assert guess.name == LANGUAGES["ps"] == "Pashto"


def test_ranking_covers_all_eleven():
    assert len(identify_language(SENTENCES["ur"]).ranking) == len(LANGUAGES) == 11


class TestRomanTagger:
    def test_code_mixed_sentence(self):
        assert tag_roman_tokens("kal meeting cancel ho gayi") == [
            ("kal", "ur"),
            ("meeting", "en"),
            ("cancel", "en"),
            ("ho", "ur"),
            ("gayi", "ur"),
        ]

    def test_pure_roman_urdu_stays_urdu(self):
        tags = [t for _, t in tag_roman_tokens("mujhe aaj bohat kaam hai lekin main kal aunga")]
        assert tags == ["ur"] * len(tags)

    def test_pure_english_stays_english(self):
        tags = [t for _, t in tag_roman_tokens("the committee will publish its report next week")]
        assert tags == ["en"] * len(tags)

    def test_words_spelled_the_same_in_both_are_decided_by_context(self):
        """`is` is اس in Roman Urdu and a verb in English. Only the words around it
        can say which."""
        assert dict(tag_roman_tokens("is ke baad woh ghar gaya"))["is"] == "ur"
        assert dict(tag_roman_tokens("this is a very good report"))["is"] == "en"

    def test_non_latin_is_skipped(self):
        assert tag_roman_tokens("یہ 123 !") == []
