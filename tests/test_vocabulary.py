"""The vocabulary stage: recovering letters Roman cannot write, from words that exist."""

from __future__ import annotations

import pytest

from urdunlp import (
    group_roman_variants,
    roman_key,
    transliterate_to_roman,
    transliterate_to_urdu,
    transliterate_with_confidence,
)
from urdunlp._channel import roman_keys, urdu_key


@pytest.mark.parametrize(
    ("roman", "urdu"),
    [
        ("baad", "بعد"),  # ع is unwritable in Roman
        ("taur", "طور"),  # ط and ت are both `t`
        ("haasil", "حاصل"),  # ح and ص
        ("mohammad", "محمد"),  # doubled m, unwritten vowels
        ("kuch", "کچھ"),  # aspiration nobody typed
        ("station", "اسٹیشن"),  # English -tion, and the initial alef Urdu adds
    ],
)
def test_words_the_rules_could_never_spell(roman, urdu):
    result = transliterate_with_confidence(roman)
    assert result.text == urdu
    assert result.sources[0][1] == "vocabulary"


def test_the_rules_alone_get_them_wrong():
    """The control: without the vocabulary these all fail, which is the point."""
    assert transliterate_to_urdu("baad", use_vocabulary=False) != "بعد"
    assert transliterate_to_urdu("taur", use_vocabulary=False) != "طور"


def test_the_curated_lexicon_still_comes_first():
    result = transliterate_with_confidence("main theek hoon")
    assert [s for _, s in result.sources] == ["lexicon"] * 3


def test_candidate_keys_line_up_across_scripts():
    """The retrieval key must be the same for a word and its romanisations."""
    assert urdu_key("بعد") in roman_keys("baad")
    assert urdu_key("اسٹیشن") in roman_keys("station")
    assert urdu_key("نہیں") in roman_keys("nahin")  # final ں is not a consonant


def test_rule_share_counts_only_the_guesses():
    result = transliterate_with_confidence("mera naam Xqzvt hai")
    assert result.rule_share == 0.25


class TestContext:
    def test_the_sentence_decides_between_homographs(self):
        """`ke` is کے (of) or کہ (that). After کہا (said) it is almost always کہ -
        which a word-by-word transliterator, taking the curated lexicon's کے, gets
        wrong every time. It was the largest single error before context."""
        text = "us ne kaha ke woh kal aayega"
        assert "کہا کہ" in transliterate_to_urdu(text)
        assert "کہا کے" in transliterate_to_urdu(text, use_context=False)

    def test_ki_can_be_that_too(self):
        """`ki` is کی (of) and, typed by many writers, کہ (that). The training lexicon
        barely attests it for کہ, so it is offered explicitly and context decides."""
        assert "کہا کہ" in transliterate_to_urdu("us ne kaha ki woh aayega")
        assert transliterate_to_urdu("is ki kitab") == "اس کی کتاب"

    def test_the_same_word_keeps_its_ordinary_reading_elsewhere(self):
        assert transliterate_to_urdu("is ke baad") == "اس کے بعد"

    def test_a_sentence_end_resets_the_context(self):
        """A full stop ends the sentence the next word is chosen in."""
        result = transliterate_with_confidence("kuch nahi hua. us ke ghar gaye")
        assert result.text == "کچھ نہیں ہوا۔ اس کے گھر گئے"

    def test_attested_spellings_reach_words_the_letter_model_misses(self):
        """The letter model gave `ke` as کہ a log-probability of -6.6; annotators
        wrote exactly that spelling for کہ, and the word's own spellings now count."""
        from urdunlp._channel import channel

        assert "کہ" in [w for w, _ in channel().candidates("ke")]


class TestRomanKey:
    def test_spelling_variants_share_a_key(self):
        assert roman_key("nahi") == roman_key("nahin") == roman_key("nhi") == "نہیں"

    def test_variants_the_curated_lexicon_does_not_list(self):
        """`naheen` and `nahin` are in the lexicon; `nahee` is not. The vocabulary
        search has to find it."""
        assert roman_key("nahee") == "نہیں"

    def test_different_words_keep_different_keys(self):
        assert roman_key("kitab") != roman_key("kutta")

    def test_grouping(self):
        groups = group_roman_variants(["nahi", "acha", "nhi", "accha", "nahi"])
        assert groups == {"نہیں": ["nahi", "nhi"], "اچھا": ["acha", "accha"]}

    def test_key_is_case_insensitive(self):
        assert roman_key("Nahi") == roman_key("nahi")


class TestKeepEnglish:
    def test_off_by_default_because_urdu_writes_loanwords_in_urdu(self):
        assert transliterate_to_urdu("station") == "اسٹیشن"

    def test_on_it_leaves_english_in_latin_script(self):
        result = transliterate_with_confidence("kal meeting cancel ho gayi", keep_english=True)
        assert result.text == "کل meeting cancel ہو گئی"
        assert dict(result.sources)["meeting"] == "english"

    def test_a_url_does_not_shift_the_tags_onto_the_wrong_words(self):
        """The first version tagged the raw text, where `http://x.co` is three
        Latin words; the transliterator treats it as one identifier. Every tag after
        the URL landed three words late - here, on `ho` and `gayi`."""
        result = transliterate_with_confidence(
            "dekho http://x.co kal meeting cancel ho gayi", keep_english=True
        )
        kinds = dict(result.sources)
        assert kinds["http://x.co"] == "identifier"
        assert kinds["meeting"] == kinds["cancel"] == "english"
        assert kinds["ho"] == kinds["gayi"] == "lexicon"


class TestLearnedRoman:
    @pytest.mark.parametrize(
        ("urdu", "roman"),
        [
            ("میں ٹھیک ہوں", "main theek hoon"),  # the rules gave `min thik hon`
            ("میرا نام علی ہے", "mera naam ali hai"),
            ("کتاب", "kitab"),  # the rules gave `katab`
            ("پاکستان", "pakistan"),  # the rules gave `pakasatan`
        ],
    )
    def test_words_come_out_the_way_people_write_them(self, urdu, roman):
        assert transliterate_to_roman(urdu) == roman

    def test_urdu_digits_are_kept(self):
        """The 0.1 mapping kept only ASCII it did not recognise, so ۱۲۳ vanished."""
        assert "123" in transliterate_to_roman("قیمت ۱۲۳ روپے")
        assert "123" in transliterate_to_roman("قیمت ۱۲۳ روپے", method="rules")

    def test_whitespace_is_kept(self):
        out = transliterate_to_roman("یہ  کتاب\nمیری ہے 😀")
        assert "  " in out and "\n" in out
        assert out.replace("😀", "").isascii()

    def test_learned_spellings_convert_back(self):
        """Readable and reversible stopped being a trade-off: on held-out tokens the
        learned spelling round-trips more often than the rules' literal mapping."""
        for word in ("کتاب", "پاکستان", "مشکل", "صرف", "حسن", "بیماریاں"):
            assert transliterate_to_urdu(transliterate_to_roman(word)) == word

    def test_an_unknown_method_is_refused(self):
        with pytest.raises(ValueError, match="method must be"):
            transliterate_to_roman("کتاب", method="phonetic")


class TestArabicArticle:
    """Roman writes the article on the word before it; Urdu on the word after."""

    @pytest.mark.parametrize(
        ("roman", "urdu"),
        [
            ("abdul rehman", "عبد الرحمن"),
            ("bainul aqwami", "بین الاقوامی"),
            ("darul uloom", "دار العلوم"),
        ],
    )
    def test_the_article_moves_to_the_next_word(self, roman, urdu):
        assert transliterate_to_urdu(roman) == urdu

    @pytest.mark.parametrize(
        ("roman", "urdu"),
        [("kabul shehar", "کابل شہر"), ("rasul allah", "رسول اللہ"), ("phool bagh", "پھول باغ")],
    )
    def test_words_that_merely_end_in_ul_are_left_alone(self, roman, urdu):
        assert transliterate_to_urdu(roman) == urdu


class TestInitialsAndTitles:
    def test_a_capital_letter_is_an_initial(self):
        assert transliterate_to_urdu("C. M. Naim").startswith("سی ایم ")

    def test_lowercase_o_is_the_conjunction(self):
        assert transliterate_to_urdu("zabt o nazm") == "ضبط و نظم"

    @pytest.mark.parametrize(
        ("roman", "urdu"),
        [("Dr. Abdul Qadeer Khan", "ڈاکٹر عبد القدیر خان"), ("Mr. Ali", "مسٹر علی")],
    )
    def test_titles_are_written_in_full_without_the_dot(self, roman, urdu):
        """Mr. Ali was میر علی - Mir Ali is a common name - until a title followed
        by its dot stopped being left to the decoder."""
        assert transliterate_to_urdu(roman) == urdu

    def test_an_abbreviation_dot_does_not_end_the_sentence(self):
        """The dot of Dr. used to reset the context, cutting it mid-name."""
        assert "کہا کہ" in transliterate_to_urdu("Dr. Khan ne kaha ke woh aayenge")
