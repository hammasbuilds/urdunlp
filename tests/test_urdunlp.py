"""Tests for urdu-nlp-toolkit.

Each one encodes a real property of Urdu text rather than a convenient example, so a
failure means the library is wrong about the language, not about a fixture.
"""

from __future__ import annotations

import pytest

from urdunlp import (
    character_ngrams,
    fix_spacing,
    is_stopword,
    is_urdu,
    normalize,
    remove_stopwords,
    remove_urls_and_mentions,
    sentences,
    transliterate_to_roman,
    transliterate_to_urdu,
    transliterate_with_confidence,
    words,
)


class TestNormalize:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("كتاب", "کتاب"),  # ARABIC KAF -> KEHEH
            ("يہ", "یہ"),  # ARABIC YEH -> FARSI YEH
            ("کِتاب", "کتاب"),  # diacritics dropped
            ("کـــتاب", "کتاب"),  # tatweel dropped
            ("أحمد", "احمد"),  # hamza above alef
        ],
    )
    def test_variants_collapse_to_one_form(self, raw, expected):
        assert normalize(raw) == expected

    def test_arabic_and_urdu_spellings_become_equal(self):
        """The whole point: the same word must compare equal after normalisation."""
        assert normalize("كتاب") == normalize("کتاب")

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("بهی", "بھی"),  # bh - aspiration
            ("تها", "تھا"),  # th
            ("کچه", "کچھ"),  # ch
            ("مجهے", "مجھے"),  # jh
            ("گهر", "گھر"),  # gh
            ("سنده", "سندھ"),  # dh, word-final
            ("نه", "نہ"),  # ن does not aspirate -> ordinary h
            ("الله", "اللہ"),  # ل does not aspirate -> ordinary h
        ],
    )
    def test_stray_arabic_heh_resolves_by_context(self, raw, expected):
        """ARABIC HEH has no single Urdu counterpart: it stands for ہ (an ordinary
        h) or ھ (aspiration) depending on what precedes it.

        The module docstring used to name ه -> ہ as one of its three headline
        examples, and the mapping table did not contain ه at all - so the case it
        advertised was the one case it did not handle. Mapping it to ہ everywhere,
        as the docstring said, is correct for 9.0% of the corpus occurrences that
        can be adjudicated; resolving by context is correct for 96.9%.

        The reason is mechanical: a keyboard missing ھ is a keyboard missing bh,
        ph, th, kh and gh, so that is where the substitution shows up.
        """
        assert normalize(raw) == expected

    def test_resolving_heh_does_not_invent_aspiration_after_a_plain_consonant(self):
        """ل is excluded from the aspirable set on purpose. Including it would
        spell الله as اللھ, which is not a word."""
        assert "ھ" not in normalize("الله")

    def test_is_urdu_and_the_tokeniser_agree_on_what_counts_as_urdu(self):
        """`is_urdu` checked two Unicode blocks, the tokeniser checked four, so a
        string could be split into Urdu word tokens and reported as not Urdu in the
        same breath.

        Arabic Presentation Forms-B is the block that matters: it is what PDF text
        layers and older systems emit. Rare in edited prose - 231 characters across
        83 of 84,581 corpus articles - but two functions in one toolkit disagreeing
        about what Urdu is, is a defect at any frequency.
        """
        presentation_forms = "ﻛﺗﺎﺏ"  # کتاب, as a PDF might emit it
        assert words(presentation_forms), "the tokeniser treats these as Urdu letters"
        assert is_urdu(presentation_forms), "so is_urdu must not call them non-Urdu"

    def test_idempotent(self):
        once = normalize("كتاب کِتاب")
        assert normalize(once) == once

    def test_urdu_digits_are_kept_by_default(self):
        """Converting them changes how the text reads, not how it is encoded."""
        assert normalize("۱۲۳") == "۱۲۳"
        assert normalize("۱۲۳", normalize_digits=True) == "123"

    def test_urdu_punctuation_is_kept_by_default(self):
        assert normalize("کیا؟") == "کیا؟"
        assert normalize("کیا؟", normalize_punctuation=True) == "کیا?"

    def test_whitespace_collapses(self):
        assert normalize("  یہ   ایک  جملہ ہے  ") == "یہ ایک جملہ ہے"

    def test_empty(self):
        assert normalize("") == ""


class TestIsUrdu:
    def test_urdu(self):
        assert is_urdu("یہ ایک جملہ ہے")

    def test_english(self):
        assert not is_urdu("this is english")

    def test_numbers_do_not_dilute_the_verdict(self):
        """Counting all characters would make a sentence with a year look less Urdu."""
        assert is_urdu("یہ 2026 کا سال ہے")

    def test_empty_is_not_urdu(self):
        assert not is_urdu("")


class TestTokenize:
    def test_splits_on_urdu_full_stop(self):
        assert len(sentences("میرا نام علی ہے۔ آپ کیسے ہیں؟")) == 2

    def test_urdu_punctuation_is_separated_when_asked(self):
        """Urdu punctuation sits inside the Arabic block, so a naive letter range
        swallows it."""
        assert words("کیا، واقعی؟", keep_punctuation=True) == ["کیا", "،", "واقعی", "؟"]

    def test_punctuation_is_dropped_by_default(self):
        assert words("کیا، واقعی؟") == ["کیا", "واقعی"]

    def test_mixed_script(self):
        assert words("Urdu اور English") == ["Urdu", "اور", "English"]

    def test_merged_compound_is_split(self):
        assert fix_spacing("اس نے کام کردیا") == "اس نے کام کر دیا"

    def test_merge_does_not_fire_on_a_substring(self):
        """An aggressive splitter does more damage than an incomplete one."""
        assert fix_spacing("کردیانہ") == "کردیانہ"

    @pytest.mark.parametrize("mark", ["۔", "،", "؟", "!", '"', ")"])
    def test_a_merge_still_fires_next_to_punctuation(self, mark):
        """These are perfective auxiliaries, so the end of a clause is where they
        sit. Splitting the input on whitespace attached the punctuation to the
        token, so "کردیا" was found and "کردیا۔" was not - and on XL-Sum Urdu the
        second form is the more common one.
        """
        assert fix_spacing(f"اس نے کام کردیا{mark}") == f"اس نے کام کر دیا{mark}"

    def test_fix_spacing_leaves_the_rest_of_the_text_alone(self):
        """It used to be `" ".join(text.split())`, which reflowed the whole input:
        newlines, indentation and runs of spaces all collapsed into one space.
        That is a surprising thing for a function that inserts a space to do, and
        it silently destroys paragraph structure in a document pipeline.
        """
        text = "پہلا جملہ۔\n\n    دوسرا  کردیا۔\n"
        assert fix_spacing(text) == "پہلا جملہ۔\n\n    دوسرا  کر دیا۔\n"

    def test_ngrams_are_padded(self):
        assert character_ngrams("کتاب", 3)[0].startswith("<")

    def test_ngram_shorter_than_n(self):
        assert character_ngrams("کا", 5) == ["<کا>"]


class TestStopwords:
    def test_function_words_are_stopwords(self):
        assert is_stopword("کا") and is_stopword("ہے")

    def test_content_words_are_not(self):
        assert not is_stopword("کتاب")

    def test_negation_survives_by_default(self):
        """A stopword list that deletes نہیں inverts every sentiment label."""
        assert "نہیں" in remove_stopwords(words("یہ اچھا نہیں ہے"))

    def test_negation_can_be_removed_explicitly(self):
        assert "نہیں" not in remove_stopwords(words("یہ اچھا نہیں ہے"), include_negation=True)


class TestTransliteration:
    @pytest.mark.parametrize(
        ("roman", "urdu"),
        [
            ("main theek hoon", "میں ٹھیک ہوں"),
            ("aap kaise hain", "آپ کیسے ہیں"),
            ("bohat shukriya", "بہت شکریہ"),
            ("yeh bohat acha kaam hai", "یہ بہت اچھا کام ہے"),
        ],
    )
    def test_common_phrases_use_the_lexicon(self, roman, urdu):
        assert transliterate_to_urdu(roman) == urdu

    def test_spelling_variants_reach_the_same_word(self):
        """Roman Urdu has no standard orthography; the lexicon absorbs the variation."""
        forms = {transliterate_to_urdu(w) for w in ("nahi", "nahin", "nhi", "naheen")}
        assert forms == {"نہیں"}

    def test_unknown_words_fall_back_to_rules(self):
        """A string no Urdu word could have produced is guessed by rule, and says so.

        This test used `Hammas`, which the vocabulary stage added in 0.2 now finds
        (حماس). A made-up word is the stable way to reach the last stage.
        """
        result = transliterate_with_confidence("Xqzvt")
        assert result.sources[0][1] == "rules"
        assert result.rule_share == 1.0
        assert result.text

    def test_coverage_is_reported_honestly(self):
        """Callers must be able to tell a lookup from a guess."""
        assert transliterate_with_confidence("main theek hoon").lexicon_coverage == 1.0
        assert transliterate_with_confidence("Zzzq Xylo").lexicon_coverage == 0.0

    def test_digits_are_not_glued_to_the_previous_word(self):
        assert transliterate_to_urdu("main 25 saal ka hoon") == "میں 25 سال کا ہوں"

    @pytest.mark.parametrize(
        "identifier",
        [
            "http://x.co",
            "https://example.com/a/b?q=1",
            "www.dawn.com",
            "@ali",
            "#lahore",
            "ali@example.com",
        ],
    )
    def test_identifiers_survive_transliteration(self, identifier):
        """A transliterated URL is a broken URL.

        The tokeniser splits `http://x.co` into `http`, `://`, `x`, `co`; three of
        those are alphabetic, so without a guard they are transliterated and the
        result no longer resolves.
        """
        assert transliterate_to_urdu(identifier) == identifier

    def test_identifier_inside_a_sentence_leaves_the_sentence_translated(self):
        """Protecting the URL must not stop the words around it converting."""
        out = transliterate_to_urdu("dekho http://x.co par")
        assert "http://x.co" in out
        assert "پر" in out

    def test_ordinary_english_words_still_transliterate(self):
        """The guard is for identifiers only.

        Roman Urdu is written in English letters, so skipping anything that looks
        English would disable the function. `lahore` is a word, not an identifier.
        """
        assert transliterate_to_urdu("lahore") == "لاہور"

    def test_identifiers_do_not_count_against_lexicon_coverage(self):
        """A URL is not a word the lexicon failed to resolve."""
        assert transliterate_with_confidence("main theek hoon").lexicon_coverage == 1.0
        assert transliterate_with_confidence("main theek hoon http://x.co").lexicon_coverage == 1.0

    def test_aspiration_does_not_take_a_vowel(self):
        """ھ marks aspiration on the letter before it - کھ is one sound, not two."""
        assert transliterate_to_roman("کھانا") == "khana"

    def test_short_vowels_are_inserted_between_consonants(self):
        """Urdu does not write short vowels, so a literal mapping gives jmlh."""
        assert transliterate_to_roman("جملہ") != transliterate_to_roman(
            "جملہ", insert_short_vowels=False
        )

    def test_roman_output_is_ascii(self):
        assert transliterate_to_roman("میرا نام علی ہے").isascii()


class TestCleaning:
    def test_urls_and_handles_are_removed(self):
        assert remove_urls_and_mentions("دیکھیں https://x.com/a @user #tag") == "دیکھیں"


class TestWrongDirection:
    """Urdu fed to the Roman->Urdu direction must say so, not pretend it worked.

    `_apply_rules` matches only Latin graphemes, so an Urdu token passed through it
    came back unchanged - the correct output, labelled `rules`, as though the rule
    engine had resolved it. `lexicon_coverage` then read 0.0, which says "guessed
    badly" rather than "this input was not Roman Urdu".
    """

    def test_urdu_input_is_labelled_rather_than_claimed_as_transliterated(self):
        result = transliterate_with_confidence("سرکاری")
        assert result.text == "سرکاری"
        assert result.sources == [("سرکاری", "already-urdu")]

    def test_already_urdu_tokens_do_not_count_against_lexicon_coverage(self):
        # All four Roman words are in the lexicon; the Urdu one is not a miss.
        result = transliterate_with_confidence("main theek hoon سرکاری")
        assert result.lexicon_coverage == 1.0
        assert result.already_urdu_share == 0.25

    def test_a_fully_urdu_string_reports_it_was_the_wrong_direction(self):
        result = transliterate_with_confidence("یہ اچھا ہے")
        assert result.already_urdu_share == 1.0

    def test_mixed_script_keeps_the_urdu_and_converts_the_roman(self):
        result = transliterate_with_confidence("mera naam علی hai")
        kinds = dict(result.sources)
        assert kinds["علی"] == "already-urdu"
        assert kinds["mera"] == "lexicon"


class TestDocumentedExamples:
    """The README's examples must be what the code returns.

    In 0.1 the README showed `Ali` resolving to علی, which is what a reader expects
    and not what the rules produced: ع cannot be written in Roman, so the rules gave
    الی. The README was corrected to الی. In 0.2 the vocabulary stage finds علی,
    because علی is a real word and الی is a far rarer one - so the README changes
    again, and both behaviours are pinned here.
    """

    def test_the_readme_transliteration_example(self):
        result = transliterate_with_confidence("mera naam Ali hai")
        assert result.text == "میرا نام علی ہے"
        assert result.lexicon_coverage == 0.5
        assert dict(result.sources)["Ali"] == "vocabulary"

    def test_without_the_vocabulary_the_rules_still_give_the_honest_guess(self):
        result = transliterate_with_confidence("mera naam Ali hai", use_vocabulary=False)
        assert result.text == "میرا نام الی ہے"
        assert dict(result.sources)["Ali"] == "rules"


class TestRoundTrip:
    def test_normalising_a_transliteration_is_stable(self):
        urdu = transliterate_to_urdu("main theek hoon")
        assert normalize(urdu) == urdu

    # Words with no ambiguous consonant, so the vowel setting is the only variable.
    # Picked by checking, not by assumption: the first draft of these tests used صرف,
    # which failed BOTH ways under the rules because ص and س both romanise to `s`.
    VOWEL_ONLY = ("کتاب", "پاکستان", "مشکل", "تقریبا", "امریکی")
    COLLAPSES = {
        "صرف": "س",  # ص and س both romanise to `s`
        "حسن": "ہ",  # ح, ہ and ھ all romanise to `h`
        "بیماریاں": "ن",  # ں (nasalisation) is written as a plain `n`
    }

    def test_under_the_rules_dropping_short_vowels_round_trips_better(self):
        """0.1, measured on 457,428 corpus tokens: 61.2% without, 44.7% with.

        Every inserted short vowel returns as an alef, so the readable Roman form is
        the one the rules cannot convert back. Still true of the rules stage alone.
        """
        for word in self.VOWEL_ONLY:
            roman = transliterate_to_roman(word, method="rules")
            bare = transliterate_to_roman(word, insert_short_vowels=False)
            assert transliterate_to_urdu(bare, use_vocabulary=False) == word
            assert transliterate_to_urdu(roman, use_vocabulary=False) != word

    def test_under_the_rules_letters_sharing_a_roman_form_cannot_round_trip(self):
        """Urdu distinguishes letters that Roman spells identically, so a mapping
        back letter by letter can only pick one - ص and س are both `s`."""
        for word, expected_substitute in self.COLLAPSES.items():
            for insert in (True, False):
                roman = transliterate_to_roman(word, method="rules", insert_short_vowels=insert)
                assert transliterate_to_urdu(roman, use_vocabulary=False) != word
            bare = transliterate_to_roman(word, insert_short_vowels=False)
            assert expected_substitute in transliterate_to_urdu(bare, use_vocabulary=False)

    @pytest.mark.parametrize("insert", [True, False])
    def test_the_vocabulary_recovers_both_losses(self, insert):
        """What the rules cannot undo, the vocabulary can: `srf` is not a word, and
        صرف is the real word whose romanisation it is most likely to be. The same
        goes for inserted vowels - `katab` finds کتاب, not کاتاب.

        docs/CORPUS.md used to call these losses "properties of the two writing
        systems, not of the implementation". For words in the vocabulary, that was
        half wrong: the information is gone from the Roman string, but not from
        the language.
        """
        for word in (*self.VOWEL_ONLY, *self.COLLAPSES):
            roman = transliterate_to_roman(word, method="rules", insert_short_vowels=insert)
            assert transliterate_to_urdu(roman) == word

    def test_a_word_with_no_ambiguity_round_trips_under_both_settings(self):
        """The control. Without it the two tests above could be describing a
        transliterator that simply never round-trips anything."""
        assert transliterate_to_urdu(transliterate_to_roman("لاہور")) == "لاہور"
        assert (
            transliterate_to_urdu(transliterate_to_roman("لاہور", insert_short_vowels=False))
            == "لاہور"
        )


def test_the_ergative_ne_is_a_stopword():
    """نے, the sixth most frequent word in BBC Urdu, was missing from the list."""
    from urdunlp import STOPWORDS, is_stopword, remove_stopwords, words

    assert is_stopword("نے")
    for word in ("جس", "اپنے", "کیونکہ", "والے", "سکتا"):
        assert word in STOPWORDS
    assert remove_stopwords(words("علی نے کتاب پڑھی")) == ["علی", "کتاب", "پڑھی"]
    assert not is_stopword("نہیں")  # negation still kept


def test_source_tags_are_exhaustive() -> None:
    """Every tag the transliterator can emit must be named in SOURCE_TAGS.

    `sources` is a public contract that callers switch on. Four of its values were
    documented and the rest were only discoverable by reading the module, so a caller
    could not tell a complete match from a partial one. This test reads the tag literals
    straight out of the source and fails if a new one appears without a description,
    which is the drift this constant exists to prevent.
    """
    import re
    from pathlib import Path

    import urdunlp
    from urdunlp import SOURCE_TAGS

    path = Path(urdunlp.__file__).parent / "translit.py"
    src = path.read_text(encoding="utf-8")
    literals = set(re.findall(r'(?:plan|rendered)\.append\(\([^)]*?"([a-z-]+)"\)\)', src))
    literals.discard("space")  # filtered out of sources by construction
    undocumented = sorted(literals - set(SOURCE_TAGS))
    assert not undocumented, f"tags emitted but not described in SOURCE_TAGS: {undocumented}"


def test_observed_source_tags_are_all_documented() -> None:
    """The same check from the outside: run real text and inspect what comes back."""
    import urdunlp
    from urdunlp import SOURCE_TAGS

    texts = [
        "mera naam Ali hai",
        "kal meeting cancel ho gayi",
        "tehreek-e-insaf ka jalsa",
        "mail test@x.com aur http://x.co dekho",
        "assalam o alaikum",
        "PIA aur NADRA",
        "yeh 125 rupay hai",
        "lol plz bhej do",
        "Dr. Ahmed sahab",
        "کتاب aur book",
        "kya?, haan. theek",
    ]
    seen = set()
    for text in texts:
        for _token, tag in urdunlp.transliterate_with_confidence(text).sources:
            seen.add(tag)
    assert seen, "no sources were produced at all"
    unknown = sorted(seen - set(SOURCE_TAGS))
    assert not unknown, f"undocumented tags reached a caller: {unknown}"
