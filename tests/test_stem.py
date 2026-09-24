"""The stemmer conflates inflected forms and leaves function words alone."""

from __future__ import annotations

import pytest

from urdunlp import stem, stem_tokens


@pytest.mark.parametrize(
    "forms",
    [
        ("کتاب", "کتابیں", "کتابوں"),  # book: plural, oblique plural
        ("لڑکا", "لڑکے", "لڑکوں", "لڑکی", "لڑکیاں", "لڑکیوں"),  # boy / girl
        ("کرتا", "کرتی", "کرتے"),  # imperfective participle
    ],
)
def test_inflected_forms_share_a_stem(forms):
    assert len({stem(f) for f in forms}) == 1


def test_the_stem_is_a_key_not_a_word():
    """لڑکا and لڑکی both become لڑک - not a word, and not meant to be shown."""
    assert stem("لڑکا") == stem("لڑکی") == "لڑک"


@pytest.mark.parametrize("word", ["سے", "کے", "تھا", "گیا", "اپنی", "گاؤں"])
def test_function_words_are_protected(word):
    """Stripping these would merge سے into a stem shared with unrelated words."""
    assert stem(word) == word


def test_never_leaves_fewer_than_min_stem_letters():
    """بات must not become ب."""
    assert stem("بات") == "بات"
    assert len(stem("کتابوں", min_stem=4)) >= 4


def test_light_mode_leaves_verbal_endings():
    assert stem("پڑھتا", light=True) == "پڑھت"  # only the gender ending goes
    assert stem("پڑھتا") == "پڑھ"


def test_the_minimum_stem_beats_a_longer_suffix():
    """کرتا minus تا would leave کر, two letters - under the default minimum of
    three, so the next suffix that fits is used instead."""
    assert stem("کرتا") == "کرت"
    assert stem("کرتا", min_stem=2) == "کر"


def test_input_is_normalised_first():
    """An Arabic kaf must not stop a match."""
    assert stem("كتابوں") == stem("کتابوں")


def test_stem_tokens_maps_over_a_list():
    assert stem_tokens(["کتابوں", "سے"]) == ["کتاب", "سے"]
