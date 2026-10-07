"""The four gaps suite-auditor proved in this package's own suite.

Mutation-tested with suite-auditor: 196 mutants scored, a 74.5% kill rate, 99.1% of
functions reached by a test, and **four proven gaps** - three of them on inputs the tests
already use. Each test below names the mutation it defeats, because a test whose purpose
is not written down is one that gets deleted in the next tidy-up for looking redundant.

Two are in `_grams`, which is the feature extractor the entire language-ID model is built
from: every count in `langid.json.gz` is a gram this function produced, so a change to
which grams it yields changes the model's vocabulary and every probability over it. The
suite checked that the model classifies correctly, which is a weaker thing - a wrong gram
set can still score well on the data it was trained from.
"""

from __future__ import annotations

import pytest

from urdunlp.cli import _looks_mangled
from urdunlp.langid import _grams


class TestGrams:
    """`_grams(text, 5)` had two surviving mutants, both on a real Arabic sentence.

    The function pads with a space either side and yields every n-gram of every length up
    to n_max, keeping a 1-gram even when it is whitespace and dropping longer grams that
    are all whitespace. Three details carry that, and none was tested: the padding, the
    `n == 1` exception, and the slice bounds.
    """

    def test_the_padding_is_part_of_the_gram_set(self) -> None:
        """Defeats `compare`: without the pad, word-initial and word-final grams vanish.

        The leading `" ه"` is what tells the model a letter starts a word, and that is
        most of what separates Urdu from Punjabi in short text.
        """
        grams = list(_grams("ab", 2))
        assert " a" in grams, "the leading pad is gone, so no gram marks a word start"
        assert "b " in grams, "the trailing pad is gone, so no gram marks a word end"
        assert grams.count(" ") == 2, "both pad characters are 1-grams"

    def test_every_gram_is_a_slice_of_the_padded_text_of_its_own_length(self) -> None:
        """Defeats `slice_lower`, which made every gram an empty string.

        The mutant produced a list of the right length full of `''`, so only the gram
        CONTENTS separate it from the original - which is what the suite never looked at.
        """
        text = "الف ب"
        padded = f" {text} "
        for n in range(1, 6):
            for gram in _grams(text, n):
                assert gram, "an empty gram carries no information"
                assert gram in padded, f"{gram!r} is not a slice of the padded text"
        for n in (1, 2, 3, 4, 5):
            produced = [g for g in _grams(text, n) if len(g) == n]
            assert produced, f"no {n}-gram was produced at all"

    def test_the_counts_are_exactly_the_sliding_windows(self) -> None:
        text = "abcd"
        padded = " abcd "
        for n_max in (1, 2, 3, 5):
            got = list(_grams(text, n_max))
            expected = [
                padded[i : i + n]
                for n in range(1, n_max + 1)
                for i in range(len(padded) - n + 1)
                if n == 1 or padded[i : i + n].strip()
            ]
            assert got == expected, n_max

    def test_whitespace_longer_than_one_character_is_dropped(self) -> None:
        """The `n == 1 or gram.strip()` rule: a lone space is a feature, `"  "` is not."""
        grams = list(_grams("a  b", 3))
        assert " " in grams
        assert "  " not in grams
        assert "   " not in grams

    def test_an_empty_text_still_yields_its_padding(self) -> None:
        assert list(_grams("", 1)) == [" ", " "]
        assert all(g.strip() or len(g) == 1 for g in _grams("", 5))


class TestLooksMangled:
    """`_looks_mangled("")` had a surviving mutant: original False, mutant True.

    It decides whether a line is text a shell has already replaced with question marks,
    and a false positive makes the CLI refuse input it could have processed. The empty
    line is the case that reaches both halves of the `and`, and nothing tested it.
    """

    @pytest.mark.parametrize("line", ["", " ", "   ", "\t"])
    def test_an_empty_or_blank_line_is_not_mangled(self, line: str) -> None:
        # Defeats `boolop`: `count("?") >= 3 and not stripped.strip(...)` mutated to
        # `or`, which makes every blank line mangled - the second half is vacuously
        # true for an empty string.
        assert _looks_mangled(line) is False

    @pytest.mark.parametrize("line", ["???", "? ? ?", "????", "???!", "??? 123"])
    def test_a_line_of_question_marks_is_mangled(self, line: str) -> None:
        assert _looks_mangled(line) is True

    @pytest.mark.parametrize("line", ["??", "? ?", "a???", "???a", "الف"])
    def test_fewer_than_three_or_any_real_text_is_not(self, line: str) -> None:
        """Both halves have to bite: three marks AND nothing else of substance."""
        assert _looks_mangled(line) is False


def test_the_channel_factory_returns_a_channel() -> None:
    """A `drop_return` mutant of `channel()` returned None and the suite passed.

    Everything in `translit` goes through it, so returning None would fail at the first
    use - which is exactly why nothing asserted it: the error would be somebody else's
    AttributeError, in a different function, with no mention of the factory.
    """
    from urdunlp._channel import Channel, channel

    got = channel()
    assert got is not None
    assert isinstance(got, Channel)
    # Cached, and the cache must hand back the same object rather than None after the
    # first call.
    assert channel() is got
