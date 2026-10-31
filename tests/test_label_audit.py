"""The label audit, checked against planted positives before its null result is believed.

`research/label_audit.py` reports, for each of the corpus's source-language pairs, the
share of its lines that appear verbatim under a different language label. On the real
corpus that rate is small, and a small number out of a detector is worth nothing until
the detector has been shown to fire: a function that returned zero unconditionally would
produce the same report. So every test here plants a duplicate it knows about and checks
that the audit finds it, and the last one checks that unrelated text does not trip it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("label_audit", ROOT / "research" / "label_audit.py")
if SPEC is None or SPEC.loader is None:  # pragma: no cover - layout changed
    pytest.skip("research/label_audit.py is not present", allow_module_level=True)
label_audit = importlib.util.module_from_spec(SPEC)
sys.modules["label_audit"] = label_audit
SPEC.loader.exec_module(label_audit)

URDU = "یہ ایک مکمل جملہ ہے جو دونوں فائلوں میں موجود ہے"
OTHER = "اسلام آباد پاکستان کا دارالحکومت ہے اور یہ ایک بڑا شہر ہے"
UNIQUE_SKR = "ٻارن دی خاطر ایہ ڳالھ وڌیک سوکھی ہے تے سمجھ وچ آوݨ والی ہے"


def corpus(tmp_path: Path, **files: list[str]) -> Path:
    """Write a throwaway corpus. Keys are `lang.source`, as the real directory names are."""
    out = tmp_path / "corpus"
    out.mkdir()
    for name, lines in files.items():
        (out / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def rate(data: dict, key: str) -> float:
    return data["sources"][key]["shared_rate_long"]


class TestItFindsAPlantedDuplicate:
    def test_one_shared_line_between_two_labels_is_reported(self, tmp_path):
        c = corpus(
            tmp_path,
            **{
                "skr.wikimedia-wiktionary": [URDU, UNIQUE_SKR],
                "ur.cc100-ur": [URDU, OTHER],
            },
        )
        data = label_audit.audit(corpus=c)
        assert data["distinct_shared_lines"] == 1
        assert rate(data, "skr.wikimedia-wiktionary") == pytest.approx(0.5)
        assert data["sources"]["skr.wikimedia-wiktionary"]["by_other_language_long"] == {"ur": 1}
        assert rate(data, "ur.cc100-ur") == pytest.approx(0.5)

    def test_the_rate_counts_repeats_not_distinct_lines(self, tmp_path):
        """A gloss repeated across 300 entries is 300 mislabelled lines, not one."""
        c = corpus(
            tmp_path,
            **{
                "skr.wikimedia-wiktionary": [URDU] * 3 + [UNIQUE_SKR],
                "ur.cc100-ur": [URDU, OTHER],
            },
        )
        data = label_audit.audit(corpus=c)
        assert data["distinct_shared_lines"] == 1
        assert data["sources"]["skr.wikimedia-wiktionary"]["shared_lines_long"] == 3
        assert rate(data, "skr.wikimedia-wiktionary") == pytest.approx(0.75)

    def test_a_third_label_is_named_too(self, tmp_path):
        c = corpus(
            tmp_path,
            **{
                "skr.wikimedia-wiki": [URDU],
                "ur.cc100-ur": [URDU],
                "pnb.wikimedia-wiki": [URDU],
            },
        )
        data = label_audit.audit(corpus=c)
        assert set(data["sources"]["skr.wikimedia-wiki"]["by_other_language_long"]) == {
            "ur",
            "pnb",
        }


class TestNormalisationDoesNotHideADuplicate:
    def test_differing_diacritics_still_count_as_the_same_line(self, tmp_path):
        """Two projects writing identical text with and without vowel marks.

        Without the folding this returns zero overlap, which is the false negative that
        would let a mislabelled source through - and it would do so silently.
        """
        marked = "یہِ ایک مکمل جملہَ ہے جو دونوں فائلوں میں موجود ہے"
        c = corpus(
            tmp_path,
            **{"skr.wikimedia-wiktionary": [marked], "ur.cc100-ur": [URDU]},
        )
        data = label_audit.audit(corpus=c)
        assert data["distinct_shared_lines"] == 1, (
            "diacritic folding is not reaching the digest: identical text written with "
            "and without vowel marks was read as two different lines"
        )

    def test_differing_whitespace_still_counts(self, tmp_path):
        c = corpus(
            tmp_path,
            **{"skr.segmented": ["  ".join(URDU.split())], "ur.cc100-ur": [URDU]},
        )
        assert label_audit.audit(corpus=c)["distinct_shared_lines"] == 1

    def test_a_tatweel_still_counts(self, tmp_path):
        c = corpus(
            tmp_path,
            **{"skr.segmented": [URDU.replace("ہے", "ہــے", 1)], "ur.cc100-ur": [URDU]},
        )
        assert label_audit.audit(corpus=c)["distinct_shared_lines"] == 1


class TestItDoesNotFireOnThingsItShouldNot:
    def test_unrelated_text_gives_a_zero_rate(self, tmp_path):
        c = corpus(
            tmp_path,
            **{"skr.wikimedia-wiki": [UNIQUE_SKR], "ur.cc100-ur": [OTHER]},
        )
        data = label_audit.audit(corpus=c)
        assert data["distinct_shared_lines"] == 0
        assert rate(data, "skr.wikimedia-wiki") == 0.0

    def test_duplication_within_one_language_is_not_a_finding(self, tmp_path):
        """`segmented` is derived from the wiki text of the same language, so intra-label
        duplication is expected and says nothing about labelling."""
        c = corpus(
            tmp_path,
            **{"skr.wikimedia-wiki": [UNIQUE_SKR], "skr.segmented": [UNIQUE_SKR]},
        )
        data = label_audit.audit(corpus=c)
        assert data["distinct_shared_lines"] == 0
        assert rate(data, "skr.segmented") == 0.0

    def test_short_lines_are_excluded_from_the_headline_rate(self, tmp_path):
        """A date or a numeral shared across labels is not a mislabelled sentence.

        Both rates are produced, so this pins that the long-line rate drops it and the
        all-lines rate still records it - a filter that quietly removed the evidence
        would be the same mistake in the other direction.
        """
        short = "۱۹۴۷"
        assert len(short) < label_audit.MIN_INTERESTING
        c = corpus(
            tmp_path,
            **{
                "skr.wikimedia-wiki": [short, UNIQUE_SKR],
                "ur.cc100-ur": [short, OTHER],
            },
        )
        data = label_audit.audit(corpus=c)
        assert data["distinct_shared_lines"] == 1
        assert rate(data, "skr.wikimedia-wiki") == 0.0
        assert data["sources"]["skr.wikimedia-wiki"]["shared_rate"] == pytest.approx(0.5)


def test_the_file_names_are_read_from_the_directory(tmp_path):
    """The source list comes from the corpus directory, not a hard-coded list.

    A hard-coded list is how a new source gets added and silently left out of the audit
    that is supposed to cover every source.
    """
    c = corpus(tmp_path, **{"xx.brand-new-source": [OTHER]})
    assert label_audit.sources(c) == [("xx", "brand-new-source", c / "xx.brand-new-source.txt")]
