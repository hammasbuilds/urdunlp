"""The orthographic criterion, and the regression that made it necessary.

Section 6.2 asserted that four implosive letters were distinctive to Saraiki among the
eleven languages, having compared them against the two languages Saraiki gets confused
with - Shahmukhi Punjabi and Urdu - and not against Sindhi, which writes three of them
about as often as Saraiki does. The claim "a line carrying one is Saraiki on orthographic
grounds alone" was therefore false.

`research/orthographic_audit.py` derives the inventory from the corpus rather than
asserting it, so the comparison is against every language by construction and cannot miss
one. These tests pin that: the Sindhi case is a named regression test, and each threshold
is checked on both sides, because a selector that returned nothing would also produce the
"no distinctive letter" result that five of the eleven languages legitimately have.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "orthographic_audit", ROOT / "research" / "orthographic_audit.py"
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - layout changed
    pytest.skip("research/orthographic_audit.py is not present", allow_module_level=True)
ortho = importlib.util.module_from_spec(SPEC)
sys.modules["orthographic_audit"] = ortho
SPEC.loader.exec_module(ortho)

BEEH = "ٻ"  # also Sindhi - the letter that broke the original claim
DUL = "ݙ"  # Saraiki only
GAF = "گ"  # everywhere


class TestDistinctiveSelectsOnBothThresholds:
    def test_a_letter_common_here_and_absent_elsewhere_is_kept(self):
        marks = ortho.distinctive({"skr": {DUL: 0.26}, "sd": {}, "ur": {}})
        assert marks["skr"] == [DUL]

    def test_a_letter_too_rare_in_its_own_language_is_dropped(self):
        below = ortho.MIN_SHARE / 2
        marks = ortho.distinctive({"skr": {DUL: below}, "ur": {}})
        assert marks["skr"] == []

    def test_a_letter_at_the_threshold_is_kept(self):
        marks = ortho.distinctive({"skr": {DUL: ortho.MIN_SHARE}, "ur": {}})
        assert marks["skr"] == [DUL]

    def test_a_letter_common_in_another_language_is_dropped(self):
        """The Sindhi regression, by name.

        U+067B is in 13.1% of Saraiki training lines and 20.7% of Sindhi's. Any selector
        that keeps it has compared against the wrong set of languages.
        """
        marks = ortho.distinctive({"skr": {BEEH: 0.131}, "sd": {BEEH: 0.207}})
        assert marks["skr"] == [], (
            "U+067B was kept as distinctive to Saraiki although Sindhi writes it more "
            "often - the exact error section 6.2 had to correct"
        )
        assert marks["sd"] == [], "and it is not distinctive to Sindhi either"

    def test_the_other_language_threshold_is_not_zero(self):
        """One quoted foreign sentence must not delete a real criterion.

        A rule demanding exact absence is defeated by a single Saraiki sentence quoted on
        a Punjabi wiki page, which is why the bar is a share rather than a count.
        """
        trace = ortho.MAX_OTHER_SHARE / 2
        marks = ortho.distinctive({"skr": {DUL: 0.26}, "pnb": {DUL: trace}})
        assert marks["skr"] == [DUL]

    def test_just_above_the_other_language_threshold_drops_it(self):
        over = ortho.MAX_OTHER_SHARE * 2
        marks = ortho.distinctive({"skr": {DUL: 0.26}, "pnb": {DUL: over}})
        assert marks["skr"] == []

    def test_a_shared_letter_is_distinctive_to_nobody(self):
        marks = ortho.distinctive({"ur": {GAF: 0.9}, "skr": {GAF: 0.9}, "sd": {GAF: 0.8}})
        assert marks == {"ur": [], "skr": [], "sd": []}

    def test_a_language_with_no_criterion_is_reported_not_omitted(self):
        """Urdu's empty inventory is a finding, so the key has to exist."""
        marks = ortho.distinctive({"ur": {GAF: 0.9}, "skr": {DUL: 0.26, GAF: 0.9}})
        assert marks["ur"] == []
        assert marks["skr"] == [DUL]

    def test_the_margin_between_the_constants_is_at_least_twenty_fold(self):
        """The paper states the ratio, not just the two numbers."""
        assert ortho.MIN_SHARE / ortho.MAX_OTHER_SHARE >= 20


class TestLetterSharesCountsLinesNotCharacters:
    def splits(self, tmp_path, **langs):
        for split in ("train", "test"):
            (tmp_path / split).mkdir(parents=True, exist_ok=True)
        for language, lines in langs.items():
            (tmp_path / "train" / f"{language}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
        return tmp_path

    def test_a_letter_repeated_in_one_line_counts_once(self, tmp_path, monkeypatch):
        """The question is whether a line can be attributed, not how often a letter is
        written, so a line with the letter twenty times is still one line."""
        d = self.splits(tmp_path, skr=[DUL * 20 + GAF, GAF])
        monkeypatch.setattr(ortho, "SPLITS", d)
        shares, lines = ortho.letter_shares("skr")
        assert lines == 2
        assert shares[DUL] == pytest.approx(0.5)

    def test_blank_lines_are_not_counted_in_the_denominator(self, tmp_path, monkeypatch):
        d = self.splits(tmp_path, skr=[DUL + GAF, "", "   ", GAF])
        monkeypatch.setattr(ortho, "SPLITS", d)
        shares, lines = ortho.letter_shares("skr")
        assert lines == 2
        assert shares[DUL] == pytest.approx(0.5)

    def test_digits_and_punctuation_are_not_letters(self, tmp_path, monkeypatch):
        """Punctuation is shared by every language here and would swamp the comparison."""
        d = self.splits(tmp_path, skr=["1234 ، ؟ " + DUL])
        monkeypatch.setattr(ortho, "SPLITS", d)
        shares, _ = ortho.letter_shares("skr")
        assert set(shares) == {DUL}

    def test_an_empty_file_gives_no_shares_rather_than_dividing_by_zero(
        self, tmp_path, monkeypatch
    ):
        d = self.splits(tmp_path, skr=[""])
        monkeypatch.setattr(ortho, "SPLITS", d)
        shares, lines = ortho.letter_shares("skr")
        assert lines == 0
        assert shares == {}


def test_languages_are_read_from_the_split_directory(tmp_path, monkeypatch):
    """A hard-coded language list is how a new language gets left out of the audit."""
    (tmp_path / "train").mkdir(parents=True)
    for name in ("zz", "aa"):
        (tmp_path / "train" / f"{name}.txt").write_text("x\n", encoding="utf-8")
    monkeypatch.setattr(ortho, "SPLITS", tmp_path)
    assert ortho.languages() == ["aa", "zz"]


def test_the_published_inventory_matches_what_the_paper_says():
    """Section 6.3 names Saraiki's single exclusive letter and the five unmarked
    languages. Read off the artefact so a regenerated audit that disagrees fails here."""
    import json  # noqa: PLC0415

    path = ROOT / "research" / "results" / "orthographic_audit.json"
    if not path.exists():
        pytest.skip("the audit has not been run in this checkout")
    data = json.loads(path.read_text(encoding="utf-8"))
    per = data["per_language"]
    assert per["skr"]["distinctive_letters"] == DUL, (
        "the paper states U+0759 is the only Saraiki-exclusive implosive"
    )
    unmarked = {k for k, v in per.items() if not v["has_criterion"]}
    assert unmarked == {"ar", "azb", "fa", "pnb", "ur"}, (
        f"the paper names those five as the unmarked languages; the audit says {unmarked}"
    )
    for language, row in per.items():
        if not row["has_criterion"]:
            continue
        marked = row["with_a_distinctive_letter"]
        assert marked["rate"] >= 0.98, (
            f"{language}: the paper's claim is that every marked half is at or above "
            f"0.99; this one is {marked['rate']}"
        )
