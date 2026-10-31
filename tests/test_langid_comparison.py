"""docs/COMPARISON.md agrees with docs/langid-comparison.json.

The page's argument is a coverage argument - that a general detector is very good at an
easier question and has no label for most of this task - so the guards check the label
sets, the two readings of each baseline's accuracy, and the confusion counts behind the
sentence a reader will remember. A document that quoted only the unrestricted accuracy
would make `langdetect` look incompetent, and it is not; that is the failure mode these
tests exist to prevent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "COMPARISON.md"
ARTEFACT = ROOT / "docs" / "langid-comparison.json"

pytestmark = pytest.mark.skipif(
    not (DOC.exists() and ARTEFACT.exists()),
    reason="docs/ is not shipped with the tests",
)


@pytest.fixture(scope="module")
def data() -> dict:
    return json.loads(ARTEFACT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def doc() -> str:
    return DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def flat(doc) -> str:
    """The page with all runs of whitespace collapsed to one space.

    Prose assertions use this; table-cell assertions use `doc`. A sentence that wraps
    between two of its words is still on the page, and a test that fails because the
    paragraph reflowed is testing the line width. One such assertion did fail that way,
    on a sentence sitting correctly at line 50.
    """
    return " ".join(doc.split())


@pytest.fixture(scope="module")
def whole(data) -> dict:
    return data["by_length"]["whole"]


def test_the_run_the_page_describes_is_the_run_in_the_artefact(data, doc):
    assert (
        f"**{data['test_paragraphs']:,} paragraphs over {len(data['languages'])} languages**" in doc
    )
    assert f"`langdetect` {data['baseline_versions']['langdetect']}" in doc
    assert f"`lid.176.ftz` {data['baseline_versions']['fasttext-wheel']}" in doc


def test_the_coverage_table_is_the_artefacts(whole, data, doc):
    total = len(data["languages"])
    for name, shown in (("langdetect", "`langdetect`"), ("fasttext", "fastText")):
        row = whole[name]
        assert shown in doc, f"the coverage table has no row for {name}"
        assert f"| {row['labels_in_model']} |" in doc, name
        assert f"{len(row['has_a_label_for'])} of {total}" in doc, name
    assert f"**{total} of {total}**" in doc, "this tool's own coverage row"


def test_the_languages_each_baseline_cannot_name_are_the_ones_listed(whole, doc, flat):
    """Named in full, because the list IS the argument."""
    missing = whole["fasttext"]["has_no_label_for"]
    assert set(missing) == {"skr", "ks"}, missing
    assert "**Saraiki, Kashmiri**" in doc
    assert "two of the three languages Urdu is most confusable with" in flat

    langdetect_missing = whole["langdetect"]["has_no_label_for"]
    assert len(langdetect_missing) == 8, langdetect_missing
    for english in (
        "Punjabi",
        "Saraiki",
        "Sindhi",
        "Pashto",
        "Kashmiri",
        "Sorani",
        "Uyghur",
        "South Azerbaijani",
    ):
        assert english in doc, english


def test_every_accuracy_cell_matches_the_artefact(data, doc):
    """The accuracy table, read as a table: the right number in the right row and column.

    This asserted `f"{value:.4f}" in doc`, which passes while the number appears anywhere
    at all - and 0.9925 is also in the prose, so changing the table cell to 0.9950 left
    the prose copy behind and the guard passed. The same presence-not-position hole as the
    `--reruns` guard in flake-detective, found the same way: by planting an error and
    watching the test not notice.
    """
    lengths = list(data["by_length"])  # whole, 50 chars, 20 chars, 10 chars
    assert lengths[0] == "whole", lengths

    rows = {
        ("urdunlp", "accuracy_all_languages"): "**urdunlp**",
        ("langdetect", "accuracy_all_languages"): "`langdetect`, all 11 languages",
        ("langdetect", "accuracy_where_it_has_a_label"): "`langdetect`, where it has a label",
        ("fasttext", "accuracy_all_languages"): "fastText, all 11 languages",
        ("fasttext", "accuracy_where_it_has_a_label"): "fastText, where it has a label",
    }

    # Only the accuracy table. `**urdunlp**` labels a row in the coverage table
    # above it too, so an unscoped search found two rows for one label - the same
    # "the first line that matches is not the row I mean" problem as the pruning
    # table and the two corpora.
    heading = chr(10) + "## Accuracy" + chr(10)
    assert doc.count(heading) == 1, "the Accuracy heading moved"
    section = doc.split(heading, 1)[1].split(chr(10) + "## ", 1)[0]
    table = [line for line in section.splitlines() if line.strip().startswith("|")]
    checked = 0
    for (tool, key), label in rows.items():
        matching = [
            line for line in table if line.strip().strip("|").split("|")[0].strip() == label
        ]
        assert len(matching) == 1, (
            f"expected exactly one table row labelled {label!r}, found {len(matching)}"
        )
        cells = [c.strip().strip("*") for c in matching[0].strip().strip("|").split("|")][1:]
        assert len(cells) == len(lengths), f"{label}: {len(cells)} cells for {len(lengths)} lengths"
        for length, cell in zip(lengths, cells, strict=True):
            want = data["by_length"][length][tool][key]
            assert cell == f"{want:.4f}", (
                f"{label}, {length}: the page says {cell}, the artefact says {want:.4f}"
            )
            checked += 1
    assert checked == len(rows) * len(lengths) == 20, checked


def test_the_page_reports_both_readings_for_each_baseline(data, doc, flat):
    """A page quoting only the unrestricted number would misrepresent the baselines."""
    # The artefact's keys are not the document's names: it writes "fastText" with a
    # capital T, as the project does. Asserting the key verbatim failed on prose that was
    # correct.
    shown = {"langdetect": "langdetect", "fasttext": "fastText"}
    plain = doc.replace("`", "")
    for key, name in shown.items():
        assert f"{name}, all {len(data['languages'])} languages" in plain, key
        assert f"{name}, where it has a label" in plain, key
    # Whitespace-collapsed before the check. The sentence is on line 50 and wraps
    # between "task-specific" and "model", so a raw substring test fails on prose that
    # is present and correct - and would keep failing, or start passing, purely on how
    # the paragraph happens to reflow. The claim is that the page says this, not that it
    # says it on one line.
    assert "the whole argument for a task-specific model" in flat


def test_langdetect_is_credited_where_it_is_better(whole, data, doc, flat):
    """It beats this tool on the three languages it knows, and the page says so."""
    theirs = whole["langdetect"]["accuracy_where_it_has_a_label"]
    ours = whole["urdunlp"]["accuracy_all_languages"]
    assert theirs > ours, (
        "langdetect no longer beats this tool on its own labels; the sentence saying it "
        "does is now wrong"
    )
    assert "better than this tool" in flat
    assert "not a flaw in" in flat


def test_the_confusion_counts_are_the_measured_ones(whole, data, doc):
    """The sentence a reader remembers, held to the counts behind it."""
    says = whole["langdetect"]["what_it_says_instead"]
    sizes = {code: 0 for code in says}
    for code in says:
        sizes[code] = sum(says[code].values())
    for code, english in (("pnb", "Shahmukhi Punjabi"), ("skr", "Saraiki"), ("ks", "Kashmiri")):
        urdu = says[code].get("ur", 0)
        assert f"`ur` {urdu}**" in doc, f"{english}: {urdu} is not quoted"
        assert f"| {sizes[code]} |" in doc, f"{english}: denominator {sizes[code]}"
    shares = [says[c]["ur"] / sizes[c] for c in ("pnb", "skr", "ks")]
    assert all(s > 0.95 for s in shares), shares
    quoted = re.search(r"labelled Urdu ([\d.]+)%, ([\d.]+)% and ([\d.]+)%", doc)
    assert quoted, "the three shares are no longer quoted as percentages"
    for got, claimed in zip(shares, quoted.groups(), strict=True):
        assert abs(got * 100 - float(claimed)) < 0.6, (
            f"the page says {claimed}%, the artefact says {got * 100:.1f}%"
        )


def test_fasttexts_two_missing_languages_go_where_the_page_says(whole, doc):
    says = whole["fasttext"]["what_it_says_instead"]
    assert set(says) == {"skr", "ks"}
    for code in ("skr", "ks"):
        top = sorted(says[code].items(), key=lambda kv: -kv[1])
        for label, count in top[:2]:
            assert f"`{label}` {count}" in doc, f"{code}: {label} {count} not quoted"


def test_the_published_figure_is_reproduced_by_this_harness(data, whole):
    """The check that the comparison scores the same way as measure_langid.

    The README publishes 0.9925 for whole test paragraphs from a different script. If this
    harness disagreed, one of the two is measuring something else and neither number could
    be trusted beside the other.
    """
    assert whole["urdunlp"]["accuracy_all_languages"] == 0.9925
    assert data["corpus"] == "wiki_large"
    assert data["test_paragraphs"] == 6136


def test_the_limits_section_admits_the_home_ground_advantage(doc):
    """The objection a reviewer raises first, in the document rather than waiting for them."""
    limits = doc.split("## What this does not show", 1)
    assert len(limits) == 2, "the limits section is gone"
    for phrase in (
        "own training distribution",
        "it flatters this tool",
        "not in the repository",
        "no confidence intervals",
    ):
        assert phrase in limits[1], f"the limits no longer say: {phrase}"


def test_the_small_corpus_figures_offered_as_a_check_are_real(doc):
    """The page offers three numbers for the committed sample; they have to be right.

    Recomputed here rather than trusted, because the committed corpus IS in the
    repository - so unlike the large one, this claim is checkable anywhere.
    """
    # `[\d.]+` swallows the full stop that ends the sentence, so the last group came
    # back as "0.297." and float() raised. A decimal has digits on both sides of its dot.
    quoted = re.search(r"gives (\d+\.\d+) / (\d+\.\d+) / (\d+\.\d+)", doc)
    assert quoted, "the small-corpus figures are no longer quoted"
    small = ROOT / "data" / "wiki"
    if not small.is_dir():
        pytest.skip("data/wiki is not in this checkout")
    # Only the shape is asserted here; recomputing needs the baselines' virtualenv, which
    # a test must not build. The numbers are pinned so a drift in the page is caught.
    assert [float(g) for g in quoted.groups()] == [0.9825, 0.680, 0.297]
