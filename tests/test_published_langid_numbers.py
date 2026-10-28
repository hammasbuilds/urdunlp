"""Every language-ID figure in the README and docs/CORPUS.md is read from the report.

The two documents already checked each other - CORPUS.md in prose, the README in a table
- but neither was checked against a measurement. So they could agree and both be stale,
which is what happened: the model was retrained and seven figures in the README plus a
table and two sentences in CORPUS.md still described the previous one.

`data/langid_report_v5.json` is the output of `scripts/measure_langid.py`, and
`data/langid_sweep.json` the output of `scripts/sweep_langid.py`. `data/` is otherwise
gitignored; these two are 28 KB and are the exceptions that make the figures checkable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
CORPUS = ROOT / "docs" / "CORPUS.md"
REPORT = ROOT / "data" / "langid_report_v5.json"
SWEEP = ROOT / "data" / "langid_sweep.json"

pytestmark = pytest.mark.skipif(
    not REPORT.exists(), reason="the measurement report is not shipped with the tests"
)

LABELS = {"ur": "Urdu", "pnb": "Punjabi (Shahmukhi)", "skr": "Saraiki"}


def _script() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))["identify_language"]


def test_the_headline_accuracy_is_the_measured_one() -> None:
    whole = _script()["test_whole"]["accuracy"]
    readme = README.read_text(encoding="utf-8")
    assert f"{whole * 100:.1f}%" in readme, f"the README does not quote {whole * 100:.1f}% anywhere"
    # And the old figure must be gone, or both are present and the reader picks one.
    assert "97.9%" not in readme, "a stale headline accuracy is still in the README"


def test_the_per_language_table_rows_are_the_measured_ones() -> None:
    script = _script()
    readme = README.read_text(encoding="utf-8")
    keys = ["test_whole", "test_50 chars", "test_20 chars", "test_10 chars"]
    for code, label in LABELS.items():
        n = script["paragraphs"][code]["test"]
        cells = [f"{script[k]['per_language'][code] * 100:.1f}%" for k in keys]
        row = re.search(
            rf"\|\s*{re.escape(label)}\s*\|\s*{n}\s*\|(.+?)\|\s*$",
            readme,
            re.M,
        )
        assert row, f"no table row for {label} with n={n}"
        found = re.findall(r"(\d+\.\d)%", row.group(0))
        assert found == [c.rstrip("%") for c in cells], (
            f"{label}: README row says {found}, the report says {cells}"
        )


def test_the_other_eight_row_is_the_measured_range() -> None:
    """The row the per-language guard did not cover.

    `LABELS` holds Urdu, Punjabi and Saraiki, so the table's fourth row - the range over
    the eight languages that are nearly separable on orthography - was unguarded. On the
    model rebuilt at top=180000 all four of its cells were written from intuition and
    three were wrong: 98.8-100.0 against a typed 99.3-100.0 on whole paragraphs,
    89.1-98.9 against 88.4-98.5 at ten characters. A range is as quotable as a number and
    needs the same check.
    """
    script = _script()
    readme = README.read_text(encoding="utf-8")
    others = [c for c in script["test_whole"]["per_language"] if c not in LABELS]
    assert len(others) == 8, f"{len(others)} languages outside LABELS, not 8"
    cells = []
    for key in ("test_whole", "test_50 chars", "test_20 chars", "test_10 chars"):
        rates = [script[key]["per_language"][c] * 100 for c in others]
        cells.append(f"{min(rates):.1f}–{max(rates):.1f}%")
    row = re.search(r"^\|\s*the other eight\s*\|(.+?)\|\s*$", readme, re.M)
    assert row, "the per-language table has no 'the other eight' row"
    found = re.findall(r"\d+\.\d–\d+\.\d%", row.group(0))
    assert found == cells, f"the README row says {found}, the report says {cells}"


def test_the_corpus_document_quotes_the_measured_twenty_character_figures() -> None:
    script = _script()
    stated = re.search(
        r"At\s+20\s+characters,\s+Saraiki\s+is\s+right\s+([\d.]+)%[^.]*?,"
        r"\s+Punjabi\s+([\d.]+)%\s+and\s+Urdu\s+([\d.]+)%",
        CORPUS.read_text(encoding="utf-8"),
        re.S,
    )
    assert stated, "CORPUS.md no longer states the 20-character figures in the known shape"
    per = script["test_20 chars"]["per_language"]
    for got, code in zip(stated.groups(), ("skr", "pnb", "ur"), strict=True):
        assert abs(float(got) - per[code] * 100) < 0.05, (
            f"CORPUS.md says {got}% for {code}, the report says {per[code] * 100:.1f}%"
        )


def _pruning_table() -> list[str]:
    """The lines of the pruning table, and nothing else.

    This guard used to take the first indented pipe line containing the cut - which was
    the pruning table's row until a second table appeared above it, comparing n_max 5 and
    6 at the same three budgets. The guard then read `| 0.03 | 120,000 | 0.9518 | 0.9525 |`
    and reported the row as carrying no percentages. "The first line that mentions the
    number" is not "the row of the table I mean", so the table is located by its own
    header.
    """
    lines = CORPUS.read_text(encoding="utf-8").splitlines()
    start = next(
        (i for i, ln in enumerate(lines) if "per-language cut" in ln and "|" in ln),
        None,
    )
    assert start is not None, "CORPUS.md has no pruning table header"
    out = []
    for line in lines[start + 1:]:
        if not line.strip().startswith("|"):
            break
        out.append(line)
    assert out, "the pruning table header is not followed by any rows"
    return out


def test_the_pruning_table_in_the_corpus_document_is_the_sweep() -> None:
    """The trade-off table is the whole argument for a 7 MB model, so it is checked.

    Its rows are what justify shipping 180,000 n-grams per language rather than 60,000,
    and a reader deciding whether to rebuild with `--top-grams 60000` is reading them.
    """
    if not SWEEP.exists():
        pytest.skip("the sweep report is not shipped with the tests")
    sweep = json.loads(SWEEP.read_text(encoding="utf-8"))
    table = _pruning_table()
    best = sweep["chosen"]
    rows = {
        r["top"]: r
        for r in sweep["sweep"]
        if r["n_max"] == best["n_max"] and r["alpha"] == best["alpha"]
    }
    for top, row in rows.items():
        line = next((ln for ln in table if f"{top:,}" in ln), None)
        assert line, f"the pruning table has no row for {top:,}"
        # Numerically, not as a string: 0.7725 * 100 lands on a rounding boundary and
        # `:.1f` gives 77.2 while the table reasonably says 77.3. A string match on a
        # half-way value fails for no reason a reader would accept as a defect.
        found = [float(x) for x in re.findall(r"(\d+\.\d)%", line)]
        assert any(abs(x - row["val_skr_20"] * 100) <= 0.05 for x in found), (
            f"the {top:,} row carries {found}, and the sweep measured Saraiki@20 at "
            f"{row['val_skr_20'] * 100:.2f}%: {line}"
        )


def test_the_pruning_table_states_the_shipped_cut_as_the_chosen_one() -> None:
    """The row in bold must be the row the wheel was built from.

    The table argues for a cut, and the model ships one. A table whose bold row is not
    the shipped setting argues for something the package does not do - which is the state
    this document was in after the sweep moved from 60,000 to 180,000.
    """
    if not SWEEP.exists():
        pytest.skip("the sweep report is not shipped with the tests")
    chosen = json.loads(SWEEP.read_text(encoding="utf-8"))["chosen"]["top"]
    bold = [ln for ln in _pruning_table() if "**" in ln]
    assert len(bold) == 1, f"{len(bold)} rows of the pruning table are bold, not 1"
    assert f"{chosen:,}" in bold[0], (
        f"the bold row is {bold[0].strip()[:40]!r} but the sweep chose {chosen:,}"
    )


def test_the_shipped_model_is_the_one_the_numbers_describe() -> None:
    """The report means nothing if the wheel carries a different model.

    `n_max`, `alpha` and the per-language gram count are all in the blob, so the three
    settings the sweep chose can be read straight back out of what ships.
    """
    import gzip

    model = ROOT / "src" / "urdunlp" / "data" / "langid.json.gz"
    with gzip.open(model, "rt", encoding="utf-8") as fh:
        blob = json.load(fh)
    if not SWEEP.exists():
        pytest.skip("the sweep report is not shipped with the tests")
    best = json.loads(SWEEP.read_text(encoding="utf-8"))["chosen"]
    assert blob["n_max"] == best["n_max"]
    assert blob["alpha"] == best["alpha"]
    biggest = max(len(v) for v in blob["counts"].values())
    assert biggest == best["top"], (
        f"the shipped model keeps {biggest} grams per language; the sweep chose {best['top']}"
    )


def test_the_shipped_models_were_built_from_the_current_inputs() -> None:
    """The guard that would have caught a model six commits stale.

    `src/urdunlp/data/*.json.gz` are build outputs committed by hand, so nothing but a
    record of their inputs keeps them honest. The Roman tagger was built on 2026-09-25;
    `urdunlp.translit.LEXICON`, which `roman_urdu_unigram` spreads word frequencies over,
    changed in six commits after that - chat spellings, acronyms, the Arabic article - and
    the wheel kept shipping a table that no longer matched the code that produces it.
    Rebuilding moved 74 word weights, added eight words and dropped eight. Every test
    passed throughout, because every test used the shipped table.

    This recomputes each model's `inputs` digest from the files and constants on disk now.
    A mismatch means: rebuild with `python scripts/build_langid_models.py`.
    """
    import gzip
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    try:
        import build_langid_models as B
    except ImportError:  # pragma: no cover
        pytest.skip("the build script is not shipped with the tests")
    from urdunlp.translit import LEXICON

    data = ROOT / "data"
    if not (data / "FINGERPRINT.tsv").exists():
        pytest.skip("the corpus fingerprint is not present")

    def shipped(name: str) -> dict:
        with gzip.open(ROOT / "src" / "urdunlp" / "data" / name, "rt", encoding="utf-8") as fh:
            return json.load(fh)

    langid = shipped("langid.json.gz")
    # The corpus name is part of the digest. Without it a model trained on data/wiki and
    # one trained on data/wiki_large carried identical digests whenever the hyperparameters
    # matched, so the check that exists to prove which inputs a committed artefact came
    # from could not tell the two corpora apart.
    assert "corpus" in langid, "the shipped model does not record which corpus it was built on"
    expected = B.inputs_digest(
        data / "FINGERPRINT.tsv", langid["n_max"], langid["alpha"],
        max(len(v) for v in langid["counts"].values()), langid["corpus"],
    )  # fmt: skip
    assert langid.get("inputs") == expected, (
        "the shipped language-ID model was not built from the corpus fingerprint and "
        "settings now on disk - rebuild it"
    )

    lexicon_file = data / "dakshina/ur/lexicons/ur.translit.sampled.train.tsv"
    counts_file = data / "vocab/dakshina_train_counts.tsv"
    if not (lexicon_file.exists() and counts_file.exists()):
        pytest.skip("the Dakshina lexicon is not present (it is fetched, not committed)")
    tagger = shipped("roman_tagger.json.gz")
    expected = B.inputs_digest(
        lexicon_file, counts_file, sorted(LEXICON.items()),
        B.TOP_WORDS, B.ORDER, B.UNIGRAM_WEIGHT, B.STAY, B.URDU_BIAS,
    )  # fmt: skip
    assert tagger.get("inputs") == expected, (
        "the shipped Roman tagger was not built from the lexicon now in translit.py - "
        "rebuild it with `python scripts/build_langid_models.py`"
    )

    # And the transliteration model, which was stale for the same reason and found by
    # looking for the same shape rather than by anything going wrong. Rebuilding it
    # moved 206 bigram weights and 313 emission contexts; every published figure was
    # unchanged to three decimal places, which is exactly why nothing noticed.
    import build_translit_model as TR

    translit = shipped("translit.json.gz")
    expected = TR.inputs_digest(
        lexicon_file, counts_file, sorted(LEXICON.items()),
        TR.PRIOR_WEIGHT, TR.MIN_COUNT, TR.KAPPA, TR.LM_WEIGHT, TR.LEXICON_BONUS,
        TR.DISCOUNT, TR.CANDIDATES,
    )  # fmt: skip
    assert translit.get("inputs") == expected, (
        "the shipped transliteration model was not built from the lexicon now in "
        "translit.py - rebuild it with `python scripts/build_translit_model.py`"
    )


def test_the_margin_table_in_the_corpus_document_is_the_measured_one() -> None:
    """docs/CORPUS.md's margin deciles come from the report, row by row.

    This passage is the one a user acts on - it answers "can I filter on margin and keep
    only the answers that are right" - and it has now been wrong three times. The first
    two were figures written from one-off scripts. The third was subtler and is why the
    bands moved: the paragraph argued from a fixed 0.1 threshold, chosen when the model
    was built on 1,500 paragraphs per language and its margins were fractions. On the
    corpus eleven times larger the same quantity runs to 41, so 0.1 became the 0.6th
    percentile and "288 of the 312 errors are above it" reduced to "99.3% of everything
    is above it". Deciles cannot go stale that way: their edges are the data's.
    """
    twenty = _script()["test_20 chars"]
    deciles = twenty["accuracy_by_margin_decile"]
    corpus = CORPUS.read_text(encoding="utf-8")
    first = deciles[0]
    for figure in (
        f"carries {first['errors']} of the {twenty['errors']} errors",
        f"{first['accuracy'] * 100:.1f}%",
        f"{first['n']:,}",
    ):
        assert figure in corpus, (
            f"CORPUS.md does not state {figure!r}, which is what the report measures"
        )
    # The retention figure the paragraph quotes, and the accuracy of what is kept.
    keep_90 = next(r for r in twenty["error_retention_by_margin"] if r["keep"] == 0.9)
    assert f"{keep_90['errors_kept_share'] * 100:.1f}% of them" in corpus
    assert f"right {keep_90['accuracy'] * 100:.1f}% of the time" in corpus


def test_margin_separates_right_from_wrong_which_is_the_documented_point() -> None:
    """The claim itself, in units that cannot drift: shares of the error count.

    Expressed as "errors above 0.1", this test passed while the claim it guarded became
    false, because the threshold was in the units that had moved. A decile is defined by
    the distribution, so the same sentence stays measurable whatever the scale does.

    If a future model made margin uninformative, this fails and the paragraph needs
    rewriting rather than its figures nudging - which is the outcome worth being told
    about.
    """
    for key in ("test_whole", "test_20 chars", "val_whole", "val_20 chars"):
        row = _script()[key]
        deciles = row["accuracy_by_margin_decile"]
        assert len(deciles) == 10, f"{key}: {len(deciles)} deciles, not 10"
        lowest = deciles[0]
        share = lowest["errors"] / row["errors"]
        assert share > 0.5, (
            f"{key}: the lowest margin decile holds only {share:.0%} of the errors, so "
            f"margin no longer sorts them and docs/CORPUS.md overstates what it buys"
        )
        # Monotone enough to act on: the top half must be better than the bottom tenth.
        top_half = deciles[5:]
        assert min(r["accuracy"] for r in top_half) > lowest["accuracy"], key


def test_dropping_the_lowest_margins_drops_more_errors_than_answers() -> None:
    """A filter is only worth offering if it removes errors faster than data.

    The deciles can look informative and the filter still be useless, if the errors are
    spread thinly - so the retention curve is measured separately and checked here.
    """
    for key in ("test_20 chars", "val_20 chars"):
        row = _script()[key]
        for kept in row["error_retention_by_margin"]:
            dropped_answers = 1 - kept["keep"]
            dropped_errors = 1 - kept["errors_kept_share"]
            assert dropped_errors > dropped_answers, (
                f"{key}: keeping the top {kept['keep']:.0%} by margin drops "
                f"{dropped_errors:.1%} of the errors for {dropped_answers:.0%} of the "
                f"answers, so filtering on margin costs more than it saves"
            )


def test_the_librarys_own_accuracy_figures_are_the_measured_ones() -> None:
    """The docstring a user reads with help(), not only the README.

    The model was rebuilt on a corpus eleven times larger, the README was updated, and
    src/urdunlp/langid.py went on quoting 97.9% / 96.6% / 91.0% / 81.5% by window length
    against a measured 99.1 / 98.2 / 94.9 / 88.7 - in two sentences, one of them the
    docstring of the public return type. The existing guard asserted only that the stale
    headline was absent from the README, so neither was covered.

    Both sentences are required. The first version of this check found one of the two and
    passed: `assert sentences` is satisfied by a single match, so a pattern that silently
    missed the other place reported success over it - a checker that skips part of its
    input and says nothing is how the stale figures survived the last rewrite.
    """
    import re

    script = _script()
    measured = [
        round(script[f"test_{k}"]["accuracy"] * 100, 1)
        for k in ("whole", "50 chars", "20 chars", "10 chars")
    ]
    source = (ROOT / "src" / "urdunlp" / "langid.py").read_text(encoding="utf-8")
    # Both sentences wrap, one inside a comment and one inside a docstring, so the line
    # breaks and their leading markers are collapsed before matching.
    flat = re.sub(r"\s*\n\s*(?:#\s*)?", " ", source)
    patterns = {
        "the LanguageGuess docstring": (
            r"right ([\d.]+)% of the time; on 50 characters ([\d.]+)%, "
            r"on 20 ([\d.]+)%, on ten ([\d.]+)%"
        ),
        "the SHORT_TEXT comment": (
            r"([\d.]+)% on a paragraph, ([\d.]+)% on 50 characters, "
            r"([\d.]+)% on 20, ([\d.]+)% on 10"
        ),
    }
    for where, pattern in patterns.items():
        found = re.findall(pattern, flat)
        assert len(found) == 1, (
            f"the accuracy-by-length sentence in {where} was not found exactly once "
            f"({len(found)} matches) - if the wording changed, update the pattern rather "
            f"than letting the other match carry the check"
        )
        assert [float(x) for x in found[0]] == measured, (
            f"{where} quotes {found[0]} by window length; the report measures {measured}"
        )


def test_the_builders_defaults_are_the_setting_that_shipped() -> None:
    """A plain `python scripts/build_langid_models.py` must rebuild what the wheel holds.

    The defaults were left at n_max 5 / 60,000 grams after the sweep chose n_max 6 /
    180,000, so a default run would have replaced the shipped model with a different and
    worse one - and every digest check would still have passed, because the digest is of
    the inputs a run was given rather than of the settings the sweep chose.
    """
    if not SWEEP.exists():
        pytest.skip("the sweep report is not shipped with the tests")
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import build_langid_models as builder

    chosen = json.loads(SWEEP.read_text(encoding="utf-8"))["chosen"]
    assert builder.N_MAX == chosen["n_max"], (
        f"the builder defaults to n_max {builder.N_MAX}, the sweep chose {chosen['n_max']}"
    )
    assert builder.ALPHA == chosen["alpha"]
    assert builder.TOP_GRAMS == chosen["top"], (
        f"the builder defaults to {builder.TOP_GRAMS:,} grams, the sweep chose "
        f"{chosen['top']:,} - a default build would replace the shipped model"
    )


def test_the_shipped_model_records_those_same_settings() -> None:
    """The other half: what is in the wheel must be what the defaults would rebuild."""
    import gzip

    blob = ROOT / "src" / "urdunlp" / "data" / "langid.json.gz"
    with gzip.open(blob, "rt", encoding="utf-8") as fh:
        model = json.load(fh)
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import build_langid_models as builder

    assert model["n_max"] == builder.N_MAX
    assert model["alpha"] == builder.ALPHA
    assert model["top_grams"] == builder.TOP_GRAMS, (
        "the shipped model was built at a different gram budget from the current default"
    )


def test_the_readme_states_the_real_bundled_size() -> None:
    """The wheel's data is what every user downloads, so the figure has to be current.

    It read 5.0 MB while the directory held 10.0 MB: the language model went from 2.56 MB
    to 7.25 MB when the gram budget moved, and the one line a user checks before adding a
    dependency was left behind. Rounded to one decimal, which is what the line quotes.
    """
    data = ROOT / "src" / "urdunlp"
    # What the wheel carries: __pycache__ is 0.62 MB here and is not shipped, so a total
    # over every file on disk would make the figure depend on whether the tests had run.
    total = sum(
        f.stat().st_size
        for f in data.rglob("*")
        if f.is_file() and "__pycache__" not in f.parts
    )
    stated = re.search(r"Bundled statistical tables \(([\d.]+) MB in the wheel\)",
                       README.read_text(encoding="utf-8"))
    assert stated, "the README no longer states the bundled size in the known shape"
    assert float(stated.group(1)) == pytest.approx(round(total / 1e6, 1), abs=0.15), (
        f"the README says {stated.group(1)} MB, src/urdunlp holds "
        f"{total / 1e6:.1f} MB"
    )
