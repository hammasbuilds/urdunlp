"""The two split sets differ in exactly one thing, and the real test rows are shared.

`research/make_splits.py` builds two corpora that are meant to be identical except that
one contains synthetic Kashmiri and the other does not. The whole synthetic experiment is
a comparison of two models trained on those two corpora and scored on the **same real
held-out rows**, so if the real rows differ between the two, the comparison measures
nothing and looks perfectly well formed while doing it.

A first version used one `random.Random(SEED)` walked across the whole run. Kashmiri's
synthetic sources consume extra `shuffle` calls, so in the `--no-synthetic` run every
language processed *after* Kashmiri drew from a different point in the stream. ur, pnb,
skr, sd, ps and ks came out identical; fa, ar, ckb, ug and azb did not - with identical
row counts, so nothing downstream would have noticed. Seeding per (language, source)
fixes it, and this pins the property rather than the fix.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WITH = ROOT / "data" / "splits"
WITHOUT = ROOT / "data" / "splits_no_synthetic"
LANGUAGES = ["ur", "pnb", "skr", "sd", "ps", "ks", "fa", "ar", "ckb", "ug", "azb"]

pytestmark = pytest.mark.skipif(
    not (WITH / "test").is_dir() or not (WITHOUT / "test").is_dir(),
    reason="the splits are built by research/make_splits.py and are not in the repository",
)


def rows(base: Path, part: str, language: str) -> list[tuple[str, str, str]]:
    """(kind, source, text) for one split file, read through its sidecar."""
    text_path = base / part / f"{language}.txt"
    src_path = base / part / f"{language}.src"
    if not text_path.exists():
        return []
    texts = text_path.read_text(encoding="utf-8", errors="replace").splitlines()
    tags = src_path.read_text(encoding="utf-8").splitlines()
    assert len(texts) == len(tags), (
        f"{base.name}/{part}/{language}: {len(texts)} lines but {len(tags)} source tags. "
        f"Every per-source accuracy computed from this file would attribute rows to the "
        f"wrong source."
    )
    out = []
    for text, tag in zip(texts, tags, strict=True):
        kind, _, source = tag.partition("\t")
        out.append((kind, source, text))
    return out


@pytest.mark.parametrize("language", LANGUAGES)
def test_the_real_test_rows_are_identical_across_both_split_sets(language):
    """The property the whole synthetic experiment rests on."""
    a = [r for r in rows(WITH, "test", language) if r[0] != "synthetic"]
    b = [r for r in rows(WITHOUT, "test", language) if r[0] != "synthetic"]
    assert a and b, f"{language}: one of the split sets has no non-synthetic test rows"
    assert len(a) == len(b), (
        f"{language}: {len(a)} non-synthetic test rows with synthetic, {len(b)} without"
    )
    assert a == b, (
        f"{language}: the two split sets hold the same number of non-synthetic test "
        f"rows ({len(a)}) but not the same rows. A model comparison across these two "
        f"corpora would be scored on different text."
    )


@pytest.mark.parametrize("language", LANGUAGES)
def test_the_real_training_rows_are_identical_too(language):
    """Otherwise the two models differ by more than the synthetic data."""
    a = [r for r in rows(WITH, "train", language) if r[0] != "synthetic"]
    b = [r for r in rows(WITHOUT, "train", language) if r[0] != "synthetic"]
    assert a == b, (
        f"{language}: the non-synthetic training rows differ between the two split "
        f"sets, so the two models differ by more than the one thing under test"
    )


def test_only_kashmiri_has_synthetic_rows():
    """Stated so that a later language gaining synthetic data cannot pass unnoticed."""
    with_synth = {
        language
        for language in LANGUAGES
        for part in ("train", "val", "test")
        if any(r[0] == "synthetic" for r in rows(WITH, part, language))
    }
    assert with_synth == {"ks"}, (
        f"synthetic rows appear for {sorted(with_synth)}; the experiment and the paper "
        f"both describe Kashmiri as the only language that needed them"
    )
    without = {
        language
        for language in LANGUAGES
        for part in ("train", "val", "test")
        if any(r[0] == "synthetic" for r in rows(WITHOUT, part, language))
    }
    assert without == set(), f"the control corpus contains synthetic rows for {without}"


@pytest.mark.parametrize("language", LANGUAGES)
def test_every_source_reaches_every_split(language):
    """The design rule the Wikipedia-only corpus broke.

    A source present only in training hides a domain the model cannot handle; a source
    present only in test judges the model on a register it never saw. The 0.9925
    in-domain score sat next to 0.2565 out of domain precisely because train and test
    came from one domain, and no number in that setup could have said so.
    """
    per_part = {
        part: {r[1] for r in rows(WITH, part, language)} for part in ("train", "val", "test")
    }
    assert per_part["train"], f"{language}: no training rows"
    assert per_part["train"] == per_part["val"] == per_part["test"], (
        f"{language}: sources differ across splits.\n"
        f"  train only: {sorted(per_part['train'] - per_part['test'])}\n"
        f"  test only : {sorted(per_part['test'] - per_part['train'])}"
    )


@pytest.mark.parametrize("language", LANGUAGES)
def test_no_test_row_appears_in_training(language):
    """Leakage would inflate every number the paper reports."""
    train = {r[2] for r in rows(WITH, "train", language)}
    test = [r[2] for r in rows(WITH, "test", language)]
    leaked = [t for t in test if t in train]
    assert not leaked, (
        f"{language}: {len(leaked)} of {len(test)} test rows also appear in training. "
        f"First: {leaked[0][:80]!r}"
    )


@pytest.mark.parametrize("language", LANGUAGES)
def test_the_corpus_clears_the_hundred_thousand_floor(language):
    """Every language, counting all three kinds, as the corpus was specified."""
    total = sum(len(rows(WITH, part, language)) for part in ("train", "val", "test"))
    assert total >= 100_000, f"{language}: {total:,} rows, below the 100,000 floor"
