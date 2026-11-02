"""docs/CORPUS.md sections 1-6 against docs/corpus-measured.json.

The corpus numbers were written into prose and nothing held them to anything. The corpus
itself - XL-Sum Urdu, 200 MB of text - is not in the repository, so there was no artefact
to hold them to either, and they quietly went stale: `normalize()` gained a letter and
control-character stripping, and the stopword list grew from 115 entries to 159, while the
document went on reporting what the package used to do. The README had been updated to 159
and this document had not.

Re-measuring on the same corpus separated the two halves cleanly. Everything that depends
only on the input - articles, characters, every count of a substituted Arabic letter - came
back byte-identical, which is what rules the corpus out as the cause. Everything that
depends on the package moved.

`docs/corpus-measured.json` is now committed as the reduction of that corpus, and this file
checks that the document still says what it says. The measurement itself is reproduced with

    python scripts/measure_corpus.py <corpus> --json docs/corpus-measured.json

which takes about thirteen minutes over 84,581 articles.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "CORPUS.md"
MEASURED = ROOT / "docs" / "corpus-measured.json"


def _doc() -> str:
    if not DOC.is_file():
        pytest.skip("docs/CORPUS.md is not shipped with the tests")
    return DOC.read_text(encoding="utf-8")


NOTE = re.compile(r"\*\*These numbers were re-measured.*?46\.5%\.", re.DOTALL)


def _body() -> str:
    """The document without its provenance note.

    The note explains the 2026-11-02 refresh and so quotes both the old values and the new
    ones. Checking the published numbers against the whole document therefore passes on the
    note alone: when this file was first written, reverting section 3 to its stale table
    still satisfied the coverage check, because "46.5%" survived in the note three
    paragraphs above. A test that the document *mentions* a number somewhere is not a test
    that the document *reports* it.
    """
    doc = _doc()
    note = NOTE.search(doc)
    assert note is not None, "the provenance note is gone; was the document rewritten?"
    return doc.replace(note.group(0), "")


def _measured() -> dict:
    if not MEASURED.is_file():
        pytest.skip("docs/corpus-measured.json is not present")
    return json.loads(MEASURED.read_text(encoding="utf-8"))


# key -> how the document prints it
COUNTS = (
    "documents",
    "characters",
    "tokens",
    "raw_whitespace_tokens",
    "distinct_normalised_types",
    "types_with_more_than_one_raw_spelling",
    "tokens_changed_by_normalize",
    "tokens_moved_onto_a_majority_spelling",
    "documents_with_substituted_arabic_letters",
    "stopword_token_hits",
)


@pytest.mark.parametrize("key", COUNTS)
def test_the_document_prints_the_measured_count(key: str):
    assert f"{_measured()[key]:,}" in _body(), (
        f"docs/CORPUS.md does not contain {_measured()[key]:,} for {key}; "
        f"re-run scripts/measure_corpus.py and update the document"
    )


@pytest.mark.parametrize(
    "key,places,label",
    [
        ("normalize_change_rate", 2, "share of tokens normalize() changes"),
        ("stopword_coverage", 1, "stopword coverage of running text"),
        ("arabic_document_rate", 1, "share of articles with a substituted letter"),
    ],
)
def test_the_document_prints_the_measured_rate(key: str, places: int, label: str):
    printed = f"{_measured()[key] * 100:.{places}f}%"
    assert printed in _body(), f"docs/CORPUS.md does not report {printed} for {label}"


def test_the_stopword_list_the_document_describes_is_the_one_that_ships():
    """The defect this file was written for: the document said 115, the package ships 159."""
    from urdunlp import STOPWORDS

    measured = _measured()
    assert measured["stopword_list_size"] == len(STOPWORDS), (
        f"docs/corpus-measured.json was measured against {measured['stopword_list_size']} "
        f"stopwords and the package now ships {len(STOPWORDS)}; re-run the measurement"
    )
    doc = _body()
    assert f"## 3. {len(STOPWORDS)} stopwords cover " in doc, (
        f"section 3 does not say {len(STOPWORDS)} stopwords"
    )
    assert f"**{measured['stopword_entries_seen']} of {len(STOPWORDS)}**" in doc, (
        f"section 3 does not say {measured['stopword_entries_seen']} of {len(STOPWORDS)}"
    )


def test_every_substituted_arabic_letter_count_is_in_the_document():
    doc = _body()
    counts = _measured()["arabic_char_counts"]
    assert counts, "the measurement recorded no Arabic substitutions at all"
    for label, count in counts.items():
        assert f"{count:,}" in doc, (label, count)


def test_the_ten_most_frequent_types_are_the_measured_ones():
    """Including their counts, which shift whenever normalisation changes."""
    doc = _body()
    for form, count in _measured()["top_30_types"][:10]:
        assert form in doc, form
        assert f"{count:,}" in doc, (form, count)


def test_the_document_does_not_still_carry_the_numbers_that_went_stale():
    """Named explicitly, because a refresh that misses one leaves the old claim standing.

    These are the published values from before `normalize()` and the stopword list
    changed. Each one is now wrong, and a document containing any of them has been
    partially updated - which is harder to spot than one that was never updated at all.
    """
    # The provenance note quotes the old values on purpose; `_body` drops it.
    body = _body()
    for stale in ("263,833", "14,611", "210,906", "18,492,484", "44,698,779", "353,358"):
        assert stale not in body, f"docs/CORPUS.md still contains the pre-refresh value {stale}"
    assert "## 3. 115 stopwords" not in body


def test_the_measurement_names_its_corpus():
    """An artefact with no provenance is a number nobody can challenge."""
    what = _measured().get("_what", "")
    assert "XL-Sum" in what and "84,581" in what, what
