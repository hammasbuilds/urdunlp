"""Section numbers are unique, and references to them resolve.

`docs/CORPUS.md` had **two sections numbered 14**: `## 14. Urdu -> Roman, scored against
people`, and then `# 14. The multi-source research corpus`, a top-level part appended later
that reused the number and carried subsections 14.1 to 14.5. The paper referred to
"`docs/CORPUS.md` section 14", meaning the second one, which a reader following the
reference would not have found - they would have landed on transliteration scoring. The
second part is now 15, and its subsections 15.1 to 15.5.

Two checks, because the defect needed both to be caught: a number used twice, and a
reference pointing at a number that does not exist. Neither is visible while reading, which
is why they are mechanical.

A note on matching. A reference to a whole section ("section 4") resolves when 4.x exist
even if a bare "4." heading does not, because that is a reference to the part rather than
to a numbered heading. And a document may mark subsections in bold rather than with a
heading level - blast-radius's manuscript does - so a bold `**4.5 Title**` counts as
defining 4.5 here too; a checker that only read `#` headings reported ten references in
that paper as dangling, every one of them fine.
"""

from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

# `## 4.5 Heading`, `# 15. Heading`
HEADING = re.compile(r"^#{1,6}[ \t]+(\d+(?:\.\d+)*)\.?[ \t]+\S", re.MULTILINE)
# `**4.5 Heading.**` at the start of a line
BOLD_HEADING = re.compile(r"^\*\*(\d+(?:\.\d+)*)\.?[ \t]+\S", re.MULTILINE)
# `section 4.5`, `sections 4.5 and`, `§4.5`
REFERENCE = re.compile(r"(?:§|\bsections?[ \t]+)(\d+(?:\.\d+)*)", re.IGNORECASE)

DOCUMENTS = ("docs/CORPUS.md", "docs/DEVELOPMENT.md", "paper/main.md", "README.md")


def _text(name: str) -> str | None:
    path = ROOT / name
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _defined(text: str) -> set[str]:
    return set(HEADING.findall(text)) | set(BOLD_HEADING.findall(text))


def _paragraph_around(text: str, position: int) -> str:
    """The blank-line-delimited paragraph containing `position`."""
    start = text.rfind("\n\n", 0, position)
    start = 0 if start == -1 else start + 2
    end = text.find("\n\n", position)
    end = len(text) if end == -1 else end
    return text[start:end]


def _resolves(target: str, defined: set[str]) -> bool:
    if target in defined:
        return True
    # "section 15" when only 15.1.. exist: a reference to the part.
    return any(number.startswith(target + ".") for number in defined)


@pytest.mark.parametrize("name", DOCUMENTS)
def test_no_section_number_is_used_twice(name: str):
    text = _text(name)
    if text is None:
        pytest.skip(f"{name} is not shipped with the tests")
    numbers = HEADING.findall(text)
    seen: dict[str, int] = {}
    for number in numbers:
        seen[number] = seen.get(number, 0) + 1
    repeated = sorted(n for n, count in seen.items() if count > 1)
    assert not repeated, (
        f"{name} numbers these sections more than once: {repeated}. A reference to one of "
        f"them is ambiguous, and a reader following it lands on whichever comes first."
    )


def test_the_corpus_document_still_has_the_two_sections_the_defect_confused():
    """Named, so a future renumbering cannot quietly merge them again."""
    text = _text("docs/CORPUS.md")
    if text is None:
        pytest.skip("docs/CORPUS.md is not shipped with the tests")
    assert "## 14. Urdu → Roman, scored against people" in text
    assert "# 15. The multi-source research corpus" in text
    for n in range(1, 6):
        assert f"## 15.{n} " in text, f"subsection 15.{n} is missing"
    # As a heading, not as a bare string: "14.1" could legitimately appear as a
    # measurement, and a test that forbids the characters would break on one.
    assert not re.search(r"^#{1,6}[ \t]+14\.\d", text, re.MULTILINE), (
        "a 14.x subsection heading survived the renumbering"
    )


def test_every_reference_into_the_corpus_document_resolves():
    """The paper points into `docs/CORPUS.md`, which is where the renumbering bit.

    The four references at issue - section 15 and 15.2, 15.3 twice - all pointed at the
    multi-source corpus and all read 14 before the fix, which resolved to the wrong
    section rather than to none, so a dangling-reference check alone would not have caught
    it. Hence the test above as well.
    """
    paper = _text("paper/main.md")
    corpus = _text("docs/CORPUS.md")
    if paper is None or corpus is None:
        pytest.skip("the paper or the corpus document is not shipped with the tests")
    defined = _defined(corpus)
    # Only the references the paper itself marks as being into that document.
    pattern = re.compile(
        r"§(\d+(?:\.\d+)*)(?=[^.]{0,40}`docs/CORPUS\.md`)|`docs/CORPUS\.md`\s*§(\d+(?:\.\d+)*)"
    )
    targets = {a or b for a, b in pattern.findall(paper)}
    assert targets, "the paper no longer references docs/CORPUS.md by section"
    unresolved = sorted(t for t in targets if not _resolves(t, defined))
    assert not unresolved, (
        f"paper/main.md points at docs/CORPUS.md {unresolved}, which do not exist"
    )


@pytest.mark.parametrize("name", DOCUMENTS)
def test_every_internal_reference_resolves(name: str):
    text = _text(name)
    if text is None:
        pytest.skip(f"{name} is not shipped with the tests")
    defined = _defined(text)
    if not defined:
        pytest.skip(f"{name} has no numbered sections")
    unresolved = set()
    for match in REFERENCE.finditer(text):
        target = match.group(1)
        # A reference into another document is the other test's business. Judged by the
        # enclosing paragraph, not by a window after the reference: the paper writes
        # "`docs/CORPUS.md` §15", with the filename *before* it, and the two references
        # that follow in the next sentences name no file of their own at all. A paragraph
        # that names another document is talking about that document.
        if ".md" in _paragraph_around(text, match.start()):
            continue
        if not _resolves(target, defined):
            unresolved.add(target)
    assert not unresolved, f"{name} refers to sections that do not exist: {sorted(unresolved)}"


def test_the_matcher_finds_both_heading_styles():
    """Otherwise every reference resolves vacuously, or none does."""
    assert _defined("## 4.5 Title\n") == {"4.5"}
    assert _defined("**4.5 Title.** body\n") == {"4.5"}
    assert _defined("# 15. Title\n") == {"15"}
    assert _defined("not a heading 4.5\n") == set()
    # A part reference resolves against its subsections, and a missing one does not.
    assert _resolves("15", {"15.1", "15.2"})
    assert not _resolves("16", {"15.1", "15.2"})
    assert not _resolves("1", {"15.1"}), "a prefix match must not count as a part"
