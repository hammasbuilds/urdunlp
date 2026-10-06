"""Every example in the README returns what the README says it returns.

Each line of a ```python block written as `expression  # result`, where the result
is a Python literal, is evaluated and compared. The README has shown wrong output
before (0.1 promised علی where the code gave الی); this makes that a test failure.

A line that is a plain statement - `r = transliterate_with_confidence(...)` - is run
first, in the order the block gives it, so an example can use the name it binds. The
first version evaluated each example on its own and failed on `r.text` with
NameError: the README was right, the harness was not.

Skipped where the README is not present, as in the installed-wheel CI job.
"""

import ast
import pathlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _examples() -> list[tuple[str, str, tuple[str, ...]]]:
    """(expression, expected literal, statements to run before it) per example."""
    if not README.exists():
        return []
    examples = []
    for block in BLOCK.findall(README.read_text(encoding="utf-8")):
        setup: list[str] = []
        for line in block.splitlines():
            code, sep, comment = line.partition("  # ")
            if not code.strip() or code.startswith(("from ", "import ", " ")):
                continue
            if not sep:
                try:
                    tree = ast.parse(code)
                except SyntaxError:
                    continue
                if all(isinstance(node, ast.Assign) for node in tree.body):
                    setup.append(code)
                continue
            # the literal is the comment up to any prose after it: `True: one word...`
            literal = re.split(r"(?<=[\]')0-9e]):| - ", comment.strip(), maxsplit=1)[0]
            try:
                ast.literal_eval(literal)
            except (ValueError, SyntaxError):
                continue
            examples.append((code.strip(), literal, tuple(setup)))
    return examples


EXAMPLES = _examples()


@pytest.mark.skipif(not README.exists(), reason="README.md is not shipped with the tests")
def test_the_readme_has_examples_to_check():
    assert len(EXAMPLES) >= 20


@pytest.mark.parametrize(
    ("code", "expected", "setup"), EXAMPLES, ids=[code for code, _, _ in EXAMPLES]
)
def test_readme_example(code, expected, setup):
    import urdunlp

    namespace = {name: getattr(urdunlp, name) for name in urdunlp.__all__}
    for statement in setup:
        exec(statement, namespace)
    assert eval(code, namespace) == ast.literal_eval(expected)


@pytest.mark.skipif(not README.exists(), reason="README.md is not shipped with the tests")
def test_the_readme_has_no_unfilled_placeholders():
    """The README is the PyPI page. A draft once shipped 34 `@@RU_TEST@@`-style
    placeholders where the accuracy figures, the demo output and the test count
    belonged; twine renders them without complaint."""
    text = README.read_text(encoding="utf-8")
    assert re.findall(r"@@\w*@@", text) == []
    assert "TODO" not in text and "TBD" not in text


DEMO = README.parent / "demo.py"


def _readme_block_after(marker: str) -> str:
    text = README.read_text(encoding="utf-8")
    return text.split(marker, 1)[1].split("```", 2)[1].strip("\n")


@pytest.mark.skipif(not DEMO.exists(), reason="demo.py is not shipped with the tests")
def test_the_readme_input_is_the_demo_input():
    """The README's Input block once said "Arabic ک and ی" while showing the Urdu
    letters; demo.py had the Arabic ones. Both must be the same string."""
    raw = re.search(r'^RAW = "(.*)"$', DEMO.read_text(encoding="utf-8"), re.M).group(1)
    assert _readme_block_after("## Input") == raw
    assert "ك" in raw and "ي" in raw  # Arabic kaf and yeh, as the README says


@pytest.mark.skipif(not DEMO.exists(), reason="demo.py is not shipped with the tests")
def test_the_readme_output_is_what_demo_py_prints():
    import os
    import subprocess
    import sys

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    done = subprocess.run(
        [sys.executable, str(DEMO)], capture_output=True, env=env, timeout=300, check=True
    )
    printed = done.stdout.decode("utf-8").replace("\r\n", "\n")
    shown = _readme_block_after("`python demo.py`")
    assert [line.rstrip() for line in shown.splitlines()] == [
        line.rstrip() for line in printed.strip("\n").splitlines()
    ]


def test_known_limits_examples_still_hold() -> None:
    """The "Known limits" block must describe the code as it behaves now.

    This block was not covered by any test and went stale in the direction that makes
    the library look worse than it is: it claimed `exam` gave اقسام and `late` gave
    لاتے, when both are now correct, and it blamed `hal` for an error that is actually
    `hai` being pulled to ہی by the word after it. Documented limitations are claims
    like any other, so they are pinned here.
    """
    import urdunlp

    # the homophone pulled by the FOLLOWING word, not by how `hal` is spelled
    assert urdunlp.transliterate_to_urdu("kya hal hai") == "کیا حال ہے"
    assert urdunlp.transliterate_to_urdu("kya hal hai bhai") == "کیا حال ہی بھائی"
    assert urdunlp.transliterate_to_urdu("hal") == urdunlp.transliterate_to_urdu("haal")

    # the English-word example, and the escape hatch the block offers
    assert urdunlp.transliterate_to_urdu("station") == "اسٹیشن"
    assert urdunlp.transliterate_to_urdu("cancel") == "کونسل"
    assert urdunlp.transliterate_to_urdu("cancel", keep_english=True) == "cancel"

    # the two examples the block used to cite are fixed, and must not regress back
    assert urdunlp.transliterate_to_urdu("exam") == "ایگزام"
    assert urdunlp.transliterate_to_urdu("late") == "لیٹ"


def test_the_per_language_table_agrees_with_the_corpus_document() -> None:
    """The README's per-language figures and CORPUS.md's prose are the same numbers.

    They were published in one pooled figure (97.9%) that mixes an 8-way orthography
    question - eight languages at exactly 100% on whole paragraphs, over 63-80 test
    paragraphs each - with the Urdu/Punjabi/Saraiki three-way, which is 96.2%. Breaking it
    out means two documents now carry the same figures, so they are checked against each
    other: CORPUS.md states them in prose, the README in a table.

    The first draft of that table took the FIRST n characters of each paragraph instead of
    the centred slice `scripts/measure_langid.py` uses, which scored Saraiki 64.2% at 20
    characters against the published 75.1%. A table that contradicts the number two rows
    above it is the defect this test exists to prevent.
    """
    corpus = (ROOT / "docs" / "CORPUS.md").read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")

    # CORPUS.md: "At 20 characters, Saraiki is right 75.1% ... Punjabi 85.7% and Urdu 93.3%"
    # Whitespace-tolerant: the sentence wraps across a newline between "Saraiki" and "is
    # right", and a regex with literal spaces silently matched nothing.
    stated = re.search(
        r"At\s+20\s+characters,\s+Saraiki\s+is\s+right\s+([\d.]+)%[^.]*?,"
        r"\s+Punjabi\s+([\d.]+)%\s+and\s+Urdu\s+([\d.]+)%",
        corpus,
        re.S,
    )
    assert stated, "CORPUS.md no longer states the 20-character figures in the known shape"
    skr, pnb, urd = (float(g) for g in stated.groups())

    rows = {}
    for line in readme.splitlines():
        match = re.match(
            r"\|\s*(Urdu|Punjabi \(Shahmukhi\)|Saraiki)\s*\|\s*(\d+)\s*\|"
            r"([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|",
            line,
        )
        if match:
            cells = [c.replace("*", "").strip().rstrip("%") for c in match.groups()[2:]]
            rows[match.group(1)] = float(cells[2])  # the 20-character column

    assert len(rows) == 3, f"expected three language rows in the README, found {sorted(rows)}"
    assert rows["Saraiki"] == skr, f"README {rows['Saraiki']}% vs CORPUS.md {skr}%"
    assert rows["Punjabi (Shahmukhi)"] == pnb, f"README {rows['Punjabi (Shahmukhi)']}% vs {pnb}%"
    assert rows["Urdu"] == urd, f"README {rows['Urdu']}% vs CORPUS.md {urd}%"


def test_the_word_accuracy_row_states_its_denominator() -> None:
    """91.3% is over the sentences that can be ALIGNED, not over the test set.

    CORPUS.md: "A sentence is scored word by word when its Urdu and Roman token counts
    agree (3,632 of 4,945 test sentences); the rest cannot be aligned without guessing."
    The README's table quoted 52,087 words and did not say that 1,313 sentences - 26.6% -
    are excluded, and the exclusion is not random: counts disagree where compounds merge
    or split and where English is inserted, which is where transliteration is hardest.
    """
    corpus = (ROOT / "docs" / "CORPUS.md").read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    stated = re.search(r"\((\d[\d,]*)\s+of\s+(\d[\d,]*)\s+test\s+sentences\)", corpus)
    assert stated, "CORPUS.md no longer states the alignable-sentence counts"
    scored, total = (g.replace(",", "") for g in stated.groups())

    row = next((ln for ln in readme.splitlines() if "word accuracy" in ln), None)
    assert row, "the README has no word-accuracy row"
    assert f"{int(scored):,}" in row and f"{int(total):,}" in row, (
        f"the word-accuracy row does not state its denominator "
        f"({int(scored):,} of {int(total):,}): {row}"
    )


def test_the_corpus_fingerprint_is_committed_and_complete() -> None:
    """An ignored corpus has to be checkable, or its numbers describe nothing.

    `data/` is gitignored, and the language-ID corpus is an unseeded `generator=random`
    draw - so the published 97.9% rested on files no reader could see and nobody could
    rebuild. The fingerprint cannot rebuild a corpus, but it can say whether the one on
    disk is the one measured, which could not be asked before.
    """
    fingerprint = ROOT / "data" / "FINGERPRINT.tsv"
    if not fingerprint.is_file():
        pytest.skip("data/ is not present, as in the installed-wheel CI job")
    rows = [
        line
        for line in fingerprint.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#") and not line.startswith("path\t")
    ]
    assert len(rows) >= 20, f"only {len(rows)} corpus files fingerprinted"
    for row in rows:
        path, digest, lines, characters = row.split("\t")
        assert len(digest) == 64, f"{path}: not a sha256"
        assert int(lines) > 0 and int(characters) > 0, f"{path}: empty"
    # The eleven language files the langid model is built from must all be in it.
    from urdunlp.langid import LANGUAGES

    listed = {row.split("\t")[0] for row in rows}
    for code in LANGUAGES:
        assert f"wiki/{code}.txt" in listed, f"wiki/{code}.txt is not fingerprinted"


def test_the_wikipedia_fetcher_can_rebuild_from_recorded_ids() -> None:
    """The draw is the server's, so page ids are the only route to reproducibility.

    Asserts the capability exists and is wired to the documented flag - not a live fetch,
    which would need the network.
    """
    script = (ROOT / "scripts" / "fetch_wikipedia_samples.py").read_text(encoding="utf-8")
    assert "--from-pages" in script, "the rebuild flag is gone"
    assert "def pages_by_id" in script, "fetching by article id is gone"
    assert "pages.tsv" in script, "article ids are no longer recorded"
    corpus_doc = (ROOT / "docs" / "CORPUS.md").read_text(encoding="utf-8")
    assert "--from-pages" in corpus_doc, "CORPUS.md does not document the rebuild route"


def test_the_python_classifiers_are_the_versions_ci_tests() -> None:
    """A version classifier is a claim pip acts on, so it has to be tested.

    These drifted in opposite directions across sibling repositories: one claimed 3.14
    without testing it, the other tested 3.14 without claiming it. Asserting the two sets
    are equal catches both, and catches the next one.
    """
    root = pathlib.Path(__file__).resolve().parent.parent
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    claimed = set(re.findall(r"Programming Language :: Python :: (3\.\d+)", pyproject))
    assert claimed, "no Python version classifiers at all"

    workflows = sorted((root / ".github" / "workflows").glob("*.yml"))
    assert workflows, "no CI workflows to compare against"
    tested: set[str] = set()
    for workflow in workflows:
        for line in workflow.read_text(encoding="utf-8").splitlines():
            if "python:" in line and "[" in line:
                tested |= set(re.findall(r"3\.\d+", line))
    assert tested, "no python matrix found in any workflow"

    assert claimed == tested, (
        f"classifiers claim {sorted(claimed)} but CI tests {sorted(tested)} - "
        f"untested: {sorted(claimed - tested)}, unclaimed: {sorted(tested - claimed)}"
    )

    # And the floor must be the lowest version actually tested.
    floor = re.search(r'requires-python\s*=\s*"[^0-9]*(\d+\.\d+)"', pyproject)
    assert floor, "no requires-python"
    assert floor.group(1) == min(tested, key=lambda v: tuple(map(int, v.split(".")))), (
        f"requires-python says {floor.group(1)} but the lowest version CI tests is "
        f"{min(tested, key=lambda v: tuple(map(int, v.split('.'))))}"
    )


def test_the_tagger_rate_is_published_with_its_lexicon_restricted_twin() -> None:
    """3.8% reads as the tagger's error rate; roughly half of it is the reference.

    Dakshina's romanised side is Wikipedia-derived and genuinely code-mixed - `county`,
    `website`, `germany` - while its gold label for every token on that side is `ur` by
    construction. Measured against the shipped unigram tables, 45.6% of the tokens the
    3.8% counts are absent from the Roman Urdu lexicon entirely, so tagging them `en` was
    right. The restricted rate is the closest thing to the tagger's own precision, and
    both have to be published or the headline is an upper bound presented as a result.
    """
    corpus = (ROOT / "docs" / "CORPUS.md").read_text(encoding="utf-8")
    assert "the Roman Urdu lexicon attests" in corpus, (
        "CORPUS.md no longer publishes the lexicon-restricted rate beside the headline"
    )
    assert "absent from the Roman Urdu lexicon" in corpus, (
        "the decomposition that explains the gap is gone"
    )
    # And the script must still produce it, or the document is quoting nothing.
    script = (ROOT / "scripts" / "measure_langid.py").read_text(encoding="utf-8")
    assert "urdu_tagged_en_lexicon_only" in script
    assert "urdu_words_in_lexicon" in script
