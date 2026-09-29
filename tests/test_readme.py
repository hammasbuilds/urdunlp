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
import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parent.parent / "README.md"
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
