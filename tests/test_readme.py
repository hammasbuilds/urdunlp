"""Every example in the README returns what the README says it returns.

Each line of a ```python block written as `expression  # result`, where the result
is a Python literal, is evaluated and compared. The README has shown wrong output
before (0.1 promised علی where the code gave الی); this makes that a test failure.
Skipped where the README is not present, as in the installed-wheel CI job.
"""

import ast
import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parent.parent / "README.md"
BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _examples() -> list[tuple[str, str]]:
    if not README.exists():
        return []
    examples = []
    for block in BLOCK.findall(README.read_text(encoding="utf-8")):
        for line in block.splitlines():
            code, sep, comment = line.partition("  # ")
            if not sep or not code.strip() or code.startswith(("from ", "import ", " ")):
                continue
            # the literal is the comment up to any prose after it: `True: one word...`
            literal = re.split(r"(?<=[\]')0-9e]):| - ", comment.strip(), maxsplit=1)[0]
            try:
                ast.literal_eval(literal)
            except (ValueError, SyntaxError):
                continue
            examples.append((code.strip(), literal))
    return examples


EXAMPLES = _examples()


@pytest.mark.skipif(not README.exists(), reason="README.md is not shipped with the tests")
def test_the_readme_has_examples_to_check():
    assert len(EXAMPLES) >= 20


@pytest.mark.parametrize(("code", "expected"), EXAMPLES, ids=[c for c, _ in EXAMPLES])
def test_readme_example(code, expected):
    import urdunlp

    namespace = {name: getattr(urdunlp, name) for name in urdunlp.__all__}
    assert eval(code, namespace) == ast.literal_eval(expected)
