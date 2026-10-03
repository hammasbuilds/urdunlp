"""Check the headline transliteration number in about a minute, with no download.

    python scripts/quick_check.py

eval/dakshina_test_sample.tsv is every 10th sentence of Dakshina's Urdu test split
(Roark et al., 2020, CC BY-SA 4.0) that aligns word for word - the same alignment
and the same scoring as scripts/measure_translit.py, which runs on all of it after
scripts/fetch_dakshina.py. A tenth of the sentences gives a number within about a
point of the full-split figure; the full split is what the README reports.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from measure_translit import align, score_in_context  # noqa: E402

SAMPLE = ROOT / "eval/dakshina_test_sample.tsv"


def main() -> int:
    started = time.time()
    sentences = []
    for line in SAMPLE.read_text(encoding="utf-8").splitlines():
        urdu, roman = line.split("\t")
        pairs = align(urdu, roman)
        if pairs is not None:
            sentences.append(pairs)
    result = score_in_context(sentences)
    print(
        f"Roman -> Urdu word accuracy on {len(sentences):,} sample sentences, "
        f"{result['n']:,} words: {result['accuracy']:.1%}  ({time.time() - started:.0f} s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
