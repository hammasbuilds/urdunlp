"""Check the headline transliteration number in about ten seconds, with no download.

    python scripts/quick_check.py            # score the committed sample
    python scripts/quick_check.py --rebuild  # regenerate the sample from the full split

eval/dakshina_test_sample.tsv is every 10th sentence (the 1st, 11th, 21st, ...) of
Dakshina's Urdu test split (Roark et al., 2020, CC BY-SA 4.0) that aligns word for
word - the same alignment and the same scoring as scripts/measure_translit.py, which
runs on all of it after scripts/fetch_dakshina.py. A tenth of the sentences gives a
number within about a point of the full-split figure; the full split is what the
README reports.

`--rebuild` needs scripts/fetch_dakshina.py to have run. It rewrites the sample from
the full split, so anyone can confirm the committed file is that selection and not a
hand-picked one: `git diff eval/` is empty afterwards.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from measure_translit import DAKSHINA, align, score_in_context  # noqa: E402

SAMPLE = ROOT / "eval/dakshina_test_sample.tsv"
EVERY = 10


def rebuild() -> int:
    """Rewrite SAMPLE as every EVERY-th aligned test sentence of the full split."""
    native = DAKSHINA / "romanized/ur.romanized.rejoined.test.native.txt"
    pairs = DAKSHINA / "romanized/ur.romanized.rejoined.tsv"
    if not native.exists() or not pairs.exists():
        print(f"{DAKSHINA} not found: run python scripts/fetch_dakshina.py first", file=sys.stderr)
        return 1
    keep = {line.strip() for line in native.read_text(encoding="utf-8").splitlines()}
    aligned = []
    for line in pairs.read_text(encoding="utf-8").splitlines():
        urdu, roman = line.split("\t")
        if urdu.strip() in keep and align(urdu, roman) is not None:
            aligned.append(line)
    sample = aligned[::EVERY]
    SAMPLE.write_text("".join(row + "\n" for row in sample), encoding="utf-8", newline="\n")
    print(f"wrote {len(sample):,} of {len(aligned):,} aligned test sentences to {SAMPLE}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--rebuild", action="store_true", help="regenerate the sample from the full test split"
    )
    if ap.parse_args().rebuild and rebuild():
        return 1
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
