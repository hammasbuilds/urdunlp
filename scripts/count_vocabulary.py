"""Count Urdu word frequencies in a corpus, for the transliterator's vocabulary.

    python scripts/count_vocabulary.py <parquet-or-txt> [--out data/vocab/counts.tsv]

Every token goes through `urdunlp.words`, so the counts are over normalised forms -
the same forms the transliterator is scored against. Tokens containing a digit or a
Latin letter are skipped: they are not candidates for Roman -> Urdu output.
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from measure_corpus import read_documents  # noqa: E402

from urdunlp import words  # noqa: E402
from urdunlp.normalize import _is_urdu_letter  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/vocab/counts.tsv"))
    ap.add_argument("--limit", type=int, default=10**9)
    ap.add_argument(
        "--exclude",
        type=Path,
        nargs="*",
        default=[],
        help="files of evaluation sentences, one per line, to leave out of the counts",
    )
    args = ap.parse_args()

    # Dakshina says its romanised sentences come from a held-out partition. Checked:
    # 456 of 4,879 dev and 460 of 4,880 test sentences also occur verbatim in the
    # training partition, so "held out" does not hold at the sentence level. A
    # vocabulary counted over those lines would have seen the evaluation sentences.
    excluded = set()
    for path in args.exclude:
        excluded.update(" ".join(line.split()) for line in path.read_text("utf-8").splitlines())
    skipped = 0

    counts: collections.Counter = collections.Counter()
    started = time.time()
    docs = 0
    for text in read_documents(args.corpus, args.limit):
        if excluded and " ".join(text.split()) in excluded:
            skipped += 1
            continue
        docs += 1
        for token in words(text):
            if all(_is_urdu_letter(c) and c.isalpha() for c in token):
                counts[token] += 1
        if docs % 20000 == 0:
            print(f"  {docs:>7,} documents  {len(counts):>9,} types", file=sys.stderr, flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for word, count in counts.most_common():
            handle.write(f"{word}\t{count}\n")
    total = sum(counts.values())
    print(
        f"{docs:,} documents ({skipped:,} excluded as evaluation sentences), "
        f"{total:,} tokens, {len(counts):,} types in {time.time() - started:.0f}s -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
