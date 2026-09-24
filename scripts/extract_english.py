"""Extract English sentences for the Roman-script language tagger.

    python scripts/extract_english.py <hotpot_qa distractor directory> [--out data/english]

HotpotQA's context paragraphs are English Wikipedia sentences. The train files give
the tagger's English side; the validation file is split in two by a hash of the
sentence into validation and test, so the English sentences a setting is chosen on
are never the ones it is reported on.

Needs pyarrow. The package does not.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sentences(path: Path, limit: int):
    import pyarrow.parquet as pq

    seen = 0
    handle = pq.ParquetFile(path)
    for batch in handle.iter_batches(batch_size=500, columns=["context"]):
        for context in batch.column(0).to_pylist():
            for paragraph in context["sentences"]:
                for sentence in paragraph:
                    sentence = " ".join(sentence.split())
                    if len(sentence.split()) >= 5:
                        yield sentence
                        seen += 1
                        if seen >= limit:
                            return


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("directory", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/english"))
    ap.add_argument("--train", type=int, default=150_000)
    ap.add_argument("--heldout", type=int, default=20_000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    train = list(
        dict.fromkeys(sentences(args.directory / "train-00000-of-00002.parquet", args.train))
    )
    held = list(
        dict.fromkeys(sentences(args.directory / "validation-00000-of-00001.parquet", args.heldout))
    )
    held = [s for s in held if s not in set(train)]
    val = [s for s in held if hashlib.sha256(s.encode()).digest()[0] % 2 == 0]
    test = [s for s in held if hashlib.sha256(s.encode()).digest()[0] % 2 == 1]
    for name, rows in (("train", train), ("val", val), ("test", test)):
        (args.out / f"{name}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"{name:<5} {len(rows):>7,} sentences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
