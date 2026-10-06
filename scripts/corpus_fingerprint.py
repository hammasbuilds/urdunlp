"""Pin the identity of the corpora the shipped models were built from.

    python scripts/corpus_fingerprint.py            # write data/FINGERPRINT.tsv
    python scripts/corpus_fingerprint.py --check     # compare what is on disk

`data/` is gitignored - it is 70 MB of other people's text - so the published measurements
rested on files no reader could see and no one could confirm. The language-ID corpus was
worse than unseen: it is an unseeded `generator=random` draw, so re-running the fetcher
produced a *different* corpus, and growing `data/wiki/ur.txt` during a review proved the
point by leaving the shipped `langid.json.gz` trained against a file that no longer
existed on disk.

`fetch_wikipedia_samples.py --from-pages` fixes that going forward by recording the
article ids. This file is for the corpus that predates it: a hash per file cannot rebuild
anything, but it can answer "is what I have the thing the numbers were measured on", which
nobody could answer before. `--check` is the question; a mismatch tells you the published
figures do not describe your disk.

Sorted-content hashes, because the paragraph ORDER in a corpus file is not part of its
identity: the split is by content hash and n-gram counts are order-independent, so a
rebuild that returns the same paragraphs in a different order is the same corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "FINGERPRINT.tsv"

# Every file a published number is measured over. Globs, because the language set and the
# Dakshina layout have both changed and a missing file must be reported, not skipped.
TRACKED = (
    "wiki/*.txt",
    "dakshina/ur/romanized/*.tsv",
    "dakshina/ur/lexicons/*.tsv",
    "english/*.txt",
    "vocab/*.tsv",
)


def fingerprint(path: Path) -> tuple[str, int, int]:
    """(sorted-content sha256, lines, characters) for one corpus file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    digest = hashlib.sha256()
    for line in sorted(lines):
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest(), len(lines), sum(len(line) for line in lines)


def collect() -> list[tuple[str, str, int, int]]:
    rows: list[tuple[str, str, int, int]] = []
    for pattern in TRACKED:
        for path in sorted(DATA.glob(pattern)):
            if path.name == OUT.name:
                continue
            digest, lines, characters = fingerprint(path)
            rows.append((path.relative_to(DATA).as_posix(), digest, lines, characters))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--check",
        action="store_true",
        help="compare the files on disk against the committed fingerprint and exit "
        "non-zero on any difference",
    )
    args = ap.parse_args()

    rows = collect()
    if not rows:
        print(f"no corpus files under {DATA} - nothing to fingerprint", file=sys.stderr)
        return 1

    if args.check:
        if not OUT.is_file():
            print(f"missing {OUT.relative_to(ROOT)}", file=sys.stderr)
            return 1
        recorded = {}
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#") and not line.startswith("path\t"):
                path, digest, lines, characters = line.split("\t")
                recorded[path] = (digest, int(lines), int(characters))
        found = {r[0]: (r[1], r[2], r[3]) for r in rows}
        problems = 0
        for path in sorted(set(recorded) | set(found)):
            want, got = recorded.get(path), found.get(path)
            if want is None:
                print(f"  EXTRA    {path}  (not in the fingerprint)")
                problems += 1
            elif got is None:
                print(f"  MISSING  {path}")
                problems += 1
            elif want[0] != got[0]:
                print(
                    f"  CHANGED  {path}  {want[1]:,} lines/{want[2]:,} chars recorded, "
                    f"{got[1]:,}/{got[2]:,} on disk"
                )
                problems += 1
        if problems:
            print(
                f"\n{problems} difference(s). The published measurements were taken over "
                "the recorded files,\nso they do not describe this disk.",
                file=sys.stderr,
            )
            return 1
        print(f"all {len(rows)} corpus file(s) match the committed fingerprint.")
        return 0

    OUT.write_text(
        "# Identity of the corpora the shipped models and published numbers were built\n"
        "# from. data/ is gitignored, so this is the only way to confirm that a corpus on\n"
        "# disk is the one measured. Check with: python scripts/corpus_fingerprint.py --check\n"
        "#\n"
        "# Hashes are over SORTED lines: paragraph order is not part of a corpus's\n"
        "# identity, because the train/val/test split is by content hash and n-gram counts\n"
        "# are order-independent. So a corpus rebuilt with\n"
        "# `fetch_wikipedia_samples.py --from-pages`, which returns articles in id order,\n"
        "# fingerprints the same as the random-order original.\n"
        "path\tsha256_sorted\tlines\tcharacters\n"
        + "\n".join(f"{p}\t{d}\t{n}\t{c}" for p, d, n, c in rows)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT.relative_to(ROOT)}: {len(rows)} file(s)")
    for path, digest, lines, characters in rows:
        print(f"  {path:<44}{lines:>8,} lines{characters:>12,} chars  {digest[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
