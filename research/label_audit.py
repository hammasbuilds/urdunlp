"""Audit every source-language pair for the defect §6.2 found in one of them.

Section 7 states the threat and then admits the gap: the Saraiki Wiktionary turned out to
be a bilingual dictionary whose Urdu glosses we had labelled `skr`, we found it by looking
at a surprising number, and *we had not audited the other thirty-five pairs the same way*.
This script closes that, and it does so without consulting the classifier.

The test is **exact line duplication across language labels**. If one and the same line of
text is filed under `skr` and under `ur`, at most one of those labels is right, whatever
any model thinks. That makes it evidence about the corpus rather than about the model - the
mistake §6.2 is careful not to make - and it needs no notion of which languages are
"close".

It is a lower bound on mislabelling, deliberately, and three things it cannot see are
worth naming:

  * a translated gloss is not a duplicated line, so a bilingual dictionary whose two
    halves differ in wording is invisible here;
  * short formulaic lines (dates, numerals, place names) duplicate across languages
    without anyone having mislabelled anything, which is why length is reported beside
    the rate and the headline rate excludes lines under `MIN_INTERESTING`;
  * a source could be wholly mislabelled with no overlap at all, if the true language has
    no corpus here.

So a high rate is evidence of a problem and a low rate is not a clean bill of health. The
output says so.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"
OUT = ROOT / "research" / "results" / "label_audit.json"

# Lines shorter than this are excluded from the headline rate. A four-character line that
# appears under two labels is a numeral or an abbreviation, not evidence of mislabelling,
# and letting those in would make every pair of languages look contaminated. Both rates
# are reported so the choice is visible rather than baked in.
MIN_INTERESTING = 24

ARABIC = re.compile(r"[ؠ-يٮ-ۓەۮۯۺ-ۿݐ-ݿ]")


def normalise(line: str) -> str:
    """Fold the differences that are not language differences.

    NFC, collapse whitespace, and drop the tatweel and the diacritics that two projects
    will write differently for identical text. Without this the audit reports zero
    overlap between corpora that differ only in whether they kept vowel marks, which is
    exactly the false negative that would let a mislabelled source through.
    """
    line = unicodedata.normalize("NFC", line).strip()
    line = re.sub(r"\s+", " ", line)
    line = line.replace("ـ", "")  # tatweel: decorative, never lexical
    return "".join(c for c in line if not unicodedata.combining(c))


def digest(line: str) -> bytes:
    return hashlib.blake2b(line.encode("utf-8"), digest_size=8).digest()


def sources(corpus: Path = CORPUS) -> list[tuple[str, str, Path]]:
    """(language, source, path) for every corpus file, from the directory not a list."""
    out = []
    for path in sorted(corpus.glob("*.txt")):
        language, _, source = path.stem.partition(".")
        out.append((language, source, path))
    return out


def index(files: list[tuple[str, str, Path]], limit: int | None):
    """digest -> set of language labels it appears under, and per-file line digests.

    Two passes over the text would double a 815 MB read, so the per-file digests are kept
    and the cross-label map is built from them.
    """
    per_file: dict[tuple[str, str], Counter] = {}
    labels: dict[bytes, set[str]] = defaultdict(set)
    for language, source, path in files:
        counts: Counter = Counter()
        with path.open(encoding="utf-8", errors="replace") as handle:
            for n, raw in enumerate(handle):
                if limit is not None and n >= limit:
                    break
                line = normalise(raw)
                if not line:
                    continue
                d = digest(line)
                counts[d] += 1
                labels[d].add(language)
        per_file[(language, source)] = counts
        print(f"  {language:5s} {source:28s} {sum(counts.values()):>9,} lines, "
              f"{len(counts):>9,} distinct", flush=True)
    return per_file, labels


def lengths(files: list[tuple[str, str, Path]], wanted: set[bytes], limit: int | None):
    """Line length for the digests that turned out to be shared, for MIN_INTERESTING.

    Only the shared digests are kept, so this is a few hundred thousand entries rather
    than three and a half million.
    """
    out: dict[bytes, int] = {}
    for _language, _source, path in files:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for n, raw in enumerate(handle):
                if limit is not None and n >= limit:
                    break
                line = normalise(raw)
                if not line:
                    continue
                d = digest(line)
                if d in wanted and d not in out:
                    out[d] = len(line)
    return out


def audit(limit: int | None = None, corpus: Path = CORPUS) -> dict:
    files = sources(corpus)
    print(f"{len(files)} corpus files")
    per_file, labels = index(files, limit)

    shared = {d for d, langs in labels.items() if len(langs) > 1}
    print(f"\n{len(shared):,} distinct lines appear under more than one language label")
    size = lengths(files, shared, limit)

    report = {}
    for (language, source), counts in sorted(per_file.items()):
        total = sum(counts.values())
        long_total = sum(c for d, c in counts.items()
                         if size.get(d, MIN_INTERESTING) >= MIN_INTERESTING)
        # `size` only holds shared digests; an unshared line is not in it and cannot be
        # cross-labelled, so counting it as long is harmless and avoids a second pass
        # over every line in the corpus.
        hit: Counter = Counter()
        hit_long: Counter = Counter()
        for d, c in counts.items():
            if d not in shared:
                continue
            others = labels[d] - {language}
            for other in others:
                hit[other] += c
                if size.get(d, 0) >= MIN_INTERESTING:
                    hit_long[other] += c
        any_other = sum(c for d, c in counts.items() if d in shared)
        any_other_long = sum(c for d, c in counts.items()
                             if d in shared and size.get(d, 0) >= MIN_INTERESTING)
        report[f"{language}.{source}"] = {
            "language": language,
            "source": source,
            "lines": total,
            "lines_at_least_min_interesting": long_total,
            "shared_lines": any_other,
            "shared_rate": round(any_other / total, 6) if total else None,
            "shared_lines_long": any_other_long,
            "shared_rate_long": round(any_other_long / long_total, 6) if long_total else None,
            "by_other_language": dict(hit.most_common()),
            "by_other_language_long": dict(hit_long.most_common()),
        }
    return {
        "min_interesting": MIN_INTERESTING,
        "files": len(files),
        "distinct_shared_lines": len(shared),
        "note": (
            "shared_rate_long is the headline: the share of this source's lines that "
            "appear verbatim under another language label, counting only lines of at "
            "least min_interesting characters. A lower bound on mislabelling - a "
            "translated gloss is not a duplicated line - so a low value is not a clean "
            "bill of health."
        ),
        "sources": report,
    }


def show(data: dict) -> None:
    rows = sorted(data["sources"].values(),
                  key=lambda r: -(r["shared_rate_long"] or 0))
    print(f"\n  {'source':34}{'lines':>10}{'shared':>9}{'rate':>9}   worst other label")
    for r in rows:
        worst = next(iter(r["by_other_language_long"].items()), None)
        tail = ""
        if worst:
            other, n = worst
            tail = f"{other} ({n:,})"
        rate = r["shared_rate_long"]
        print(f"  {r['language'] + '.' + r['source']:34}{r['lines']:>10,}"
              f"{r['shared_lines_long']:>9,}{(rate if rate is not None else 0):>9.4f}   {tail}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="lines per file, for a smoke run")
    ap.add_argument("--json", type=Path, default=OUT)
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    args = ap.parse_args()

    data = audit(args.limit, args.corpus)
    show(data)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwritten to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
