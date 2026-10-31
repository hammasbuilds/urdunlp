"""Split every source - real, segmented and synthetic - 80/10/10 within itself.

Two rules, both from the failure this is fixing.

**Every source reaches every split.** A source only in training hides a domain the model
cannot handle; a source only in test judges the model on a register it never saw. The
Wikipedia-only setup had one domain in both and so could not see a 0.9925 in-domain score
sitting next to 0.2565 out of domain.

**Synthetic and segmented rows are tagged and splittable apart.** They get their own
80/10/10 like any other source, so a synthetic test chunk exists - but `kind` is recorded
per row, so accuracy on **real** test rows can be reported separately. That is the whole
point: the question "does synthetic training data improve real-text accuracy" is only
answerable if the real test rows are identifiable.

    .venv-baselines/Scripts/python research/make_splits.py --out data/splits
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 20261031
SPLIT = (0.8, 0.1, 0.1)
LANGUAGES = ["ur", "pnb", "skr", "sd", "ps", "ks", "fa", "ar", "ckb", "ug", "azb"]


def kind_of(name: str) -> str:
    if ".synthetic-" in name:
        return "synthetic"
    if name.endswith(".segmented.txt"):
        return "segmented"
    return "real"


def gather(language: str, directories: list[Path]) -> dict[str, tuple[str, list[str]]]:
    """{source name: (kind, lines)} for one language, across every directory given."""
    out: dict[str, tuple[str, list[str]]] = {}
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob(f"{language}.*.txt")):
            source = path.name[len(language) + 1: -len(".txt")]
            lines = [l for l in path.read_text(encoding="utf-8",
                                               errors="replace").splitlines() if l.strip()]
            if len(lines) >= 20:
                out[source] = (kind_of(path.name), lines)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "splits")
    ap.add_argument("--no-synthetic", action="store_true",
                    help="build the splits from real and segmented rows only, so the "
                         "two training sets differ in exactly one thing")
    args = ap.parse_args()

    directories = [ROOT / "data" / "corpus", ROOT / "data" / "synthetic"]
    report: dict = {"seed": SEED, "split": list(SPLIT),
                    "synthetic_included": not args.no_synthetic, "languages": {}}

    for language in LANGUAGES:
        sources = gather(language, directories)
        if args.no_synthetic:
            sources = {k: v for k, v in sources.items() if v[0] != "synthetic"}
        if not sources:
            continue
        chunks: dict[str, dict[str, list[tuple[str, str]]]] = {
            "train": {}, "val": {}, "test": {}}
        for source, (kind, lines) in sources.items():
            pool = list(lines)
            # Seeded per (language, source), not from one generator walked across the
            # whole run. With a single shared generator, `--no-synthetic` performs fewer
            # shuffles for Kashmiri and every language processed after it draws from a
            # different point in the stream: ur through ks came out identical and fa,
            # ar, ckb, ug and azb did not, with the same row counts and different rows.
            # A paired real-row comparison across the two runs would then have been
            # measuring two different test sets while looking perfectly well formed.
            random.Random(f"{SEED}|{language}|{source}").shuffle(pool)
            n = len(pool)
            a, b = int(SPLIT[0] * n), int(SPLIT[0] * n) + int(SPLIT[1] * n)
            for part, rows in (("train", pool[:a]), ("val", pool[a:b]),
                               ("test", pool[b:])):
                chunks[part][source] = [(kind, text) for text in rows]

        for part in ("train", "val", "test"):
            directory = args.out / part
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / f"{language}.txt").open("w", encoding="utf-8") as text_fh, \
                 (directory / f"{language}.src").open("w", encoding="utf-8") as src_fh:
                for source, rows in sorted(chunks[part].items()):
                    for kind, text in rows:
                        text_fh.write(text + "\n")
                        src_fh.write(f"{kind}\t{source}\n")

        counts = {
            part: {
                "total": sum(len(v) for v in chunks[part].values()),
                "real": sum(1 for v in chunks[part].values() for k, _ in v if k == "real"),
                "segmented": sum(1 for v in chunks[part].values()
                                 for k, _ in v if k == "segmented"),
                "synthetic": sum(1 for v in chunks[part].values()
                                 for k, _ in v if k == "synthetic"),
            }
            for part in ("train", "val", "test")
        }
        report["languages"][language] = {"sources": len(sources), "counts": counts}
        print(f"  {language:5} {len(sources):>2} source(s)  "
              f"train {counts['train']['total']:>7,}  "
              f"val {counts['val']['total']:>6,}  "
              f"test {counts['test']['total']:>6,}  "
              f"(test: {counts['test']['real']:,} real, "
              f"{counts['test']['segmented']:,} segmented, "
              f"{counts['test']['synthetic']:,} synthetic)")

    path = ROOT / "research" / "results" / (
        "splits_no_synthetic.json" if args.no_synthetic else "splits.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwritten to {args.out} and {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
