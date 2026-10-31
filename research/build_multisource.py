"""Build a multi-source corpus, split BY SOURCE, so the test set spans every domain.

The reason this exists: the shipped model trained on Wikipedia and was tested on
Wikipedia, scored 0.9925, and then scored **0.2565** on real out-of-domain Urdu - calling
2,972 of 4,000 Tanzil sentences Shahmukhi Punjabi. A split that draws train and test from
one domain cannot see that. It had no way to.

So every source contributes to **all three** splits. A source that only ever appears in
training hides a domain the model cannot handle; a source that only appears in test is
being judged on text whose register the model was never shown. Both are unfair in
different directions, and splitting each source 80/10/10 avoids both.

    research/build_multisource.py --cap 60000
    research/build_multisource.py --languages ur,pnb,skr --cap 10000

Downloads are streamed and truncated, not fetched whole: NLLB's Arabic monolingual file is
3.8 GB and the first 60,000 lines of it are as representative as all of it for this
purpose. The gzip stream is read until the cap and then closed.

What this cannot fix, and says so: **Shahmukhi Punjabi and Saraiki have no OPUS source at
all.** Across eighteen corpora the sweep found zero for both. With Wikipedia and Tatoeba's
1.3 KB and 2.0 KB they stay single-domain, and no amount of restructuring changes that -
it is a fact about the languages' available resources. Kashmiri has 132 KB, from
localisation files, which is a third domain but a narrow one.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "multisource"
REPORT = ROOT / "research" / "results" / "multisource.json"
SEED = 20261031

# Per language, the OPUS corpora that exist with a usable amount of text, found by
# sweeping eighteen corpora. Wikipedia is listed last and is local, not downloaded.
# Order matters only for reporting.
# Derived from the measured sweep in .baselines/opus_sources.json, not written by
# hand: hand-written codes had TED2020:ckb and CCAligned:ckb, which 404 - those
# corpora carry Sorani under `ku`. The sweep probed both codes per language, so the
# availability it recorded is the availability that exists.
SOURCES: dict[str, list[tuple[str, str]]] = {
    "ur": [("NLLB", "ur"), ("CCMatrix", "ur"), ("CCAligned", "ur"), ("MultiCCAligned", "ur"), ("XLEnt", "ur"), ("Tanzil", "ur"), ("TED2020", "ur"), ("Mozilla-I10n", "ur")],
    "pnb": [],
    "skr": [],
    "sd": [("NLLB", "sd"), ("CCMatrix", "sd"), ("XLEnt", "sd"), ("Tanzil", "sd")],
    "ps": [("NLLB", "ps"), ("CCAligned", "ps"), ("MultiCCAligned", "ps"), ("XLEnt", "ps"), ("TED2020", "ps")],
    "ks": [("Mozilla-I10n", "ks"), ("GNOME", "ks")],
    "fa": [("CCMatrix", "fa"), ("NLLB", "fa"), ("CCAligned", "fa"), ("MultiCCAligned", "fa"), ("XLEnt", "fa"), ("TED2020", "fa"), ("Tanzil", "fa"), ("infopankki", "fa"), ("Mozilla-I10n", "fa"), ("GNOME", "fa")],
    "ar": [("NLLB", "ar"), ("CCMatrix", "ar"), ("CCAligned", "ar"), ("MultiCCAligned", "ar"), ("XLEnt", "ar"), ("TED2020", "ar"), ("bible-uedin", "ar"), ("Tanzil", "ar"), ("infopankki", "ar"), ("Mozilla-I10n", "ar"), ("GNOME", "ar")],
    "ckb": [("CCAligned", "ku"), ("MultiCCAligned", "ku"), ("TED2020", "ku"), ("Tanzil", "ku"), ("Mozilla-I10n", "ckb")],
    "ug": [("NLLB", "ug"), ("Tanzil", "ug"), ("XLEnt", "ug"), ("TED2020", "ug"), ("GNOME", "ug")],
    "azb": [("NLLB", "az"), ("CCMatrix", "az"), ("CCAligned", "az"), ("MultiCCAligned", "az"), ("XLEnt", "az"), ("Tanzil", "az"), ("TED2020", "az"), ("Mozilla-I10n", "az")],
}
# Sources whose script must be checked before use - a Latin-script file for a
# Perso-Arabic language would be separable by script alone and would flatter every
# aggregate. Tanzil `az` is 100% Latin and GNOME `ks` is 98% Devanagari; both are
# rejected at runtime by the same check rather than by this list, which only records
# that the check is load-bearing.
MIN_ARABIC_SHARE = 0.5
SPLIT = (0.8, 0.1, 0.1)


def arabic_share(texts: list[str]) -> float:
    sample = " ".join(texts[:300])
    arab = sum(1 for c in sample if "؀" <= c <= "ۿ" or "ݐ" <= c <= "ݿ")
    latn = sum(1 for c in sample if c.isascii() and c.isalpha())
    deva = sum(1 for c in sample if "ऀ" <= c <= "ॿ")
    return arab / (arab + latn + deva or 1)


def stream(corpus: str, code: str, cap: int, min_chars: int = 40) -> list[str]:
    """The first `cap` distinct lines of an OPUS monolingual file, then stop.

    Streamed because NLLB's Arabic file is 3.8 GB. Opening the gzip stream and closing it
    after `cap` lines transfers a few megabytes instead.
    """
    url = f"https://object.pouta.csc.fi/OPUS-{corpus}/v1/mono/{code}.txt.gz"
    out: list[str] = []
    seen: set[str] = set()
    try:
        with urllib.request.urlopen(url, timeout=180) as response:
            with gzip.GzipFile(fileobj=response) as fh:
                for raw in fh:
                    text = raw.decode("utf-8", "replace").strip()
                    if len(text) >= min_chars and text not in seen:
                        seen.add(text)
                        out.append(text)
                        if len(out) >= cap:
                            break
    except (urllib.error.HTTPError, urllib.error.URLError, EOFError, OSError) as exc:
        print(f"      {corpus}:{code} failed: {type(exc).__name__}")
    return out


def wikipedia(language: str, cap: int) -> list[str]:
    path = ROOT / "data" / "wiki_large" / f"{language}.txt"
    if not path.exists():
        return []
    out, seen = [], set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if len(text) >= 40 and text not in seen:
            seen.add(text)
            out.append(text)
            if len(out) >= cap:
                break
    return out


def split(texts: list[str], rng: random.Random) -> dict[str, list[str]]:
    """80/10/10 within one source, shuffled once with a fixed seed.

    Within the source, so every source reaches every split. The shuffle is seeded so the
    same source produces the same three chunks on a rebuild.
    """
    pool = list(texts)
    rng.shuffle(pool)
    n = len(pool)
    a = int(SPLIT[0] * n)
    b = a + int(SPLIT[1] * n)
    return {"train": pool[:a], "val": pool[a:b], "test": pool[b:]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cap", type=int, default=60000,
                    help="lines taken per source per language (60000)")
    ap.add_argument("--languages", default=",".join(SOURCES))
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    wanted = [l.strip() for l in args.languages.split(",") if l.strip()]
    rng = random.Random(SEED)

    report: dict = {"cap_per_source": args.cap, "split": list(SPLIT), "seed": SEED,
                    "languages": {}}
    for language in wanted:
        print(f"\n{language}:")
        per_source: dict[str, dict[str, list[str]]] = {}
        rejected: dict[str, str] = {}

        local = wikipedia(language, args.cap)
        if local:
            per_source["wikipedia"] = split(local, rng)
            print(f"    wikipedia         {len(local):>7,}")

        for corpus, code in SOURCES.get(language, []):
            texts = stream(corpus, code, args.cap)
            if not texts:
                continue
            share = arabic_share(texts)
            if share < MIN_ARABIC_SHARE:
                rejected[f"{corpus}:{code}"] = (
                    f"{share:.0%} Perso-Arabic characters - wrong script for this "
                    "language, so it would be separable by script alone"
                )
                print(f"    {corpus + ':' + code:18}{len(texts):>7,}  REJECTED "
                      f"({share:.0%} Arabic)")
                continue
            per_source[f"{corpus}:{code}"] = split(texts, rng)
            print(f"    {corpus + ':' + code:18}{len(texts):>7,}  "
                  f"{share:.0%} Arabic")

        total = sum(len(v["train"]) + len(v["val"]) + len(v["test"])
                    for v in per_source.values())
        report["languages"][language] = {
            "sources_used": sorted(per_source),
            "sources_rejected": rejected,
            "lines_total": total,
            "per_source": {k: {s: len(v) for s, v in d.items()}
                           for k, d in per_source.items()},
            "meets_50k": total >= 50_000,
        }
        print(f"    {'total':18}{total:>7,}"
              f"{'  (under 50,000)' if total < 50_000 else ''}")

        # Written as one file per split per language, with a sidecar naming the source of
        # every line - so a per-source accuracy can be computed afterwards.
        for part in ("train", "val", "test"):
            path = args.out / part / f"{language}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            tags = args.out / part / f"{language}.sources.txt"
            with path.open("w", encoding="utf-8") as fh, tags.open("w", encoding="utf-8") as sh:
                for source, chunks in sorted(per_source.items()):
                    for text in chunks[part]:
                        fh.write(text.replace("\n", " ") + "\n")
                        sh.write(source + "\n")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print("\nsummary:")
    for language, row in report["languages"].items():
        mark = "" if row["meets_50k"] else "   SHORT"
        print(f"  {language:5} {len(row['sources_used'])} source(s)  "
              f"{row['lines_total']:>7,} line(s){mark}")
    short = [l for l, r in report["languages"].items() if not r["meets_50k"]]
    if short:
        print(f"\n  {len(short)} language(s) under 50,000 from real sources: "
              f"{', '.join(short)}")
    print(f"\nwritten to {args.out} and {REPORT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
