"""An out-of-domain validation set, and what it can and cannot settle.

The first objection to every number in this work is that the model trained on Wikipedia
and was evaluated on Wikipedia. This builds a test set from somewhere else entirely:
**Tatoeba**, crowd-sourced single sentences, CC-BY 2.0 FR - a different domain, a
different register, and much shorter texts than a Wikipedia paragraph.

What it can settle, and for which languages, is decided by the data rather than by hope.
Tatoeba's coverage of these eleven is wildly uneven:

    fa  468 KB     ar  769 KB     ckb 146 KB     ug  119 KB     ur   39 KB
    kas 3.4 KB     skr 2.0 KB     pnb 1.3 KB     snd 242 bytes
    ps  absent     azb absent

So this is a genuine out-of-domain check for Urdu, Persian, Arabic, Sorani and Uyghur, and
a token one at best for the languages the whole task is about - Shahmukhi Punjabi, Saraiki,
Sindhi, Kashmiri. Reporting one accuracy over the union would hide that, so every figure
here is per language with its n beside it, and no rate is quoted for a language whose n
cannot carry one.

That limitation is the finding as much as the accuracies are: **there is no out-of-domain
corpus for Saraiki or Shahmukhi Punjabi of a size that can validate a classifier.** A paper
can say that, and it is more useful than a number computed over nine sentences.

    .venv-baselines/Scripts/python research/out_of_domain.py --fetch
    .venv-baselines/Scripts/python research/out_of_domain.py
"""

from __future__ import annotations

import argparse
import bz2
import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
RAW = ROOT / ".baselines" / "tatoeba"
OUT = ROOT / "research" / "results" / "out_of_domain.json"

# Our code -> Tatoeba's ISO 639-3 directory. Pashto and South Azerbaijani have no export
# under any code tried (pbu, pus, pbt / azb), and are recorded as absent rather than
# silently dropped.
TATOEBA = {
    "ur": "urd", "pnb": "pnb", "skr": "skr", "sd": "snd", "ks": "kas",
    "fa": "pes", "ar": "ara", "ckb": "ckb", "ug": "uig",
}
ABSENT = {"ps": ("pbu", "pus", "pbt"), "azb": ("azb",)}
# Below this, a per-language accuracy is reported as a count and not as a rate: a
# proportion over nine items has an interval wider than the range it lives in.
MIN_FOR_A_RATE = 30


def wilson(k: int, n: int, z: float = 1.96) -> dict | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"rate": round(p, 4), "lo": round(max(0.0, centre - half), 4),
            "hi": round(min(1.0, centre + half), 4), "n": n, "correct": k}


def fetch() -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    got = {}
    for ours, theirs in TATOEBA.items():
        path = RAW / f"{theirs}.tsv.bz2"
        if not path.exists() or path.stat().st_size == 0:
            url = (f"https://downloads.tatoeba.org/exports/per_language/"
                   f"{theirs}/{theirs}_sentences.tsv.bz2")
            try:
                urllib.request.urlretrieve(url, path)
            except urllib.error.HTTPError as exc:
                print(f"  {ours}: HTTP {exc.code}")
                continue
        got[ours] = path.stat().st_size
        print(f"  {ours:4} {theirs:4} {got[ours]:>9,} bytes")
    for ours, tried in ABSENT.items():
        print(f"  {ours:4} absent from Tatoeba (tried {', '.join(tried)})")
    return got


def sentences(path: Path) -> list[str]:
    """Tatoeba's export is id, language, text - the text column, deduplicated."""
    out, seen = [], set()
    with bz2.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            text = parts[2].strip()
            if len(text) >= 10 and text not in seen:
                seen.add(text)
                out.append(text)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    if args.fetch:
        fetch()
        print()

    from urdunlp.langid import identify_language  # noqa: PLC0415

    corpus: dict[str, list[str]] = {}
    for ours, theirs in TATOEBA.items():
        path = RAW / f"{theirs}.tsv.bz2"
        if path.exists():
            corpus[ours] = sentences(path)

    print(f"{sum(len(v) for v in corpus.values()):,} out-of-domain sentence(s) "
          f"over {len(corpus)} of 11 language(s)\n")
    print(f"  {'lang':6}{'n':>7}  {'accuracy':>10}  {'95% interval':>18}   what it answered instead")
    results = {}
    for ours in ["ur", "pnb", "skr", "sd", "ps", "ks", "fa", "ar", "ckb", "ug", "azb"]:
        if ours not in corpus:
            print(f"  {ours:6}{'—':>7}  {'—':>10}  {'absent from Tatoeba':>18}")
            results[ours] = {"available": False,
                             "why": f"absent from Tatoeba (tried {ABSENT.get(ours, ())})"}
            continue
        texts = corpus[ours]
        got = [identify_language(t).language for t in texts]
        right = sum(1 for g in got if g == ours)
        wrong = {}
        for g in got:
            if g != ours:
                wrong[g] = wrong.get(g, 0) + 1
        top = ", ".join(f"{k} {v}" for k, v in
                        sorted(wrong.items(), key=lambda kv: -kv[1])[:3]) or "—"
        row = {"available": True, "n": len(texts), "correct": right,
               "confusions": dict(sorted(wrong.items(), key=lambda kv: -kv[1]))}
        if len(texts) >= MIN_FOR_A_RATE:
            row["accuracy"] = wilson(right, len(texts))
            w = row["accuracy"]
            print(f"  {ours:6}{len(texts):>7}  {w['rate']:>10.4f}  "
                  f"[{w['lo']:.4f}, {w['hi']:.4f}]   {top}")
        else:
            row["accuracy"] = None
            row["withheld"] = (
                f"{len(texts)} sentence(s) is below {MIN_FOR_A_RATE}; a rate over this "
                "many has an interval wider than the range it lives in, so the count is "
                "reported and the rate is not"
            )
            print(f"  {ours:6}{len(texts):>7}  {right:>4} right  "
                  f"{'rate withheld':>18}   {top}")
        results[ours] = row

    scored = [r for r in results.values() if r.get("accuracy")]
    overall_n = sum(r["n"] for r in results.values() if r.get("available"))
    overall_right = sum(r["correct"] for r in results.values() if r.get("available"))
    payload = {
        "source": "Tatoeba sentence exports (CC-BY 2.0 FR), downloads.tatoeba.org",
        "why": "a non-Wikipedia domain, to test the in-domain objection",
        "languages_available": sorted(corpus),
        "languages_absent": sorted(ABSENT),
        "min_n_for_a_rate": MIN_FOR_A_RATE,
        "per_language": results,
        "overall": wilson(overall_right, overall_n),
        "languages_with_a_reportable_rate": len(scored),
        "the_limitation": (
            "Tatoeba has 1.3 KB of Shahmukhi Punjabi, 2.0 KB of Saraiki and 242 bytes of "
            "Sindhi. So this validates the easy half of the task out of domain and cannot "
            "validate the half the task exists for. There is no out-of-domain corpus of a "
            "size that can validate a classifier on Saraiki or Shahmukhi Punjabi, which "
            "is a fact about the languages' resources rather than about this model."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\n  {len(scored)} of 11 language(s) have enough sentences for a rate")
    print(f"  overall over the {overall_n:,} available: "
          f"{payload['overall']['rate']:.4f} "
          f"[{payload['overall']['lo']:.4f}, {payload['overall']['hi']:.4f}]")
    print(f"\nwritten to {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
