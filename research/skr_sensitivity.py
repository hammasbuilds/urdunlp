"""Is Saraiki's figure depressed by mislabelled rows, and how much?

`skr` scores 0.8688 on the multi-source test split while every other language is above
0.96. Two things are going on, and they compound.

**The source skews short.** Saraiki's two sources are its Wikipedia (median 211
characters) and its Wiktionary (median 116). Accuracy falls with length for every
language, so a source made of dictionary entries scores lower for that reason alone.

**The short Wiktionary rows are partly Urdu.** The Saraiki Wiktionary is a bilingual
Saraiki-Urdu dictionary: an entry gives a Saraiki definition and an Urdu gloss, each
extracted as its own line and both labelled `skr`. At matched length the gap is 0.047
above 200 characters and **0.193 below 80**, which is where a bare gloss sits.

Filtering those rows with the classifier would be circular - it would remove exactly the
rows the model gets wrong and inflate the score by construction. So the criterion here is
lexical and model-independent: the four implosive letters that are distinctive to Saraiki
among these eleven languages. They appear in 38% of Saraiki Wikipedia lines and in 0.01%
of Shahmukhi Punjabi and 0.00% of Urdu, so a line carrying one is Saraiki on orthographic
grounds alone, whatever any model says.

This reports accuracy on that subset beside the whole. The subset is biased towards
longer and more distinctively Saraiki text, so it is an **upper** bound, not a corrected
figure - both are printed and neither replaces the other.

    .venv-baselines/Scripts/python research/skr_sensitivity.py
"""

from __future__ import annotations

import collections
import gzip
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMPLOSIVES = set("ٻڄڳݙ")
MODEL = ROOT / "models" / "with_synthetic" / "langid.json.gz"


def wilson(k: int, n: int, z: float = 1.96) -> dict | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"rate": round(p, 4), "lo": round(max(0.0, c - h), 4),
            "hi": round(min(1.0, c + h), 4), "n": n}


def main() -> int:
    import urdunlp.langid as langid

    payload = json.loads(gzip.decompress(MODEL.read_bytes()))
    langid._load = lambda name, _p=payload: _p  # noqa: SLF001
    langid._script_model.cache_clear()  # noqa: SLF001

    texts = (ROOT / "data" / "splits" / "test" / "skr.txt").read_text(
        encoding="utf-8").splitlines()
    tags = (ROOT / "data" / "splits" / "test" / "skr.src").read_text(
        encoding="utf-8").splitlines()

    buckets: dict[tuple[str, bool], list[int]] = collections.defaultdict(
        lambda: [0, 0])
    for text, tag in zip(texts, tags):
        source = tag.split("\t")[-1]
        marked = bool(IMPLOSIVES & set(text))
        ok = int(langid.identify_language(text).language == "skr")
        buckets[(source, marked)][0] += ok
        buckets[(source, marked)][1] += 1

    report: dict = {
        "criterion": "line contains one of the Saraiki implosives " + "".join(
            sorted(IMPLOSIVES)),
        "why_not_the_model": (
            "Filtering with the classifier would remove exactly the rows it gets wrong "
            "and inflate the result by construction. This criterion is orthographic and "
            "independent of any model."
        ),
        "is_an_upper_bound": (
            "Lines carrying an implosive are longer and more distinctively Saraiki than "
            "the rest, so this is an upper bound on Saraiki accuracy, not a corrected "
            "figure. Both are reported."
        ),
        "per_source": {},
    }

    print(f"  {'source':26}{'implosive':>11}{'accuracy':>11}{'n':>9}")
    for (source, marked), (right, total) in sorted(buckets.items()):
        row = wilson(right, total)
        report["per_source"][f"{source}|{'with' if marked else 'without'}"] = row
        print(f"  {source:26}{'yes' if marked else 'no':>11}"
              f"{row['rate']:>11.4f}{total:>9,}")

    with_imp = [v for (s, m), v in buckets.items() if m]
    without = [v for (s, m), v in buckets.items() if not m]
    a = wilson(sum(v[0] for v in with_imp), sum(v[1] for v in with_imp))
    b = wilson(sum(v[0] for v in without), sum(v[1] for v in without))
    whole = wilson(sum(v[0] for v in buckets.values()),
                   sum(v[1] for v in buckets.values()))
    report["with_implosive"] = a
    report["without_implosive"] = b
    report["whole"] = whole

    print(f"\n  whole skr test split         {whole['rate']:.4f} "
          f"[{whole['lo']:.4f}, {whole['hi']:.4f}]  n={whole['n']:,}")
    print(f"  lines with an implosive      {a['rate']:.4f} "
          f"[{a['lo']:.4f}, {a['hi']:.4f}]  n={a['n']:,}   <- upper bound")
    print(f"  lines without one            {b['rate']:.4f} "
          f"[{b['lo']:.4f}, {b['hi']:.4f}]  n={b['n']:,}")

    out = ROOT / "research" / "results" / "skr_sensitivity.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwritten to {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
