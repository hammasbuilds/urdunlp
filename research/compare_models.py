"""Put the trained models side by side on identical rows, and answer two questions.

Every model here is scored by `evaluate_multisource.py` on the **same** 358,720-row
multi-source test split, so the columns are comparable by construction.

**Question 1: is the gain source diversity, or just more data?**
The shipped model differs from the multi-source one in two ways at once - it sees only
Wikipedia, and it sees 12,000 lines per language instead of 120,000. Either could explain
an improvement. So two further models are trained at a matched 12,000 lines per language,
one on Wikipedia and one on the multi-source corpus. The difference between *those two* is
diversity with size held constant, and it is the only honest form of the claim.

**Question 2: did the built Kashmiri rows earn their place?**
`with_synthetic` and `no_synthetic` differ in exactly one thing, and are scored on
identical real rows (`tests/test_split_integrity.py` checks that). The comparison is made
on **real test rows only**: scoring the built rows too would let a model that had learned
the generator's artefacts certify its own success. If real-row accuracy does not improve,
the rows should be dropped and that is the finding.

    .venv-baselines/Scripts/python research/compare_models.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "research" / "results"
ORDER = ["ur", "pnb", "skr", "sd", "ps", "ks", "fa", "ar", "ckb", "ug", "azb"]

MODELS = {
    "shipped": ("eval_shipped_on_new_split.json", "Wikipedia only, 12k/lang (shipped)"),
    "wikipedia_12k": ("eval_wikipedia_12k.json", "Wikipedia only, 12k/lang (retrained)"),
    "multisource_12k": ("eval_multisource_12k.json", "multi-source, 12k/lang"),
    "no_synthetic": ("eval_no_synthetic.json", "multi-source, 120k/lang, no built rows"),
    "with_synthetic": ("eval_with_synthetic.json", "multi-source, 120k/lang, + built ks"),
}


def load() -> dict[str, dict]:
    out = {}
    for key, (name, _) in MODELS.items():
        path = RESULTS / name
        if path.exists():
            out[key] = json.loads(path.read_text(encoding="utf-8"))
    return out


def rate(payload: dict, language: str | None, real_only: bool) -> dict | None:
    if language is None:
        return payload.get("overall_real_rows_only" if real_only else "overall")
    key = "per_language_real_rows_only" if real_only else "per_language"
    return (payload.get(key) or {}).get(language)


def two_proportion_z(a: dict, b: dict) -> tuple[float, float]:
    """Unpaired z on two rates. Approximate - the rows are the same, so this is
    conservative relative to a paired test, and it is only used for a sanity direction.
    """
    p1, n1 = a["rate"], a["n"]
    p2, n2 = b["rate"], b["n"]
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) or 1e-12
    z = (p2 - p1) / se
    return z, math.erfc(abs(z) / math.sqrt(2))


def table(have: dict, keys: list[str], real_only: bool, title: str) -> None:
    keys = [k for k in keys if k in have]
    if len(keys) < 2:
        print(f"\n{title}\n  (needs {', '.join(MODELS[k][0] for k in keys)})")
        return
    print(f"\n{title}")
    print(f"  {'lang':6}{'n':>9}" + "".join(f"{k[:14]:>16}" for k in keys)
          + f"{'change':>10}")
    for language in [*ORDER, None]:
        cells = [rate(have[k], language, real_only) for k in keys]
        if any(c is None for c in cells):
            continue
        name = language or "OVERALL"
        change = cells[-1]["rate"] - cells[0]["rate"]
        print(f"  {name:6}{cells[0]['n']:>9,}"
              + "".join(f"{c['rate']:>16.4f}" for c in cells)
              + f"{change:>+10.4f}")


def main() -> int:
    have = load()
    if not have:
        raise SystemExit(f"no evaluation results in {RESULTS}")
    missing = [k for k in MODELS if k not in have]
    print(f"{len(have)} of {len(MODELS)} model(s) scored"
          + (f"; still missing: {', '.join(missing)}" if missing else ""))

    rows = {k: v.get("test_rows") for k, v in have.items()}
    if len(set(rows.values())) != 1:
        raise SystemExit(
            f"the models were scored on different numbers of rows ({rows}); the columns "
            f"are not comparable and no table should be printed from them"
        )
    print(f"all scored on the same {next(iter(rows.values())):,} test rows\n")

    table(have, ["wikipedia_12k", "multisource_12k"], False,
          "Q1. Diversity at matched size (12,000 lines per language, all test rows)")
    table(have, ["shipped", "no_synthetic"], False,
          "    For reference: shipped against the full multi-source model")
    table(have, ["no_synthetic", "with_synthetic"], True,
          "Q2. Built Kashmiri rows, scored on REAL test rows only")

    if "no_synthetic" in have and "with_synthetic" in have:
        a = rate(have["no_synthetic"], "ks", True)
        b = rate(have["with_synthetic"], "ks", True)
        if a and b:
            z, p = two_proportion_z(a, b)
            verdict = ("the built rows improved real Kashmiri"
                       if b["rate"] > a["rate"] else
                       "the built rows did NOT improve real Kashmiri")
            print(f"\n  Kashmiri, real test rows only: {a['rate']:.4f} -> {b['rate']:.4f} "
                  f"over {a['n']:,} rows (z={z:+.2f}, p={p:.3g})")
            print(f"  {verdict}.")
            if b["rate"] <= a["rate"]:
                print("  On the stated criterion the rows have not earned their place "
                      "and should be dropped.")

    out = RESULTS / "model_comparison.json"
    out.write_text(json.dumps(
        {k: {"label": MODELS[k][1],
             "overall": v.get("overall"),
             "overall_real_rows_only": v.get("overall_real_rows_only"),
             "per_language": v.get("per_language"),
             "per_language_real_rows_only": v.get("per_language_real_rows_only"),
             "per_kind": v.get("per_kind")}
         for k, v in have.items()}, indent=1), encoding="utf-8")
    print(f"\nwritten to {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
