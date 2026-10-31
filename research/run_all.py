"""Regenerate every number and figure the paper uses, from committed artefacts.

    .venv-research/Scripts/python research/run_all.py

Reads `research/results/predictions.json.gz` (per-item, written by
`scripts/compare_langid.py`), `docs/langid-comparison.json` (the aggregates) and
`data/langid_report_v5.json` (the margin deciles and the retention curve). None of those
needs the 132,000-paragraph corpus, so every table and figure rebuilds from the
repository.

Writes `research/results/*.json` and `research/figures/*.{pdf,png}`, and prints each
figure's caption so a caption cannot drift from the figure it belongs to.

Nothing here is imported by the package. `research/` ships in neither the wheel nor the
sdist; `tests/test_research_is_not_packaged.py` checks that.
"""

from __future__ import annotations

import json
from pathlib import Path

import figures
import metrics

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "research" / "results"
TOOLS = ("urdunlp", "pali", "fasttext", "langdetect")
# PALI is scored by research/pali_baseline.py, which needs its model and the corpus; the
# rest come from the committed comparison artefact. A missing PALI file is reported and
# skipped rather than silently dropping a column.
PALI_PREDICTIONS = ROOT / "research" / "results" / "pali_predictions.json.gz"


def show_p(row: dict) -> str:
    """A p-value as text, including when it is too small to be a float.

    `mcnemar` returns None for `p_value` when the exact two-sided tail underflows - with
    1,725 discordant pairs and 8 on one side it does - and reports log10 instead. Printing
    "0" there would be publishing a p-value of exactly zero, which no test returns.
    """
    if row.get("p_value") is not None:
        return f"{row['p_value']:.3g}"
    return f"< {row['p_value_below']} (log10 p = {row['log10_p_value']})"


def write(name: str, payload: dict | list) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


def main() -> int:
    per_item = metrics.load_predictions()
    comparison = metrics.load_comparison()
    report = json.loads(
        (ROOT / "data" / "langid_report_v5.json").read_text(encoding="utf-8")
    )["identify_language"]

    truth = per_item["truth"]
    labels = per_item["languages"]
    whole = dict(per_item["by_length"]["whole"])

    tools = list(TOOLS)
    if PALI_PREDICTIONS.exists():
        import gzip  # noqa: PLC0415

        with gzip.open(PALI_PREDICTIONS, "rb") as fh:
            pali = json.loads(fh.read().decode("utf-8"))
        # The two runs build their item order the same way - languages sorted, test rows
        # in order - but that is an assumption worth refusing to make silently.
        assert pali["truth"] == truth, (
            "PALI's item order differs from the comparison's; the two cannot be paired"
        )
        whole["pali"] = pali["by_length"]["whole"]
    else:
        tools.remove("pali")
        print(f"  (no {PALI_PREDICTIONS.name}; run research/pali_baseline.py to add it)\n")
    print(f"{len(truth):,} test items, {len(labels)} languages, "
          f"{len(per_item['by_length'])} text lengths\n")

    # --- tables ---------------------------------------------------------------
    scores = {}
    for tool in tools:
        predicted = whole[tool]
        scored = metrics.per_class(truth, predicted, labels)
        scored["coverage"] = metrics.coverage(predicted, labels)
        scored["bootstrap"] = metrics.bootstrap(truth, predicted, labels)
        scores[tool] = scored
        boot = scored["bootstrap"]["accuracy_95"]
        print(f"  {tool:11} accuracy {scored['micro_accuracy']:.4f} "
              f"[{boot['lo']:.4f}, {boot['hi']:.4f}]  "
              f"macro-F1 {scored['macro_f1']:.4f} over {scored['macro_f1_over']} class(es)"
              f"  covers {scored['coverage']['covered']}/{scored['coverage']['of']}")
    write("per_class_whole", scores)

    # Paired tests: this tool against each baseline, on identical items.
    pairs = {}
    for tool in [t for t in tools if t != "urdunlp"]:
        pairs[f"urdunlp_vs_{tool}"] = metrics.mcnemar(truth, whole["urdunlp"], whole[tool])
        row = pairs[f"urdunlp_vs_{tool}"]
        print(f"\n  McNemar urdunlp vs {tool}: urdunlp-only right {row['a_only']:,}, "
              f"{tool}-only right {row['b_only']:,}, p = {show_p(row)}")
    write("mcnemar_whole", pairs)

    # The same, restricted to the classes the baseline can name - the fair comparison.
    restricted = {}
    for tool in [t for t in tools if t != "urdunlp"]:
        known = set(scores[tool]["coverage"]["has_a_label_for"])
        keep = [i for i, t in enumerate(truth) if t in known]
        t = [truth[i] for i in keep]
        a = [whole["urdunlp"][i] for i in keep]
        b = [whole[tool][i] for i in keep]
        restricted[tool] = {
            "n": len(keep),
            "classes": sorted(known),
            "urdunlp_accuracy": round(sum(1 for x, y in zip(t, a) if x == y) / len(keep), 4),
            f"{tool}_accuracy": round(sum(1 for x, y in zip(t, b) if x == y) / len(keep), 4),
            "mcnemar": metrics.mcnemar(t, a, b),
        }
        row = restricted[tool]
        print(f"  restricted to its {len(known)} label(s), n={row['n']:,}: "
              f"urdunlp {row['urdunlp_accuracy']:.4f} vs "
              f"{tool} {row[f'{tool}_accuracy']:.4f}, "
              f"p = {show_p(row['mcnemar'])}")
    write("restricted_whole", restricted)

    confusions = {tool: metrics.confusion(truth, whole[tool], labels) for tool in tools}
    write("confusion_whole", confusions)

    # --- per language, every system ------------------------------------------
    # The result rather than a diagnostic: the claim is about WHICH languages a general
    # detector collapses, so this table is what a reviewer reads first.
    order = ["ur", "pnb", "skr", "sd", "ps", "ks", "fa", "ar", "ckb", "ug", "azb"]
    assert set(order) == set(labels), (sorted(order), sorted(labels))
    per_language = {}
    print("\nper-language F1 on whole paragraphs "
          "(— = the system has no label for that language):")
    header = "  lang  support  " + "  ".join(f"{t:>11}" for t in tools)
    print(header)
    for code in order:
        row = {}
        cells = []
        for tool in tools:
            entry = scores[tool]["per_class"][code]
            row[tool] = {
                "precision": entry["precision"],
                "recall": entry["recall"],
                "f1": entry["f1"],
            }
            cells.append("          —" if entry["f1"] is None else f"{entry['f1']:>11.4f}")
        support = scores[tools[0]]["per_class"][code]["support"]
        per_language[code] = {"support": support, **row}
        print(f"  {code:5} {support:>7}  " + "  ".join(cells))
    write("per_language_whole", per_language)

    print("\nthe same, as recall - how much of each language each system recovers:")
    for code in order:
        cells = []
        for tool in tools:
            r = scores[tool]["per_class"][code]["recall"]
            cells.append("          —" if r is None else f"{r:>11.4f}")
        print(f"  {code:5} {'':>7}  " + "  ".join(cells))

    # --- figures --------------------------------------------------------------
    print("\nfigures:")
    captions: dict[str, str] = {}
    made = [
        figures.coverage_figure({t: scores[t]["coverage"] for t in tools
                                 if t in ("urdunlp", "fasttext", "langdetect")}),
        figures.length_figure(comparison["by_length"]),
        figures.per_class_figure(scores),
        figures.misattribution_figure(
            comparison["by_length"]["whole"]["langdetect"]["what_it_says_instead"],
            "langdetect",
        ),
        figures.confusion_figure(confusions["langdetect"]["rows"], "langdetect", 5),
        figures.margin_figure(report["test_whole"]["accuracy_by_margin_decile"]),
        figures.selective_figure(report["test_whole"]["error_retention_by_margin"]),
    ]
    for pdf, png, caption in made:
        captions[pdf.stem] = caption
        print(f"  {pdf.name:36} {caption[:88]}...")
    write("captions", captions)

    print(f"\n{len(made)} figure(s) and 5 table(s) written to research/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
