"""What the comparison looks like when the test set stops being balanced.

Every accuracy in section 5 is measured over a test set with roughly equal numbers of
each of the eleven languages. That is the standard way to build a language-identification
benchmark and it is not how text arrives. An experience report on OpenLID-v3 makes the
point directly: precision and F1 on standard LID datasets can mislead because models face
much larger language imbalance in real web crawls. Arabic and Persian are not 1/11 of
Perso-Arabic text on the web; Kashmiri and Saraiki are nowhere near it either.

The imbalance matters in a specific direction. Accuracy under a balanced prior is the
unweighted mean of per-class recall, so a system that is excellent on the eight rare
languages and mediocre on Arabic scores well. Under a realistic prior the mediocre
majority class dominates. Precision moves the other way and harder: a rare class whose
predictions are mostly correct on a balanced set can be mostly wrong on an imbalanced one,
because the false positives leaking in from a class a hundred times larger swamp its true
positives. That is Bayes, not a property of any model, and it is why a balanced benchmark
flatters a rare-language classifier.

**The method.** Reweighting, not re-running. Given a confusion matrix `C[t][p]` from the
balanced evaluation, per-class recall `C[t][t] / n_t` is unchanged by a change of prior
*provided the class-conditional distribution of text is unchanged* - the standard
prior-shift assumption, which this script states rather than hides. Under a target prior
`pi`, the reweighted joint is `W[t][p] = pi(t) * C[t][p] / n_t`, and accuracy, precision,
recall and F1 all follow from `W`. Nothing is re-predicted, so the numbers are exactly the
same predictions read under a different population.

**The priors.** Three, because picking one would be picking an answer:

  uniform     the balanced test set as published - the baseline, not a claim.
  wikipedia   article counts per language edition, a public, checkable proxy for how
              much text of each language is written down. All eleven have an edition.
  crawl       CC-100 bytes per language, the corpus this project draws from. It covers
              only six of the eleven: azb, ckb, ks, pnb and skr are absent from CC-100
              entirely, which is itself the finding - the languages a balanced benchmark
              over-represents most are the ones a crawl-built corpus does not contain.
              They are given the smallest observed share rather than zero, so the prior
              stays defined and the direction of the approximation is stated.

Run:  .venv-research/Scripts/python research/prior_shift.py
Writes research/results/prior_shift.json.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "research" / "results"
CONFUSION = RESULTS / "confusion_whole.json"

# Article counts, meta.wikimedia.org/wiki/List_of_Wikipedias/Table, read 2026-10-31.
WIKIPEDIA_ARTICLES = {
    "ar": 1_189_316,
    "azb": 242_515,
    "ckb": 43_616,
    "fa": 934_025,
    "ks": 2_268,
    "pnb": 66_324,
    "ps": 16_319,
    "sd": 15_276,
    "skr": 5_392,
    "ug": 5_405,
    "ur": 178_690,
}

# CC-100 per-language sizes, data.statmt.org/cc-100, read 2026-10-31, in megabytes.
# The five absent languages are recorded as None and filled with the smallest observed
# share by `_crawl_prior`, which is an approximation and is reported as one.
CC100_MEGABYTES: dict[str, float | None] = {
    "ar": 5_400.0,
    "azb": None,
    "ckb": None,
    "fa": 20_000.0,
    "ks": None,
    "pnb": None,
    "ps": 107.0,
    "sd": 67.0,
    "skr": None,
    "ug": 46.0,
    "ur": 884.0,
}


def _normalise(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def _crawl_prior(labels: list[str]) -> tuple[dict[str, float], list[str]]:
    present = {k: v for k, v in CC100_MEGABYTES.items() if v is not None and k in labels}
    absent = [k for k in labels if CC100_MEGABYTES.get(k) is None]
    floor = min(present.values())
    weights = {k: CC100_MEGABYTES[k] or floor for k in labels}
    return _normalise(weights), sorted(absent)


def priors(labels: list[str]) -> dict[str, dict]:
    uniform = _normalise(dict.fromkeys(labels, 1.0))
    wiki = _normalise({k: float(WIKIPEDIA_ARTICLES[k]) for k in labels})
    crawl, absent = _crawl_prior(labels)
    return {
        "uniform": {"weights": uniform, "note": "the balanced test set as published"},
        "wikipedia": {
            "weights": wiki,
            "note": "article counts per Wikipedia edition, read 2026-10-31",
        },
        "crawl": {
            "weights": crawl,
            "note": (
                "CC-100 bytes per language; "
                + ", ".join(absent)
                + " are absent from CC-100 and are given the smallest observed share, "
                "which understates nothing and overstates their presence"
            ),
            "approximated": absent,
        },
    }


def reweight(rows: dict[str, dict[str, int]], prior: dict[str, float]) -> dict:
    """Metrics under `prior`, from a confusion matrix measured under another one."""
    labels = sorted(rows)
    joint: dict[str, dict[str, float]] = {}
    for true in labels:
        n = sum(rows[true].values())
        if not n:
            joint[true] = dict.fromkeys(labels, 0.0)
            continue
        joint[true] = {p: prior[true] * rows[true].get(p, 0) / n for p in labels}

    accuracy = sum(joint[t][t] for t in labels)
    per_class = {}
    for label in labels:
        tp = joint[label][label]
        predicted = sum(joint[t][label] for t in labels)
        actual = sum(joint[label].values())
        precision = tp / predicted if predicted else None
        recall = tp / actual if actual else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall and (precision + recall)
            else 0.0
        )
        per_class[label] = {
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4),
            "share_of_the_population": round(prior[label], 6),
        }

    scored = [v for v in per_class.values() if v["precision"] is not None]
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(sum(v["f1"] for v in per_class.values()) / len(per_class), 4),
        "macro_precision": round(sum(v["precision"] for v in scored) / len(scored), 4)
        if scored
        else None,
        "per_class": per_class,
    }


def main() -> int:
    payload_in = json.loads(CONFUSION.read_text(encoding="utf-8"))
    systems = sorted(payload_in)
    labels = sorted(payload_in[systems[0]]["rows"])
    all_priors = priors(labels)

    out: dict = {
        "method": (
            "Prior shift applied to the published confusion matrices. Per-class recall "
            "is held fixed and the class prior is replaced; accuracy, precision and F1 "
            "are recomputed from the reweighted joint. Nothing is re-predicted. The "
            "assumption is that the distribution of text *within* a language does not "
            "change with the prior, which is the standard prior-shift assumption and is "
            "not tested here."
        ),
        "why": (
            "Every accuracy in the comparison is over a roughly balanced test set, and "
            "text does not arrive balanced. Precision on a rare class is the metric a "
            "balanced benchmark flatters most, because the false positives leaking in "
            "from a class a hundred times larger are not in the test set."
        ),
        "labels": labels,
        "priors": {
            k: {kk: vv for kk, vv in v.items() if kk != "weights"} for k, v in all_priors.items()
        },
        "prior_weights": {k: v["weights"] for k, v in all_priors.items()},
        "systems": {},
    }

    for system in systems:
        rows = payload_in[system]["rows"]
        out["systems"][system] = {
            name: reweight(rows, spec["weights"]) for name, spec in all_priors.items()
        }

    width = max(len(s) for s in systems)
    print(f"{'system':{width}}  " + "  ".join(f"{p:>22}" for p in all_priors))
    for system in systems:
        cells = []
        for name in all_priors:
            m = out["systems"][system][name]
            cells.append(f"acc {m['accuracy']:.4f} mP {m['macro_precision']:.4f}")
        print(f"{system:{width}}  " + "  ".join(f"{c:>22}" for c in cells))

    path = RESULTS / "prior_shift.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwritten to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
