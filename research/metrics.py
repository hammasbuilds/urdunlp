"""Metrics for the language-identification experiment, with their uncertainty.

Accuracy over eleven classes is one number and a reviewer's first question is what it
hides. So everything here is per-class or carries an interval:

  * **per-class precision, recall, F1** and support, plus **macro and micro** averages.
    Macro is the one to read: the classes are near-balanced by construction here, but a
    macro average is what makes a failure on Saraiki visible when ten other languages are
    easy.
  * **bootstrap confidence intervals** on accuracy and macro-F1, by resampling the test
    items. 10,000 resamples, a fixed seed, percentile interval. Reported because a single
    accuracy over 6,136 items invites "is that difference real".
  * **McNemar's exact test** between two systems on the same items. Paired, because the
    systems are scored on identical paragraphs, and an unpaired comparison of two rates
    throws away exactly the information that makes small differences decidable.
  * **the confusion matrix**, which for this task is the result rather than a diagnostic:
    the claim is about *which* languages a general detector collapses into Urdu.

One deliberate omission: no accuracy is reported for a baseline on a class it has no
label for, except where the page says so explicitly. A detector with no Saraiki label is
not wrong about Saraiki in any sense a metric should average over - it was asked a
question it cannot answer. Those items are reported separately, never folded in.
"""

from __future__ import annotations

import gzip
import json
import math
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS = ROOT / "research" / "results" / "predictions.json.gz"
COMPARISON = ROOT / "docs" / "langid-comparison.json"

BOOTSTRAP_RESAMPLES = 10_000
SEED = 20261031


def load_predictions(path: Path = PREDICTIONS) -> dict:
    with gzip.open(path, "rb") as fh:
        return json.loads(fh.read().decode("utf-8"))


def load_comparison(path: Path = COMPARISON) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# --- per-class -----------------------------------------------------------------------


def per_class(truth: list[str], predicted: list[str], labels: list[str]) -> dict:
    """Precision, recall, F1 and support for each label, plus macro and micro.

    Written out rather than taken from scikit-learn so the convention for an empty
    denominator is explicit and visible: a class the system never predicts has precision
    **undefined**, not 0.0. Reporting 0.0 there reads as "it predicted this class and got
    it wrong every time", which is a different failure from "it never used this label" -
    and the difference is the whole point of the coverage argument this experiment makes.
    """
    assert len(truth) == len(predicted), (len(truth), len(predicted))
    out: dict[str, dict] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(truth, predicted) if t == label and p == label)
        fp = sum(1 for t, p in zip(truth, predicted) if t != label and p == label)
        fn = sum(1 for t, p in zip(truth, predicted) if t == label and p != label)
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / support if support else None
        if precision and recall:
            f1 = 2 * precision * recall / (precision + recall)
        elif precision is None or recall is None:
            f1 = None
        else:
            f1 = 0.0
        out[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted_as_this": tp + fp,
        }

    scored = [v for v in out.values() if v["f1"] is not None]
    total = len(truth)
    correct = sum(1 for t, p in zip(truth, predicted) if t == p)
    return {
        "per_class": out,
        # Macro over the classes that have an F1 at all, with the count stated: a macro
        # average over "the classes it can name" is a different number from one over all
        # eleven, and a paper that does not say which it used cannot be checked.
        "macro_f1": (sum(v["f1"] for v in scored) / len(scored)) if scored else None,
        "macro_f1_over": len(scored),
        "classes": len(labels),
        "micro_accuracy": correct / total if total else None,
        "n": total,
    }


# --- uncertainty ---------------------------------------------------------------------


def bootstrap(
    truth: list[str],
    predicted: list[str],
    labels: list[str],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = SEED,
) -> dict:
    """Percentile intervals for accuracy and macro-F1, by resampling items.

    Items, not classes: the sampling unit has to be the thing that was independently
    drawn, and these are independently drawn paragraphs. A fixed seed, so the interval is
    the same number twice.
    """
    rng = random.Random(seed)
    n = len(truth)
    indices = range(n)
    accuracies: list[float] = []
    macros: list[float] = []
    for _ in range(resamples):
        sample = [rng.choice(indices) for _ in range(n)]
        t = [truth[i] for i in sample]
        p = [predicted[i] for i in sample]
        accuracies.append(sum(1 for a, b in zip(t, p) if a == b) / n)
        scored = per_class(t, p, labels)
        if scored["macro_f1"] is not None:
            macros.append(scored["macro_f1"])

    def interval(values: list[float]) -> dict | None:
        if not values:
            return None
        values = sorted(values)
        lo = values[int(0.025 * len(values))]
        hi = values[min(len(values) - 1, int(0.975 * len(values)))]
        return {"lo": round(lo, 4), "hi": round(hi, 4), "width": round(hi - lo, 4)}

    return {
        "resamples": resamples,
        "seed": seed,
        "accuracy_95": interval(accuracies),
        "macro_f1_95": interval(macros),
    }


def mcnemar(truth: list[str], a: list[str], b: list[str]) -> dict:
    """Exact McNemar on the items where exactly one of the two systems is right.

    Paired, because both systems answered the same paragraphs. The discordant pairs are
    the whole of the evidence: items both get right or both get wrong say nothing about
    which is better, and an unpaired comparison of two accuracies discards precisely the
    pairing that makes a small difference decidable.

    Exact binomial rather than the chi-square approximation: with a strongly lopsided
    table - which is what a coverage gap produces - the approximation is the wrong tool.
    """
    assert len(truth) == len(a) == len(b)
    a_only = sum(1 for t, x, y in zip(truth, a, b) if x == t and y != t)
    b_only = sum(1 for t, x, y in zip(truth, a, b) if x != t and y == t)
    discordant = a_only + b_only
    if discordant == 0:
        return {"a_only": 0, "b_only": 0, "discordant": 0, "p_value": 1.0,
                "note": "the two systems are right and wrong on exactly the same items"}
    # Two-sided exact: P(X <= min) * 2 under Binomial(n, 0.5).
    #
    # In log space. Done directly, `sum(comb(n, k)) / 2**n` underflows to 0.0 for a
    # lopsided table - 1,725 discordant pairs with 8 on one side gave `p = 0`, and no
    # test returns a p-value of exactly zero. log10 is always reported; `p_value` is the
    # float where one exists and None where it does not, with `p_value_below` carrying
    # the bound instead.
    smaller = min(a_only, b_only)
    log10_2 = math.log10(2.0)
    terms = []
    for k in range(smaller + 1):
        log_comb = (
            math.lgamma(discordant + 1)
            - math.lgamma(k + 1)
            - math.lgamma(discordant - k + 1)
        ) / math.log(10.0)
        terms.append(log_comb - discordant * log10_2)
    # log10 of a sum of terms given their log10s, without leaving log space.
    biggest = max(terms)
    log10_tail = biggest + math.log10(sum(10 ** (t - biggest) for t in terms))
    log10_p = min(0.0, log10_tail + log10_2)

    exact: float | None
    try:
        exact = min(1.0, 10 ** log10_p)
        if exact == 0.0:
            exact = None
    except OverflowError:  # pragma: no cover - defensive
        exact = None
    return {
        "a_only": a_only,
        "b_only": b_only,
        "discordant": discordant,
        "p_value": exact,
        "log10_p_value": round(log10_p, 2),
        # What to write in a paper when the float cannot hold it.
        "p_value_below": None if exact is not None else f"1e{math.ceil(log10_p):.0f}",
    }


# --- confusion -----------------------------------------------------------------------


def confusion(truth: list[str], predicted: list[str], labels: list[str]) -> dict:
    """Counts of (true, predicted), including labels outside the task's eleven.

    A baseline answers `fa` for Pashto and `pa` for nothing at all; those columns are kept
    rather than collapsed into "other", because which wrong answer a detector gives is the
    finding. `__refused__` is kept too: langdetect raises on input it cannot profile, and
    a refusal is not a wrong answer.
    """
    rows: dict[str, dict[str, int]] = {}
    for label in labels:
        counts = Counter(p for t, p in zip(truth, predicted) if t == label)
        rows[label] = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
    predicted_labels = sorted({p for p in predicted})
    return {
        "rows": rows,
        "labels_true": labels,
        "labels_predicted": predicted_labels,
        "outside_the_task": sorted(set(predicted_labels) - set(labels)),
    }


def coverage(predicted_labels: list[str], labels: list[str]) -> dict:
    """Which of the task's classes a system can name at all."""
    known = [c for c in labels if c in set(predicted_labels)]
    return {
        "has_a_label_for": known,
        "has_no_label_for": [c for c in labels if c not in set(predicted_labels)],
        "covered": len(known),
        "of": len(labels),
    }
