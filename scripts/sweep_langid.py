r"""Sweep the script model's three settings, choose on validation, report on test.

    python scripts/sweep_langid.py
    python scripts/sweep_langid.py --n-max 5 6 --alpha 0.1 0.03 --top 20000 60000

`build_langid_models.py` carries the settings as constants with the sweep behind them
written in a comment: `N_MAX = 5  # 3 -> 92.9%, 4 -> 94.6%, 5 -> 96.2% unpruned`. That is
the right thing to record, and it leaves two questions open. 5 was the largest value
tried, so it is the top of the range rather than a measured optimum; and `TOP_GRAMS =
20000` is a package-size decision whose cost - 96.2% unpruned against 94.9% pruned - is
stated but never traded off against the other two settings together.

The weakness this is aimed at is specific. Overall test accuracy is 97.9%, and Saraiki is
92.6% whole and **75.1% on 20-character windows**. The dominant error is one direction:
`skr -> pnb`, Saraiki read as Shahmukhi Punjabi, 39 of 56 errors at 20 characters. The two
are close - same script, Indo-Aryan, overlapping vocabulary - and Saraiki's four implosive
letters, which would settle it, are shared with Sindhi and absent from most short windows.

Selection is on **validation** and the winner is then reported on **test**, once. Picking
on test and quoting test is how a sweep manufactures an improvement that does not exist.
The per-language column matters more than the mean here: a setting that adds 0.2 points
overall by trading Saraiki away is not an improvement to this model.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_langid_models import distinctive_letters, wiki_splits  # noqa: E402
from measure_langid import window  # noqa: E402

from urdunlp.langid import _grams, _ScriptModel  # noqa: E402

WINDOWS = (None, 50, 20, 10)


def score(model: _ScriptModel, splits: dict, split: str, length: int | None) -> dict:
    """Accuracy overall and per language on one split at one window length."""
    per: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    confusion: collections.Counter = collections.Counter()
    outcomes = bytearray()
    for code, parts in splits.items():
        for paragraph in parts[split]:
            # The repository's own centred slice, not the first N characters. A leading
            # slice reads 0.836 where the published val figure is 0.904, because the
            # start of a Wikipedia intro is disproportionately a title and a date - and
            # a sweep measured one way and compared against numbers measured the other
            # chooses on a difference that is not there.
            text = window(paragraph, length)
            if text is None or not text.strip():
                continue
            scores = model.log_likelihoods(text)
            guess = max(scores, key=scores.get)
            per[code][1] += 1
            outcomes.append(1 if guess == code else 0)
            if guess == code:
                per[code][0] += 1
            else:
                confusion[(code, guess)] += 1
    right = sum(v[0] for v in per.values())
    total = sum(v[1] for v in per.values())
    return {
        "accuracy": right / total if total else 0.0,
        "n": total,
        "per_language": {k: round(v[0] / v[1], 4) for k, v in sorted(per.items()) if v[1]},
        "top_confusions": [[a, b, n] for (a, b), n in confusion.most_common(5)],
        # One byte per scored paragraph, in iteration order, so two settings can be
        # compared paragraph by paragraph instead of rate against rate. Which paragraphs
        # are scored depends on `window()` alone - a property of the text, not of the
        # model - so two settings produce the same order and the same length, which
        # separable() asserts before comparing. Not written to the report: it is 79,388
        # bytes per setting and it is evidence for a choice, not a published figure.
        "outcomes": bytes(outcomes),
    }


# Selection, and why it is not accuracy.
#
# This file's docstring says "the per-language column matters more than the mean here: a
# setting that adds 0.2 points overall by trading Saraiki away is not an improvement".
# That principle is right, and the rule that used to act on it was
#
#     max(rows, key=lambda r: (r["val_mean"], r["val_skr_20"]))
#
# which reads Saraiki only when two means are exactly equal as floats - across 27 settings,
# never. So it was effectively mean-only.
#
# Selecting on Saraiki instead was tried, and it picked a model that is worse on test:
# n_max 6 / alpha 0.03 / top 90000 won on validation Saraiki@20 by 0.54 points and then
# lost on test Saraiki@20 by 1.10 points, while being slightly worse at three of the four
# windows overall. The reason is resolution, not bad luck. Saraiki has 1,827 validation and
# 549 test paragraphs, so at p ~ 0.80 the standard error is 0.95 and 1.65 points; both the
# gain and the loss sit inside one standard error. The column cannot separate these settings.
#
# Seven of the 27 settings are within 0.2 points of the best mean, and within noise of each
# other on Saraiki. When accuracy cannot tell them apart, the remaining difference that IS
# measured exactly is model size, which is wheel size for every user and nothing else. So:
# the mean defines a band, and inside the band the smallest model wins.
#
# "Cannot tell them apart" was a constant, MEAN_TOLERANCE = 0.002, and the paragraph above
# justifies it with the standard error of a SINGLE rate - 0.95 points for Saraiki on 1,827
# paragraphs. That is the right error bar for "how well do we know this model's accuracy"
# and the wrong one for "is A better than B", because both settings are scored on the same
# paragraphs. A paired comparison resolves differences far smaller than either model's own
# interval, and it did: the rule picked top=120000 over top=180000 on a val_mean gap of
# 0.0020, while on the discordant validation paragraphs at 20 characters 180000 wins 149
# and loses 98, an exact p of 0.0014. The ladder had not flattened; the tolerance could
# not see that it had not.
#
# So the band is defined by a test rather than a number: a setting is in it when no window
# length shows the ceiling significantly better, Holm-corrected across the four lengths.
ALPHA_LEVEL = 0.05


def mcnemar(a: bytes, b: bytes) -> tuple[int, int, float]:
    """Wins for `a`, wins for `b`, and the exact two-sided p over the discordant pairs.

    Exact rather than chi-squared: the discordant count can be small for whole paragraphs
    (40 of 19,847 here) and the continuity-corrected approximation is unreliable there.
    """
    if len(a) != len(b):
        raise ValueError(
            f"outcome vectors of different lengths ({len(a)} and {len(b)}) cannot be "
            f"paired - the scored paragraphs must be the same ones in the same order"
        )
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    b_only = sum(1 for x, y in zip(a, b) if y and not x)
    discordant = a_only + b_only
    if not discordant:
        return a_only, b_only, 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(discordant, i) for i in range(k + 1)) / 2 ** discordant
    return a_only, b_only, min(1.0, 2 * tail)


def separable(candidate: dict, ceiling: dict) -> dict | None:
    """The window length at which the ceiling beats the candidate, or None.

    Holm across the four lengths, because four tests are run and the question asked is
    "is this setting worse at any length". Only a loss counts: a candidate that is
    significantly BETTER somewhere is not excluded for it - the ceiling is the setting
    with the best mean, and a smaller model that wins a length is exactly what is wanted.
    """
    tests = []
    for length, outcomes in candidate["outcomes"].items():
        wins, losses, p = mcnemar(outcomes, ceiling["outcomes"][length])
        if losses > wins:  # the ceiling is ahead at this length
            tests.append((p, length, wins, losses))
    # By p only. Sorting the tuples compares `length` whenever two p-values tie, and
    # `None` is one of the four lengths, so an exact tie between the whole-paragraph test
    # and any other raised TypeError - in the selection rule, after the sweep's work.
    tests.sort(key=lambda t: t[0])
    for i, (p, length, wins, losses) in enumerate(tests):
        if p <= ALPHA_LEVEL / (len(tests) - i):
            return {"length": length, "p": p, "candidate_wins": wins,
                    "ceiling_wins": losses}
    return None


def choose(rows: list[dict]) -> dict:
    """The smallest model no window length shows to be worse than the best mean.

    Falls back to the old behaviour - a tolerance band on the mean - for rows that carry
    no outcome vectors, so a report loaded from disk can still be re-chosen from.
    """
    ceiling = max(rows, key=lambda r: r["val_mean"])
    if not all(r.get("outcomes") for r in rows):
        close = [r for r in rows if ceiling["val_mean"] - r["val_mean"] <= 0.002]
        return min(close, key=lambda r: (r["json_bytes"], -r["val_skr_20"]))
    for row in sorted(rows, key=lambda r: (r["json_bytes"], -r["val_skr_20"])):
        if row is ceiling or separable(row, ceiling) is None:
            return row
    return ceiling


def _published(row: dict) -> dict:
    """A score row without its outcome vector, which is not JSON and not a figure."""
    return {k: v for k, v in row.items() if k != "outcomes"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-max", type=int, nargs="+", default=[4, 5, 6])
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.3, 0.1, 0.03])
    ap.add_argument("--top", type=int, nargs="+", default=[20000, 60000])
    ap.add_argument(
        "--wiki-dir",
        type=Path,
        help="corpus directory to sweep over (default data/wiki). The builder and the "
        "measurement both take this; without it here the settings would be chosen on "
        "the 1,500-per-language sample and then applied to a corpus 11x larger",
    )
    ap.add_argument("--json", type=Path, default=ROOT / "data" / "langid_sweep.json")
    args = ap.parse_args()

    splits = wiki_splits(args.wiki_dir)
    print(
        "paragraphs: "
        + ", ".join(f"{k} {len(v['train'])}/{len(v['val'])}/{len(v['test'])}"
                    for k, v in sorted(splits.items()))
    )  # fmt: skip
    print()

    rows = []
    # Gram counting is the expensive part and depends only on n_max, so it is done once
    # per n_max rather than once per combination.
    for n_max in args.n_max:
        started = time.time()
        counts = {}
        for code, parts in splits.items():
            c: collections.Counter = collections.Counter()
            for paragraph in parts["train"]:
                c.update(_grams(paragraph, n_max))
            counts[code] = c
        counted = time.time() - started
        distinctive = distinctive_letters(counts)
        for top in args.top:
            kept = {k: dict(c.most_common(top)) for k, c in counts.items()}
            size = len(json.dumps(kept))
            vocabulary = len(set().union(*kept.values()))
            for alpha in args.alpha:
                model = _ScriptModel({
                    "n_max": n_max, "alpha": alpha,
                    "vocabulary_size": vocabulary, "counts": kept,
                    "distinctive": distinctive,
                })  # fmt: skip
                val = {str(w): score(model, splits, "val", w) for w in WINDOWS}
                # The mean over window lengths, so a setting cannot win by being good
                # only on whole paragraphs - the short-window case is the weak one.
                mean = sum(v["accuracy"] for v in val.values()) / len(val)
                skr20 = val["20"]["per_language"].get("skr", 0.0)
                rows.append({
                    "n_max": n_max, "alpha": alpha, "top": top,
                    "val_mean": round(mean, 5),
                    "val_whole": round(val["None"]["accuracy"], 5),
                    "val_20": round(val["20"]["accuracy"], 5),
                    "val_skr_20": round(skr20, 5),
                    "json_bytes": size,
                    # Dropped before the report is written; see score().
                    "outcomes": {w: val[str(w)]["outcomes"] for w in WINDOWS},
                })  # fmt: skip
                print(
                    f"  n_max {n_max}  alpha {alpha:<5} top {top:>6}  "
                    f"val mean {mean:.4f}  whole {val['None']['accuracy']:.4f}  "
                    f"20ch {val['20']['accuracy']:.4f}  skr@20 {skr20:.4f}  "
                    f"{size / 1e6:.1f} MB json",
                    flush=True,
                )
        print(f"  (n_max {n_max}: counted in {counted:.0f}s)")

    best = choose(rows)
    print()
    # Without the outcome vectors: adding them to the row made this one line dump
    # 79,388 bytes per window into the log, 300 KB in all.
    print(f"best on validation: {_published(best)}")

    # And now, once, on test.
    counts = {}
    for code, parts in splits.items():
        c = collections.Counter()
        for paragraph in parts["train"]:
            c.update(_grams(paragraph, best["n_max"]))
        counts[code] = c
    kept = {k: dict(c.most_common(best["top"])) for k, c in counts.items()}
    model = _ScriptModel({
        "n_max": best["n_max"], "alpha": best["alpha"],
        "vocabulary_size": len(set().union(*kept.values())),
        "counts": kept, "distinctive": distinctive_letters(counts),
    })  # fmt: skip
    # Why this setting and not the one with the best mean: the paired comparison that
    # decided it, published rather than left in the terminal. A tolerance band records no
    # evidence at all, which is how a choice made on a 0.0020 gap in the mean went
    # unexamined while a paired test on the same paragraphs separated the two at p=0.0014.
    ceiling = max(rows, key=lambda r: r["val_mean"])
    chosen_against_ceiling = None
    if best is not ceiling and best.get("outcomes") and ceiling.get("outcomes"):
        chosen_against_ceiling = {
            "ceiling": {k: ceiling[k] for k in ("n_max", "alpha", "top", "val_mean")},
            "by_window": {},
        }
        for w in WINDOWS:
            wins, losses, pvalue = mcnemar(
                best["outcomes"][w], ceiling["outcomes"][w]
            )
            chosen_against_ceiling["by_window"][str(w)] = {
                "chosen_only_right": wins,
                "ceiling_only_right": losses,
                "exact_p": round(pvalue, 6),
            }

    report = {
        # Which corpus these settings were chosen on. The committed sweep carried
        # `corpus: None`, so a sweep over data/wiki and one over data/wiki_large -
        # eleven times larger, and giving a different winner - were indistinguishable
        # files. The model blob records its corpus for the same reason.
        "corpus": (args.wiki_dir or (ROOT / "data" / "wiki")).name,
        "chosen": {k: v for k, v in best.items() if k != "outcomes"},
        "chosen_against_best_mean": chosen_against_ceiling,
        # The outcome vectors are evidence for the choice, not a published figure, and
        # they are 79,388 bytes per setting - so they are dropped here rather than
        # written. choose() falls back to the old tolerance band for rows without them,
        # which is what a report reloaded from this file has.
        "sweep": [{k: v for k, v in r.items() if k != "outcomes"} for r in rows],
        "train": {str(w): _published(score(model, splits, "train", w)) for w in WINDOWS},
        "val": {str(w): _published(score(model, splits, "val", w)) for w in WINDOWS},
        "test": {str(w): _published(score(model, splits, "test", w)) for w in WINDOWS},
    }
    for split in ("train", "val", "test"):
        print(
            f"  {split:5} "
            + "  ".join(f"{w or 'whole'}: {report[split][str(w)]['accuracy']:.4f}" for w in WINDOWS)
        )
    print(
        "  test skr:", {str(w): report["test"][str(w)]["per_language"].get("skr") for w in WINDOWS}
    )
    args.json.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
