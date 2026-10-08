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
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-max", type=int, nargs="+", default=[4, 5, 6])
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.3, 0.1, 0.03])
    ap.add_argument("--top", type=int, nargs="+", default=[20000, 60000])
    ap.add_argument("--json", type=Path, default=ROOT / "data" / "langid_sweep.json")
    args = ap.parse_args()

    splits = wiki_splits()
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
                })  # fmt: skip
                print(
                    f"  n_max {n_max}  alpha {alpha:<5} top {top:>6}  "
                    f"val mean {mean:.4f}  whole {val['None']['accuracy']:.4f}  "
                    f"20ch {val['20']['accuracy']:.4f}  skr@20 {skr20:.4f}  "
                    f"{size / 1e6:.1f} MB json",
                    flush=True,
                )
        print(f"  (n_max {n_max}: counted in {counted:.0f}s)")

    best = max(rows, key=lambda r: (r["val_mean"], r["val_skr_20"]))
    print()
    print(f"best on validation: {best}")

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
    report = {
        "chosen": best,
        "sweep": rows,
        "train": {str(w): score(model, splits, "train", w) for w in WINDOWS},
        "val": {str(w): score(model, splits, "val", w) for w in WINDOWS},
        "test": {str(w): score(model, splits, "test", w) for w in WINDOWS},
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
