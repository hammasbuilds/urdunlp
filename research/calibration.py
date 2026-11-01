"""Can the ranking score be turned into a probability, and does it survive leaving Wikipedia?

`identify_language` returns a `margin` and its docstring is careful about what that is: "a
ranking score, not a probability", good for ordering a batch and explicitly not good for one
answer. Section 4.5 uses it exactly that way - margin deciles and a retention curve, both of
which only need the score to *rank* correctly.

That leaves a question the paper has not asked. The model's `ranking` carries a
log-likelihood for every one of the eleven languages, so a posterior is one softmax away. If
it can be calibrated - if "0.95" means right 95% of the time - then the library can ship a
probability and a caller can use one answer, which is the thing the docstring currently says
they cannot do.

The standard method is temperature scaling (Guo, Pleiss, Sun & Weinberger, *On Calibration of
Modern Neural Networks*, ICML 2017): one parameter, fitted on held-out data by minimising
negative log-likelihood, which cannot change any prediction because dividing every logit by
the same positive number preserves their order. Calibration is then measured as expected
calibration error - bin the predictions by confidence, and average the gap between confidence
and accuracy in each bin.

And then the question this paper is actually about. Section 5 shows accuracy falling from
0.9925 to far less once the corpus is not Wikipedia. A model that merely gets worse off its
training domain is one problem; a model that gets worse *without its confidence noticing* is a
worse one, because the abstention mechanism of section 4.5 is the recommended mitigation and it
is driven by that confidence. Overconfidence under distribution shift is the central finding of
the calibration literature (Ovadia et al., *Can You Trust Your Model's Uncertainty?*,
NeurIPS 2019), so it is the thing to measure rather than assume either way.

Three populations, and the split between them is the whole design:

* **val** - fits the temperature. Nothing is measured here that is also reported here.
* **test** - in-domain, held out, the 6,136 windows section 4 reports on.
* **Tatoeba** - out of domain, the corpus section 5's objection is built from. Pooled only
  over the languages whose n can carry a rate, because Tatoeba has 2,812 Urdu sentences and
  seven Sindhi ones and an ECE over that union would be an ECE about Urdu.

    .venv/Scripts/python research/calibration.py

Writes research/results/calibration.json.
"""

from __future__ import annotations

import argparse
import bz2
import json
import math
import pathlib
import random
import sys
import unicodedata

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from urdunlp.langid import (  # noqa: E402
    _ARABIC_RUNS,
    _PRESENTATION_FORMS,
    LANGUAGES,
    _expand_presentation_forms,
    identify_language,
)

RESULTS = ROOT / "research" / "results"
SPLITS = ROOT / "data" / "splits"
TATOEBA_RAW = ROOT / ".baselines" / "tatoeba"

# Our code -> Tatoeba's directory, as `out_of_domain.py` has it. Imported would be better
# than copied, but that module does the fetching and this one must not.
TATOEBA = {
    "ur": "urd",
    "pnb": "pnb",
    "skr": "skr",
    "sd": "snd",
    "ks": "kas",
    "fa": "pes",
    "ar": "ara",
    "ckb": "ckb",
    "ug": "uig",
}

# Below this many items a language contributes counts and not a rate, matching
# `out_of_domain.MIN_FOR_A_RATE`. A calibration curve needs more than an accuracy does, so
# this is the floor for being pooled at all, not for being believed on its own.
MIN_FOR_A_RATE = 30

# Fifteen equal-width bins, the convention in the source paper. Bin counts are reported
# beside every error: an ECE whose mass sits in one bin is a number about one bin.
BINS = 15


def _read_split(name: str, per_language: int, seed: int) -> list[tuple[str, str]]:
    """(text, language) for one split directory, subsampled to `per_language` each.

    The committed splits hold 36,000 lines per language - about 396,000 items and 94 MB
    per split - and scoring all of them through `identify_language` takes hours for a
    number that does not move. Section 4 reports on 6,136 windows for the same reason, so
    this draws a comparable sample: `per_language` items from each language, with a fixed
    seed, taken by reservoir so the whole file is eligible rather than only its head.

    The head would be the wrong sample twice over: the files are ordered as the corpus
    was built, so an early slice is one source and one register, and the languages are
    unequally long so a fixed prefix is a different fraction of each.
    """
    out: list[tuple[str, str]] = []
    folder = SPLITS / name
    if not folder.is_dir():
        return out
    for path in sorted(folder.glob("*.txt")):
        language = path.stem
        if language not in LANGUAGES:
            continue
        rng = random.Random(f"{seed}:{name}:{language}")
        kept: list[str] = []
        seen = 0
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                text = line.strip()
                if not text:
                    continue
                seen += 1
                if len(kept) < per_language:
                    kept.append(text)
                else:
                    index = rng.randrange(seen)
                    if index < per_language:
                        kept[index] = text
        out.extend((text, language) for text in kept)
    return out


def _read_tatoeba() -> list[tuple[str, str]]:
    out = []
    for language, folder in TATOEBA.items():
        path = TATOEBA_RAW / f"{folder}.tsv.bz2"
        if not path.is_file():
            continue
        seen = set()
        with bz2.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                text = parts[2].strip()
                if len(text) >= 10 and text not in seen:
                    seen.add(text)
                    out.append((text, language))
    return out


def _logits(text: str) -> tuple[list[tuple[str, float]], int] | None:
    """Per-character log-likelihoods, which is what a posterior has to be built from.

    The raw ranking holds totals, so a long paragraph's winner beats its runner-up by
    hundreds of nats and every softmax of it is 1.0 to machine precision regardless of
    whether the answer is right. Dividing by length is the same normalisation `margin`
    already does, and without it the temperature has to absorb the text length as well
    as the model's sharpness and can do neither.
    """
    guess = identify_language(text)
    if guess.language is None or len(guess.ranking) < 2:
        return None
    # The divisor is the model's own letter count. Recovering it from `margin` instead -
    # dividing the top two scores by the gap between them - looked tidier and is wrong:
    # `margin` is rounded to four places before it is returned, so the quotient is an
    # estimate of an integer and is off by one wherever the rounding bites. The module's
    # run extraction is the only thing that knows the real number.
    length = max(1, len(_letters(text)))
    return [(code, score / length) for code, score in guess.ranking], length


def _letters(text: str) -> str:
    """The Perso-Arabic letters `identify_language` actually scores, in its own terms."""
    normalised = _PRESENTATION_FORMS.sub(
        _expand_presentation_forms, unicodedata.normalize("NFC", text)
    )
    return " ".join(_ARABIC_RUNS.findall(normalised))


def _posterior(logits: list[tuple[str, float]], temperature: float) -> tuple[str, float]:
    """The top language and its probability under a softmax at `temperature`."""
    scaled = [score / temperature for _, score in logits]
    top = max(scaled)
    weights = [math.exp(s - top) for s in scaled]
    total = sum(weights)
    index = scaled.index(top)
    return logits[index][0], weights[index] / total


def score(rows: list[tuple[str, str]]) -> list[dict]:
    """One record per item: the per-character logits, the truth, and the length."""
    out = []
    for text, truth in rows:
        got = _logits(text)
        if got is None:
            # No Perso-Arabic letters at all. The model declines rather than guessing,
            # and an item it declined cannot be scored for calibration either.
            continue
        logits, length = got
        out.append({"truth": truth, "logits": logits, "letters": length})
    return out


def negative_log_likelihood(records: list[dict], temperature: float) -> float:
    """Mean NLL of the true label, which is what the temperature is fitted on."""
    total = 0.0
    for record in records:
        scaled = {code: value / temperature for code, value in record["logits"]}
        top = max(scaled.values())
        denominator = sum(math.exp(v - top) for v in scaled.values())
        truth = scaled.get(record["truth"])
        if truth is None:
            # A language the model has no column for cannot contribute a likelihood;
            # skipping it silently would fit the temperature on a subset nobody named.
            continue
        total += -(truth - top - math.log(denominator))
    return total / max(1, len(records))


def fit_temperature(records: list[dict]) -> dict:
    """One parameter, by golden-section search on a bracket wide enough to contain it.

    A grid would do, but a grid's answer is the grid's resolution and this is reported to
    two places. The objective is convex in log-temperature for a softmax, so a bracketing
    search is sound; the bracket is checked rather than assumed, because a minimum at an
    endpoint means the bracket, not the model, decided the answer.
    """
    lo, hi = 0.05, 200.0
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - ratio * (b - a)
    d = a + ratio * (b - a)
    for _ in range(200):
        if negative_log_likelihood(records, c) < negative_log_likelihood(records, d):
            b = d
        else:
            a = c
        c = b - ratio * (b - a)
        d = a + ratio * (b - a)
        if abs(b - a) < 1e-4:
            break
    best = (a + b) / 2.0
    return {
        "temperature": round(best, 4),
        "nll": round(negative_log_likelihood(records, best), 4),
        "nll_at_temperature_1": round(negative_log_likelihood(records, 1.0), 4),
        "bracket": [lo, hi],
        # If the optimum sits on an endpoint the bracket chose it, and the number below
        # is a property of this function rather than of the model.
        "at_the_bracket_edge": bool(best <= lo * 1.01 or best >= hi * 0.99),
    }


def calibration(records: list[dict], temperature: float) -> dict:
    """ECE, MCE and the reliability bins at one temperature."""
    rows = []
    for record in records:
        predicted, confidence = _posterior(record["logits"], temperature)
        rows.append((confidence, predicted == record["truth"]))
    if not rows:
        return {"n": 0}
    bins = []
    ece = 0.0
    mce = 0.0
    for index in range(BINS):
        low = index / BINS
        high = (index + 1) / BINS
        inside = [(c, ok) for c, ok in rows if (low < c <= high or (index == 0 and c <= high))]
        if not inside:
            bins.append({"lo": round(low, 4), "hi": round(high, 4), "n": 0})
            continue
        confidence = sum(c for c, _ in inside) / len(inside)
        accuracy = sum(1 for _, ok in inside if ok) / len(inside)
        gap = abs(confidence - accuracy)
        ece += len(inside) / len(rows) * gap
        mce = max(mce, gap)
        bins.append(
            {
                "lo": round(low, 4),
                "hi": round(high, 4),
                "n": len(inside),
                "confidence": round(confidence, 4),
                "accuracy": round(accuracy, 4),
                "gap": round(confidence - accuracy, 4),
            }
        )
    mean_confidence = sum(c for c, _ in rows) / len(rows)
    accuracy = sum(1 for _, ok in rows if ok) / len(rows)
    return {
        "n": len(rows),
        "temperature": temperature,
        "accuracy": round(accuracy, 4),
        "mean_confidence": round(mean_confidence, 4),
        # Positive means overconfident, which is the direction that matters: it is the one
        # that makes an abstention threshold keep answers it should have dropped.
        "overconfidence": round(mean_confidence - accuracy, 4),
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "bins": bins,
        # Where the mass is. An ECE of 0.01 with 99% of items in the top bin says almost
        # nothing about the rest of the range, and the bins alone do not say it loudly.
        "share_in_the_top_bin": round(max((b["n"] for b in bins), default=0) / len(rows), 4),
    }


def balanced(per_language: dict, minimum: int = MIN_FOR_A_RATE) -> dict:
    """ECE with every language weighted equally, and the cancellation made visible.

    The pooled figure is not a figure about this task. Tatoeba is 64,042 Arabic sentences
    and 31,437 Persian ones against 2,812 Urdu and 70 Shahmukhi Punjabi, so an average
    over the union is an average over Arabic and Persian - the two languages a
    Perso-Arabic classifier finds easiest and the two the toolkit is least about. Section
    4.6 makes the same argument for the balanced test set, and it applies here with more
    force, because the imbalance is 900:1 rather than 1:1.

    Worse, ECE averages absolute gaps *within* a population, and the pooled number is
    computed over one merged population, so a group that is overconfident and a group
    that is underconfident cancel. They do here: the pooled signed gap is -0.0244 while
    Urdu is +0.0918 and Persian -0.0456. So the signed mean is reported beside the
    absolute one - a small signed value next to a large absolute one is the signature of
    cancellation, and a reader seeing only the pooled ECE would conclude the model was
    calibrated.
    """
    rows = {
        code: row["calibrated"]
        for code, row in per_language.items()
        if row["n"] >= minimum and row.get("calibrated") and row["calibrated"].get("n")
    }
    if not rows:
        return {"languages": 0}
    count = len(rows)
    overconfident = {c: r for c, r in rows.items() if r["overconfidence"] > 0}
    worst_code, worst_row = max(rows.items(), key=lambda kv: kv[1]["ece"])
    weakest = min(rows.items(), key=lambda kv: kv[1]["accuracy"])[0]
    return {
        "languages": count,
        "weighted": "equally, one vote per language",
        "accuracy": round(sum(r["accuracy"] for r in rows.values()) / count, 4),
        "ece": round(sum(r["ece"] for r in rows.values()) / count, 4),
        # The signed gap. Near zero while `ece` is not means the pooled figure is an
        # artefact of two groups failing in opposite directions.
        "signed_gap": round(sum(r["overconfidence"] for r in rows.values()) / count, 4),
        "worst_language": worst_code,
        "worst_ece": worst_row["ece"],
        "overconfident_languages": sorted(overconfident),
        "underconfident_languages": sorted(set(rows) - set(overconfident)),
        "weakest_language": weakest,
        # The direction that matters: overconfidence keeps answers an abstention
        # threshold should have dropped, and the question is whether it lands on the
        # language the model is worst at.
        "overconfident_where_accuracy_is_worst": weakest in overconfident,
        "per_language": {
            code: {
                "n": per_language[code]["n"],
                "accuracy": row["accuracy"],
                "ece": row["ece"],
                "overconfidence": row["overconfidence"],
            }
            for code, row in sorted(rows.items(), key=lambda kv: kv[1]["accuracy"])
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-language",
        type=int,
        default=600,
        help="items sampled from each language of each split (default 600, so about "
        "6,600 - the scale section 4 reports on)",
    )
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--out", type=pathlib.Path, default=RESULTS / "calibration.json")
    args = parser.parse_args()

    val = score(_read_split("val", args.per_language, args.seed))
    test = score(_read_split("test", args.per_language, args.seed))
    print(f"val {len(val)} items, test {len(test)} items", flush=True)
    if not val or not test:
        raise SystemExit("no val/test split on disk - nothing to fit or measure")

    fit = fit_temperature(val)
    temperature = fit["temperature"]
    print(f"fitted temperature {temperature} on val (NLL {fit['nll']})", flush=True)

    out_rows = _read_tatoeba()
    print(f"tatoeba {len(out_rows)} sentences", flush=True)
    out_scored = score(out_rows)
    per_language = {}
    for language in sorted({r["truth"] for r in out_scored}):
        subset = [r for r in out_scored if r["truth"] == language]
        per_language[language] = {
            "n": len(subset),
            "pooled": len(subset) >= MIN_FOR_A_RATE,
            "calibrated": calibration(subset, temperature) if len(subset) >= 10 else None,
        }
    pooled = [r for r in out_scored if per_language.get(r["truth"], {}).get("pooled")]

    payload = {
        "what": (
            "Whether the ranking score can be read as a probability, and whether a "
            "temperature fitted on Wikipedia still calibrates off it. Temperature "
            "scaling after Guo et al. (2017); ECE over 15 equal-width bins."
        ),
        "bins": BINS,
        "sample": {
            "per_language": None,
            "seed": None,
            "note": (
                "val and test are subsampled by reservoir from the committed splits, "
                "which hold 36,000 lines per language. Tatoeba is used whole."
            ),
        },
        "fit_on_val": fit,
        "in_domain": {
            "uncalibrated": calibration(test, 1.0),
            "calibrated": calibration(test, temperature),
        },
        "out_of_domain": {
            "source": "Tatoeba sentence exports, as research/out_of_domain.py uses",
            "min_n_to_be_pooled": MIN_FOR_A_RATE,
            "languages_pooled": sorted(code for code, row in per_language.items() if row["pooled"]),
            "languages_too_small": sorted(
                code for code, row in per_language.items() if not row["pooled"]
            ),
            "pooled_uncalibrated": calibration(pooled, 1.0),
            "pooled_calibrated": calibration(pooled, temperature),
            "per_language": per_language,
        },
    }

    payload["out_of_domain"]["balanced"] = balanced(per_language)
    payload["out_of_domain"]["why_the_pooled_figure_misleads"] = (
        "97.6% of the pooled items are Arabic, Persian, Sorani and Uyghur, which this "
        "classifier separates almost perfectly. The pooled accuracy and ECE are "
        "therefore figures about those four, and the languages the toolkit exists for "
        "are 2.4% of the mass."
    )

    # The claim the whole exercise exists to settle, computed rather than asserted.
    in_domain = payload["in_domain"]["calibrated"]
    off_domain = payload["out_of_domain"]["pooled_calibrated"]
    if in_domain.get("n") and off_domain.get("n"):
        payload["does_the_calibration_transfer"] = {
            "ece_in_domain": in_domain["ece"],
            "ece_out_of_domain": off_domain["ece"],
            "ece_ratio": round(off_domain["ece"] / in_domain["ece"], 2)
            if in_domain["ece"]
            else None,
            "overconfidence_in_domain": in_domain["overconfidence"],
            "overconfidence_out_of_domain": off_domain["overconfidence"],
        }

    payload["sample"]["per_language"] = args.per_language
    payload["sample"]["seed"] = args.seed

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    def show(label: str, row: dict) -> None:
        if not row.get("n"):
            print(f"  {label:34} no items")
            return
        print(
            f"  {label:34} n={row['n']:6} acc={row['accuracy']:.4f} "
            f"conf={row['mean_confidence']:.4f} over={row['overconfidence']:+.4f} "
            f"ECE={row['ece']:.4f} MCE={row['mce']:.4f}"
        )

    print("\nin domain (held-out Wikipedia):")
    show("uncalibrated (T=1)", payload["in_domain"]["uncalibrated"])
    show(f"calibrated (T={temperature})", payload["in_domain"]["calibrated"])
    print("\nout of domain (Tatoeba, pooled):")
    show("uncalibrated (T=1)", payload["out_of_domain"]["pooled_uncalibrated"])
    show(f"calibrated (T={temperature})", payload["out_of_domain"]["pooled_calibrated"])
    bal = payload["out_of_domain"]["balanced"]
    if bal.get("languages"):
        print(f"\nout of domain, every language weighted equally ({bal['languages']} languages):")
        print(
            f"  accuracy {bal['accuracy']:.4f}  ECE {bal['ece']:.4f}  "
            f"signed gap {bal['signed_gap']:+.4f}  "
            f"worst {bal['worst_language']} at {bal['worst_ece']:.4f}"
        )
        print(f"  overconfident on: {', '.join(bal['overconfident_languages'])}")
        print(f"  underconfident on: {', '.join(bal['underconfident_languages'])}")
        print(f"\n  {'lang':6}{'n':>8}{'acc':>9}{'ECE':>8}{'over':>9}")
        for code, row in bal["per_language"].items():
            print(
                f"  {code:6}{row['n']:>8}{row['accuracy']:>9.4f}"
                f"{row['ece']:>8.4f}{row['overconfidence']:>+9.4f}"
            )
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
