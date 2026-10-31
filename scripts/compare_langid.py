"""Compare `identify_language` against two general-purpose language detectors.

The task is Urdu against the ten languages it is actually confusable with - Shahmukhi
Punjabi, Saraiki, Sindhi, Pashto, Kashmiri, Persian, Arabic, Sorani, Uyghur, South
Azerbaijani - all written in the Perso-Arabic script. Telling Urdu from English is not the
problem; telling it from Saraiki is.

The baselines:

  * **langdetect** (a port of Google's language-detection library), 55 languages.
  * **fastText `lid.176.ftz`**, 176 languages, the usual off-the-shelf choice.

The first thing to measure is not accuracy, it is **label coverage**: a detector with no
label for Saraiki cannot be right about Saraiki, and cannot be scored as wrong in a way
that means anything either. So three numbers are reported per baseline, and the first one
is the one that decides how to read the other two:

  1. which of the 11 languages it has a label for at all,
  2. its accuracy restricted to those languages - the fair comparison,
  3. what it says instead for the languages it cannot name, which is what a user would
     actually get.

Both sides are scored on the same rows in the same run - the test split of
`data/wiki_large`, reproduced by `measure_langid.wiki_splits` - so no number here is taken
from a different corpus than the one beside it. The corpus is gitignored (132,000
paragraphs); `docs/langid-comparison.json` is the committed result.

    python scripts/compare_langid.py
    python scripts/compare_langid.py --wiki-dir data/wiki --out /tmp/small.json
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from measure_langid import LANGUAGES, wiki_splits  # noqa: E402

from urdunlp.langid import identify_language  # noqa: E402

VENV = ROOT / ".venv-baselines"
MODEL = ROOT / ".baselines" / "lid.176.ftz"
LENGTHS = (None, 50, 20, 10)

# Runs in the baselines' interpreter: both detectors over the same rows, one subprocess.
BASELINE_SIDE = """
import json, sys, time

payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
model_path = sys.argv[2]
rows = payload["rows"]

out = {"labels": {}, "predictions": {}, "seconds": {}}

# --- langdetect -------------------------------------------------------------------
from langdetect import detect, detect_langs, DetectorFactory
from langdetect.detector_factory import PROFILES_DIRECTORY
import os as _os
DetectorFactory.seed = 0
out["labels"]["langdetect"] = sorted(_os.listdir(PROFILES_DIRECTORY))

start = time.perf_counter()
preds = []
for text in rows:
    try:
        preds.append(detect(text))
    except Exception:
        # langdetect raises LangDetectException on input it cannot profile at all.
        # Recorded as a refusal, not silently mapped to anything.
        preds.append("__refused__")
out["predictions"]["langdetect"] = preds
out["seconds"]["langdetect"] = round(time.perf_counter() - start, 1)

# --- fastText ---------------------------------------------------------------------
import fasttext
fasttext.FastText.eprint = lambda *a, **k: None
model = fasttext.load_model(model_path)
out["labels"]["fasttext"] = sorted(
    label.replace("__label__", "") for label in model.get_labels()
)

start = time.perf_counter()
preds = []
for text in rows:
    # fastText reads a newline as a document boundary.
    labels, _ = model.predict(text.replace("\\n", " "), k=1)
    preds.append(labels[0].replace("__label__", "") if labels else "__refused__")
out["predictions"]["fasttext"] = preds
out["seconds"]["fasttext"] = round(time.perf_counter() - start, 1)

print("__JSON__" + json.dumps(out))
"""


def python_in(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def ensure_baselines() -> dict[str, str]:
    python = python_in(VENV)
    if not python.exists():
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
    versions = {}
    for package in ("langdetect", "fasttext-wheel"):
        shown = subprocess.run(
            [str(python), "-m", "pip", "show", package], capture_output=True, text=True
        )
        if shown.returncode != 0:
            subprocess.run(
                [str(python), "-m", "pip", "install", "-q", "--disable-pip-version-check", package],
                check=True,
            )
            shown = subprocess.run(
                [str(python), "-m", "pip", "show", package],
                capture_output=True,
                text=True,
                check=True,
            )
        for line in shown.stdout.splitlines():
            if line.startswith("Version:"):
                versions[package] = line.split(":", 1)[1].strip()
    if not MODEL.exists():
        raise SystemExit(
            f"no fastText model at {MODEL}. Fetch it once:\n"
            "  curl -o .baselines/lid.176.ftz "
            "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
        )
    return versions


def cut(text: str, length: int | None) -> str:
    return text if length is None else text[:length]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wiki-dir", type=Path, default=ROOT / "data" / "wiki_large")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "langid-comparison.json")
    ap.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / "research" / "results" / "predictions.json.gz",
        help="per-item predictions for every tool and length, gzipped. The "
        "aggregates in --out are what the package's guard tests read; "
        "this is experiment input - a confusion matrix, a per-class "
        "recall, a bootstrap interval and a McNemar test all need to "
        "know what each tool said about each item",
    )
    ap.add_argument("--no-predictions", action="store_true")
    args = ap.parse_args()

    versions = ensure_baselines()
    splits = wiki_splits(args.wiki_dir)
    truth, rows = [], []
    for code, parts in splits.items():
        for paragraph in parts["test"]:
            truth.append(code)
            rows.append(paragraph)
    print(
        f"{len(rows):,} test paragraph(s) over {len(splits)} language(s) "
        f"from {args.wiki_dir.name}\n",
        flush=True,
    )

    per_item: dict = {"truth": truth, "languages": sorted(splits), "by_length": {}}
    result: dict = {
        "measured": time.strftime("%Y-%m-%d"),
        "corpus": args.wiki_dir.name,
        "languages": {code: LANGUAGES[code] for code in splits},
        "test_paragraphs": len(rows),
        "baseline_versions": versions,
        "fasttext_model": MODEL.name,
        "by_length": {},
    }

    for length in LENGTHS:
        label = "whole" if length is None else f"{length} chars"
        texts = [cut(t, length) for t in rows]

        start = time.perf_counter()
        # `.language`, not the object: identify_language returns a LanguageGuess
        # (language, name, margin, evidence, short), and comparing the dataclass to an
        # ISO code scored this tool at 0.0000 on its own task.
        ours = [identify_language(t).language for t in texts]
        our_seconds = time.perf_counter() - start

        payload = args.out.parent / f".langid-payload-{label.replace(' ', '')}.json"
        payload.parent.mkdir(parents=True, exist_ok=True)
        payload.write_text(json.dumps({"rows": texts}), encoding="utf-8")
        proc = subprocess.run(
            [str(python_in(VENV)), "-c", BASELINE_SIDE, str(payload), str(MODEL)],
            capture_output=True,
            text=True,
        )
        payload.unlink(missing_ok=True)
        marker = proc.stdout.find("__JSON__")
        if marker < 0:
            raise SystemExit(f"baseline side failed:\n{(proc.stderr or proc.stdout)[-800:]}")
        side = json.loads(proc.stdout[marker + len("__JSON__") :].splitlines()[0])

        entry: dict = {"n": len(rows), "assay": None}
        # --- this tool ---------------------------------------------------------
        right = sum(1 for got, want in zip(ours, truth, strict=True) if got == want)
        entry["urdunlp"] = {
            "accuracy_all_languages": round(right / len(rows), 4),
            "seconds": round(our_seconds, 1),
            "labels_covering_the_task": len(splits),
        }
        # --- the baselines -----------------------------------------------------
        for name in ("langdetect", "fasttext"):
            preds = side["predictions"][name]
            known = {code for code in splits if code in set(side["labels"][name])}
            unknown = [code for code in splits if code not in known]

            all_right = sum(1 for got, want in zip(preds, truth, strict=True) if got == want)
            in_known = [(g, w) for g, w in zip(preds, truth, strict=True) if w in known]
            known_right = sum(1 for g, w in in_known if g == w)
            says_for_unknown: dict[str, dict[str, int]] = {}
            for code in unknown:
                counts = collections.Counter(
                    g for g, w in zip(preds, truth, strict=True) if w == code
                )
                says_for_unknown[code] = dict(counts.most_common(3))

            entry[name] = {
                "labels_in_model": len(side["labels"][name]),
                "has_a_label_for": sorted(known),
                "has_no_label_for": unknown,
                "accuracy_all_languages": round(all_right / len(rows), 4),
                "accuracy_where_it_has_a_label": (
                    round(known_right / len(in_known), 4) if in_known else None
                ),
                "n_where_it_has_a_label": len(in_known),
                "what_it_says_instead": says_for_unknown,
                "seconds": side["seconds"][name],
            }
        per_item["by_length"][label] = {
            "urdunlp": ours,
            "langdetect": side["predictions"]["langdetect"],
            "fasttext": side["predictions"]["fasttext"],
        }
        result["by_length"][label] = entry

        print(
            f"  {label:<10} urdunlp {entry['urdunlp']['accuracy_all_languages']:.4f}"
            f" | langdetect {entry['langdetect']['accuracy_all_languages']:.4f}"
            f" ({entry['langdetect']['accuracy_where_it_has_a_label']} where labelled)"
            f" | fasttext {entry['fasttext']['accuracy_all_languages']:.4f}"
            f" ({entry['fasttext']['accuracy_where_it_has_a_label']} where labelled)",
            flush=True,
        )

    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    if not args.no_predictions:
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        # mtime=0: gzip stores a timestamp, and a file identical except for when it was
        # written is a diff on every run.
        with gzip.GzipFile(args.predictions, "wb", mtime=0) as fh:
            fh.write(json.dumps(per_item).encode("utf-8"))
        rows = len(truth) * len(per_item["by_length"]) * 3
        print(
            f"{rows:,} per-item prediction(s) -> {args.predictions.name} "
            f"({args.predictions.stat().st_size / 1024:.0f} KB)"
        )
    whole = result["by_length"]["whole"]
    print("\nlabel coverage of the 11-language task:")
    for name in ("langdetect", "fasttext"):
        row = whole[name]
        print(
            f"  {name:<11} {len(row['has_a_label_for'])}/{len(splits)} "
            f"- no label for: {', '.join(row['has_no_label_for']) or 'none'}"
        )
    print(f"\nwritten to {args.out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
