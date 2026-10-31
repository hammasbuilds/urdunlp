"""Score a model on the multi-source test split, per language AND per source.

This is the evaluation the Wikipedia-only setup could not do. The shipped model scored
0.9925 on Wikipedia and 0.2565 on real out-of-domain Urdu, and no number in that setup
could have warned of it: train and test came from one domain.

Here every source contributed 80/10/10, so the test split spans every domain the training
split saw. Two numbers per language follow, and the gap between them is the thing worth
reading:

  * accuracy on the whole language's test chunk, and
  * accuracy **per source**, so a language that holds up on one domain and collapses on
    another is visible instead of averaged.

Two models are compared on identical rows: the shipped Wikipedia-only model and the one
trained on the multi-source training split.

    .venv-baselines/Scripts/python research/evaluate_multisource.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLIT = ROOT / "data" / "multisource" / "test"
OUT = ROOT / "research" / "results" / "multisource_eval.json"
ORDER = ["ur", "pnb", "skr", "sd", "ps", "ks", "fa", "ar", "ckb", "ug", "azb"]


def wilson(k: int, n: int, z: float = 1.96) -> dict | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"rate": round(p, 4), "lo": round(max(0.0, c - h), 4),
            "hi": round(min(1.0, c + h), 4), "n": n, "correct": k}


def load_split(split: Path = SPLIT) -> list[tuple[str, str, str]]:
    """(language, source, text) for every test row, as (language, kind|source, text).

    The sidecar carries `kind\tsource` per row. Both are kept, joined, so accuracy on
    **real** rows can be separated from accuracy on built ones - which is the entire
    point of tagging them, and the only way the synthetic-data question is answerable.
    """
    rows = []
    for language in ORDER:
        text_path = split / f"{language}.txt"
        src_path = split / f"{language}.src"
        if not text_path.exists():
            continue
        texts = text_path.read_text(encoding="utf-8", errors="replace").splitlines()
        sources = (src_path.read_text(encoding="utf-8").splitlines()
                   if src_path.exists() else ["unknown"] * len(texts))
        if len(sources) != len(texts):
            raise SystemExit(
                f"{language}: {len(texts)} test lines but {len(sources)} source tags - "
                "the sidecar and the text have gone out of step, and a per-source "
                "accuracy computed from them would attribute rows to the wrong source"
            )
        rows += [(language, s.replace("\t", "|"), t)
                 for s, t in zip(sources, texts) if t.strip()]
    return rows


def kind_of(tagged_source: str) -> str:
    """`real`, `segmented` or `synthetic` from a `kind|source` tag."""
    return tagged_source.split("|", 1)[0]


def score(rows: list[tuple[str, str, str]], predict) -> dict:
    """Accuracy per language, per source, and - separately - per row kind.

    The per-kind split is the one the synthetic experiment is decided on. If training
    with built Kashmiri does not improve accuracy on **real** held-out Kashmiri, the
    built rows have not earned their place, and that is the result rather than a
    disappointment. Reporting one figure over real and built rows together would let a
    model that had learned the generator's artefacts certify its own success.
    """
    per_language: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    per_source: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    per_kind: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    real_only: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    confusions: dict[str, Counter] = defaultdict(Counter)

    for language, source, text in rows:
        got = predict(text)
        ok = int(got == language)
        kind = kind_of(source)
        per_language[language][0] += ok
        per_language[language][1] += 1
        per_source[(language, source)][0] += ok
        per_source[(language, source)][1] += 1
        per_kind[kind][0] += ok
        per_kind[kind][1] += 1
        if kind != "synthetic":
            real_only[language][0] += ok
            real_only[language][1] += 1
        if not ok:
            confusions[language][str(got)] += 1

    total_right = sum(v[0] for v in per_language.values())
    total = sum(v[1] for v in per_language.values())
    real_right = sum(v[0] for v in real_only.values())
    real_total = sum(v[1] for v in real_only.values())

    return {
        "overall": wilson(total_right, total),
        "overall_real_rows_only": wilson(real_right, real_total),
        "per_language": {k: wilson(v[0], v[1]) for k, v in per_language.items()},
        "per_language_real_rows_only": {k: wilson(v[0], v[1])
                                        for k, v in real_only.items()},
        "per_kind": {k: wilson(v[0], v[1]) for k, v in sorted(per_kind.items())},
        "per_source": {f"{k[0]}|{k[1]}": wilson(v[0], v[1])
                       for k, v in sorted(per_source.items())},
        "confusions": {k: dict(v.most_common(4)) for k, v in confusions.items()},
    }


def predictor(model_dir: Path | None):
    """identify_language backed by a given model directory, or the shipped one.

    `langid._load` reads from the package's data directory through importlib.resources and
    has no path override - an earlier version of this function set a
    `_MODEL_PATH_OVERRIDE` attribute that does not exist, so it silently scored the
    shipped model twice and would have reported "no difference" for a model that had never
    been loaded. The loader is patched instead, and `_script_model`'s lru_cache is cleared
    so the patch is in force before the first call.
    """
    import gzip
    import json

    import urdunlp.langid as langid

    if model_dir is not None:
        blob = Path(model_dir) / "langid.json.gz"
        if not blob.exists():
            raise SystemExit(f"no model at {blob}")
        payload = json.loads(gzip.decompress(blob.read_bytes()))

        def _load(name: str, _payload=payload):
            if name == "langid.json.gz":
                return _payload
            raise SystemExit(f"this evaluation does not supply {name}")

        langid._load = _load  # noqa: SLF001
        langid._script_model.cache_clear()
        # Prove the patch took: the model's own language set must be readable through it.
        loaded = langid._script_model()  # noqa: SLF001
        assert loaded is not None, "the patched loader produced no model"

    return lambda text: langid.identify_language(text).language


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=None,
                    help="a model directory; omit for the shipped model")
    ap.add_argument("--label", default="shipped")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--split", type=Path, default=SPLIT,
                    help="the test split directory (default data/multisource/test)")
    args = ap.parse_args()

    rows = load_split(args.split)
    sources = {s for _, s, _ in rows}
    print(f"{len(rows):,} test row(s), {len(sources)} distinct source(s), "
          f"model: {args.label}\n")

    result = score(rows, predictor(args.model))
    o = result["overall"]
    print(f"overall {o['rate']:.4f} [{o['lo']:.4f}, {o['hi']:.4f}] over {o['n']:,}")
    r = result["overall_real_rows_only"]
    if r and r["n"] != o["n"]:
        print(f"real rows only {r['rate']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}] over "
              f"{r['n']:,}  <- the figure the synthetic question is decided on")
    for kind, krow in result["per_kind"].items():
        print(f"  {kind:11}{krow['rate']:>9.4f} over {krow['n']:>7,}")
    print()
    print(f"  {'lang':6}{'n':>7}{'accuracy':>11}{'real only':>11}   per-source spread")
    for language in ORDER:
        row = result["per_language"].get(language)
        if not row:
            continue
        mine = {k.split("|")[1]: v for k, v in result["per_source"].items()
                if k.startswith(language + "|")}
        spread = ""
        if len(mine) > 1:
            lo = min(mine.items(), key=lambda kv: kv[1]["rate"])
            hi = max(mine.items(), key=lambda kv: kv[1]["rate"])
            spread = (f"{lo[1]['rate']:.3f} ({lo[0]}, n={lo[1]['n']}) .. "
                      f"{hi[1]['rate']:.3f} ({hi[0]})")
        else:
            spread = f"one source only ({next(iter(mine), '-')})"
        real = result["per_language_real_rows_only"].get(language)
        real_cell = "        —" if not real else f"{real['rate']:>11.4f}"
        print(f"  {language:6}{row['n']:>7}{row['rate']:>11.4f}{real_cell}   {spread}")

    payload = {"model": args.label, "test_rows": len(rows),
               "sources": sorted(sources), **result}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwritten to {args.out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
