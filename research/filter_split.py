"""Build a split set restricted to chosen sources, so size and diversity can be separated.

Section 6 compares a Wikipedia-trained model against a multi-source one and attributes the
gain to source diversity. The obvious reviewer objection is that the two also differ in
**size**: the shipped model was trained on a Wikipedia sample of a few thousand paragraphs
per language, and the multi-source corpus holds hundreds of thousands. More data alone
could explain an improvement, and the comparison as it stood could not tell the two apart.

This writes a third training set from the *same* corpus, restricted to the Wikimedia
sources, so a model can be trained on Wikipedia text at the same per-language cap, with
the same hyperparameters, and scored on the same held-out rows. The only thing that then
differs between it and the multi-source model is which sources the training rows came
from.

Test and validation splits are **not** filtered: every model is scored on the identical
full multi-source test set, which is the whole point of building it that way.

    .venv-baselines/Scripts/python research/filter_split.py \
        --match wikimedia --out data/splits_wikipedia_only
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LANGUAGES = ["ur", "pnb", "skr", "sd", "ps", "ks", "fa", "ar", "ckb", "ug", "azb"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", type=Path, default=ROOT / "data" / "splits")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--match", required=True,
                    help="keep only rows whose source tag contains this substring")
    ap.add_argument("--part", default="train",
                    help="which split to filter (default train; val and test are left "
                         "alone on purpose, so every model is scored on identical rows)")
    args = ap.parse_args()

    out = args.out / args.part
    out.mkdir(parents=True, exist_ok=True)
    print(f"keeping {args.part} rows whose source contains {args.match!r}\n")
    kept_any = False

    for language in LANGUAGES:
        text_path = args.split / args.part / f"{language}.txt"
        src_path = args.split / args.part / f"{language}.src"
        if not text_path.exists():
            continue
        texts = text_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tags = src_path.read_text(encoding="utf-8").splitlines()
        if len(texts) != len(tags):
            raise SystemExit(
                f"{language}: {len(texts)} lines but {len(tags)} tags - the sidecar and "
                f"the text have gone out of step and the filter would keep wrong rows"
            )
        pairs = [(t, g) for t, g in zip(texts, tags) if args.match in g]
        sources = sorted({g.split("\t", 1)[-1] for _, g in pairs})

        # A language with no matching source gets an empty file rather than a missing
        # one, and says so. Silently skipping it would train the control on ten
        # languages and score it on eleven, which reads as a collapse on the eleventh.
        (out / f"{language}.txt").write_text(
            "\n".join(t for t, _ in pairs) + ("\n" if pairs else ""), encoding="utf-8")
        (out / f"{language}.src").write_text(
            "\n".join(g for _, g in pairs) + ("\n" if pairs else ""), encoding="utf-8")
        kept_any = kept_any or bool(pairs)
        note = "" if pairs else "   <- NO MATCHING SOURCE"
        print(f"  {language:5}{len(pairs):>9,} of {len(texts):>9,}  "
              f"{', '.join(sources) if sources else '-'}{note}")

    if not kept_any:
        raise SystemExit(f"no row in any language matched {args.match!r}")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
