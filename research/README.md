# research/ — experiments for the paper, not part of the package

Everything in this directory exists to support a manuscript. **None of it ships.**

| | ships in the wheel | ships in the sdist |
|---|:--:|:--:|
| `src/urdunlp/` | ✅ | ✅ |
| `tests/`, `scripts/`, `docs/`, `eval/` | ❌ | ✅ |
| **`research/`, `paper/`** | ❌ | ❌ |

The wheel is `packages = ["src/urdunlp"]` and the sdist has an explicit `include` list
that names neither directory, so the separation is enforced by the build config rather
than by convention. `tests/test_research_is_not_packaged.py` checks it.

Why keep them apart at all: the package has **zero runtime dependencies**, and that is a
property worth protecting. The experiments here need matplotlib, scikit-learn and scipy,
and the baselines need langdetect and fastText. A reader who installs `urdunlp` should get
none of that.

## Layout

```
research/
  metrics.py        per-class P/R/F1, macro/micro, bootstrap CIs, McNemar
  figures.py        every figure in the paper, one function each
  run_all.py        regenerates results/ and figures/ from the committed artefacts
  results/          numbers, as JSON, one file per table
  figures/          PDF and PNG, one file per figure
paper/
  main.md           the manuscript
```

## Environment

```
python -m venv .venv-research
.venv-research/Scripts/python -m pip install matplotlib scikit-learn scipy
python research/run_all.py
```

`.venv-research/` is gitignored. The experiments read
`docs/langid-comparison.json` and `data/langid_report_v5.json`, both committed, so every
figure can be rebuilt without the 132,000-paragraph corpus.

## What is deliberately not here

Raw corpora. `data/wiki_large/` is 132,000 paragraphs and gitignored;
`scripts/fetch_wikipedia_samples.py` rebuilds it. A figure that cannot be rebuilt from
committed inputs is marked as such in its own docstring.
