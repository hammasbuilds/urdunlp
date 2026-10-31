"""Does the built Kashmiri data help where there was room to help: short text?

The headline comparison says the built rows changed real Kashmiri accuracy by -0.0005
(z=-0.24, p=0.811) - no effect. But it was run in a regime with almost no room: the
control already scored 0.9903 on real Kashmiri whole paragraphs. A null result against a
ceiling does not distinguish "this augmentation does not transfer" from "there was
nothing left to win", and reporting it as the former would overclaim.

Short text is where the headroom is. The project already reports accuracy at 50, 20 and
10 characters, and every language degrades there. If resampling a language's own
vocabulary teaches a character n-gram model anything useful, a short window is where it
should show, because that is where the model has fewest grams to work with.

Scored on **real** Kashmiri test rows only, cut to centred windows, both models on
identical rows.

    .venv-baselines/Scripts/python research/ks_short_text.py
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WIDTHS = (10, 20, 50, 100, None)


def wilson(k: int, n: int, z: float = 1.96) -> dict | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"rate": round(p, 4), "lo": round(max(0.0, c - h), 4),
            "hi": round(min(1.0, c + h), 4), "n": n, "correct": k}


def window(text: str, width: int | None) -> str:
    """A centred slice - the same measure the rest of the project uses.

    Taking the FIRST n characters reads several points lower, because the start of an
    encyclopaedia line is disproportionately a title and a date.
    """
    if width is None or len(text) <= width:
        return text
    start = (len(text) - width) // 2
    return text[start:start + width]


def predictor(model_dir: Path):
    import urdunlp.langid as langid

    payload = json.loads(gzip.decompress((model_dir / "langid.json.gz").read_bytes()))
    langid._load = lambda name, _p=payload: _p  # noqa: SLF001
    langid._script_model.cache_clear()  # noqa: SLF001
    assert langid._script_model() is not None  # noqa: SLF001
    return lambda text: langid.identify_language(text).language


def mcnemar(a_right: list[bool], b_right: list[bool]) -> dict:
    """Exact two-sided McNemar over discordant pairs, in log space.

    The rows are identical between the two models, so the paired test is the right one -
    an unpaired z throws away the pairing and is less sensitive, which matters when the
    effect being looked for is small.
    """
    a_only = sum(1 for x, y in zip(a_right, b_right) if x and not y)
    b_only = sum(1 for x, y in zip(a_right, b_right) if y and not x)
    n = a_only + b_only
    if n == 0:
        return {"a_only": 0, "b_only": 0, "discordant": 0, "p_value": 1.0}
    k = min(a_only, b_only)
    log_p = -math.inf
    for i in range(k + 1):
        term = (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                - n * math.log(2))
        log_p = term if log_p == -math.inf else max(log_p, term) + math.log1p(
            math.exp(min(log_p, term) - max(log_p, term)))
    log10_p = (log_p + math.log(2)) / math.log(10)
    # A two-sided tail this lopsided underflows: 113 discordant pairs split 113/0 gives
    # a p around 1e-34, and `round(p, 6)` prints 0.0. A published p of exactly zero is a
    # floating-point artefact, not a result, so the exponent is carried instead.
    out = {"a_only": a_only, "b_only": b_only, "discordant": n,
           "log10_p_value": round(log10_p, 2)}
    if log10_p > -300:
        out["p_value"] = min(1.0, 2 * math.exp(log_p))
    else:
        out["p_value"] = None
        out["p_value_below"] = "1e-300"
    return out


def shown(test: dict) -> str:
    """A p-value a reader can trust, never a rounded-to-zero one."""
    p = test.get("p_value")
    if p is None:
        return f"<{test['p_value_below']}"
    if p < 1e-6:
        return f"1e{test['log10_p_value']:.0f}"
    return f"{p:.3g}"


def main() -> int:
    texts = (ROOT / "data" / "splits" / "test" / "ks.txt").read_text(
        encoding="utf-8").splitlines()
    tags = (ROOT / "data" / "splits" / "test" / "ks.src").read_text(
        encoding="utf-8").splitlines()
    real = [t for t, g in zip(texts, tags) if not g.startswith("synthetic")]
    print(f"{len(real):,} real Kashmiri test rows, both models on identical rows\n")

    report: dict = {"rows": len(real), "widths": {}}
    print(f"  {'width':>7}{'no synthetic':>15}{'with synthetic':>16}"
          f"{'change':>10}{'discordant':>12}{'p':>12}")
    for width in WIDTHS:
        cut = [window(t, width) for t in real]
        results = {}
        for label, directory in (("no", ROOT / "models" / "no_synthetic"),
                                 ("with", ROOT / "models" / "with_synthetic")):
            predict = predictor(directory)
            results[label] = [predict(t) == "ks" for t in cut]
        a = wilson(sum(results["no"]), len(cut))
        b = wilson(sum(results["with"]), len(cut))
        test = mcnemar(results["no"], results["with"])
        name = "whole" if width is None else str(width)
        report["widths"][name] = {"no_synthetic": a, "with_synthetic": b,
                                  "mcnemar": test}
        print(f"  {name:>7}{a['rate']:>15.4f}{b['rate']:>16.4f}"
              f"{b['rate'] - a['rate']:>+10.4f}{test['discordant']:>12,}"
              f"{shown(test):>12}")

    out = ROOT / "research" / "results" / "ks_short_text.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwritten to {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
