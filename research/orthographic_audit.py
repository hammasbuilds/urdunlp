"""The Saraiki instrument from section 6.2, generalised to every language that has one.

Section 6.2 split the Saraiki test rows on whether a line carries one of four implosive
letters, and the split separated 0.9918 from 0.7186. That criterion is **orthographic**:
it is decided before any prediction is read, so it cannot inflate a score the way
filtering with the classifier would. Section 7 then admits the gap this script closes -
the same audit was never run on the other thirty-five source-language pairs.

Two questions, both answered without consulting the classifier to choose the rows:

  1. **Does the language have a distinctive orthography at all?** For each character,
     the share of each language's training lines that contain it. A character counts as
     distinctive to a language when it is common in that language and near-absent from
     every other (`MIN_SHARE`, `MAX_OTHER_SHARE`). Some languages have no such character
     - Urdu, written in the plainest Perso-Arabic inventory of the eleven, is the
     unmarked case against which the others are marked - and for those the instrument
     does not exist. That is reported rather than worked around, and it is the reason
     section 6.2's defect is invisible to this audit: an Urdu gloss mislabelled `skr`
     carries no Urdu-exclusive letter, because there is no such letter.

  2. **Is a source's accuracy carried by its orthographically unambiguous lines?** The
     test rows of each language are split on whether they carry one of its own
     distinctive characters, and accuracy is measured within each half, per source. A
     large gap means the label is doing work the text does not support.

And one direct check that needs no model at all: a line labelled `L` that carries a
character distinctive to a *different* language `M`, and none of `L`'s own. That is
evidence about the corpus rather than about the classifier, and it is the one finding here
that would be worth acting on by relabelling.

Both derivation and evaluation are kept apart on purpose: the character statistics come
from the **train** split and the accuracies from the **test** split, so the criterion is
not chosen on the rows it is scored on.

    .venv/Scripts/python research/orthographic_audit.py
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "splits"
MODEL = ROOT / "models" / "with_synthetic" / "langid.json.gz"
OUT = ROOT / "research" / "results" / "orthographic_audit.json"

# A character is distinctive to a language when at least this share of that language's
# training lines carry it. 2% is low enough to admit a letter that only appears in part of
# the vocabulary - the Saraiki implosives are in 37.9% of Wikipedia lines but far fewer
# Wiktionary ones - and high enough to exclude a character that occurs in a handful of
# borrowed names.
MIN_SHARE = 0.02

# ... and when no other language carries it in more than this share of its lines. 0.1%
# rather than 0 because every large corpus contains some quoted foreign text, and a
# criterion that demanded exactly zero would be defeated by one Saraiki sentence quoted on
# a Punjabi Wikipedia page. The ratio between the two constants is the real claim: a
# distinctive character is at least twenty times more common in its own language.
MAX_OTHER_SHARE = 0.001


# The letters this script prints are the point of it, and a Windows console defaults to
# cp1252, which cannot encode any of them - the run died in the middle of the report with
# a UnicodeEncodeError after doing all of the work. Reconfigured here rather than left to
# the caller's environment.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")


def wilson(k: int, n: int, z: float = 1.96) -> dict | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"rate": round(p, 4), "lo": round(max(0.0, c - h), 4),
            "hi": round(min(1.0, c + h), 4), "n": n}


def languages() -> list[str]:
    """Read from the split directory, not a list, so a new language cannot be left out."""
    return sorted(p.stem for p in (SPLITS / "train").glob("*.txt"))


def letter_shares(language: str, split: str = "train") -> tuple[dict[str, float], int]:
    """Share of this language's lines that contain each letter.

    Letters only: `str.isalpha` drops digits, punctuation and the Arabic comma, which are
    shared by every language here and would swamp the comparison. Shares are over lines
    rather than over characters, because the question is "can this line be attributed",
    not "how often is this letter written".
    """
    path = SPLITS / split / f"{language}.txt"
    counts: collections.Counter = collections.Counter()
    lines = 0
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = unicodedata.normalize("NFC", raw.strip())
            if not line:
                continue
            lines += 1
            counts.update({c for c in line if c.isalpha()})
    return ({c: n / lines for c, n in counts.items()} if lines else {}), lines


def distinctive(shares: dict[str, dict[str, float]]) -> dict[str, list[str]]:
    """Characters common in one language and near-absent from every other."""
    out: dict[str, list[str]] = {}
    for language, own in shares.items():
        picked = []
        for char, share in own.items():
            if share < MIN_SHARE:
                continue
            other = max((s.get(char, 0.0) for lang, s in shares.items()
                         if lang != language), default=0.0)
            if other <= MAX_OTHER_SHARE:
                picked.append(char)
        out[language] = sorted(picked)
    return out


def load_model():
    import urdunlp.langid as langid

    payload = json.loads(gzip.decompress(MODEL.read_bytes()))
    langid._load = lambda name, _p=payload: _p  # noqa: SLF001
    langid._script_model.cache_clear()  # noqa: SLF001
    return langid


def audit(limit: int | None = None) -> dict:
    langs = languages()
    print(f"{len(langs)} languages: {' '.join(langs)}\n")

    shares: dict[str, dict[str, float]] = {}
    train_lines: dict[str, int] = {}
    for language in langs:
        shares[language], train_lines[language] = letter_shares(language)
        print(f"  {language:5s} {train_lines[language]:>9,} train lines, "
              f"{len(shares[language]):>4} distinct letters", flush=True)

    marks = distinctive(shares)
    print("\n  distinctive letters (common here, near-absent everywhere else):")
    for language in langs:
        chars = marks[language]
        if chars:
            own = min(shares[language][c] for c in chars)
            print(f"    {language:5s} {''.join(chars):20s} "
                  f"(rarest of them in {own:.1%} of its train lines)")
        else:
            print(f"    {language:5s} {'-':20s} (no distinctive letter: the unmarked case)")

    langid = load_model()
    per_source: dict[str, dict] = {}
    foreign: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    # A line can carry the inventories of two other languages, so the per-other counts
    # above sum to more than the number of lines. Counted separately rather than summed,
    # because "129 lines contradict their label" would have been wrong by however many
    # lines carry two foreign inventories, and nothing in the per-other breakdown says
    # which.
    foreign_lines: collections.Counter = collections.Counter()
    buckets: dict[tuple[str, str, bool], list[int]] = collections.defaultdict(
        lambda: [0, 0])

    for language in langs:
        own = set(marks[language])
        texts = (SPLITS / "test" / f"{language}.txt").read_text(
            encoding="utf-8").splitlines()
        tags = (SPLITS / "test" / f"{language}.src").read_text(
            encoding="utf-8").splitlines()
        if limit is not None:
            texts, tags = texts[:limit], tags[:limit]
        for text, tag in zip(texts, tags, strict=True):
            source = tag.split("\t")[-1]
            chars = set(text)
            marked = bool(own & chars)
            ok = int(langid.identify_language(text).language == language)
            buckets[(language, source, marked)][0] += ok
            buckets[(language, source, marked)][1] += 1
            if not marked:
                others = [o for o in langs
                          if o != language and marks[o] and set(marks[o]) & chars]
                for other in others:
                    foreign[language][other] += 1
                if others:
                    foreign_lines[language] += 1
        print(f"  scored {language}: {len(texts):,} test lines", flush=True)

    for (language, source, marked) in sorted(buckets):
        right, total = buckets[(language, source, marked)]
        key = f"{language}.{source}|{'with' if marked else 'without'}"
        per_source[key] = wilson(right, total)

    summary: dict[str, dict] = {}
    for language in langs:
        with_m = [v for (lg, _s, m), v in buckets.items() if lg == language and m]
        without = [v for (lg, _s, m), v in buckets.items() if lg == language and not m]
        summary[language] = {
            "distinctive_letters": "".join(marks[language]),
            "has_criterion": bool(marks[language]),
            "with_a_distinctive_letter": wilson(sum(v[0] for v in with_m),
                                                sum(v[1] for v in with_m)),
            "without_one": wilson(sum(v[0] for v in without),
                                  sum(v[1] for v in without)),
            "whole": wilson(sum(v[0] for v in with_m + without),
                            sum(v[1] for v in with_m + without)),
            "unmarked_lines_carrying_another_languages_letter": dict(
                foreign[language].most_common()),
            "unmarked_lines_carrying_any_foreign_letter": foreign_lines[language],
        }

    return {
        "min_share": MIN_SHARE,
        "max_other_share": MAX_OTHER_SHARE,
        "criterion_derived_from": "train split",
        "accuracies_measured_on": "test split",
        "why_not_the_model": (
            "The split is orthographic and fixed before any prediction is read. Choosing "
            "rows with the classifier would remove exactly the rows it gets wrong."
        ),
        "is_an_upper_bound": (
            "Lines carrying a distinctive letter are longer and more distinctively "
            "written than the rest, so the with-letter figure is an upper bound on the "
            "language's accuracy rather than a corrected value. Both halves and the "
            "whole are reported."
        ),
        "what_this_cannot_see": (
            "A language with no distinctive letter has no criterion here, and a line "
            "mislabelled as such a language is undetectable by this instrument. Urdu is "
            "that case, which is why section 6.2's Urdu glosses under a Saraiki label do "
            "not show up in the foreign-letter counts."
        ),
        "train_lines": train_lines,
        "foreign_letter_lines_total": sum(foreign_lines.values()),
        "per_language": summary,
        "per_source": per_source,
    }


def show(data: dict) -> None:
    print(f"\n  {'lang':6}{'letters':14}{'whole':>9}{'with':>9}{'without':>9}"
          f"{'gap':>8}   foreign-letter lines")
    for language, r in data["per_language"].items():
        w, a, b = r["whole"], r["with_a_distinctive_letter"], r["without_one"]
        gap = (a["rate"] - b["rate"]) if (a and b) else None
        foreign = r["unmarked_lines_carrying_another_languages_letter"]
        tail = ", ".join(f"{k}:{v}" for k, v in list(foreign.items())[:3]) or "-"
        print(f"  {language:6}{r['distinctive_letters'][:12] or '-':14}"
              f"{w['rate']:>9.4f}"
              f"{(a['rate'] if a else float('nan')):>9.4f}"
              f"{(b['rate'] if b else float('nan')):>9.4f}"
              f"{(gap if gap is not None else float('nan')):>8.4f}   {tail}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="test lines per language")
    ap.add_argument("--json", type=Path, default=OUT)
    args = ap.parse_args()
    data = audit(args.limit)
    show(data)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwritten to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
