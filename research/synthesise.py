"""Synthetic lines for the languages real text cannot fill, built from their own words.

The honest objection to synthetic training data is that a model learns the generator's
artefacts, scores well on synthetic test data, and fails on real text. That is not a
hypothetical here - it is what this project already measured, when a Wikipedia-trained
model scored 0.9925 in domain and 0.2565 on real out-of-domain Urdu.

So nothing here is authored. Three generators, each a resampling or perturbation of text
real speakers wrote, and each tagged so no figure can mix it with real rows:

**`shuffle`** - words drawn from that language's own corpus, frequency-weighted, assembled
into lines of realistic length. Every token is attested in the language. For a character
n-gram classifier this preserves within-word n-grams, which carry most of the signal, and
scrambles cross-word n-grams, which carry some. That trade-off is the thing to measure,
not to assume.

**`bigram`** - word *pairs* drawn from the real corpus and chained, so cross-word n-grams
at the join are real too. Closer to real text than `shuffle` and less diverse.

**`noise`** - real lines with character-level perturbation: a dropped diacritic, a
transposed pair, a repeated character. Standard augmentation, and it targets the thing a
deployed classifier actually meets - text with typos.

Whether any of it helps is an experiment, not a claim. The synthetic rows get their own
80/10/10 split, and `evaluate_multisource.py` reports accuracy on **real** test rows
separately, so the comparison is: does training with synthetic data improve accuracy on
real held-out text? If it does not, that is the finding and the data should be dropped.

    .venv-baselines/Scripts/python research/synthesise.py --languages skr,ks,ckb --target 100000
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"
OUT = ROOT / "data" / "synthetic"
REPORT = ROOT / "research" / "results" / "synthetic.json"
SEED = 20261031

ARABIC = re.compile(r"[؀-ۿݐ-ݿ]")


def arabic_word(word: str) -> bool:
    """Is this token majority Perso-Arabic *letters*?

    The obvious test - `ARABIC.search(word)` - is wrong, and wrong in a way that only
    shows up in the generated text. The U+0600 block contains Arabic **punctuation**, so
    a word like `विकिपीडिया؛` carries one in-range character and passes. 3,435 of
    Kashmiri's 96,917 vocabulary entries pass that way, and a first run put Devanagari
    into synthetic lines labelled Kashmiri - the exact failure the corpus builder's
    script-share check exists to prevent, reintroduced one layer down because the rule
    was re-typed here instead of shared.

    So: count alphabetic characters only, and require at least half of them to be in the
    Perso-Arabic range. One predicate, called by both the vocabulary and the bigram
    builder, so the two cannot drift apart.
    """
    letters = [c for c in word if c.isalpha()]
    if not letters:
        return False
    return sum(1 for c in letters if ARABIC.match(c)) / len(letters) >= 0.5
# Diacritics that are routinely omitted in real writing, so dropping one produces a
# spelling a reader would accept rather than a corruption.
DIACRITICS = "ًٌٍَُِّْٰٓٔ"
MIN_CHARS = 40
MAX_CHARS = 240
# The same share the corpus builder demands of a real line. Applied again to every
# generated line, because word-level filtering is necessary and not sufficient: a line
# assembled entirely from accepted words can still carry a run of Latin from an
# accepted-but-mixed token.
MIN_ARABIC = 0.6


def acceptable(line: str, seen: set[str]) -> bool:
    """Length, novelty, and majority Perso-Arabic - the gate every generator writes through."""
    if not (MIN_CHARS <= len(line) <= MAX_CHARS) or line in seen:
        return False
    letters = [c for c in line if c.isalpha()]
    if not letters:
        return False
    return sum(1 for c in letters if ARABIC.match(c)) / len(letters) >= MIN_ARABIC


def real_lines(language: str) -> list[str]:
    """Every real line collected for this language, segmented files excluded.

    Segmented spans are real, but they are already derived from these paragraphs; using
    them as a word source would weight the vocabulary towards whatever was segmented.
    """
    out: list[str] = []
    for path in sorted(CORPUS.glob(f"{language}.*.txt")):
        if path.name.endswith(".segmented.txt"):
            continue
        out += path.read_text(encoding="utf-8", errors="replace").splitlines()
    return [line for line in out if line.strip()]


def vocabulary(lines: list[str]) -> tuple[list[str], list[int]]:
    counts = Counter()
    for line in lines:
        for word in line.split():
            if len(word) >= 2 and arabic_word(word):
                counts[word] += 1
    words = list(counts)
    return words, [counts[w] for w in words]


def bigrams(lines: list[str]) -> dict[str, list[str]]:
    following: dict[str, list[str]] = {}
    for line in lines:
        words = [w for w in line.split() if len(w) >= 2 and arabic_word(w)]
        for first, second in zip(words, words[1:]):
            following.setdefault(first, []).append(second)
    return following


def lengths(lines: list[str]) -> list[int]:
    """Word counts of the real lines, so synthetic lines have a realistic shape."""
    return [len(line.split()) for line in lines if 3 <= len(line.split()) <= 60]


def make_shuffle(words, weights, shape, want, rng, seen) -> list[str]:
    out = []
    while len(out) < want and shape:
        n = rng.choice(shape)
        line = " ".join(rng.choices(words, weights=weights, k=n))
        if acceptable(line, seen):
            seen.add(line)
            out.append(line)
    return out


def make_bigram(following, words, weights, shape, want, rng, seen) -> list[str]:
    starts = [w for w in following]
    if not starts:
        return []
    out = []
    while len(out) < want and shape:
        n = rng.choice(shape)
        word = rng.choice(starts)
        chain = [word]
        for _ in range(n - 1):
            nexts = following.get(chain[-1])
            word = rng.choice(nexts) if nexts else rng.choices(words, weights=weights)[0]
            chain.append(word)
        line = " ".join(chain)
        if acceptable(line, seen):
            seen.add(line)
            out.append(line)
    return out


def make_noise(lines, want, rng, seen) -> list[str]:
    out = []
    attempts = 0
    while len(out) < want and attempts < want * 40 and lines:
        attempts += 1
        text = rng.choice(lines)
        chars = list(text)
        if not chars:
            continue
        for _ in range(max(1, len(chars) // 50)):
            kind = rng.random()
            i = rng.randrange(len(chars))
            if kind < 0.4 and chars[i] in DIACRITICS:
                chars.pop(i)                       # a dropped diacritic
            elif kind < 0.7 and i + 1 < len(chars):
                chars[i], chars[i + 1] = chars[i + 1], chars[i]   # a transposition
            else:
                chars.insert(i, chars[i])          # a repeated character
        line = " ".join("".join(chars).split())
        if acceptable(line, seen):
            seen.add(line)
            out.append(line)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--languages", required=True)
    ap.add_argument("--target", type=int, default=100_000,
                    help="total lines wanted per language, real included")
    args = ap.parse_args()
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    report: dict = {"seed": SEED, "target": args.target, "languages": {},
                    "generators": {
                        "shuffle": "words resampled from this language's own corpus, "
                                   "frequency-weighted",
                        "bigram": "word pairs from the real corpus, chained",
                        "noise": "real lines with dropped diacritics, transpositions and "
                                 "repeated characters",
                    },
                    "nothing_authored": (
                        "Every token in every synthetic line is a word attested in that "
                        "language's real corpus. No sentence was written by the author or "
                        "by a model."
                    )}

    for language in [l.strip() for l in args.languages.split(",") if l.strip()]:
        lines = real_lines(language)
        segmented = (CORPUS / f"{language}.segmented.txt")
        segmented_n = (len(segmented.read_text(encoding="utf-8").splitlines())
                       if segmented.exists() else 0)
        have = len(lines) + segmented_n
        want = max(0, args.target - have)
        print(f"\n{language}: {len(lines):,} real + {segmented_n:,} segmented = "
              f"{have:,}; need {want:,}")
        if not lines:
            print("    no real text, nothing to resample from")
            continue
        if want == 0:
            print("    already at target")
            report["languages"][language] = {"real": len(lines),
                                             "segmented": segmented_n, "synthetic": 0}
            continue

        words, weights = vocabulary(lines)
        shape = lengths(lines)
        following = bigrams(lines)
        print(f"    vocabulary {len(words):,} word(s), "
              f"{len(following):,} word(s) with a successor")

        seen = set(lines)
        share = want // 3
        made = {
            "shuffle": make_shuffle(words, weights, shape, share, rng, seen),
            "bigram": make_bigram(following, words, weights, shape, share, rng, seen),
            "noise": make_noise(lines, want - 2 * share, rng, seen),
        }
        for kind, texts in made.items():
            if texts:
                (OUT / f"{language}.synthetic-{kind}.txt").write_text(
                    "\n".join(texts) + "\n", encoding="utf-8")
                print(f"    {kind:10}{len(texts):>9,}")
        total_synthetic = sum(len(v) for v in made.values())
        report["languages"][language] = {
            "real": len(lines),
            "segmented": segmented_n,
            "synthetic": total_synthetic,
            "synthetic_by_kind": {k: len(v) for k, v in made.items()},
            "vocabulary": len(words),
            "total": have + total_synthetic,
            "meets_target": have + total_synthetic >= args.target,
        }
        row = report["languages"][language]
        print(f"    {'TOTAL':10}{row['total']:>9,}"
              f"{'' if row['meets_target'] else '   still short'}")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwritten to {OUT} and {REPORT.name}")
    print("\nWhether this helps is measured, not assumed: these rows get their own "
          "80/10/10 split and\nevaluate_multisource.py reports accuracy on REAL test "
          "rows separately. If real-text accuracy\ndoes not improve, the data should be "
          "dropped and that is the result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
