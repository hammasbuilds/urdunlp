"""Score Roman -> Urdu transliteration and `roman_key` against human romanisations.

    python scripts/fetch_dakshina.py
    python scripts/measure_translit.py [--json out.json]

Until this script, the toolkit had no accuracy figure for transliteration at all:
docs/CORPUS.md said human-checked pairs "do not exist for Urdu at any useful scale".
They do. Dakshina (Roark et al., LREC 2020) had native speakers romanise Urdu
Wikipedia: a word lexicon with every attested spelling and its annotator count, and
~10,000 whole sentences romanised by hand.

Reported, for the 0.1 pipeline (lexicon then rules) and the 0.2 pipeline (lexicon,
vocabulary, rules):

  * **word accuracy on sentences** - Dakshina's own dev and test split. Sentences are
    paired word by word where the Urdu and Roman token counts agree, which holds for
    about three quarters of them; the rest cannot be aligned without guessing. 42
    sentences appear in both splits and are dropped from dev.
  * **word accuracy on the lexicon** - train, dev and test, weighted by how many
    annotators wrote each spelling. This is harder than sentences: lexicon words are
    sampled across the vocabulary, so rare words are over-represented relative to
    running text, and the frequency prior helps them least.

A prediction is correct when it equals the gold word after `normalize` on both
sides - so a missing diacritic is not an error, but ہ for ھ is.

Then `roman_key`, scored as clustering: every spelling in the test lexicon is an
item, and its gold cluster is the Urdu word it spells. B-cubed precision and recall
compare `roman_key` with two baselines - exact lowercase match, and a consonant
skeleton over the Roman letters.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from urdunlp import normalize, roman_key, words  # noqa: E402
from urdunlp._channel import roman_keys, urdu_key  # noqa: E402
from urdunlp.translit import LEXICON, _apply_rules, transliterate_with_confidence  # noqa: E402

DAKSHINA = ROOT / "data/dakshina/ur"


def lexicon(split: str) -> list[tuple[str, str, int]]:
    rows = []
    path = DAKSHINA / f"lexicons/ur.translit.sampled.{split}.tsv"
    for line in path.read_text(encoding="utf-8").splitlines():
        urdu, roman, count = line.split("\t")
        if roman.isascii() and roman.isalpha():
            rows.append((normalize(urdu), roman.lower(), int(count)))
    return rows


def sentence_words(split: str) -> tuple[list[tuple[str, str]], int, int]:
    def native(name: str) -> set[str]:
        path = DAKSHINA / f"romanized/ur.romanized.rejoined.{name}.native.txt"
        return {line.strip() for line in path.read_text(encoding="utf-8").splitlines()}

    keep = native(split)
    if split == "dev":
        keep -= native("test")  # val != test
    pairs: list[tuple[str, str]] = []
    used = total = 0
    rows = (DAKSHINA / "romanized/ur.romanized.rejoined.tsv").read_text("utf-8").splitlines()
    for line in rows:
        urdu, roman = line.split("\t")
        if urdu.strip() not in keep:
            continue
        total += 1
        u = [w for w in words(urdu) if not w.isdigit()]
        r = [w for w in re.findall(r"[A-Za-z]+|\d+", roman) if not w.isdigit()]
        if len(u) == len(r):
            used += 1
            pairs.extend((a, b) for a, b in zip(u, r, strict=True) if b.isalpha())
    return pairs, used, total


class Scorer:
    def __init__(self, use_vocabulary: bool) -> None:
        self.use_vocabulary = use_vocabulary
        self.cache: dict[str, tuple[str, str]] = {}

    def __call__(self, roman: str) -> tuple[str, str]:
        if roman not in self.cache:
            result = transliterate_with_confidence(roman, use_vocabulary=self.use_vocabulary)
            self.cache[roman] = (normalize(result.text), result.sources[0][1])
        return self.cache[roman]


def score(items, scorer) -> dict:
    right = total = 0
    by_source: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for urdu, roman, weight in items:
        got, source = scorer(roman)
        ok = got == urdu
        right += weight * ok
        total += weight
        by_source[source][0] += weight * ok
        by_source[source][1] += weight
    return {
        "accuracy": round(right / total, 4),
        "n": total,
        "by_source": {
            s: {"share": round(n / total, 4), "accuracy": round(k / n, 4)}
            for s, (k, n) in sorted(by_source.items())
        },
    }


def skeleton(roman: str) -> str:
    return min(roman_keys(roman), default="") or roman


def bcubed(items: list[tuple[str, str]], key) -> dict:
    """B-cubed precision/recall of `key` over (spelling, gold word) items."""
    predicted: dict[str, set[str]] = collections.defaultdict(set)
    gold: dict[str, set[str]] = collections.defaultdict(set)
    keys = {}
    for spelling, word in items:
        k = key(spelling)
        keys[spelling] = k
        predicted[k].add(spelling)
        gold[word].add(spelling)
    word_of = dict(items)
    precision = recall = 0.0
    for spelling, word in items:
        cluster = predicted[keys[spelling]]
        same = sum(1 for s in cluster if word_of[s] == word)
        precision += same / len(cluster)
        recall += same / len(gold[word])
    p, r = precision / len(items), recall / len(items)
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", type=Path)
    ap.add_argument("--skip-train", action="store_true", help="skip the 106k-pair train lexicon")
    args = ap.parse_args()
    started = time.time()
    report: dict = {}

    old, new = Scorer(False), Scorer(True)
    print("Sentences (word accuracy, Dakshina's own split)")
    for split in ("dev", "test"):
        pairs, used, total = sentence_words(split)
        items = [(u, r.lower(), 1) for u, r in pairs]
        a, b = score(items, old), score(items, new)
        report[f"sentences_{split}"] = {
            "aligned_sentences": used,
            "sentences": total,
            "words": len(items),
            "v0.1": a,
            "v0.2": b,
        }
        print(
            f"  {split:<5} {used:,}/{total:,} sentences aligned, {len(items):,} words: "
            f"0.1 {a['accuracy']:.3f}  ->  0.2 {b['accuracy']:.3f}"
        )
        for source, row in b["by_source"].items():
            print(
                f"        {source:<11} {row['share']:>6.1%} of words, {row['accuracy']:.3f} right"
            )

    print("\nLexicon (word accuracy, weighted by annotator count)")
    for split in ("dev", "test") if args.skip_train else ("train", "dev", "test"):
        items = lexicon(split)
        a, b = score(items, old), score(items, new)
        report[f"lexicon_{split}"] = {"v0.1": a, "v0.2": b}
        print(
            f"  {split:<5} {len(items):,} spellings: 0.1 {a['accuracy']:.3f}  ->  "
            f"0.2 {b['accuracy']:.3f}",
            flush=True,
        )

    test = lexicon("test")
    in_bucket = sum(c for u, r, c in test if urdu_key(u) in roman_keys(r)) / sum(
        c for _, _, c in test
    )
    report["candidate_key_recall_test"] = round(in_bucket, 4)
    print(
        f"\n  the gold word's key is among the Roman word's keys for {in_bucket:.1%} of "
        "test spellings (the ceiling on vocabulary lookup)"
    )

    print("\nroman_key as clustering (test lexicon, B-cubed)")
    items = sorted({(r, u) for u, r, _ in test})
    for name, key in (
        ("exact spelling", lambda s: s),
        ("consonant skeleton", skeleton),
        ("rules only (0.1)", lambda s: normalize(LEXICON.get(s) or _apply_rules(s))),
        ("roman_key", roman_key),
    ):
        row = bcubed(items, key)
        report.setdefault("roman_key_test", {})[name] = row
        print(
            f"  {name:<20} precision {row['precision']:.3f}  recall {row['recall']:.3f}  "
            f"F1 {row['f1']:.3f}"
        )

    report["seconds"] = round(time.time() - started)
    if args.json:
        args.json.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n[{report['seconds']}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
