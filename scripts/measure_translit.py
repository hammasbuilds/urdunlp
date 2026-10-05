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

from urdunlp import normalize, roman_key, translit, transliterate_to_roman, words  # noqa: E402
from urdunlp._channel import channel, roman_keys, urdu_key  # noqa: E402
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


def sentence_words(split: str) -> tuple[list[list[tuple[str, str]]], int, int]:
    def native(name: str) -> set[str]:
        path = DAKSHINA / f"romanized/ur.romanized.rejoined.{name}.native.txt"
        return {line.strip() for line in path.read_text(encoding="utf-8").splitlines()}

    keep = native(split)
    if split == "dev":
        keep -= native("test")  # val != test
    sentences: list[list[tuple[str, str]]] = []
    used = total = 0
    rows = (DAKSHINA / "romanized/ur.romanized.rejoined.tsv").read_text("utf-8").splitlines()
    for line in rows:
        urdu, roman = line.split("\t")
        if urdu.strip() not in keep:
            continue
        total += 1
        pairs = align(urdu, roman)
        if pairs is not None:
            used += 1
            sentences.append(pairs)
    return sentences, used, total


def align(urdu: str, roman: str) -> list[tuple[str, str]] | None:
    """(urdu, roman) word pairs of one sentence, or None when the counts disagree."""
    # A number is dropped from both sides. words() keeps 1,500 as one token since
    # 0.2; the pattern drops exactly what isdigit() dropped when it split it.
    u = [w for w in words(urdu) if not re.fullmatch(r"\d+(?:[.,:/٫٬-]\d+)*", w)]
    r = [w for w in re.findall(r"[A-Za-z]+|\d+", roman) if not w.isdigit()]
    if len(u) != len(r):
        return None
    return [(a, b.lower()) for a, b in zip(u, r, strict=True) if b.isalpha()]


class Scorer:
    """One Roman word at a time, with no context - the 0.1 pipeline or 0.2 word mode."""

    def __init__(self, use_vocabulary: bool) -> None:
        self.use_vocabulary = use_vocabulary
        self.cache: dict[str, tuple[str, str]] = {}

    def __call__(self, roman: str) -> tuple[str, str]:
        if roman not in self.cache:
            result = transliterate_with_confidence(
                roman, use_vocabulary=self.use_vocabulary, use_context=False
            )
            self.cache[roman] = (normalize(result.text), result.sources[0][1])
        return self.cache[roman]


def _public_path(romans: list[str]) -> list[tuple[str, str]]:
    """(urdu, source) per word, exactly as transliterate_to_urdu produces it.

    Until round 2 this called the Viterbi decoder directly, which is not quite what
    a user gets: the public function sends a one-word sentence down the word-by-word
    path, and since round 2 it also reads stretched letters, keeps clearly English
    words it cannot resolve, and so on. Scoring the decoder measured a function
    nobody calls. For a version without `_render` (0.2 before round 2) the old
    public behaviour is rebuilt: the decoder for two words or more, else the
    word-by-word path.
    """
    render = getattr(translit, "_render", None)
    if render is not None:
        out = [(u, s) for _, u, s in render(" ".join(romans)) if s != "space"]
        assert len(out) == len(romans), romans
        return out
    if len(romans) > 1:
        return channel().decode(romans, LEXICON)
    result = transliterate_with_confidence(romans[0])
    return [(result.text, result.sources[0][1])]


def score_in_context(sentences: list[list[tuple[str, str]]]) -> dict:
    """Transliterate each sentence as a whole, the way transliterate_to_urdu does."""
    right = total = 0
    by_source: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for sentence in sentences:
        chosen = _public_path([r for _, r in sentence])
        for (gold, _), (urdu, source) in zip(sentence, chosen, strict=True):
            ok = normalize(urdu) == gold
            right += ok
            total += 1
            by_source[source][0] += ok
            by_source[source][1] += 1
    return {
        "accuracy": round(right / total, 4),
        "n": total,
        "by_source": {
            s: {"share": round(n / total, 4), "accuracy": round(k / n, 4)}
            for s, (k, n) in sorted(by_source.items())
        },
    }


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

    # Say what is missing and how to get it, the way build_langid_models.py does. This
    # used to reach the user as a bare FileNotFoundError naming a path inside a
    # gitignored directory, which reads like a broken script rather than a download
    # nobody has run yet - and this script is named in the README.
    if not DAKSHINA.is_dir():
        raise SystemExit(
            f"missing {DAKSHINA}: run scripts/fetch_dakshina.py first.\n"
            "For the headline Roman->Urdu number with no download, use "
            "scripts/quick_check.py instead - it scores the committed sample."
        )

    started = time.time()
    report: dict = {}

    # Speed, first, before any scoring warms a cache: the README's "words a second" is
    # this line. The model loads on the first call, which is timed separately.
    dev_sentences, _, _ = sentence_words("dev")
    romans = [" ".join(r for _, r in sentence) for sentence in dev_sentences]
    n_words = sum(len(sentence) for sentence in dev_sentences)
    cpu = time.process_time()
    transliterate_with_confidence("mera naam")
    load = time.process_time() - cpu
    cpu = time.process_time()
    for roman in romans:
        transliterate_with_confidence(roman)
    cpu = time.process_time() - cpu
    report["speed"] = {"load_cpu_s": round(load, 2), "words": n_words, "cpu_s": round(cpu, 1)}
    print(
        f"Speed: model load {load:.1f} s of CPU, then {n_words / cpu:,.0f} words a second "
        f"of CPU on {n_words:,} words of dev sentences\n"
    )

    old, new = Scorer(False), Scorer(True)
    print("Sentences (word accuracy, Dakshina's own split)")
    for split in ("dev", "test"):
        sentences, used, total = sentence_words(split)
        items = [(u, r, 1) for sentence in sentences for u, r in sentence]
        a, b = score(items, old), score(items, new)
        c = score_in_context(sentences)
        report[f"sentences_{split}"] = {
            "aligned_sentences": used,
            "sentences": total,
            "words": len(items),
            "v0.1": a,
            "v0.2_word_by_word": b,
            "v0.2_in_context": c,
        }
        print(
            f"  {split:<5} {used:,}/{total:,} sentences aligned, {len(items):,} words: "
            f"0.1 {a['accuracy']:.3f}  ->  word by word {b['accuracy']:.3f}  ->  "
            f"in context {c['accuracy']:.3f}",
            flush=True,
        )
        for source, row in c["by_source"].items():
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

    print("\nUrdu -> Roman: does the output match a spelling a person wrote?")
    methods = {
        "rules, short vowels (0.1 default)": lambda u: transliterate_to_roman(u, method="rules"),
        "rules, literal": lambda u: transliterate_to_roman(u, insert_short_vowels=False),
        "learned (0.2 default)": transliterate_to_roman,
    }
    for split in ("dev", "test"):
        attested: dict[str, set[str]] = collections.defaultdict(set)
        for urdu, roman, _ in lexicon(split):
            attested[urdu].add(roman)
        sentences, _, _ = sentence_words(split)
        pairs = [pair for sentence in sentences for pair in sentence]
        written: dict[str, set[str]] = collections.defaultdict(set)
        for urdu, roman in pairs:
            written[urdu].add(roman)
        for name, fn in methods.items():
            cache: dict[str, str] = {}

            def spell(u: str, fn=fn, cache=cache) -> str:
                if u not in cache:
                    cache[u] = fn(u)
                return cache[u]

            row = {
                "lexicon_words_matching_a_human_spelling": round(
                    sum(spell(u) in rs for u, rs in attested.items()) / len(attested), 4
                ),
                "sentence_words_exactly_as_written": round(
                    sum(spell(u) == r for u, r in pairs) / len(pairs), 4
                ),
                "sentence_words_matching_any_spelling_of_the_word": round(
                    sum(spell(u) in written[u] for u, _ in pairs) / len(pairs), 4
                ),
            }
            report.setdefault(f"to_roman_{split}", {})[name] = row
            lexicon_rate, exact, any_spelling = row.values()
            print(
                f"  {split:<5} {name:<34} lexicon {lexicon_rate:.3f}   "
                f"sentence exact {exact:.3f}   any spelling {any_spelling:.3f}",
                flush=True,
            )

    report["seconds"] = round(time.time() - started)
    if args.json:
        args.json.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n[{report['seconds']}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
