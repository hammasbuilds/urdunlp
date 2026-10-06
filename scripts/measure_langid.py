"""Score `identify_language` and `tag_roman_tokens` on held-out data.

    python scripts/measure_langid.py [--json out.json]

**identify_language.** Each language's Wikipedia sample is split 80/15/5 by a hash of
the paragraph (build_langid_models.py trains on the 80). Reported on the validation
and test paragraphs, whole and cut to a window of 50, 20 and 10 characters from the
middle of each paragraph - the start is often a name in Latin script or a title.
Alongside it: how often `is_urdu`, the script check, says yes to each language. And
whether `margin` means anything: accuracy by margin band.

**tag_roman_tokens.** Three held-out sets, because no one set answers the question:

  * Dakshina's romanised Urdu test sentences - every word is Urdu by the label, so
    anything tagged `en` is a false alarm. Except that it is not: the annotators
    kept names and English words in English spelling ("New York", "website",
    "film"), so the raw rate overstates the error. The most frequent flagged words
    are printed so a reader can see which is which.
  * HotpotQA English validation/test sentences - anything tagged `ur` is an error.
  * Synthetic code-mixing: a 1-3 word English span spliced into a Roman Urdu test
    sentence at a random point, so every tag is known. Synthetic by construction;
    real code-mixed text switches at grammatical boundaries this does not model.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_langid_models import wiki_splits  # noqa: E402

from urdunlp import LANGUAGES, identify_language, is_urdu  # noqa: E402
from urdunlp.langid import _tagger  # noqa: E402

DATA = ROOT / "data"
_WORD = re.compile(r"[A-Za-z]+")


def window(text: str, length: int | None) -> str | None:
    if length is None:
        return text
    if len(text) < length:
        return None
    start = (len(text) - length) // 2
    return text[start : start + length]


def script_report() -> dict:
    splits = wiki_splits()
    report: dict = {"paragraphs": {k: {s: len(v) for s, v in d.items()} for k, d in splits.items()}}

    raw = {
        code: (DATA / "wiki" / f"{code}.txt").read_text("utf-8").splitlines() for code in LANGUAGES
    }
    report["is_urdu_says_yes"] = {
        code: round(sum(is_urdu(p) for p in rows) / len(rows), 4) for code, rows in raw.items()
    }
    print("is_urdu (a script check) says yes to:")
    for code, rate in report["is_urdu_says_yes"].items():
        print(f"  {LANGUAGES[code]:<22} {rate:.1%}")

    for part in ("val", "test"):
        print(f"\nidentify_language, {part} paragraphs")
        for length in (None, 50, 20, 10):
            right = total = 0
            confusion: collections.Counter = collections.Counter()
            per_language: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
            margins = []
            for code, parts in splits.items():
                for paragraph in parts[part]:
                    text = window(paragraph, length)
                    if text is None:
                        continue
                    guess = identify_language(text)
                    ok = guess.language == code
                    right += ok
                    total += 1
                    per_language[code][0] += ok
                    per_language[code][1] += 1
                    margins.append((guess.margin, ok))
                    if not ok:
                        confusion[(code, guess.language)] += 1
            label = "whole" if length is None else f"{length} chars"
            bands = {}
            for lo, hi in ((0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 99)):
                inside = [ok for m, ok in margins if lo <= m < hi]
                if inside:
                    bands[f"{lo}-{hi}"] = {
                        "n": len(inside),
                        "accuracy": round(sum(inside) / len(inside), 4),
                    }
            report[f"{part}_{label}"] = {
                "accuracy": round(right / total, 4),
                "n": total,
                "per_language": {c: round(k / n, 4) for c, (k, n) in per_language.items()},
                "top_confusions": [[a, b, n] for (a, b), n in confusion.most_common(6)],
                "accuracy_by_margin": bands,
            }
            worst = min(per_language, key=lambda c: per_language[c][0] / per_language[c][1])
            print(
                f"  {label:<9} {right / total:.3f} of {total:,}   weakest: {LANGUAGES[worst]} "
                f"{per_language[worst][0] / per_language[worst][1]:.3f}   confusions: "
                + ", ".join(f"{a}->{b} {n}" for (a, b), n in confusion.most_common(3))
            )
            if part == "test" and length == 20:
                for band, row in bands.items():
                    n, accuracy = row["n"], row["accuracy"]
                    print(f"      margin {band:<10} {n:>5} texts, accuracy {accuracy:.3f}")
    return report


def roman_sentences(split: str) -> list[list[str]]:
    def native(name: str) -> set[str]:
        path = DATA / f"dakshina/ur/romanized/ur.romanized.rejoined.{name}.native.txt"
        return {line.strip() for line in path.read_text("utf-8").splitlines()}

    keep = native(split)
    if split == "dev":
        keep -= native("test")
    out = []
    rows = (DATA / "dakshina/ur/romanized/ur.romanized.rejoined.tsv").read_text("utf-8")
    for line in rows.splitlines():
        urdu, roman = line.split("\t")
        if urdu.strip() in keep:
            out.append(_WORD.findall(roman))
    return out


def english_sentences(split: str) -> list[list[str]]:
    path = DATA / f"english/{split}.txt"
    return [_WORD.findall(line) for line in path.read_text("utf-8").splitlines()]


def tagger_report() -> dict:
    tagger = _tagger()
    report: dict = {}
    for urdu_split, english_split in (("dev", "val"), ("test", "test")):
        urdu = roman_sentences(urdu_split)
        english = english_sentences(english_split)

        flagged: collections.Counter = collections.Counter()
        false_en = urdu_words = 0
        # The same count restricted to tokens the Roman Urdu lexicon attests. The
        # unrestricted rate includes tokens with NO Urdu lexical evidence - county,
        # website, afghanistan - which Dakshina's Wikipedia-derived romanised side really
        # contains while labelling every token on it `ur`. Those are reference-label
        # problems, and they are 45.6% of the unrestricted count, so a figure that calls
        # itself the tagger's error rate has to separate them.
        attested = attested_en = 0
        lexicon = tagger.urdu.unigram
        for sentence in urdu:
            tags = tagger.tag(sentence)
            urdu_words += len(tags)
            for word, tag in zip(sentence, tags, strict=True):
                known = word.lower() in lexicon
                if known:
                    attested += 1
                if tag == "en":
                    false_en += 1
                    flagged[word.lower()] += 1
                    if known:
                        attested_en += 1

        false_ur = english_words = 0
        for sentence in english:
            tags = tagger.tag(sentence)
            english_words += len(tags)
            false_ur += tags.count("ur")

        rng = random.Random(0)
        right = total = found = spans = 0
        pool = [s for s in english if len(s) >= 3]
        for sentence in urdu:
            if len(sentence) < 4:
                continue
            span_len = rng.randint(1, 3)
            donor = rng.choice(pool)
            start = rng.randint(0, len(donor) - span_len)
            at = rng.randint(1, len(sentence) - 1)
            mixed = sentence[:at] + donor[start : start + span_len] + sentence[at:]
            gold = ["ur"] * at + ["en"] * span_len + ["ur"] * (len(sentence) - at)
            tags = tagger.tag(mixed)
            right += sum(a == b for a, b in zip(tags, gold, strict=True))
            total += len(gold)
            found += sum(a == b == "en" for a, b in zip(tags, gold, strict=True))
            spans += span_len

        row = {
            "urdu_words": urdu_words,
            "urdu_tagged_en": round(false_en / urdu_words, 4),
            # An upper bound: the denominator includes code-mixed reference tokens the
            # gold labels call `ur` with no Urdu lexical evidence.
            "urdu_words_in_lexicon": attested,
            "urdu_tagged_en_lexicon_only": (
                round(attested_en / attested, 4) if attested else None
            ),
            "english_words": english_words,
            "english_tagged_ur": round(false_ur / english_words, 4),
            "mixed_token_accuracy": round(right / total, 4),
            "mixed_english_recall": round(found / spans, 4),
            "most_flagged_in_urdu": flagged.most_common(60),
        }
        report[urdu_split] = row
        print(f"\ntag_roman_tokens, {urdu_split}")
        print(f"  Roman Urdu words tagged en   {row['urdu_tagged_en']:.2%} of {urdu_words:,}")
        if row["urdu_tagged_en_lexicon_only"] is not None:
            print(
                f"    ... of those the lexicon has {row['urdu_tagged_en_lexicon_only']:.2%} "
                f"of {attested:,}  (the rest of the gap is the reference's own code-mixing)"
            )
        print(f"  English words tagged ur      {row['english_tagged_ur']:.2%} of {english_words:,}")
        print(
            f"  synthetic mix: token accuracy {row['mixed_token_accuracy']:.3f}, "
            f"English recall {row['mixed_english_recall']:.3f}"
        )
        print("  most flagged 'Urdu' words:", ", ".join(w for w, _ in flagged.most_common(40)))
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    started = time.time()
    report = {"identify_language": script_report(), "tag_roman_tokens": tagger_report()}
    report["seconds"] = round(time.time() - started)
    if args.json:
        args.json.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n[{report['seconds']}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
