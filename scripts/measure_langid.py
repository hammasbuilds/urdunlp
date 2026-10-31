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

from build_langid_models import wiki_splits, window  # noqa: E402

from urdunlp import LANGUAGES, identify_language, is_urdu  # noqa: E402
from urdunlp.langid import _tagger  # noqa: E402

DATA = ROOT / "data"
_WORD = re.compile(r"[A-Za-z]+")


def margin_bands(margins: list[tuple[float, bool]]) -> dict:
    """Accuracy per fixed margin band, the shape earlier reports used.

    The edges - 0, 0.02, 0.05, 0.1 - were chosen when the model was built on 1,500
    paragraphs per language and its margins were fractions. On the 130,791-paragraph
    corpus the same quantity runs from 0.001 to 41, with a median of 11.9, so 0.1 is the
    0.6th percentile: 6,093 of 6,136 windows land in the top band and the three bands
    below it hold 9, 10 and 24 texts between them.

    That is how docs/CORPUS.md came to say "6,093 of 6,136 windows carry a margin above
    0.1, and so do 288 of the 312 errors" as evidence that margin is not informative. The
    sentence is arithmetically true and says nothing: when 99.3% of everything is above a
    threshold, so are almost all the errors. Band edges have to come from the
    distribution - see margin_deciles() - or they measure the scale rather than the model.
    """
    bands = {}
    for lo, hi in ((0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 99)):
        inside = [ok for m, ok in margins if lo <= m < hi]
        if inside:
            bands[f"{lo}-{hi}"] = {
                "n": len(inside),
                "accuracy": round(sum(inside) / len(inside), 4),
            }
    return bands


def margin_deciles(margins: list[tuple[float, bool]]) -> list[dict]:
    """Accuracy per tenth of the margin distribution, lowest first.

    Edges from the data, so the answer cannot be an artefact of a scale that moved. Each
    row carries the margins it spans, so a reader can turn a decile back into a threshold
    for their own text.
    """
    rows = sorted(margins)
    n = len(rows)
    out = []
    for i in range(10):
        chunk = rows[i * n // 10 : (i + 1) * n // 10]
        if not chunk:
            continue
        out.append(
            {
                "decile": i + 1,
                "margin_from": round(chunk[0][0], 4),
                "margin_to": round(chunk[-1][0], 4),
                "n": len(chunk),
                "accuracy": round(sum(ok for _, ok in chunk) / len(chunk), 4),
                "errors": sum(1 for _, ok in chunk if not ok),
            }
        )
    return out


def error_retention(margins: list[tuple[float, bool]]) -> list[dict]:
    """What keeping only the highest margins buys, which is the question a caller asks.

    "Is accuracy worse at low margin" is answered by the deciles. A caller who can drop
    an answer asks the other question: if I keep the confident ones, how many of the
    errors do I keep? A signal can look informative per band and still be useless here,
    if the errors are spread evenly.
    """
    rows = sorted(margins)
    n = len(rows)
    total_errors = sum(1 for _, ok in rows if not ok)
    out = []
    for keep in (0.5, 0.75, 0.9, 0.95, 0.99):
        kept = rows[int(n * (1 - keep)) :]
        if not kept:
            continue
        errors = sum(1 for _, ok in kept if not ok)
        out.append(
            {
                "keep": keep,
                "n": len(kept),
                "accuracy": round(sum(ok for _, ok in kept) / len(kept), 4),
                "errors_kept": errors,
                "errors_total": total_errors,
                "errors_kept_share": round(errors / total_errors, 4) if total_errors else None,
            }
        )
    return out


def script_report(wiki_dir: Path | None = None) -> dict:
    """Score identify_language on `wiki_dir`, default data/wiki.

    The corpus directory has to be an argument here, not a constant. build_langid_models.py
    grew a --wiki-dir option so a larger corpus could be "trained and scored", but the
    scoring half was never wired up: this function read data/wiki whatever the model was
    trained on. A model built on the 132,000-paragraph corpus was therefore still reported
    against the 1,500-per-language sample, and the corpus upgrade could not reach the number.
    """
    folder = wiki_dir or (DATA / "wiki")
    splits = wiki_splits(wiki_dir)
    report: dict = {
        "corpus": folder.name,
        "paragraphs": {k: {s: len(v) for s, v in d.items()} for k, d in splits.items()},
    }

    raw = {code: (folder / f"{code}.txt").read_text("utf-8").splitlines() for code in LANGUAGES}
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
            bands = margin_bands(margins)
            deciles = margin_deciles(margins)
            retention = error_retention(margins)
            report[f"{part}_{label}"] = {
                "accuracy": round(right / total, 4),
                "n": total,
                "per_language": {c: round(k / n, 4) for c, (k, n) in per_language.items()},
                "top_confusions": [[a, b, n] for (a, b), n in confusion.most_common(6)],
                # Kept for continuity with earlier reports, and no longer the figure any
                # document quotes: see margin_bands() for why fixed edges stopped
                # describing this model.
                "accuracy_by_margin": bands,
                "errors": total - right,
                "accuracy_by_margin_decile": deciles,
                "error_retention_by_margin": retention,
            }
            worst = min(per_language, key=lambda c: per_language[c][0] / per_language[c][1])
            print(
                f"  {label:<9} {right / total:.3f} of {total:,}   weakest: {LANGUAGES[worst]} "
                f"{per_language[worst][0] / per_language[worst][1]:.3f}   confusions: "
                + ", ".join(f"{a}->{b} {n}" for (a, b), n in confusion.most_common(3))
            )
            if part == "test" and length == 20:
                print("      margin decile (1 = lowest margin)")
                for row in deciles:
                    print(
                        f"        {row['decile']:>2}  {row['margin_from']:7.3f}-"
                        f"{row['margin_to']:<8.3f} {row['n']:>5} texts, accuracy "
                        f"{row['accuracy']:.3f}, {row['errors']:>3} errors"
                    )
                for row in retention:
                    print(
                        f"      keeping the top {row['keep']:>4.0%} by margin: accuracy "
                        f"{row['accuracy']:.4f}, {row['errors_kept']} of "
                        f"{row['errors_total']} errors kept "
                        f"({row['errors_kept_share']:.1%})"
                    )
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
            "urdu_tagged_en_lexicon_only": (round(attested_en / attested, 4) if attested else None),
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
    # Defaulted, not optional. `--json` had no default, so this script printed a full
    # measurement, exited 0, and wrote nothing: the committed report stayed at whatever
    # the last person to remember the flag produced. That is how
    # data/langid_report_v5.json went stale while the model was retrained, and how the
    # six tests that read it sat skipped because the figures they guard were never
    # refreshed. The report records which corpus it measured, so a --wiki-dir run that
    # lands here is identifiable rather than silently confusing.
    ap.add_argument(
        "--json",
        type=Path,
        default=DATA / "langid_report_v5.json",
        help="where to write the report (default data/langid_report_v5.json, the file "
        "tests/test_published_langid_numbers.py reads)",
    )
    ap.add_argument(
        "--wiki-dir",
        type=Path,
        help="corpus directory to score on (default data/wiki). Must match the "
        "--wiki-dir the model was built with, or the number describes neither corpus",
    )
    args = ap.parse_args()
    started = time.time()
    report = {
        "identify_language": script_report(args.wiki_dir),
        "tag_roman_tokens": tagger_report(),
    }
    report["seconds"] = round(time.time() - started)
    if args.json:
        args.json.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.json.name} (corpus: {report['identify_language']['corpus']})")
    print(f"\n[{report['seconds']}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
