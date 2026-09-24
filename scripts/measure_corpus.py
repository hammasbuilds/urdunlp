"""Measure this toolkit against a real Urdu corpus.

    python scripts/measure_corpus.py <corpus>  [--limit N]  [--json out.json]

`<corpus>` is a directory of `.txt` files, a single `.txt` or `.txt.gz` file (one
document per line), or a `.parquet` file with a text column. The published numbers in
docs/CORPUS.md come from XL-Sum Urdu: 84,581 BBC Urdu news articles.

Why this exists: the README made four claims about real Urdu text - that Arabic
codepoints get substituted for Urdu ones, that a small closed-class lexicon covers
most running text, that 115 stopwords are enough, that transliteration is lossy -
and quoted no number for any of them. A toolkit that states no corpus cannot be
shown to work on anything but its own examples.

Reading a parquet needs `pyarrow`. Nothing else here has a dependency, and the
toolkit itself still has none - this is a script, not part of the package.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from urdunlp import (  # noqa: E402
    STOPWORDS,
    is_stopword,
    normalize,
    transliterate_to_roman,
    transliterate_to_urdu,
    words,
)
from urdunlp.translit import LEXICON, RULES  # noqa: E402

# Arabic letters that stand in for their Urdu counterparts in scraped text.
# Unicode NFC leaves these alone - they are genuinely distinct codepoints for
# distinct languages - which is exactly why the problem survives the normalisation
# people assume handles it.
SUBSTITUTED = {
    "ي": "ARABIC YEH -> FARSI YEH",
    "ك": "ARABIC KAF -> KEHEH",
    "ة": "TEH MARBUTA -> HEH GOAL",
    "ى": "ALEF MAKSURA -> FARSI YEH",
    # Added after the fact. This set and normalize()'s table were written from the
    # same list, so the letter missing from one was missing from the other, and the
    # published 8.8% was measuring only the letters the code already handled. A
    # measurement that shares its blind spot with the thing it measures cannot find
    # the gap; it reports a clean bill of health for the part nobody looked at.
    "ه": "ARABIC HEH -> DOACHASHMEE HE or HEH GOAL, by context",
}


def read_documents(path: Path, limit: int):
    """Yield documents from a directory of .txt, one .txt, or a .parquet."""
    seen = 0
    if path.is_dir():
        sources = sorted(path.rglob("*.txt")) + sorted(path.rglob("*.parquet"))
        if not sources:
            raise SystemExit(f"no .txt or .parquet files under {path}")
    else:
        sources = [path]

    for source in sources:
        if source.suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError:
                raise SystemExit("reading a .parquet needs pyarrow: pip install pyarrow") from None
            handle = pq.ParquetFile(source)
            names = handle.schema_arrow.names
            column = "text" if "text" in names else names[-1]
            for batch in handle.iter_batches(batch_size=1000, columns=[column]):
                for value in batch.column(0).to_pylist():
                    if value:
                        yield value
                        seen += 1
                        if seen >= limit:
                            return
        else:
            if source.suffix == ".gz":
                with gzip.open(source, "rt", encoding="utf-8") as handle:
                    lines = handle.read().splitlines()
            else:
                lines = source.read_text(encoding="utf-8").splitlines()
            for line in lines:
                if line.strip():
                    yield line
                    seen += 1
                    if seen >= limit:
                        return


def measure(corpus: Path, limit: int, every: int = 97) -> dict:
    docs = chars = raw_tokens = tok_tokens = changed = 0
    arabic_docs = arabic_tokens = 0
    arabic_counts: collections.Counter = collections.Counter()
    spellings: dict[str, set[str]] = collections.defaultdict(set)
    raw_freq: collections.Counter = collections.Counter()
    freq: collections.Counter = collections.Counter()
    stop_hits = 0
    stop_seen: set[str] = set()

    seen = rt_tried = rt_vowels = rt_novowels = rt_vocab_vowels = rt_vocab_novowels = 0
    rt_types: set[str] = set()
    examples: list[dict] = []
    started = time.time()

    for text in read_documents(corpus, limit):
        docs += 1
        chars += len(text)

        hit = False
        for char in SUBSTITUTED:
            count = text.count(char)
            if count:
                arabic_counts[char] += count
                hit = True
        arabic_docs += hit

        # Normalisation is measured on RAW whitespace tokens, before the tokeniser
        # touches them. `words()` normalises internally, so measuring `normalize(t)
        # != t` on its output is false for every token by construction - which reads
        # like a finding and is a tautology.
        for raw in text.split():
            raw_tokens += 1
            raw_freq[raw] += 1
            norm = normalize(raw)
            spellings[norm].add(raw)
            if norm != raw:
                changed += 1
                if any(c in raw for c in SUBSTITUTED):
                    arabic_tokens += 1

        tokens = words(text)
        tok_tokens += len(tokens)
        for token in tokens:
            freq[token] += 1
            if is_stopword(token):
                stop_hits += 1
                stop_seen.add(token)

            # Round-tripping every token would take hours, so sample every Nth.
            # Spread across the whole corpus rather than capping a running total,
            # which silently takes the sample from the first few hundred documents.
            seen += 1
            if seen % every == 0 and len(token) > 1 and token.isalpha():
                rt_tried += 1
                rt_types.add(token)
                roman = transliterate_to_roman(token)
                bare = transliterate_to_roman(token, insert_short_vowels=False)
                # The 0.1 pipeline - lexicon, then rules - is what the published
                # 44.7% / 61.2% measured, so it is kept as its own column. The
                # vocabulary stage added in 0.2 is measured beside it.
                back = transliterate_to_urdu(roman, use_vocabulary=False)
                back_bare = transliterate_to_urdu(bare, use_vocabulary=False)
                rt_vowels += back == token
                rt_novowels += back_bare == token
                rt_vocab_vowels += transliterate_to_urdu(roman) == token
                rt_vocab_novowels += transliterate_to_urdu(bare) == token
                if back != token and len(examples) < 40:
                    examples.append(
                        {
                            "urdu": token,
                            "roman": roman,
                            "back": back,
                            "roman_bare": bare,
                            "back_bare": back_bare,
                        }
                    )

        if docs % 5000 == 0:
            print(f"  {docs:>7,} documents  {tok_tokens:>12,} tokens", file=sys.stderr, flush=True)

    if not docs:
        raise SystemExit("no documents read")

    multi = {k: v for k, v in spellings.items() if len(v) > 1}
    moved = 0
    for forms in multi.values():
        counts = sorted((raw_freq[f] for f in forms), reverse=True)
        moved += sum(counts[1:])

    return {
        "documents": docs,
        "characters": chars,
        "raw_whitespace_tokens": raw_tokens,
        "tokens": tok_tokens,
        "distinct_normalised_types": len(spellings),
        "types_with_more_than_one_raw_spelling": len(multi),
        "tokens_changed_by_normalize": changed,
        "normalize_change_rate": round(changed / raw_tokens, 6) if raw_tokens else 0,
        "tokens_moved_onto_a_majority_spelling": moved,
        "documents_with_substituted_arabic_letters": arabic_docs,
        "arabic_document_rate": round(arabic_docs / docs, 6),
        "arabic_char_counts": {
            f"U+{ord(k):04X} {SUBSTITUTED[k]}": v for k, v in arabic_counts.most_common()
        },
        "tokens_fixed_by_character_unification": arabic_tokens,
        "stopword_list_size": len(STOPWORDS),
        "stopword_entries_seen": len(stop_seen),
        "stopword_token_hits": stop_hits,
        "stopword_coverage": round(stop_hits / tok_tokens, 6) if tok_tokens else 0,
        "top_30_types": [[w, n] for w, n in freq.most_common(30)],
        "translit_lexicon_size": len(LEXICON),
        "translit_rule_count": len(RULES),
        "roundtrip_sampled_every": every,
        "roundtrip_tried": rt_tried,
        "roundtrip_distinct_types": len(rt_types),
        "roundtrip_rate_with_short_vowels": round(rt_vowels / rt_tried, 6) if rt_tried else 0,
        "roundtrip_rate_without_short_vowels": round(rt_novowels / rt_tried, 6) if rt_tried else 0,
        "roundtrip_rate_with_short_vowels_vocabulary": (
            round(rt_vocab_vowels / rt_tried, 6) if rt_tried else 0
        ),
        "roundtrip_rate_without_short_vowels_vocabulary": (
            round(rt_vocab_novowels / rt_tried, 6) if rt_tried else 0
        ),
        "roundtrip_failure_examples": examples,
        "seconds": round(time.time() - started, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("corpus", type=Path, help="directory of .txt, a .txt, or a .parquet")
    ap.add_argument("--limit", type=int, default=10**9, help="stop after N documents")
    ap.add_argument("--every", type=int, default=97, help="round-trip every Nth token")
    ap.add_argument("--json", type=Path, help="also write the full result here")
    args = ap.parse_args()

    if not args.corpus.exists():
        raise SystemExit(f"no such path: {args.corpus}")

    result = measure(args.corpus, args.limit, args.every)
    if args.json:
        args.json.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")

    skip = {"top_30_types", "roundtrip_failure_examples", "arabic_char_counts"}
    for key, value in result.items():
        if key not in skip:
            print(f"{key:<44} {value}")
    print("\nArabic codepoints substituted for Urdu ones:")
    for key, value in result["arabic_char_counts"].items():
        print(f"  {key:<34} {value:>10,}")
    if args.json:
        print(f"\nwritten to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
