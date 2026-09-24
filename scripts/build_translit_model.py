"""Train the Roman -> Urdu channel model and write src/urdunlp/data/translit.json.gz.

    python scripts/fetch_dakshina.py
    python scripts/count_vocabulary.py data/dakshina/ur/native_script_wikipedia/\\
        ur.wiki-filt.train.text.shuf.txt.gz --out data/vocab/dakshina_train_counts.tsv \\
        --exclude data/dakshina/ur/romanized/ur.romanized.rejoined.{dev,test}.native.txt
    python scripts/build_translit_model.py

Two ingredients, both from Dakshina (Roark et al. 2020, CC BY-SA 4.0):

  * **Emissions** - P(Roman string | Urdu letter, position, next letter is a vowel
    letter), trained with EM over every monotone alignment of the 106,260 attested
    (Urdu word, romanisation, count) triples in the *training* lexicon. Each Urdu
    letter may emit 0-4 Roman characters, which is how an unwritten short vowel gets
    attached to the consonant before it. The (letter, position, next) table backs
    off to (letter, position) and then to the letter alone.
  * **Vocabulary** - words seen at least MIN_COUNT times in Dakshina's Urdu Wikipedia
    training text, counted with the evaluation sentences excluded.

The settings below were chosen on the dev lexicon and dev sentences; the test
partitions were not looked at until scripts/measure_translit.py reported them.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from urdunlp import normalize  # noqa: E402
from urdunlp._channel import MAX_EMIT, _context  # noqa: E402

LEXICON = ROOT / "data/dakshina/ur/lexicons/ur.translit.sampled.train.tsv"
COUNTS = ROOT / "data/vocab/dakshina_train_counts.tsv"
OUT = ROOT / "src/urdunlp/data/translit.json.gz"

ITERATIONS = 10
MIN_COUNT = 5  # 42,498 words; dev sentence accuracy was flat from 2 (84,783) to 5
PRIOR_WEIGHT = 1.0  # plain Bayes; 0.5 scored 0.862 on dev sentences, 1.0 scored 0.870
SMOOTHING = 2.0
FLOOR = 1e-5


def load_pairs(path: Path) -> list[tuple[str, str, int]]:
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        urdu, roman, count = line.split("\t")
        urdu, roman = normalize(urdu), roman.lower()
        if urdu and roman.isascii() and roman.isalpha():
            pairs.append((urdu, roman, int(count)))
    return pairs


def smooth(counts: dict[str, float], prior: dict[str, float], k: float) -> dict[str, float]:
    total = sum(counts.values())
    row = {}
    for s in set(counts) | set(prior):
        p = (counts.get(s, 0.0) + k * prior.get(s, 0.0)) / (total + k)
        if p > FLOOR:
            row[s] = p
    return row


def train(pairs: list[tuple[str, str, int]]) -> tuple[dict, dict, dict]:
    emit: dict = {}
    mid: dict = {}
    letter: dict = {}

    def table(key):
        if not emit:
            return None  # first iteration: every segmentation equally likely
        return emit.get(key) or mid.get(key[:2]) or letter.get(key[0], {})

    for iteration in range(ITERATIONS):
        expected: dict = collections.defaultdict(lambda: collections.defaultdict(float))
        loglik = 0.0
        for urdu, roman, weight in pairs:
            n, m = len(urdu), len(roman)
            contexts = [_context(urdu, i) for i in range(n)]
            probs = []
            for i in range(n):
                t = table(contexts[i])
                row = {}
                for j in range(m + 1):
                    for length in range(min(MAX_EMIT, m - j) + 1):
                        p = 1.0 if t is None else t.get(roman[j : j + length], 0.0)
                        if p:
                            row[(j, length)] = p
                probs.append(row)
            fwd = [dict() for _ in range(n + 1)]
            fwd[0][0] = 1.0
            for i in range(n):
                for (j, length), p in probs[i].items():
                    if j in fwd[i]:
                        fwd[i + 1][j + length] = fwd[i + 1].get(j + length, 0.0) + fwd[i][j] * p
            z = fwd[n].get(m, 0.0)
            if z <= 0:
                continue
            loglik += weight * math.log(z)
            bwd = [dict() for _ in range(n + 1)]
            bwd[n][m] = 1.0
            for i in range(n - 1, -1, -1):
                for (j, length), p in probs[i].items():
                    after = bwd[i + 1].get(j + length)
                    if after:
                        bwd[i][j] = bwd[i].get(j, 0.0) + p * after
            for i in range(n):
                for (j, length), p in probs[i].items():
                    before, after = fwd[i].get(j), bwd[i + 1].get(j + length)
                    if before and after:
                        expected[contexts[i]][roman[j : j + length]] += (
                            weight * before * p * after / z
                        )

        by_mid: dict = collections.defaultdict(lambda: collections.defaultdict(float))
        by_letter: dict = collections.defaultdict(lambda: collections.defaultdict(float))
        for (ch, position, _following), row in expected.items():
            for s, v in row.items():
                by_mid[(ch, position)][s] += v
                by_letter[ch][s] += v
        letter = {ch: {s: v / sum(r.values()) for s, v in r.items()} for ch, r in by_letter.items()}
        mid = {key: smooth(r, letter[key[0]], SMOOTHING) for key, r in by_mid.items()}
        emit = {key: smooth(r, mid[key[:2]], SMOOTHING) for key, r in expected.items()}
        print(
            f"  EM iteration {iteration + 1}/{ITERATIONS}  log-likelihood {loglik:,.0f}", flush=True
        )
    return emit, mid, letter


def rounded(table: dict[str, float]) -> dict[str, float]:
    return {s: float(f"{p:.4g}") for s, p in sorted(table.items(), key=lambda kv: -kv[1])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    started = time.time()

    pairs = load_pairs(LEXICON)
    print(f"{len(pairs):,} training pairs from {LEXICON.name}")
    emit, mid, letter = train(pairs)

    vocabulary = []
    for line in COUNTS.read_text(encoding="utf-8").splitlines():
        word, count = line.split("\t")
        if int(count) < MIN_COUNT:
            break
        vocabulary.append([word, int(count)])

    model = {
        "source": "Dakshina v1.0 Urdu (Roark et al., LREC 2020), CC BY-SA 4.0",
        "prior_weight": PRIOR_WEIGHT,
        "min_count": MIN_COUNT,
        "emit": {"|".join(k): rounded(v) for k, v in sorted(emit.items())},
        "emit_mid": {"|".join(k): rounded(v) for k, v in sorted(mid.items())},
        "emit_letter": {k: rounded(v) for k, v in sorted(letter.items())},
        "vocabulary": vocabulary,
    }
    blob = json.dumps(model, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(gzip.compress(blob, compresslevel=9, mtime=0))
    print(
        f"{len(vocabulary):,} vocabulary words, {len(emit):,} emission contexts -> "
        f"{args.out} ({args.out.stat().st_size / 1024:.0f} KB) in {time.time() - started:.0f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
