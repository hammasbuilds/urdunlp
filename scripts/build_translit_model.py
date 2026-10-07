"""Train the Roman -> Urdu channel model and write src/urdunlp/data/translit.json.gz.

    python scripts/fetch_dakshina.py
    python scripts/count_vocabulary.py data/dakshina/ur/native_script_wikipedia/\\
        ur.wiki-filt.train.text.shuf.txt.gz --out data/vocab/dakshina_train_counts.tsv \\
        --exclude data/dakshina/ur/romanized/ur.romanized.rejoined.{dev,test}.native.txt
    python scripts/build_translit_model.py

Four ingredients, all from Dakshina (Roark et al. 2020, CC BY-SA 4.0):

  * **Emissions** - P(Roman string | Urdu letter, position, next letter is a vowel
    letter), trained with EM over every monotone alignment of the 106,260 attested
    (Urdu word, romanisation, count) triples in the *training* lexicon. Each Urdu
    letter may emit 0-4 Roman characters, which is how an unwritten short vowel gets
    attached to the consonant before it. The (letter, position, next) table backs
    off to (letter, position) and then to the letter alone.
  * **Vocabulary** - words seen at least MIN_COUNT times in Dakshina's Urdu Wikipedia
    training text, counted with the evaluation sentences excluded.
  * **Attested spellings** - for the 25,000 Urdu words in the training lexicon, how
    often annotators wrote each Roman spelling. The letter model learns that a final ہ
    is rarely typed as `e`, which is true of letters and false of کہ, which people
    type as `ke` all the time. A word's own spellings are mixed with the letter model
    (weight KAPPA), so evidence about the word beats evidence about its letters.
  * **Word bigrams** - over the same training text, eval sentences excluded, kept if
    seen at least BIGRAM_MIN_COUNT times. The transliterator decodes a sentence as a
    whole with them, which is what separates کے from کہ after کہا.

The settings below were chosen on the dev lexicon and dev sentences; the test
partitions were not looked at until scripts/measure_translit.py reported them.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from urdunlp import normalize, words  # noqa: E402
from urdunlp._channel import MAX_EMIT, _context  # noqa: E402
from urdunlp.normalize import _is_urdu_letter  # noqa: E402
from urdunlp.translit import LEXICON as CURATED  # noqa: E402

LEXICON = ROOT / "data/dakshina/ur/lexicons/ur.translit.sampled.train.tsv"

def inputs_digest(*parts: object) -> str:
    """A digest of everything this model is built from. See build_langid_models.py."""
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, Path):
            h.update(part.read_bytes())
        else:
            h.update(repr(part).encode("utf-8"))
        h.update(b"")
    return h.hexdigest()[:32]
COUNTS = ROOT / "data/vocab/dakshina_train_counts.tsv"
OUT = ROOT / "src/urdunlp/data/translit.json.gz"

ITERATIONS = 10
TEXT = ROOT / "data/dakshina/ur/native_script_wikipedia/ur.wiki-filt.train.text.shuf.txt.gz"
EVAL_SENTENCES = [
    ROOT / f"data/dakshina/ur/romanized/ur.romanized.rejoined.{split}.native.txt"
    for split in ("dev", "test")
]

ITERATIONS = 10
# Every setting below was chosen on the dev sentences (Dakshina's dev split, 51,764
# aligned words). Word by word, vocabulary size barely mattered (flat from 42k to
# 85k words); decoding with context, 60,638 words beat 42,498 by 0.2 points.
MIN_COUNT = 3
PRIOR_WEIGHT = 1.0  # word by word: plain Bayes; 0.5 scored 0.862, 1.0 scored 0.870
SMOOTHING = 2.0
FLOOR = 1e-5
KAPPA = 10.0  # 1 -> 0.907, 3 -> 0.910, 10 -> 0.912, 30 -> 0.910, 100 -> 0.909
BIGRAM_MIN_COUNT = 3  # 2 -> +0.1 point for 68% more bigrams; 5 -> -0.4; 10 -> -0.8
LM_WEIGHT = 0.6  # 0.3 -> 0.886, 0.6 -> 0.898, 1.0 -> 0.895 (before attested spellings)
LEXICON_BONUS = 0.0  # 2.0 -> -0.1 point; -2.0 -> -0.7
DISCOUNT = 0.75
CANDIDATES = 5  # 3 -> -0.2 point, 10 -> +0.05


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


def count_bigrams(keep: set[str]) -> tuple[dict, dict]:
    """Bigram counts over the training text, and each context's total and type count.

    Totals and type counts are taken before pruning, so the discounted estimate is
    the same one the full table would give.
    """
    excluded = set()
    for path in EVAL_SENTENCES:
        excluded.update(" ".join(line.split()) for line in path.read_text("utf-8").splitlines())
    bigrams: collections.Counter = collections.Counter()
    with gzip.open(TEXT, "rt", encoding="utf-8") as handle:
        for line in handle:
            if " ".join(line.split()) in excluded:
                continue
            previous = "<s>"
            for token in words(line):
                if not all(_is_urdu_letter(c) and c.isalpha() for c in token):
                    previous = "<unk>"
                    continue
                bigrams[(previous, token)] += 1
                previous = token
    context: dict[str, list[int]] = {}
    for (v, _), c in bigrams.items():
        if v in keep:
            row = context.setdefault(v, [0, 0])
            row[0] += c
            row[1] += 1
    kept = {
        (v, w): c
        for (v, w), c in bigrams.items()
        if c >= BIGRAM_MIN_COUNT and v in keep and w in keep
    }
    return kept, context


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
        word, count = line.split("	")
        if int(count) < MIN_COUNT:
            break
        vocabulary.append([word, int(count)])

    attested: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for urdu, roman, count in pairs:
        attested[urdu][roman] = attested[urdu].get(roman, 0) + count

    # Word ids: the vocabulary, then the curated lexicon's words, then the two markers.
    ids = [w for w, _ in vocabulary]
    known = set(ids)
    for value in CURATED.values():
        for part in normalize(value).split():
            if part not in known:
                known.add(part)
                ids.append(part)
    ids += ["<s>", "<unk>"]
    index = {w: i for i, w in enumerate(ids)}
    bigrams, context = count_bigrams(set(ids))
    table: dict[int, list[int]] = collections.defaultdict(list)
    for (v, w), c in sorted(bigrams.items(), key=lambda kv: (index[kv[0][0]], index[kv[0][1]])):
        table[index[v]] += [index[w], c]

    model = {
        "source": "Dakshina v1.0 Urdu (Roark et al., LREC 2020), CC BY-SA 4.0",
        "prior_weight": PRIOR_WEIGHT,
        "min_count": MIN_COUNT,
        "kappa": KAPPA,
        "lm_weight": LM_WEIGHT,
        "lexicon_bonus": LEXICON_BONUS,
        "discount": DISCOUNT,
        "candidates": CANDIDATES,
        "emit": {"|".join(k): rounded(v) for k, v in sorted(emit.items())},
        "emit_mid": {"|".join(k): rounded(v) for k, v in sorted(mid.items())},
        "emit_letter": {k: rounded(v) for k, v in sorted(letter.items())},
        "vocabulary": vocabulary,
        "attested": {u: dict(sorted(r.items())) for u, r in sorted(attested.items())},
        "words": ids,
        "context": {str(index[v]): context[v] for v in sorted(context, key=index.__getitem__)},
        "bigrams": {str(k): v for k, v in table.items()},
        # What this was built from, so a shipped model that no longer matches its own
        # source can be detected. The Roman tagger was six commits stale before this
        # existed; so was this model - `CURATED` is `urdunlp.translit.LEXICON`, which
        # changed in those same commits. Rebuilding moved 206 bigram weights and 313
        # emission contexts, added 11 entries and dropped 46. Every published figure was
        # unchanged to three decimal places, which is the point: nothing could tell
        # without rebuilding, and "it probably does not matter" is not a measurement.
        "inputs": inputs_digest(
            LEXICON, COUNTS, sorted(CURATED.items()),
            PRIOR_WEIGHT, MIN_COUNT, KAPPA, LM_WEIGHT, LEXICON_BONUS, DISCOUNT,
            CANDIDATES,
        ),
    }
    blob = json.dumps(model, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(gzip.compress(blob, compresslevel=9, mtime=0))
    print(
        f"{len(vocabulary):,} vocabulary words, {len(attested):,} attested words, "
        f"{len(bigrams):,} bigrams, {len(emit):,} emission contexts -> {args.out} "
        f"({args.out.stat().st_size / 1024:.0f} KB) in {time.time() - started:.0f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
