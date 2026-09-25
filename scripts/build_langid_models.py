"""Train the two language models in `urdunlp.langid` and write them into the package.

    python scripts/fetch_wikipedia_samples.py      # data/wiki/<code>.txt
    python scripts/fetch_dakshina.py               # Roman Urdu lexicon + sentences
    python scripts/extract_english.py <hotpot dir> # data/english/{train,val,test}.txt
    python scripts/build_langid_models.py

**Script model** (`identify_language`): multinomial naive Bayes over character 1-5
grams, one table per language, from the *train* split of each language's Wikipedia
sample. Splits are by a hash of the paragraph, 80/15/5 - under 10,000 paragraphs per
language, so the house 80/15/5 policy applies. Each table is cut to its TOP_GRAMS
most frequent grams to keep the package small; the cost of that cut is printed on
the validation split.

**Roman tagger** (`tag_roman_tokens`): P(word | language) for Roman Urdu and English,
each a word-frequency table interpolated with a Witten-Bell character 4-gram model,
then a two-state Viterbi pass over the sentence. English frequencies come from
HotpotQA's English Wikipedia sentences. Roman Urdu has no running text to count, so
its frequencies are *estimated*: each Urdu word's corpus frequency is spread over the
romanisations annotators wrote for it in Dakshina's training lexicon (plus the
toolkit's own curated lexicon), so `ke` inherits the frequency of کے.

The settings (unigram weight 0.5, stay 0.8, Urdu bias 1.0) were chosen on Dakshina's
dev sentences and HotpotQA validation sentences; see measure_langid.py for the
held-out numbers.
"""

from __future__ import annotations

import collections
import gzip
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from urdunlp import normalize  # noqa: E402
from urdunlp.langid import LANGUAGES, _grams  # noqa: E402
from urdunlp.translit import LEXICON  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "src/urdunlp/data"

N_MAX = 5  # 3 -> 92.9% on 20-char val windows, 4 -> 94.6%, 5 -> 96.2% unpruned
ALPHA = 0.1  # 0.5 -> 0.1 added ~0.3 points at every length
TOP_GRAMS = 20000  # 5-grams unpruned are 5.4 MB; top 20k per language is 848 KB and 94.9%

ORDER = 4
UNIGRAM_WEIGHT = 0.5
STAY = 0.8
URDU_BIAS = 1.0
TOP_WORDS = 30000

_ARABIC_RUNS = re.compile("[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿‌]+")
_ROMAN_WORD = re.compile(r"[A-Za-z]+")


def clean(text: str) -> str:
    return " ".join(_ARABIC_RUNS.findall(unicodedata.normalize("NFC", text)))


def split_of(paragraph: str) -> str:
    h = hashlib.sha256(paragraph.encode("utf-8")).digest()[0] / 256
    return "train" if h < 0.80 else ("val" if h < 0.95 else "test")


def wiki_splits() -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    for code in LANGUAGES:
        path = DATA / "wiki" / f"{code}.txt"
        if not path.exists():
            raise SystemExit(f"missing {path}: run scripts/fetch_wikipedia_samples.py")
        parts: dict[str, list[str]] = collections.defaultdict(list)
        for paragraph in path.read_text(encoding="utf-8").splitlines():
            text = clean(paragraph)
            if len(text) >= 20:
                parts[split_of(paragraph)].append(text)
        out[code] = parts
    return out


def distinctive_letters(counts: dict[str, collections.Counter]) -> dict[str, list[str]]:
    """Letters common in at most three languages and essentially absent elsewhere."""
    share = {}
    for lang, c in counts.items():
        letters = {g: n for g, n in c.items() if len(g) == 1 and g.strip()}
        total = sum(letters.values())
        share[lang] = {g: n / total for g, n in letters.items()}
    every = set().union(*share.values())
    out: dict[str, list[str]] = collections.defaultdict(list)
    for letter in sorted(every):
        users = [lang for lang in share if share[lang].get(letter, 0) >= 1e-3]
        others = [lang for lang in share if lang not in users]
        if 1 <= len(users) <= 3 and all(share[o].get(letter, 0) < 2e-5 for o in others):
            for lang in users:
                out[lang].append(letter)
    return dict(out)


def build_script_model() -> None:
    splits = wiki_splits()
    counts = {}
    for code, parts in splits.items():
        c: collections.Counter = collections.Counter()
        for paragraph in parts["train"]:
            c.update(_grams(paragraph, N_MAX))
        counts[code] = c
    distinctive = distinctive_letters(counts)

    from urdunlp.langid import _ScriptModel

    def model_for(top: int | None) -> _ScriptModel:
        kept = {k: dict(c.most_common(top)) for k, c in counts.items()}
        vocabulary = set().union(*kept.values())
        return _ScriptModel(
            {
                "n_max": N_MAX,
                "alpha": ALPHA,
                "vocabulary_size": len(vocabulary),
                "counts": kept,
                "distinctive": distinctive,
            }
        )

    for top in (40000, TOP_GRAMS, 6000):
        model = model_for(top)
        right = total = 0
        for code, parts in splits.items():
            for paragraph in parts["val"]:
                window = paragraph[:20]
                scores = model.log_likelihoods(window)
                right += max(scores, key=scores.get) == code
                total += 1
        print(
            f"  script model, top {top or 'all'} grams: validation accuracy on 20-char "
            f"windows {right / total:.4f}"
        )

    model = model_for(TOP_GRAMS)
    blob = {
        "n_max": N_MAX,
        "alpha": ALPHA,
        "vocabulary_size": model.vocabulary_size,
        "counts": model.counts,
        "distinctive": distinctive,
        "source": "Wikipedia article intros, one random sample per language (CC BY-SA 4.0)",
    }
    write(OUT / "langid.json.gz", blob)
    print("  distinctive letters:", {k: "".join(v) for k, v in distinctive.items()})


def roman_urdu_unigram() -> dict[str, float]:
    frequency = {}
    for line in (DATA / "vocab/dakshina_train_counts.tsv").read_text("utf-8").splitlines():
        word, count = line.split("\t")
        frequency[word] = int(count)
    attested: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    lexicon = DATA / "dakshina/ur/lexicons/ur.translit.sampled.train.tsv"
    for line in lexicon.read_text("utf-8").splitlines():
        urdu, roman, count = line.split("\t")
        roman = roman.lower()
        if roman.isascii() and roman.isalpha():
            attested[normalize(urdu)][roman] += int(count)
    for roman, urdu in LEXICON.items():
        attested[normalize(urdu)][roman] += 3
    unigram: collections.Counter = collections.Counter()
    for urdu, romans in attested.items():
        f = frequency.get(urdu, 0)
        if f:
            total = sum(romans.values())
            for roman, count in romans.items():
                unigram[roman] += f * count / total
    return unigram


def char_counts(words: list[str]) -> dict[str, dict[str, int]]:
    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for word in words:
        padded = "^" * (ORDER - 1) + word + "$"
        for j in range(ORDER - 1, len(padded)):
            for o in range(ORDER):
                counts[padded[j - o : j]][padded[j]] += 1
    return {ctx: dict(c) for ctx, c in counts.items()}


def build_roman_tagger() -> None:
    urdu_types = []
    lexicon = DATA / "dakshina/ur/lexicons/ur.translit.sampled.train.tsv"
    for line in lexicon.read_text("utf-8").splitlines():
        roman = line.split("\t")[1].lower()
        if roman.isascii() and roman.isalpha():
            urdu_types.append(roman)
    urdu_types = sorted(set(urdu_types))
    english: collections.Counter = collections.Counter()
    for sentence in (DATA / "english/train.txt").read_text("utf-8").splitlines():
        english.update(w.lower() for w in _ROMAN_WORD.findall(sentence))
    english_types = sorted(w for w, c in english.items() if c >= 2)

    urdu_unigram = roman_urdu_unigram()

    def side(unigram, types):
        top = dict(sorted(unigram.items(), key=lambda kv: -kv[1])[:TOP_WORDS])
        return {
            "unigram": {w: round(c, 1) for w, c in top.items()},
            "order": ORDER,
            "chars": char_counts(types),
            "unigram_weight": UNIGRAM_WEIGHT,
        }

    blob = {
        "ur": side(urdu_unigram, urdu_types),
        "en": side(english, english_types),
        "stay": STAY,
        "urdu_bias": URDU_BIAS,
        "source": "Roman Urdu: Dakshina v1.0 (CC BY-SA 4.0). English: HotpotQA (CC BY-SA 4.0).",
    }
    write(OUT / "roman_tagger.json.gz", blob)


def write(path: Path, blob: dict) -> None:
    raw = json.dumps(blob, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(raw.encode("utf-8"), compresslevel=9, mtime=0))
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    build_script_model()
    build_roman_tagger()
