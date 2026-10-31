r"""Train the two language models in `urdunlp.langid` and write them into the package.

    python scripts/fetch_wikipedia_samples.py      # data/wiki/<code>.txt, 1,500 each
    python scripts/fetch_wikipedia_samples.py --only ur pnb skr \
        --paragraphs 5000 --max-requests 1500      # the three that get confused
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
import pathlib
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from urdunlp import normalize  # noqa: E402
from urdunlp.langid import LANGUAGES, _grams  # noqa: E402
from urdunlp.translit import LEXICON  # noqa: E402

# These scripts print Urdu letters, and a Windows console defaults to cp1252,
# where that raises UnicodeEncodeError. The crash came AFTER the model was
# written and after four minutes of work, from a progress line - and the test
# that fails tells you to run this very command.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

DATA = ROOT / "data"
OUT = ROOT / "src/urdunlp/data"

# Chosen by `scripts/sweep_langid.py` on the VALIDATION split, 24 combinations, and
# reported on test once. Full table in data/langid_sweep.json.
#
# N_MAX: 4 -> 5 is worth 1.3 points of validation mean; 6 and 7 are both WORSE than 5 at
# every alpha and both pruning levels (6 at top 60k: 0.9193 against 5's 0.9243). Longer
# grams are rarer, so pruning hits them harder and the tail they would need is the part
# that gets cut. 5 is now a measured optimum rather than the largest value anyone tried.
#
# TOP_GRAMS was the setting that mattered. The cut to 20k cost far more than the comment
# here admitted, and it cost it where the model was already weakest:
#
# Chosen by scripts/sweep_langid.py over data/wiki_large, on validation, by a paired test
# rather than a tolerance on the mean - see that file for why the two differ. The whole
# ladder, at n_max 6 and alpha 0.03 so only the cut varies, with the size the wheel
# carries and a McNemar test of each rung against the one below on 19,847 paragraphs:
#
#   top      val mean   val whole   val 20ch   Saraiki@20   wheel (gzip)   vs below, 20ch
#   20,000     0.9376      0.9838     0.9321       0.7274       0.94 MB     -
#   60,000     0.9478      0.9880     0.9458       0.7887       2.62 MB     p = 9e-27
#   90,000     0.9505      0.9894     0.9484       0.7969       3.82 MB     p = 0.005
#   120,000    0.9525      0.9898     0.9508       0.8041       4.98 MB     p = 0.004
#   180,000    0.9545      0.9902     0.9534       0.8144       7.25 MB     p = 0.001
#   240,000    0.9550      0.9907     0.9534       0.8139       9.44 MB     p = 1
#   300,000    0.9558      0.9909     0.9542       0.8210      11.60 MB     p = 0.185
#
# 180,000 is the last rung that separates from the one below it. Past it the wheel grows
# about 2.2 MB per rung for differences 19,847 validation paragraphs cannot resolve - at
# 240,000 the 20-character column is identical, 81 wins each way. Rebuild with
# `--top-grams 60000` for a 2.62 MB model at 0.7 points of 20-character accuracy and 2.5
# points of Saraiki.
#
# These defaults are the setting that shipped, which a test enforces: they were left at
# n_max 5 / 60,000 after the sweep moved, so a plain `python scripts/build_langid_models.py`
# would have replaced the artefact with a different and worse model while every digest
# check still passed, because the digest is of the inputs a run was given.
#
# alpha: the previous comment read "0.3 -> 0.1 is worth 0.7 points at top 60k; 0.03 is a
# wash (0.9242)", measured on the 1,500-per-language corpus. On 130,791 paragraphs 0.03
# is the best of the three at every n_max and every gram budget, though only by 0.0005 on
# the mean, which is inside noise.
#
# n_max: 5 used to be the largest value tried, so it was the top of the range rather than
# a measured optimum, and 6 was worse at 8 of 9 matched budgets - all of them at cuts of
# 90,000 or below. The reason given was that longer grams are rarer, so the cut hits them
# hardest; past 120,000 the tail survives and 6 beats 5 at every matched budget.
N_MAX = 6
ALPHA = 0.03
TOP_GRAMS = 180000

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


def wiki_splits(
    wiki_dir: pathlib.Path | None = None, max_lines: int | None = None
) -> dict[str, dict[str, list[str]]]:
    """The splits, from `data/wiki` unless another corpus directory is named.

    The path was hard-coded, so a corpus built at a different size could not be
    trained on without overwriting the one the shipped model was built from - which
    is exactly the mistake FINGERPRINT.tsv exists to catch.

    `max_lines` caps the paragraphs read per language. It exists because the
    multi-source corpus is roughly twenty times the Wikipedia one and holding eleven
    full n-gram counters at once is several gigabytes on a machine that is shared with
    other work. The cap takes **every k-th line, not the first n**: the split files are
    written one source after another, so the first n lines of a Kashmiri file are
    entirely synthetic-shuffle rows and the first n of an Urdu file are entirely CC-100.
    A head-of-file cap would therefore have trained each language on one source while
    the file it came from spanned seven, and nothing downstream would have said so.
    """
    folder = wiki_dir or (DATA / "wiki")
    out: dict[str, dict[str, list[str]]] = {}
    for code in LANGUAGES:
        path = folder / f"{code}.txt"
        if not path.exists():
            raise SystemExit(f"missing {path}: run scripts/fetch_wikipedia_samples.py")
        lines = path.read_text(encoding="utf-8").splitlines()
        if max_lines and len(lines) > max_lines:
            step = len(lines) / max_lines
            lines = [lines[int(i * step)] for i in range(max_lines)]
        parts: dict[str, list[str]] = collections.defaultdict(list)
        for paragraph in lines:
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


def inputs_digest(*parts: object) -> str:
    """A digest of everything a model is built from.

    Stored in the model and checked by tests/test_published_langid_numbers.py, because
    the shipped Roman tagger was six commits stale and nothing could tell. It was built
    on 2026-09-25; `urdunlp.translit.LEXICON`, which `roman_urdu_unigram` spreads word
    frequencies over, changed in six commits after that - chat spellings, acronyms, the
    Arabic article - and the wheel kept shipping a table that no longer matched the code
    that produces it. Rebuilding moved 74 word weights, added eight words and dropped
    eight.

    A model artefact in git is a build output committed by hand, so the only thing that
    keeps it honest is a record of what it was built from.
    """
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, Path):
            # Normalised: these are text files, and git stores them with LF while
            # a Windows checkout can hold CRLF. Hashing the raw bytes made the
            # digest depend on the machine that built the model, so a value
            # computed on Windows matched there and could never match on CI's
            # Linux runner.
            h.update(part.read_bytes().replace(b"\r\n", b"\n"))
        else:
            h.update(repr(part).encode("utf-8"))
        # A separator, so two adjacent parts cannot run together into the same
        # bytes that one longer part would produce.
        h.update(b"")
    return h.hexdigest()[:32]


def window(text: str, length: int | None) -> str | None:
    """A CENTRED slice of `length` characters, or None if the text is shorter.

    Centred, not leading: the start of a Wikipedia intro is disproportionately a title
    and a date, and taking the first N characters reads about five points lower. This
    used to live only in measure_langid.py while the build script sliced `paragraph[:20]`
    itself, so the two printed different numbers under the same name.
    """
    if length is None:
        return text
    if len(text) < length:
        return None
    start = (len(text) - length) // 2
    return text[start : start + length]


def build_script_model(
    n_max: int = N_MAX,
    alpha: float = ALPHA,
    top_grams: int = TOP_GRAMS,
    wiki_dir: pathlib.Path | None = None,
    out_dir: pathlib.Path | None = None,
    max_lines: int | None = None,
) -> None:
    """Build the script model. `out_dir` defaults to the shipped src/urdunlp/data.

    out_dir exists so a hyperparameter sweep does not overwrite the shipped model once
    per candidate. Without it, comparing top-grams settings means replacing the artefact
    the package installs on every iteration, and a sweep interrupted half way leaves the
    wheel carrying whichever candidate happened to run last.
    """
    splits = wiki_splits(wiki_dir, max_lines)
    counts = {}
    for code, parts in splits.items():
        c: collections.Counter = collections.Counter()
        for paragraph in parts["train"]:
            c.update(_grams(paragraph, n_max))
        counts[code] = c
    distinctive = distinctive_letters(counts)

    from urdunlp.langid import _ScriptModel

    def model_for(top: int | None) -> _ScriptModel:
        kept = {k: dict(c.most_common(top)) for k, c in counts.items()}
        vocabulary = set().union(*kept.values())
        return _ScriptModel(
            {
                "n_max": n_max,
                "alpha": alpha,
                "vocabulary_size": len(vocabulary),
                "counts": kept,
                "distinctive": distinctive,
            }
        )

    for top in dict.fromkeys((40000, top_grams, 6000)):
        model = model_for(top)
        right = total = 0
        for code, parts in splits.items():
            for paragraph in parts["val"]:
                # The same centred slice measure_langid.py uses. This took the FIRST 20
                # characters, which reads about five points lower - the start of a
                # Wikipedia intro is disproportionately a title and a date - so the
                # figure printed here was not comparable with the one in the README even
                # though both were called "validation accuracy on 20-char windows".
                text = window(paragraph, 20)
                if text is None:
                    continue
                scores = model.log_likelihoods(text)
                right += max(scores, key=scores.get) == code
                total += 1
        print(
            f"  script model, top {top or 'all'} grams: validation accuracy on 20-char "
            f"windows {right / total:.4f}"
        )

    model = model_for(top_grams)
    blob = {
        "n_max": n_max,
        "alpha": alpha,
        # Recorded, not only hashed. top_grams is in the inputs digest, so a model built
        # at a different budget is detectably different - but the blob did not say which
        # budget it was, and vocabulary_size is a proxy a reader has to decode. `corpus`
        # was added for this reason after a sweep over data/wiki and one over
        # data/wiki_large produced indistinguishable files; the gram budget is the other
        # thing a sweep varies.
        "top_grams": top_grams,
        "vocabulary_size": model.vocabulary_size,
        "counts": model.counts,
        "distinctive": distinctive,
        "inputs": inputs_digest(
            DATA / "FINGERPRINT.tsv", n_max, alpha, top_grams, (wiki_dir or (DATA / "wiki")).name
        ),
        "corpus": (wiki_dir or (DATA / "wiki")).name,
        "source": "Wikipedia article intros, one random sample per language (CC BY-SA 4.0)",
    }
    target = (out_dir or OUT) / "langid.json.gz"
    target.parent.mkdir(parents=True, exist_ok=True)
    write(target, blob)
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
        "inputs": inputs_digest(
            lexicon,
            DATA / "vocab/dakshina_train_counts.tsv",
            sorted(LEXICON.items()),
            TOP_WORDS,
            ORDER,
            UNIGRAM_WEIGHT,
            STAY,
            URDU_BIAS,
        ),
        "source": "Roman Urdu: Dakshina v1.0 (CC BY-SA 4.0). English: HotpotQA (CC BY-SA 4.0).",
    }
    write(OUT / "roman_tagger.json.gz", blob)


def write(path: Path, blob: dict) -> None:
    raw = json.dumps(blob, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(raw.encode("utf-8"), compresslevel=9, mtime=0))
    # relative_to raises for any path outside the repo, and --out-dir exists precisely so
    # a sweep can write elsewhere. Reported as written and then crashed on the line
    # announcing it, after the bytes were safely on disk - the same shape as the manifest
    # writer that exited 1 on its own success message.
    try:
        shown = path.relative_to(ROOT)
    except ValueError:
        shown = path
    print(f"wrote {shown} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    import argparse

    # Settable, because the README and docs/CORPUS.md both say the cut is a package-size
    # decision the reader can take differently - and a documented knob that is actually
    # a module constant is a claim the code does not honour.
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-max", type=int, default=N_MAX, help=f"longest n-gram ({N_MAX})")
    ap.add_argument("--alpha", type=float, default=ALPHA, help=f"smoothing ({ALPHA})")
    ap.add_argument(
        "--top-grams",
        type=int,
        default=TOP_GRAMS,
        help=f"n-grams kept per language ({TOP_GRAMS:,}), the last rung validation can "
        "separate from the one below. 60000 gives a 2.62 MB model instead of 7.25 MB, at "
        "0.7 points of 20-character accuracy; see the ladder in docs/CORPUS.md",
    )
    ap.add_argument(
        "--max-lines",
        type=int,
        help="cap paragraphs read per language, sampled evenly across the file so "
        "every source in it is represented (default: no cap)",
    )
    ap.add_argument("--skip-tagger", action="store_true", help="only the script model")
    ap.add_argument(
        "--out-dir",
        type=pathlib.Path,
        help="where to write the model (default src/urdunlp/data). Lets a sweep compare "
        "candidates without replacing the artefact the package ships",
    )
    ap.add_argument(
        "--wiki-dir",
        type=pathlib.Path,
        help="corpus directory to train on (default data/wiki). Lets a larger "
        "corpus be trained and scored without overwriting the one the shipped "
        "model and FINGERPRINT.tsv were built from",
    )
    a = ap.parse_args()
    build_script_model(
        n_max=a.n_max,
        alpha=a.alpha,
        top_grams=a.top_grams,
        wiki_dir=a.wiki_dir,
        out_dir=a.out_dir,
        max_lines=a.max_lines,
    )
    if not a.skip_tagger:
        build_roman_tagger()
