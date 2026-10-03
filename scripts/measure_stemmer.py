"""Does stemming help Urdu retrieval? Measured, on a task with exact ground truth.

    python scripts/measure_stemmer.py <urdu_articles.parquet> [--row-groups N] [--json out.json]

The published figures (docs/CORPUS.md section 12) are the first 6,000 articles of
Hugging Face's `wikimedia/wikipedia`, config `20231101.ur` - the first six row groups
of its one public parquet file (168 MB):

    https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.ur/train-00000-of-00001.parquet
    python scripts/measure_stemmer.py train-00000-of-00001.parquet --row-groups 6

Of those 6,000, the 5,016 with at least 60 tokens of body are kept. The same six row
groups were first cut out for nlp-lab project 23's benchmark; that file, passed without
`--row-groups`, gives the same numbers.

The task is title-to-body retrieval over Urdu Wikipedia, borrowed from nlp-lab
project 23 so the numbers are comparable with its plain-word baseline: each article's
title is the query, its body is the single relevant document, and the title is
removed from the body first so that nothing wins by matching a copy of itself.

Queries are split in two by a hash of the title. Settings are chosen on the
validation half and reported on the test half, with both shown - a stemmer tuned and
scored on the same queries would be graded on its own homework.

BM25 is implemented here (k1=1.5, b=0.75) so the script needs nothing but pyarrow
to read the parquet.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from urdunlp import words  # noqa: E402
from urdunlp.stem import stem  # noqa: E402


class BM25:
    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.lengths = [len(d) for d in documents]
        self.average = sum(self.lengths) / len(documents)
        self.postings: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
        for index, document in enumerate(documents):
            for term, count in collections.Counter(document).items():
                self.postings[term].append((index, count))
        n = len(documents)
        self.idf = {
            t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()
        }

    def top(self, query: list[str], k: int = 10) -> list[int]:
        scores: dict[int, float] = collections.defaultdict(float)
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, tf in self.postings[term]:
                norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average)
                scores[index] += idf * tf * (self.k1 + 1) / (tf + norm)
        return sorted(scores, key=scores.__getitem__, reverse=True)[:k]


def load(
    path: Path, min_tokens: int = 60, row_groups: int | None = None
) -> tuple[list[str], list[str]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if row_groups is None:
        table = pq.read_table(path, columns=["title", "text"])
    else:
        reader = pq.ParquetFile(path)
        groups = range(min(row_groups, reader.num_row_groups))
        table = pa.concat_tables(
            reader.read_row_group(i, columns=["title", "text"]) for i in groups
        )
    titles, bodies = [], []
    titles_raw, texts_raw = table.column("title").to_pylist(), table.column("text").to_pylist()
    for title, text in zip(titles_raw, texts_raw, strict=True):
        title = str(title).strip()
        body = str(text).replace(title, " ") if title else str(text)
        if title and len(words(body)) >= min_tokens:
            titles.append(title)
            bodies.append(body)
    return titles, bodies


# stem()'s defaults: light=False, min_stem=3.
DEFAULT = "full min3"


def split_of(title: str) -> str:
    return "val" if hashlib.sha256(title.encode()).digest()[0] % 2 == 0 else "test"


def lead_task(bodies: list[str]) -> tuple[list[str], list[str]]:
    """Queries that inflect: each article's first sentence, removed from its body.

    Titles are mostly names - people, places, organisations - and names do not
    inflect, so title retrieval barely exercises a stemmer at all. A lead sentence is
    ordinary prose, full of the plural, oblique and verbal endings stemming exists to
    conflate. It is cut from the body, so the query never matches itself.
    """
    queries, rest = [], []
    for body in bodies:
        cut = min((i for i in (body.find("۔"), body.find(".")) if i > 20), default=-1)
        if cut < 0:
            queries.append("")
            rest.append(body)
            continue
        queries.append(body[: cut + 1])
        rest.append(body[cut + 1 :])
    return queries, rest


def sign_test(helped: int, hurt: int) -> float:
    """Two-sided exact sign test over the queries a setting changed."""
    n = helped + hurt
    if n == 0:
        return 1.0
    k = min(helped, hurt)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("corpus", type=Path)
    ap.add_argument(
        "--row-groups", type=int, help="read only the first N row groups (6 = published run)"
    )
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    started = time.time()

    titles, bodies = load(args.corpus, row_groups=args.row_groups)
    splits = [split_of(t) for t in titles]
    leads, rests = lead_task(bodies)
    tasks = {
        "title -> body": ([words(t) for t in titles], [words(b) for b in bodies]),
        "lead sentence -> rest of article": ([words(q) for q in leads], [words(b) for b in rests]),
    }
    print(
        f"{len(bodies):,} articles; {splits.count('val'):,} validation and "
        f"{splits.count('test'):,} test queries"
    )

    settings = [("words", None)]
    for light in (True, False):
        for min_stem in (2, 3):
            settings.append((f"{'light' if light else 'full'} min{min_stem}", (light, min_stem)))

    report = {}
    for task, (queries, tokenised) in tasks.items():
        print(f"\n{task}")
        rows, found = [], {}
        for name, config in settings:
            if config is None:
                docs, qs = tokenised, queries
            else:
                light, min_stem = config
                cache: dict[str, str] = {}

                def s(t: str, a=light, m=min_stem, c=cache) -> str:
                    if t not in c:
                        c[t] = stem(t, light=a, min_stem=m)
                    return c[t]

                docs = [[s(t) for t in d] for d in tokenised]
                qs = [[s(t) for t in q] for q in queries]
            index = BM25(docs)
            hit = [bool(q) and i in index.top(q) for i, q in enumerate(qs)]
            found[name] = hit
            valid = [i for i, q in enumerate(queries) if q]
            row = {"setting": name, "types": len(index.postings)}
            for part in ("val", "test"):
                members = [i for i in valid if splits[i] == part]
                row[f"{part}_recall@10"] = round(sum(hit[i] for i in members) / len(members), 4)
            if config is not None:
                test = [i for i in valid if splits[i] == "test"]
                helped = sum(hit[i] and not found["words"][i] for i in test)
                hurt = sum(found["words"][i] and not hit[i] for i in test)
                row.update(test_helped=helped, test_hurt=hurt, sign_test_p=sign_test(helped, hurt))
            rows.append(row)
            extra = (
                (
                    f"   helped {row['test_helped']:>3} hurt {row['test_hurt']:>3} "
                    f"p={row['sign_test_p']:.3g}"
                )
                if config
                else ""
            )
            print(
                f"  {name:<18} types {row['types']:>8,}   val {row['val_recall@10']:.4f}   "
                f"test {row['test_recall@10']:.4f}{extra}",
                flush=True,
            )
        best = max(rows[1:], key=lambda r: r["val_recall@10"])
        base = rows[0]
        print(
            f"  chosen on validation: {best['setting']};  test {base['test_recall@10']:.4f} -> "
            f"{best['test_recall@10']:.4f} ({best['test_recall@10'] - base['test_recall@10']:+.4f},"
            f" sign test p={best['sign_test_p']:.3g})"
        )
        # What a user of `stem()` gets. Validation can tie to four places (it does on
        # title -> body: light min3 and full min3 both 0.4454), and max() then takes
        # the first in list order, so the chosen line alone can name a setting the
        # package does not ship. The README reports this line.
        default = next(r for r in rows if r["setting"] == DEFAULT)
        print(
            f"  shipped default ({DEFAULT}): test {base['test_recall@10']:.4f} -> "
            f"{default['test_recall@10']:.4f} "
            f"({default['test_recall@10'] - base['test_recall@10']:+.4f},"
            f" sign test p={default['sign_test_p']:.3g})"
        )
        report[task] = {"rows": rows, "chosen": best["setting"], "default": DEFAULT}

    print(f"[{time.time() - started:.0f}s]")
    if args.json:
        args.json.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
