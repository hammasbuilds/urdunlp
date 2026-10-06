"""Fetch random Wikipedia article intros for every language written in Urdu's script.

    python scripts/fetch_wikipedia_samples.py [--out data/wiki] [--paragraphs 1500]

`is_urdu` used to answer one question - "is this Perso-Arabic script?" - and call the
answer Urdu. Eleven Wikipedias are written in that script, several of them by people
in Pakistan. `identify_language` has to be measured against all of them, and from the
same source, so that a difference between languages is not a difference between a
news register and an encyclopaedia register.

Output is one paragraph per line in `<out>/<code>.txt`. Paragraphs are de-duplicated
within each language: small Wikipedias are full of bot-written stubs ("X is a village
in Y district") and without de-duplication one template can be most of a sample.

Standard library only, and polite: one request at a time with a pause between them.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

LANGUAGES = {
    "ur": "Urdu",
    "pnb": "Punjabi (Shahmukhi)",
    "skr": "Saraiki",
    "sd": "Sindhi",
    "ps": "Pashto",
    "ks": "Kashmiri",
    "fa": "Persian",
    "ar": "Arabic",
    "ckb": "Central Kurdish",
    "ug": "Uyghur",
    "azb": "South Azerbaijani",
}

USER_AGENT = "urdunlp-measurement/0.2 (https://github.com/hammasbuilds/urdunlp)"


def random_intros(code: str) -> list[tuple[str, int, str]]:
    """Paragraphs with the article each came from.

    Returns (paragraph, pageid, title). The ids are what make a corpus rebuildable: the
    draw is the server's `generator=random`, so no seed on this side can reproduce it, and
    without the ids a corpus can only be described, never re-fetched.
    """
    query = urllib.parse.urlencode(
        {
            "action": "query",
            "generator": "random",
            "grnnamespace": 0,
            "grnlimit": 20,
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
            "exlimit": 20,
            "format": "json",
        }
    )
    request = urllib.request.Request(
        f"https://{code}.wikipedia.org/w/api.php?{query}", headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        pages = json.load(response).get("query", {}).get("pages", {}).values()
    paragraphs = []
    for page in pages:
        pageid = int(page.get("pageid") or 0)
        title = str(page.get("title") or "")
        for paragraph in page.get("extract", "").split("\n"):
            paragraph = " ".join(paragraph.split())
            if len(paragraph) >= 40:
                paragraphs.append((paragraph, pageid, title))
    return paragraphs


def pages_by_id(code: str, ids: list[int]) -> list[tuple[str, int, str]]:
    """The same extracts, fetched by page id instead of at random.

    This is what makes a recorded corpus reproducible: `--from-pages` reads the committed
    ids and asks for exactly those articles.
    """
    out: list[tuple[str, int, str]] = []
    for start in range(0, len(ids), 20):
        batch = ids[start : start + 20]
        query = urllib.parse.urlencode(
            {
                "action": "query",
                "pageids": "|".join(str(i) for i in batch),
                "prop": "extracts",
                "exintro": 1,
                "explaintext": 1,
                "exlimit": 20,
                "format": "json",
            }
        )
        request = urllib.request.Request(
            f"https://{code}.wikipedia.org/w/api.php?{query}",
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            pages = json.load(response).get("query", {}).get("pages", {}).values()
        for page in pages:
            pageid = int(page.get("pageid") or 0)
            title = str(page.get("title") or "")
            for paragraph in page.get("extract", "").split("\n"):
                paragraph = " ".join(paragraph.split())
                if len(paragraph) >= 40:
                    out.append((paragraph, pageid, title))
        time.sleep(0.3)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("data/wiki"))
    ap.add_argument("--paragraphs", type=int, default=1500)
    # Characters, not paragraphs, because the n-gram counts are taken over characters and
    # Wikipedia paragraph lengths differ by language: equal paragraph counts gave Saraiki
    # 2.24x more training text than Urdu, and a 7.74x spread across all eleven.
    ap.add_argument(
        "--chars",
        type=int,
        help="stop when the language's file reaches this many characters "
        "(use instead of --paragraphs to balance a corpus in the unit the model reads)",
    )
    ap.add_argument("--max-requests", type=int, default=400)
    ap.add_argument("--only", nargs="*", help="language codes to fetch (default: all)")
    ap.add_argument(
        "--from-pages",
        action="store_true",
        help="rebuild each language from the article ids recorded in "
        "data/wiki/<code>.pages.tsv instead of drawing new random articles - this is "
        "what makes a corpus reproducible, since the draw is the server's and no seed "
        "here can control it",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for code, name in LANGUAGES.items():
        if args.only and code not in args.only:
            continue
        target = args.out / f"{code}.txt"
        pages_file = args.out / f"{code}.pages.tsv"
        # paragraph -> the page id it came from, 0 when it was loaded from an existing
        # file written before ids were recorded.
        seen: dict[str, int] = {}
        if target.exists():
            seen = dict.fromkeys(target.read_text(encoding="utf-8").splitlines(), 0)
        # Article id -> title, for every article a paragraph came from. Accumulated across
        # runs exactly as the paragraphs are, and committed, so the corpus can be rebuilt.
        articles: dict[int, str] = {}
        recorded_count = 0
        if pages_file.exists():
            for line in pages_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("# paragraphs:"):
                    recorded_count = int(line.split(":", 1)[1].strip() or 0)
                elif line and not line.startswith("#"):
                    parts = line.split("\t")
                    if parts[0].isdigit():
                        articles[int(parts[0])] = parts[1] if len(parts) > 1 else ""

        if args.from_pages:
            if not articles:
                print(f"{code:<4} no {pages_file.name}; nothing to rebuild from", flush=True)
                continue
            # Deterministic: articles in id order, paragraphs in the order each article
            # returns them, truncated to the count the manifest records. Two rebuilds of
            # the same ids therefore produce the same file byte for byte.
            rebuilt: dict[str, None] = {}
            for paragraph, pageid, title in pages_by_id(code, sorted(articles)):
                rebuilt.setdefault(paragraph, None)
                articles[pageid] = title or articles.get(pageid, "")
            want = recorded_count if recorded_count else len(rebuilt)
            lines = list(rebuilt)[:want]
            target.write_text("\n".join(lines) + "\n", encoding="utf-8")
            characters = sum(len(line) for line in lines)
            short = "" if len(rebuilt) >= want else f"  SHORT by {want - len(rebuilt)}"
            print(
                f"{code:<4} {name:<22} rebuilt from {len(articles):>5} article(s): "
                f"{len(lines):>5} paragraphs  {characters:>9,} chars{short}",
                flush=True,
            )
            continue

        requests = stale = 0

        # Inline rather than a closure over `seen`: the budget is re-read every turn of
        # the loop, and a nested function capturing the loop variable is the kind of thing
        # that keeps working until someone moves it.
        while requests < args.max_requests and stale < 15:
            if args.chars is not None:
                if sum(len(line) for line in seen) >= args.chars:
                    break
            elif len(seen) >= args.paragraphs:
                break
            requests += 1
            try:
                batch = random_intros(code)
            except Exception as error:  # noqa: BLE001 - a flaky request is retried
                print(f"  {code}: {error}")
                time.sleep(5)
                continue
            before = len(seen)
            for paragraph, pageid, title in batch:
                seen.setdefault(paragraph, pageid)
                if pageid:
                    articles[pageid] = title
            stale = stale + 1 if len(seen) == before else 0
            time.sleep(0.3)
        if args.chars is not None:
            lines = []
            used = 0
            for line in seen:
                if used >= args.chars:
                    break
                lines.append(line)
                used += len(line)
        else:
            lines = list(seen)[: args.paragraphs]
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        # Only the articles that contributed a paragraph the file actually KEPT. Recording
        # every article the fetcher saw meant the budget could cut an article's paragraphs
        # while its id stayed, and a rebuild from 60 ids then produced 80 paragraphs
        # against the 68 written.
        kept_ids = {seen[line] for line in lines if seen.get(line)}
        contributing = {i: articles[i] for i in sorted(kept_ids) if i in articles}
        pages_file.write_text(
            "# Articles every paragraph in "
            + target.name
            + " came from. Rebuild with --from-pages.\n"
            + "# The draw is the server's generator=random, so these ids are the only way\n"
            + "# to reproduce a corpus; a seed on this side cannot.\n"
            + f"# paragraphs: {len(lines)}\n"
            + "pageid\ttitle\n"
            + "\n".join(f"{i}\t{contributing[i]}" for i in contributing)
            + "\n",
            encoding="utf-8",
        )
        characters = sum(len(line) for line in lines)
        # `stale` hitting its limit means this Wikipedia has no more distinct articles to
        # offer, which is the normal outcome for the smaller ones and is not an error - but
        # it is the difference between "budget met" and "this is all there is".
        exhausted = " EXHAUSTED - no more distinct articles" if stale >= 15 else ""
        print(
            f"{code:<4} {name:<22} {len(lines):>5} paragraphs  {characters:>9,} chars  "
            f"{len(articles):>5} articles  ({requests} requests){exhausted}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
