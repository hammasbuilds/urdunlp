"""Extract a large corpus from full Wikimedia dumps, for the languages OPUS has nothing for.

Across eighteen OPUS corpora, Tatoeba and CC-100's 120 languages there is no
non-Wikipedia Perso-Arabic text for Shahmukhi Punjabi, Saraiki, Kashmiri or South
Azerbaijani. That claim stands. What did **not** stand is the implication drawn from it:
the earlier corpus used 12,000 paragraphs per language, and the dumps are far larger.

    pnb   75,798 articles   111.1 MB   + 0.7 MB Wiktionary
    skr   25,164 articles    25.1 MB   + 16.2 MB Wiktionary
    ks    11,908 articles    10.7 MB   + 0.6 MB Wiktionary
    azb  244,931 articles    94.3 MB

So the size limit was the sampling, not the language's resources. Wiktionary is also a
different register from encyclopaedia prose - dictionary entries, inflection tables,
usage lines - so for Saraiki, whose Wiktionary is two thirds the size of its Wikipedia, it
is a second Wikimedia-family source rather than more of the same.

This does not solve domain diversity. Wikipedia and Wiktionary are both Wikimedia, both
written by volunteers, often by overlapping ones. A paper must say that plainly: these
languages get **size** here, and the domain-robustness question stays open for them.

    .venv-baselines/Scripts/python research/build_wikimedia.py --languages pnb,skr,ks,azb
"""

from __future__ import annotations

import argparse
import bz2
import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / ".baselines" / "wikimedia"
OUT = ROOT / "data" / "wikimedia"
REPORT = ROOT / "research" / "results" / "wikimedia.json"
UA = {"User-Agent": "urdunlp-research/0.1 (language-ID corpus study; "
                    "github.com/hammasbuilds/urdunlp)"}

PROJECTS = ("wiki", "wiktionary", "wikisource", "wikibooks")
# Wikitext constructs to remove before paragraphs are taken. Order matters: templates and
# tables nest, so they go first.
STRIP = [
    (re.compile(r"\{\{[^{}]*\}\}"), " "),          # templates, innermost first
    (re.compile(r"\{\|.*?\|\}", re.S), " "),       # tables
    (re.compile(r"<ref[^>]*>.*?</ref>", re.S), " "),
    (re.compile(r"<ref[^>]*/>"), " "),
    (re.compile(r"<[^>]+>"), " "),                 # any remaining html
    (re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]"), r"\1"),  # links, keep the label
    (re.compile(r"\[https?://\S+\s*([^\]]*)\]"), r"\1"),
    (re.compile(r"https?://\S+"), " "),
    (re.compile(r"^[*:#;]+", re.M), " "),          # list and indent markers
    (re.compile(r"'{2,}"), ""),                     # bold and italic
    (re.compile(r"^=+.*?=+$", re.M), " "),         # headings
    (re.compile(r"&[a-z]+;|&#\d+;"), " "),
]
TEXT = re.compile(r"<text\b[^>]*>(.*?)</text>", re.S)
MIN_CHARS = 40
ARABIC = re.compile(r"[؀-ۿݐ-ݿ]")


def download(code: str, project: str) -> Path | None:
    name = f"{code}{project}"
    path = RAW / f"{name}.xml.bz2"
    if path.exists() and path.stat().st_size > 10_000:
        return path
    url = f"https://dumps.wikimedia.org/{name}/latest/{name}-latest-pages-articles.xml.bz2"
    RAW.mkdir(parents=True, exist_ok=True)
    try:
        request = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(request, timeout=600) as response:
            if int(response.headers.get("Content-Length") or 0) < 10_000:
                return None
            with path.open("wb") as fh:
                while chunk := response.read(1 << 20):
                    fh.write(chunk)
    except Exception as exc:  # noqa: BLE001 - absence is the common case, not an error
        print(f"      {name}: {type(exc).__name__}")
        return None
    return path


def paragraphs(path: Path, cap: int) -> list[str]:
    """Plain-text paragraphs from a dump, markup stripped.

    Streamed and truncated: `pnbwiki` is 111 MB compressed and the first `cap` usable
    paragraphs are enough. Lines must be majority Perso-Arabic to be kept - a dump
    contains English templates, transliteration tables and bot-written stubs, and those
    would otherwise enter a corpus as if they were the language.
    """
    out: list[str] = []
    seen: set[str] = set()
    buffer = ""
    with bz2.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            buffer += line
            if len(buffer) < 1 << 16 and "</text>" not in buffer:
                continue
            for body in TEXT.findall(buffer):
                for pattern, replacement in STRIP:
                    previous = None
                    # Templates nest; repeat until the innermost pass changes nothing.
                    while previous != body and pattern.pattern.startswith("\\{\\{"):
                        previous = body
                        body = pattern.sub(replacement, body)
                    body = pattern.sub(replacement, body)
                for chunk in body.split("\n"):
                    text = " ".join(chunk.split())
                    if len(text) < MIN_CHARS or text in seen:
                        continue
                    letters = sum(1 for c in text if c.isalpha())
                    if not letters or len(ARABIC.findall(text)) / letters < 0.6:
                        continue
                    # Reject residue rather than strip harder. A first pass left 521 of
                    # 20,000 Kashmiri lines still carrying `{{`, `[[`, a tag or a URL -
                    # 2.6% that would teach the model wikitext instead of the language.
                    # Rejecting is safe; a cleverer stripper would keep mangling text.
                    if any(bad in text for bad in ("{{", "}}", "[[", "]]", "<", ">",
                                                   "http", "|", "=")):
                        continue
                    seen.add(text)
                    out.append(text)
                    if len(out) >= cap:
                        return out
            buffer = buffer[buffer.rfind("</text>") + 7:] if "</text>" in buffer else ""
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--languages", default="pnb,skr,ks,azb")
    ap.add_argument("--cap", type=int, default=60000,
                    help="paragraphs per project per language (60000)")
    args = ap.parse_args()

    report: dict = {"cap_per_project": args.cap, "min_chars": MIN_CHARS,
                    "min_arabic_share": 0.6, "languages": {}}
    OUT.mkdir(parents=True, exist_ok=True)
    for code in [c.strip() for c in args.languages.split(",") if c.strip()]:
        print(f"\n{code}:")
        per_project: dict[str, list[str]] = {}
        for project in PROJECTS:
            path = download(code, project)
            if path is None:
                continue
            texts = paragraphs(path, args.cap)
            if len(texts) < 50:
                print(f"    {project:12} {len(texts):>7,}  too few, dropped")
                continue
            per_project[project] = texts
            print(f"    {project:12} {len(texts):>7,}")
        total = sum(len(v) for v in per_project.values())
        report["languages"][code] = {
            "projects": {k: len(v) for k, v in per_project.items()},
            "total": total,
        }
        print(f"    {'total':12} {total:>7,}")
        for project, texts in per_project.items():
            path = OUT / f"{code}.{project}.txt"
            path.write_text("\n".join(texts) + "\n", encoding="utf-8")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("\nsummary:")
    for code, row in report["languages"].items():
        print(f"  {code:5} {row['total']:>7,} from {len(row['projects'])} project(s): "
              f"{', '.join(row['projects'])}")
    print(f"\nwritten to {OUT} and {REPORT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
