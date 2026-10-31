"""Build a large corpus for all eleven languages, from every real source available.

Target: **300,000 lines per language**, over the 100,000 floor, from four source families.
Each is streamed and truncated rather than downloaded whole, because CC-100's Urdu file is
884 MB and the first slice of it is as representative as all of it for n-gram language
identification.

    CC-100          data.statmt.org, .txt.xz, web crawl        ur sd ps ug fa ar
    OPUS            object.pouta.csc.fi, .txt.gz, 18 corpora   everything it has
    Wikimedia       dumps.wikimedia.org, .xml.bz2, 4 projects  all eleven
    Tatoeba         downloads.tatoeba.org, .tsv.bz2            9 of eleven

Two languages cannot reach 300,000 from distinct real text at all: Saraiki's Wikipedia is
25,164 articles and Kashmiri's is 11,908. For those, and only those, the shortfall is made
up by **sentence segmentation** of the real paragraphs already collected - a 174-character
paragraph yields two to four sentences, each a real span of the language written by a
person. That is more training examples from the same text, not new text, and it matches
what the classifier is asked to do, since accuracy at 10, 20 and 50 characters is already
reported. Segmented lines are written to a separate file and tagged `segmented:<source>`
so no figure can mix them with unsegmented rows without saying so.

Nothing here is generated. Where real text runs out, the count is reported short rather
than filled.

    .venv-baselines/Scripts/python research/build_corpus.py --target 300000
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import json
import lzma
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / ".baselines"
OUT = ROOT / "data" / "corpus"
REPORT = ROOT / "research" / "results" / "corpus.json"
UA = {"User-Agent": "urdunlp-research/0.1 (language-ID corpus study; "
                    "github.com/hammasbuilds/urdunlp)"}

LANGUAGES = ["ur", "pnb", "skr", "sd", "ps", "ks", "fa", "ar", "ckb", "ug", "azb"]
CC100 = {"ur": "ur", "sd": "sd", "ps": "ps", "ug": "ug", "fa": "fa", "ar": "ar"}

# Languages whose OPUS entries are a different language in a different script, recorded
# here so the build does not rediscover it by streaming gigabytes. `opus_sources.json` is
# keyed by our code, and the sweep that built it matched `azb` to OPUS's `az` - but `az`
# is Latin-script Azerbaijani (North Azerbaijani), a different language from the
# Perso-Arabic South Azerbaijani this corpus labels `azb`. Every `az` line fails the
# script filter, correctly, so the only effect of trying is a very long download.
#
# `WRONG_SCRIPT_AFTER` would catch each of these anyway; this skips them outright, and
# states the reason where a reader will find it.
NO_OPUS_SOURCE = {
    "azb": "OPUS `az` is Latin-script North Azerbaijani, a different language",
}
WIKI_PROJECTS = ("wiki", "wiktionary", "wikisource", "wikibooks")
MIN_CHARS = 40
MIN_ARABIC = 0.6
ARABIC = re.compile(r"[؀-ۿݐ-ݿ]")
MARKUP = ("{{", "}}", "[[", "]]", "<", ">", "http", "|", "=")
# Sentence enders used across these orthographies: the Urdu full stop, the Arabic
# question mark, and the Latin ones that appear in mixed text.
SENTENCE = re.compile(r"(?<=[۔؟.!?])\s+")

TEXT = re.compile(r"<text\b[^>]*>(.*?)</text>", re.S)
STRIP = [
    (re.compile(r"\{\{[^{}]*\}\}"), " "),
    (re.compile(r"\{\|.*?\|\}", re.S), " "),
    (re.compile(r"<ref[^>]*>.*?</ref>", re.S), " "),
    (re.compile(r"<[^>]+>"), " "),
    (re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]"), r"\1"),
    (re.compile(r"\[https?://\S+\s*([^\]]*)\]"), r"\1"),
    (re.compile(r"'{2,}"), ""),
    (re.compile(r"^=+.*?=+$", re.M), " "),
    (re.compile(r"^[*:#;]+", re.M), " "),
]


def usable(text: str) -> bool:
    if len(text) < MIN_CHARS or any(bad in text for bad in MARKUP):
        return False
    letters = sum(1 for c in text if c.isalpha())
    return bool(letters) and len(ARABIC.findall(text)) / letters >= MIN_ARABIC


def open_remote(url: str):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600)


# How many lines to read from a stream before giving up on it if none were usable.
# An OPUS language code is not a guarantee of script: `az` is Latin-script Azerbaijani,
# not the Perso-Arabic South Azerbaijani this corpus wants, and `ks` on some corpora is
# Devanagari Kashmiri rather than the Perso-Arabic orthography. `out_of_domain_two.py`
# already rejects such sources on script share; this file did not, and the omission cost
# a run: with no usable line ever found, the cap is never reached, and the build sat for
# forty minutes streaming NLLB `az` - 1.1 billion lines - discarding every one of them.
#
# A real source yields usable lines almost immediately. 20,000 read with nothing kept
# means the wrong script or the wrong language, so stop and say which.
WRONG_SCRIPT_AFTER = 20_000


def from_stream(url: str, opener, cap: int, seen: set[str]) -> list[str]:
    """Lines from a compressed remote stream, stopping at `cap`.

    Also stops when the stream is plainly the wrong script - see `WRONG_SCRIPT_AFTER`.
    """
    out: list[str] = []
    read = 0
    try:
        with open_remote(url) as response, opener(response) as fh:
            for raw in fh:
                read += 1
                text = " ".join(raw.decode("utf-8", "replace").split())
                if usable(text) and text not in seen:
                    seen.add(text)
                    out.append(text)
                    if len(out) >= cap:
                        break
                elif not out and read >= WRONG_SCRIPT_AFTER:
                    print(f"        rejected: {read:,} lines read, none in this script")
                    break
    except Exception as exc:  # noqa: BLE001 - absence is routine, not an error
        if out:
            print(f"        partial ({type(exc).__name__}) after {len(out):,}")
    return out


def from_wikimedia(code: str, project: str, cap: int, seen: set[str]) -> list[str]:
    name = f"{code}{project}"
    path = RAW / "wikimedia" / f"{name}.xml.bz2"
    url = (f"https://dumps.wikimedia.org/{name}/latest/"
           f"{name}-latest-pages-articles.xml.bz2")
    if not path.exists() or path.stat().st_size < 10_000:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open_remote(url) as response:
                if int(response.headers.get("Content-Length") or 0) < 10_000:
                    return []
                with path.open("wb") as fh:
                    while chunk := response.read(1 << 20):
                        fh.write(chunk)
        except Exception:  # noqa: BLE001
            return []
    out: list[str] = []
    buffer = ""
    try:
        with bz2.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                buffer += line
                if len(buffer) < (1 << 16) and "</text>" not in buffer:
                    continue
                for body in TEXT.findall(buffer):
                    for pattern, replacement in STRIP:
                        for _ in range(3):  # templates nest
                            body = pattern.sub(replacement, body)
                    for chunk in body.split("\n"):
                        text = " ".join(chunk.split())
                        if usable(text) and text not in seen:
                            seen.add(text)
                            out.append(text)
                            if len(out) >= cap:
                                return out
                buffer = buffer[buffer.rfind("</text>") + 7:] if "</text>" in buffer else ""
    except OSError:
        pass
    return out


def segment(texts: list[str], want: int, seen: set[str]) -> list[str]:
    """Sentences from real paragraphs, for the languages real text cannot fill.

    Not generated text: each span was written by a person and appears in the source. It
    multiplies examples and shortens them, which is what the classifier is scored on
    anyway. Kept in its own file and tagged, so a figure cannot silently mix these with
    unsegmented rows.
    """
    out: list[str] = []
    for paragraph in texts:
        for piece in SENTENCE.split(paragraph):
            text = " ".join(piece.split())
            if usable(text) and text not in seen:
                seen.add(text)
                out.append(text)
                if len(out) >= want:
                    return out
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=300_000)
    ap.add_argument("--languages", default=",".join(LANGUAGES))
    ap.add_argument("--per-source", type=int, default=120_000)
    args = ap.parse_args()

    opus = json.loads((RAW / "opus_sources.json").read_text(encoding="utf-8"))
    report: dict = {"target": args.target, "per_source_cap": args.per_source,
                    "min_chars": MIN_CHARS, "languages": {}}
    OUT.mkdir(parents=True, exist_ok=True)

    for code in [c.strip() for c in args.languages.split(",") if c.strip()]:
        print(f"\n{code}:")
        seen: set[str] = set()
        by_source: dict[str, list[str]] = {}

        if code in CC100:
            url = f"https://data.statmt.org/cc-100/{CC100[code]}.txt.xz"
            got = from_stream(url, lzma.open, args.per_source, seen)
            if got:
                by_source[f"cc100:{CC100[code]}"] = got
                print(f"    {'cc100':26}{len(got):>8,}")

        if code in NO_OPUS_SOURCE:
            print(f"    {'opus':26}skipped - {NO_OPUS_SOURCE[code]}")
        for key in sorted(opus.get(code, {}) if code not in NO_OPUS_SOURCE else {},
                          key=lambda k: -opus[code][k]):
            if sum(len(v) for v in by_source.values()) >= args.target:
                break
            corpus, lang = key.split(":", 1)
            url = f"https://object.pouta.csc.fi/OPUS-{corpus}/v1/mono/{lang}.txt.gz"
            got = from_stream(url, gzip.open, args.per_source, seen)
            if len(got) >= 50:
                by_source[f"opus-{key}"] = got
                print(f"    {'opus ' + key:26}{len(got):>8,}")

        for project in WIKI_PROJECTS:
            if sum(len(v) for v in by_source.values()) >= args.target:
                break
            got = from_wikimedia(code, project, args.per_source, seen)
            if len(got) >= 50:
                by_source[f"wikimedia:{project}"] = got
                print(f"    {'wikimedia ' + project:26}{len(got):>8,}")

        real = sum(len(v) for v in by_source.values())
        segmented: list[str] = []
        if real < args.target:
            pool = [t for v in by_source.values() for t in v]
            segmented = segment(pool, args.target - real, seen)
            if segmented:
                print(f"    {'segmented (real spans)':26}{len(segmented):>8,}")

        for name, texts in by_source.items():
            path = OUT / f"{code}.{name.replace(':', '-')}.txt"
            path.write_text("\n".join(texts) + "\n", encoding="utf-8")
        if segmented:
            (OUT / f"{code}.segmented.txt").write_text(
                "\n".join(segmented) + "\n", encoding="utf-8")

        report["languages"][code] = {
            "real": real,
            "real_by_source": {k: len(v) for k, v in by_source.items()},
            "segmented": len(segmented),
            "total": real + len(segmented),
            "meets_100k": real + len(segmented) >= 100_000,
            "real_meets_100k": real >= 100_000,
        }
        row = report["languages"][code]
        print(f"    {'TOTAL':26}{row['total']:>8,}  "
              f"({row['real']:,} real + {row['segmented']:,} segmented)"
              f"{'' if row['meets_100k'] else '   UNDER 100k'}")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("\nsummary:")
    for code, row in report["languages"].items():
        mark = "" if row["real_meets_100k"] else (
            "  (segmentation needed)" if row["meets_100k"] else "  SHORT")
        print(f"  {code:5} {row['total']:>8,} total, {row['real']:>8,} real, "
              f"{len(row['real_by_source'])} source(s){mark}")
    print(f"\nwritten to {OUT} and {REPORT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
