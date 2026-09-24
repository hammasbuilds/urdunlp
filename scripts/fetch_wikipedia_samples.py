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


def random_intros(code: str) -> list[str]:
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
        for paragraph in page.get("extract", "").split("\n"):
            paragraph = " ".join(paragraph.split())
            if len(paragraph) >= 40:
                paragraphs.append(paragraph)
    return paragraphs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("data/wiki"))
    ap.add_argument("--paragraphs", type=int, default=1500)
    ap.add_argument("--max-requests", type=int, default=400)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for code, name in LANGUAGES.items():
        target = args.out / f"{code}.txt"
        seen: dict[str, None] = {}
        if target.exists():
            seen = dict.fromkeys(target.read_text(encoding="utf-8").splitlines())
        requests = stale = 0
        while len(seen) < args.paragraphs and requests < args.max_requests and stale < 15:
            requests += 1
            try:
                batch = random_intros(code)
            except Exception as error:  # noqa: BLE001 - a flaky request is retried
                print(f"  {code}: {error}")
                time.sleep(5)
                continue
            before = len(seen)
            seen.update(dict.fromkeys(batch))
            stale = stale + 1 if len(seen) == before else 0
            time.sleep(0.3)
        lines = list(seen)[: args.paragraphs]
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{code:<4} {name:<22} {len(lines):>5} paragraphs  ({requests} requests)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
