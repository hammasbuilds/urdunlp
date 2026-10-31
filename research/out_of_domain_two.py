"""A second, independent out-of-domain source - to adjudicate the Urdu collapse.

Tatoeba put Urdu at 0.5302, against 0.9896 in domain, with 1,255 of 2,812 sentences
labelled Shahmukhi Punjabi. Two readings, and they need separating rather than choosing
between:

  A. the model does not transfer out of domain, or
  B. Tatoeba's crowd-sourced labels are noisy on exactly the Urdu/Punjabi boundary this
     task is about - the one place a volunteer is most likely to mislabel.

A second source with *professional* labels separates them. If Urdu is also ~0.53 on it,
reading A. If Urdu is high, reading B, and the Tatoeba figure is partly a fact about
Tatoeba.

**Tanzil** (OPUS mono, Quran translations) is that source: published human translation,
one language per file, nothing crowd-sourced, and as far from Wikipedia as text gets.
Pashto is not in it, so Pashto comes from **TED2020** - transcribed talks.

Every candidate's script is checked before use, and two were rejected on that basis:

  * OPUS Tanzil `az` is **100% Latin** - Azerbaijani in Latin script, not the South
    Azerbaijani in Perso-Arabic this task uses. Using it would have added a class
    separable by script alone and inflated every aggregate.
  * OPUS GNOME `ks` is **98% Devanagari** - Kashmiri in Devanagari, not the Perso-Arabic
    orthography here.

So South Azerbaijani has no out-of-domain source at all, and Kashmiri still has only
Tatoeba's 210 sentences. Stated rather than filled with something that would look like
coverage.

    .venv-baselines/Scripts/python research/out_of_domain_two.py --fetch
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
RAW = ROOT / ".baselines" / "opus"
OUT = ROOT / "research" / "results" / "out_of_domain_two.json"
TATOEBA_RESULT = ROOT / "research" / "results" / "out_of_domain.json"

# our code -> (OPUS corpus, OPUS language code, what the text is)
SOURCES = {
    "ur": ("Tanzil", "ur", "Quran translation"),
    "sd": ("Tanzil", "sd", "Quran translation"),
    "fa": ("Tanzil", "fa", "Quran translation"),
    "ar": ("Tanzil", "ar", "Quran, original"),
    "ug": ("Tanzil", "ug", "Quran translation"),
    "ckb": ("Tanzil", "ku", "Quran translation, Arabic-script Kurdish"),
    "ps": ("TED2020", "ps", "transcribed talks"),
}
REJECTED = {
    "azb": "OPUS Tanzil `az` is 100% Latin script - Azerbaijani, not South Azerbaijani "
           "in Perso-Arabic. No Perso-Arabic source found.",
    "ks": "OPUS GNOME `ks` is 98% Devanagari - Kashmiri in Devanagari, not the "
          "Perso-Arabic orthography used here. Tatoeba's 210 sentences remain the only "
          "out-of-domain Kashmiri.",
    "pnb": "No source found beyond Tatoeba's 1.3 KB.",
    "skr": "No source found beyond Tatoeba's 2.0 KB.",
}
MIN_FOR_A_RATE = 30
CAP = 4000  # sentences per language, so Persian's 4.7 MB does not dominate the aggregate


def wilson(k: int, n: int, z: float = 1.96) -> dict | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"rate": round(p, 4), "lo": round(max(0.0, c - h), 4),
            "hi": round(min(1.0, c + h), 4), "n": n, "correct": k}


def arabic_share(text: str) -> float:
    arab = sum(1 for c in text if "؀" <= c <= "ۿ" or "ݐ" <= c <= "ݿ")
    latn = sum(1 for c in text if c.isascii() and c.isalpha())
    deva = sum(1 for c in text if "ऀ" <= c <= "ॿ")
    return arab / (arab + latn + deva or 1)


def fetch() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for ours, (corpus, code, _) in SOURCES.items():
        path = RAW / f"{corpus}-{code}.txt.gz"
        if path.exists() and path.stat().st_size:
            print(f"  {ours:4} have {path.name}")
            continue
        url = f"https://object.pouta.csc.fi/OPUS-{corpus}/v1/mono/{code}.txt.gz"
        try:
            urllib.request.urlretrieve(url, path)
            print(f"  {ours:4} {path.name} {path.stat().st_size:,} bytes")
        except urllib.error.HTTPError as exc:
            print(f"  {ours:4} HTTP {exc.code} for {url}")


def lines(path: Path, cap: int = CAP) -> list[str]:
    out, seen = [], set()
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            text = line.strip()
            if len(text) >= 20 and text not in seen:
                seen.add(text)
                out.append(text)
            if len(out) >= cap:
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    if args.fetch:
        fetch()
        print()

    from urdunlp.langid import identify_language  # noqa: PLC0415

    print(f"  {'lang':6}{'source':22}{'n':>6}{'arabic':>8}  {'accuracy':>10}  "
          f"{'95% interval':>18}   confusions")
    results = {}
    for ours, (corpus, code, what) in SOURCES.items():
        path = RAW / f"{corpus}-{code}.txt.gz"
        if not path.exists():
            results[ours] = {"available": False, "why": f"{path.name} not fetched"}
            continue
        texts = lines(path)
        share = arabic_share(" ".join(texts[:200]))
        if share < 0.5:
            results[ours] = {"available": False,
                             "why": f"{corpus} {code} is not Perso-Arabic "
                                    f"({share:.0%} Arabic-range characters)"}
            print(f"  {ours:6}{corpus + ' ' + code:22}{len(texts):>6}{share:>7.0%}   "
                  f"REJECTED: wrong script")
            continue
        got = [identify_language(t).language for t in texts]
        right = sum(1 for g in got if g == ours)
        wrong: dict[str, int] = {}
        for g in got:
            if g != ours:
                wrong[str(g)] = wrong.get(str(g), 0) + 1
        top = ", ".join(f"{k} {v}" for k, v in
                        sorted(wrong.items(), key=lambda kv: -kv[1])[:3]) or "—"
        row = {"available": True, "source": f"OPUS {corpus} {code}", "what": what,
               "n": len(texts), "correct": right, "arabic_share": round(share, 3),
               "confusions": dict(sorted(wrong.items(), key=lambda kv: -kv[1]))}
        row["accuracy"] = wilson(right, len(texts)) if len(texts) >= MIN_FOR_A_RATE else None
        results[ours] = row
        if row["accuracy"]:
            w = row["accuracy"]
            print(f"  {ours:6}{corpus + ' ' + code:22}{len(texts):>6}{share:>7.0%}  "
                  f"{w['rate']:>10.4f}  [{w['lo']:.4f}, {w['hi']:.4f}]   {top}")
        else:
            print(f"  {ours:6}{corpus + ' ' + code:22}{len(texts):>6}{share:>7.0%}  "
                  f"{right:>4} right  {'rate withheld':>18}   {top}")

    # --- the adjudication -----------------------------------------------------
    verdict = None
    if TATOEBA_RESULT.exists() and results.get("ur", {}).get("accuracy"):
        tatoeba = json.loads(TATOEBA_RESULT.read_text(encoding="utf-8"))
        theirs = tatoeba["per_language"]["ur"]["accuracy"]
        mine = results["ur"]["accuracy"]
        gap = mine["rate"] - theirs["rate"]
        verdict = {
            "tatoeba_urdu": theirs,
            "tanzil_urdu": mine,
            "difference": round(gap, 4),
            "reading": (
                "The model does not transfer: Urdu is low on both independent "
                "out-of-domain sources."
                if mine["rate"] < 0.75 else
                "Tatoeba's Urdu/Punjabi labels are implicated: Urdu holds on a source "
                "with professional labels, so the Tatoeba figure is partly a fact about "
                "Tatoeba rather than only about the model."
            ),
        }
        print(f"\nadjudication of the Urdu collapse:")
        print(f"  Tatoeba (crowd-sourced) {theirs['rate']:.4f} "
              f"[{theirs['lo']:.4f}, {theirs['hi']:.4f}]  n={theirs['n']:,}")
        print(f"  Tanzil  (professional)  {mine['rate']:.4f} "
              f"[{mine['lo']:.4f}, {mine['hi']:.4f}]  n={mine['n']:,}")
        print(f"  -> {verdict['reading']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "sources": {k: f"OPUS {v[0]} {v[1]} ({v[2]})" for k, v in SOURCES.items()},
        "rejected_on_script_or_absence": REJECTED,
        "cap_per_language": CAP,
        "min_n_for_a_rate": MIN_FOR_A_RATE,
        "per_language": results,
        "urdu_adjudication": verdict,
    }, indent=1), encoding="utf-8")
    print(f"\nwritten to {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
