"""Score PALI, the published Perso-Arabic language-ID benchmark, on this task.

PALI (Ahmadi et al., 2023, arXiv:2304.01322, MIT) is the real competitor rather than an
off-the-shelf one: it targets Perso-Arabic language identification directly, and its own
confusion analysis clusters Urdu, Kashmiri, Punjabi, Saraiki and Sindhi together - which
is this task. It ships trained fastText models, so it can be run rather than cited.

Two things this script is careful about.

**The label mapping is derived, not guessed.** PALI's `configs.json` uses ISO 639-3
(`urd`, `pnb`, `arb`) and its model labels are different strings (`ud`, `pa`, `ar`), so
assuming `ud` means Urdu would be a guess that silently decides the result. Instead the
mapping is learned on the **validation** split - for each of our eleven languages, the
PALI label it most often assigns - and then applied unchanged to the **test** split. Using
test items to build the mapping would make "PALI predicts our label" partly definitional.

**The confound is stated and its direction is known.** PALI's corpora are Wikipedia dumps
(`*-wiki-20230120`) for several of these languages, and our test paragraphs are Wikipedia
too. Our test rows may therefore appear in PALI's training data. That biases the comparison
**in PALI's favour**, so a result where this tool still leads is conservative, and a result
where PALI leads cannot be read as a clean win for it. There is no way to check from here:
PALI's training file is 61 MB and its corpora are not the same snapshot as ours.

    .venv-baselines/Scripts/python research/pali_baseline.py
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

MODELS = ROOT / ".baselines" / "pali"
OUT = ROOT / "research" / "results" / "pali.json"
LENGTHS = (None, 50, 20, 10)


def load(name: str):
    import fasttext

    fasttext.FastText.eprint = lambda *a, **k: None
    path = MODELS / f"LID_model_{name}.ftz"
    if not path.exists():
        raise SystemExit(
            f"no PALI model at {path}. Fetch it once (MIT licensed, 2.1 MB):\n"
            f"  gh api repos/sinaahmadi/PersoArabicLID/contents/models/LID_model_{name}.ftz"
            f" -H 'Accept: application/vnd.github.raw' > {path}"
        )
    return fasttext.load_model(str(path))


def predict(model, texts: list[str]) -> list[str]:
    out = []
    for text in texts:
        labels, _ = model.predict(text.replace("\n", " "), k=1)
        out.append(labels[0].replace("__label__", "") if labels else "__refused__")
    return out


def principled_mapping(model, splits: dict, languages: list[str]) -> dict:
    """Our code -> PALI's label: identity where it exists, derived only where it cannot.

    The first version of this took the majority vote for every language and produced
    `fa -> mzn` and `ckb -> sdh`. PALI's label set contains both `fa` and `ckb`, so those
    were not naming differences - they were PALI's own confusions, and mapping them that
    way scored its errors as correct. Identity first, and a derivation only where the
    identity label is genuinely absent.
    """
    available = {l.replace("__label__", "") for l in model.get_labels()}
    mapping, how = {}, {}
    derived = derive_mapping(model, splits, languages)
    for code in languages:
        if code in available:
            mapping[code] = code
            how[code] = "identity"
        else:
            mapping[code] = derived["mapping"][code]
            how[code] = "derived on val"
    return {
        "mapping": mapping,
        "how": how,
        "identity_for": sorted(c for c in languages if how[c] == "identity"),
        "derived_for": sorted(c for c in languages if how[c] != "identity"),
        "generous_mapping": derived["mapping"],
        "evidence": derived["evidence"],
        "collisions": derived["collisions"],
    }


def derive_mapping(model, splits: dict, languages: list[str]) -> dict:
    """Our code -> PALI's label, by majority vote on the VALIDATION split.

    Reported with the vote share, so a mapping that rests on a plurality rather than a
    majority is visible instead of silent - and collisions are reported rather than
    resolved, because two of our languages mapping to one PALI label is a fact about its
    label set, not a problem to paper over.
    """
    mapping, evidence = {}, {}
    for code in languages:
        texts = splits[code]["val"]
        counts = collections.Counter(predict(model, texts))
        best, hits = counts.most_common(1)[0]
        mapping[code] = best
        evidence[code] = {
            "pali_label": best,
            "share": round(hits / len(texts), 4),
            "n": len(texts),
            "runner_up": counts.most_common(2)[1] if len(counts) > 1 else None,
        }
    collisions = {
        label: sorted(c for c, l in mapping.items() if l == label)
        for label in set(mapping.values())
        if sum(1 for l in mapping.values() if l == label) > 1
    }
    return {"mapping": mapping, "evidence": evidence, "collisions": collisions}


def main() -> int:
    from measure_langid import LANGUAGES, wiki_splits  # noqa: PLC0415

    splits = wiki_splits(ROOT / "data" / "wiki_large")
    languages = sorted(splits)
    model_name = "merged"  # the only one whose labels span all eleven
    model = load(model_name)
    labels_in_model = sorted(l.replace("__label__", "") for l in model.get_labels())
    print(f"PALI LID_model_{model_name}: {len(labels_in_model)} labels\n")

    derived = principled_mapping(model, splits, languages)
    print("label mapping - identity where PALI has our code, derived only where it does not:")
    for code in languages:
        row = derived["evidence"][code]
        how = derived["how"][code]
        flag = ""
        if how == "identity" and row["pali_label"] != code:
            # The generous mapping disagrees: PALI's most common answer for this language
            # is a DIFFERENT language it also has a label for. That is its error, and it
            # is reported rather than mapped away.
            flag = f"   [majority vote said {row['pali_label']} - PALI's own confusion]"
        print(f"  {code:4} -> {derived['mapping'][code]:5} ({how}){flag}")
    if derived["collisions"]:
        print(f"\n  collisions in the generous mapping: {derived['collisions']}")

    # Score on test, at every length, with the mapping fixed.
    # The SAME iteration order as scripts/compare_langid.py, which walks splits.items().
    # This walked sorted(splits) instead, so the two prediction vectors described
    # different items and a paired test would have compared one language's answers with
    # another's. run_all.py asserts the orders match, which is what caught it.
    truth, texts = [], []
    for code, parts in splits.items():
        for paragraph in parts["test"]:
            truth.append(code)
            texts.append(paragraph)

    inverse = {v: k for k, v in derived["mapping"].items()}
    generous_inverse = {v: k for k, v in derived["generous_mapping"].items()}
    by_length = {}
    for length in LENGTHS:
        label = "whole" if length is None else f"{length} chars"
        cut = [t if length is None else t[:length] for t in texts]
        raw = predict(model, cut)
        # PALI's label back to our code, or the raw label when it is outside the mapping.
        got = [inverse.get(p, p) for p in raw]
        right = sum(1 for g, t in zip(got, truth) if g == t)
        per_language = {}
        for code in languages:
            idx = [i for i, t in enumerate(truth) if t == code]
            per_language[code] = round(
                sum(1 for i in idx if got[i] == code) / len(idx), 4
            )
        generous = [generous_inverse.get(p, p) for p in raw]
        generous_right = sum(1 for g, t in zip(generous, truth) if g == t)
        by_length[label] = {
            "accuracy": round(right / len(truth), 4),
            "accuracy_generous_mapping": round(generous_right / len(truth), 4),
            "n": len(truth),
            "per_language": per_language,
            "predictions": got,
        }
        row = by_length[label]
        print(f"  {label:10} accuracy {row['accuracy']:.4f}  "
              f"(generous mapping {row['accuracy_generous_mapping']:.4f}, "
              f"+{row['accuracy_generous_mapping'] - row['accuracy']:.4f} from crediting "
              f"its own confusions)")

    payload = {
        "model": f"LID_model_{model_name}.ftz",
        "source": "https://github.com/sinaahmadi/PersoArabicLID (MIT)",
        "paper": "arXiv:2304.01322",
        "labels_in_model": labels_in_model,
        "mapping_learned_on": "validation split",
        **derived,
        "confound": (
            "PALI's corpora are Wikipedia dumps for several of these languages and this "
            "test set is Wikipedia, so these test paragraphs may appear in PALI's "
            "training data. The bias is in PALI's favour; a lead for this tool is "
            "therefore conservative and a lead for PALI cannot be read as clean."
        ),
        "by_length": {k: {kk: vv for kk, vv in v.items() if kk != "predictions"}
                      for k, v in by_length.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    # Per-item predictions go beside the others so McNemar can pair them.
    import gzip

    path = ROOT / "research" / "results" / "pali_predictions.json.gz"
    with gzip.GzipFile(path, "wb", mtime=0) as fh:
        fh.write(json.dumps({
            "truth": truth,
            "by_length": {k: v["predictions"] for k, v in by_length.items()},
        }).encode("utf-8"))
    print(f"\nwritten to {OUT.name} and {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
