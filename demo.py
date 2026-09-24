"""The toolkit on deliberately messy input, one stage at a time.

    python demo.py

Prints what went in and what each stage produced. No arguments, no network.
"""

import sys

if hasattr(sys.stdout, "reconfigure"):  # Urdu will not survive a cp1252 console
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, "src")

from urdunlp import (  # noqa: E402
    find_numbers,
    format_number,
    group_roman_variants,
    identify_language,
    normalize,
    remove_stopwords,
    remove_urls_and_mentions,
    stem_tokens,
    tag_roman_tokens,
    transliterate_to_roman,
    transliterate_to_urdu,
    transliterate_with_confidence,
    words,
)

# Deliberately messy: Arabic kaf and yeh rather than Urdu, doubled spaces, a URL,
# an English mention.
RAW = "میں  کل  لاہور  سے  آیا  ہوں۔ http://x.co @ali"

stages = []
clean = remove_urls_and_mentions(RAW)
stages.append(("remove_urls_and_mentions", clean))

norm = normalize(clean)
stages.append(("normalize", norm))

toks = words(norm)
stages.append(("words", toks))

content = remove_stopwords(toks)
stages.append(("remove_stopwords", content))

roman = transliterate_to_roman(norm)
stages.append(("transliterate_to_roman", roman))

print("INPUT")
print(f"   {RAW}")
print()
print("OUTPUT")
for name, value in stages:
    shown = " ".join(value) if isinstance(value, list) else value
    print(f"   {name:24} {shown}")
print()
print(
    f"   {len(toks)} tokens in, {len(content)} content words out "
    f"({len(toks) - len(content)} stopwords removed)"
)

# Roman -> Urdu, reporting which stage resolved each word. `Ali` needs ع, which
# Roman cannot write; the vocabulary stage finds the real word that has it.
print()
print("   Roman -> Urdu, and which stage answered:")
for probe in ("mera naam Ali hai", "is ke baad taur par haasil", "dekho http://x.co par"):
    result = transliterate_with_confidence(probe)
    stages_used = " ".join(kind for _, kind in result.sources)
    print(f"      {probe:28} -> {result.text}    [{stages_used}]")

print()
print("   Spelling variants, grouped by the Urdu word they spell:")
for key, group in group_roman_variants(["nahi", "acha", "nhi", "accha", "naheen", "achha"]).items():
    print(f"      {key:8} {', '.join(group)}")

print()
print("   English inside Roman Urdu:")
mixed = "kal meeting cancel ho gayi"
print(f"      {mixed:28} -> " + " ".join(f"{w}/{t}" for w, t in tag_roman_tokens(mixed)))
print(f"      {'keep_english=True':28} -> {transliterate_to_urdu(mixed, keep_english=True)}")

print()
print("   Script is not language - every one of these passes is_urdu():")
# The last one is shown because it is wrong. Four short words are where the
# model is weakest; docs/CORPUS.md has accuracy by length.
for text, truth in (
    ("یہ کتاب میری ہے اور میں اسے پڑھتا ہوں", "Urdu"),
    ("هذا الكتاب لي وأنا أقرأه", "Arabic"),
    ("هي ڪتاب منهنجو آهي", "Sindhi"),
    ("دا کتاب زما دی او زه یې هره ماښام لولم", "Pashto"),
    ("دا کتاب زما دی", "Pashto"),
):
    guess = identify_language(text)
    evidence = " ".join(f"{k}:{''.join(v)}" for k, v in guess.evidence.items())
    mark = "right" if guess.name == truth else f"WRONG, it is {truth} - four words is too few"
    print(f"      {text:40} -> {guess.name:<8} {mark}  {evidence}")

print()
print("   Inflected forms, stemmed to one retrieval key:")
forms = words("کتاب کتابیں کتابوں لڑکا لڑکے لڑکیاں")
print(f"      {' '.join(forms):36} -> {' '.join(stem_tokens(forms))}")

print()
print("   Numbers, with the fractions English has no word for:")
for span in find_numbers("اس نے ڈیڑھ لاکھ روپے اور سوا دو کروڑ کا قرض لیا، aur 15 lakh baqi"):
    print(f"      {span.text:14} = {format_number(span.value)}")
