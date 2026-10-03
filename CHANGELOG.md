# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — unreleased

The first version with an accuracy figure for transliteration. Every number below was
measured on data held out from what it was built from; `docs/CORPUS.md` has each one
with its method, and the scripts that reproduce them are listed there.

### Added

- `scripts/quick_check.py` and `eval/dakshina_test_sample.tsv`: the headline Roman → Urdu
  figure re-measured on a committed tenth of the test sentences in about a minute, no
  download (91.4% on the sample against 91.3% on the full split).
- **A vocabulary stage in Roman → Urdu transliteration**, between the curated lexicon
  and the rules: a noisy-channel search over 60,638 Urdu words for the one most likely
  to have been typed as the Roman string. Letter emissions were trained with EM on
  106,260 attested romanisations from Google's Dakshina lexicon, and mixed with each
  word's own attested spellings. Sentences are decoded as a whole with a word-bigram
  model. On 52,087 words of held-out hand-romanised sentences, word accuracy goes from
  **43.1% to 91.3%** (88.5% word by word, `use_context=False`). It finds the letters
  Roman cannot write — `baad` → بعد, `taur` → طور, `Ali` → علی — and reads `ke` as کہ
  after کہا. `use_vocabulary=False` restores the 0.1 pipeline. New source label
  `vocabulary`, new property `rule_share`.
- `roman_key()` / `group_roman_variants()` — group Roman spellings by the Urdu word they
  stand for: `nahi`, `nhi`, `naheen`, `nahee` → نہیں. B-cubed F1 **0.831** on the
  held-out lexicon, against 0.375 for exact matching and 0.497 for a consonant skeleton.
- `identify_language()` — which of eleven Perso-Arabic-script languages a text is in:
  Urdu, Punjabi (Shahmukhi), Saraiki, Sindhi, Pashto, Kashmiri, Persian, Arabic,
  Central Kurdish, Uyghur, South Azerbaijani. **97.9%** on 1,254 held-out Wikipedia
  paragraphs, 91.0% on 20 characters, 81.5% on 10 - character 1-5-grams, with 5,000
  paragraphs each of the three closest languages. The 1-3-gram, 1,500-paragraph first
  version scored 96.4 / 89.8 / 81.5 on the same test set. Reports the distinctive letters it
  saw as evidence.
- `tag_roman_tokens()` and `keep_english=True` — label each word of Roman Urdu as `ur`
  or `en`, smoothed over the sentence. English recall 0.873 on synthetic code-mixed
  sentences, 1.6% of English words mislabelled `ur`.
- `parse_number()`, `find_numbers()`, `number_to_words()`, `format_number()` — Urdu and
  Roman Urdu number words, including ڈیڑھ, ڈھائی, سوا, ساڑھے and پونے, lakh and crore
  scales, and 3-then-2 digit grouping (`12,34,567`). Round-trips every integer below
  200,000 and 20,000 random ones up to 10¹³.
- **Urdu → Roman spelled the way people spell it** (`transliterate_to_roman`, now the
  default; `method="rules"` keeps the 0.1 letter mapping). On held-out Dakshina test
  words, a spelling some annotator wrote 33.4% → **54.1%** of the time; in running text,
  exactly the annotator's spelling 28.6% → **54.7%**; and the round trip back to Urdu
  90.9% → **93.9%** - readable and reversible stopped being a trade-off.
- The Arabic article moves across the word boundary: `abdul rehman` → عبد الرحمن,
  `bainul aqwami` → بین الاقوامی, `darul uloom` → دار العلوم. Capital initials are
  spelled by letter name (`C. M.` → سی ایم), titles written in full without their dot
  (`Dr.` → ڈاکٹر, `Mr.` → مسٹر), lowercase `o` is و, and `ki`/`ke` may be کہ when the
  sentence says so. Together: 90.7% → 91.2% on the test sentences.
- `parse_ordinal()`, and ordinals in `find_numbers` with `ordinal=True`: پہلا دوسرا
  تیسرا چوتھا چھٹا, یکم, any cardinal + واں/ویں, and `5ویں`. پہلے ("before") and
  دوسرے ("other") are not read as ordinals on their own.
- `stem()` / `stem_tokens()` — a rule-based suffix stripper. It helps, a little:
  +0.008 recall@10 on title retrieval (p = 0.019), +0.005 on lead-sentence retrieval
  (not significant). Both are reported.
- Scripts that fetch the evaluation data and reproduce every number:
  `fetch_dakshina.py` (walks a 2 GB remote tar by byte range and takes the 34 MB it needs),
  `fetch_wikipedia_samples.py`, `extract_english.py`, `count_vocabulary.py`,
  `build_translit_model.py`, `build_langid_models.py`, `measure_translit.py`,
  `measure_langid.py`, `measure_stemmer.py`.
- **An `urdunlp` command** (also `python -m urdunlp`): `to-urdu`, `to-roman`, `langid`,
  `normalize` and `words`, each taking text as arguments or line by line on standard
  input, with `--help` and `--version`.
- `urdu_punctuation=True` (the default) in Roman → Urdu: `?` `,` `;` and a full stop
  after a word become ؟ ، ؛ ۔ - `ye kitab hai?` gives یہ کتاب ہے؟. Marks inside `2.5`,
  `...` or `:)` are left alone. New source label `punctuation`.
- The merged Roman future is written the standard way, as two words: `karunga` → کروں
  گا, `dekhenge` → دیکھیں گے, `milega` → ملے گا. It came out as non-words before
  (کرؤنگ) or fell to the letter rules.
- Chat spellings in the lexicon: `h` (ہے), `kse`, `bhot`, `kro`, `pata` (پتہ), `gaari`,
  `gaadi`, `gadi` (گاڑی). Checked on dev sentences first: five dev words changed, all
  for the better.
- Chat typed in capitals is not read as acronyms: `KYA HAAL HAI` gave کے وائی اے حال
  ہے. A word the curated lexicon knows is never an acronym, and in text written in
  capitals throughout only dotted acronyms (`U.S.A`) are spelled; `BBC TV` still is.
- `LanguageGuess.short`: True when the text had fewer than 20 Perso-Arabic letters,
  where held-out accuracy is 91% or less.

### Changed

- `transliterate_to_urdu("mera naam Ali hai")` now returns میرا نام **علی** ہے. The
  README example and its test are updated; the 0.1 behaviour is pinned under
  `use_vocabulary=False`.
- Round-tripping Urdu → Roman → Urdu on held-out sentences with the 0.1 letter rules:
  **42.0% → 90.9%** with the default short-vowel setting, 61.7% → 92.7% without it. The losses docs/CORPUS.md
  called "properties of the two writing systems" are mostly recoverable for words that
  exist - صرف comes back from `srf`.
- `measure_corpus.py` reads `.txt.gz` and reports both pipelines' round-trip rates.

### Changed in the release candidate, after a user-view audit

- `transliterate_to_roman` keeps emoji, symbols and letters of other scripts. It used
  to delete every character without a Roman form: `میں خوش ہوں 😀` gave
  `main khush hoon `. The output is therefore no longer guaranteed ASCII; only Urdu
  diacritics and zero-width marks inside Urdu words are dropped. `٪` becomes `%`.
- `words()` keeps URLs, emails, `@mentions`, `#hashtags` and numbers with separators
  (`2.5`, `1,500`, `3:30`, `۱۲٫۵`) as single tokens, the way the transliterator always
  had. `test@x.com` used to become `test`, `x`, `com`.
- `remove_urls_and_mentions` removes emails whole (it removed `@x` from `test@x.com`
  as a mention, leaving `test .com`), and a URL no longer takes the full stop after it.
- The list-taking functions name themselves and the offending type or item:
  `remove_stopwords(b"x")` said "is_stopword() expects a str, got int". They also accept
  any iterable of strings, a generator included.
- A curated word the vocabulary never counted lost to any vocabulary word in a
  sentence: `kal parso` gave کل پرشو. Such words now get a median word frequency.
- Faster first load: `import urdunlp` no longer imports `gzip`/`json`, and the model
  builds its bigram rows on demand. The first transliteration still costs about 1 s of
  CPU; the README's "about 0.6 s" was never measured and is replaced by measured figures.
- Two numbers corrected after re-running every measurement for the release: the learned
  Urdu → Roman round trip is 93.9%, not 94.3%, and sentence words spelled exactly as the
  annotator did are 54.7%, not 54.9%. Both earlier figures came from older runs and
  had not been updated. The 91.2% headline was re-measured and did not move.

### Fixed

- `parse_number` rejected every hundred that was not round (پانچ سو تیس), and read
  ایک ہزار کروڑ as 10,010,000,000. Both caught while writing the tests, before release.
- Transliteration output keeps the input's spacing and line breaks exactly; numbers,
  phone numbers, dates and times (`2.5`, `0300-1234567`, `25-12-2024`, `3:30`) and codes
  like `5th` pass through whole; all-capital acronyms are spelled by letter name
  (`BBC` → بی بی سی); words with accented letters (`café`) pass through.
- `find_numbers` offsets index the caller's text, not the normalised text; punctuation
  ends a number phrase; `٫` and `٬` are read as decimal and grouping marks; a bare scale
  word after a finished group (`کروڑ ہزار`) is rejected.
- `format_number` handled no float printed in exponent form (`1e-05` raised
  `IndexError`), and accepted `True`, NaN and infinity.
- Every public function raises the same `TypeError`, naming itself, for a non-string -
  and the list-taking ones for a string.
- Transliterating one long run of words and scanning a long run of number words were
  both superlinear; both are linear now.
- The model loader failed on a zipped install under Python 3.10.
- `transliterate_to_roman` silently dropped Urdu digits: ۱۲۳ vanished from the output.
- CI now runs `mypy --strict`, the docstring examples, and the suite against the built
  wheel installed outside `src/`.

### Data

The bundled models (3.4 MB) are derived from Dakshina (CC BY-SA 4.0), Wikipedia
(CC BY-SA 4.0) and HotpotQA (CC BY-SA 4.0). The code remains MIT.

## [0.1.0] — 2026-09-24 (never published to PyPI)

First release.

### Added

- `normalize()` — Arabic↔Urdu codepoint unification, diacritics, tatweel, zero-width
  characters, digits and punctuation. Unicode NFC does **not** merge `U+064A` ARABIC YEH
  with `U+06CC` FARSI YEH, because they are genuinely different letters used by different
  languages, which is why the problem survives the normalisation people assume handles it.
- `words()` / `sentences()` — tokenisation whose letter ranges skip each Urdu punctuation
  codepoint individually. Urdu punctuation sits *inside* the Arabic block, so the obvious
  range swallows `،` `؟` `۔` into word tokens.
- `transliterate_to_urdu()` / `transliterate_to_roman()` / `transliterate_with_confidence()`
  — with per-token provenance, so a caller can tell a lexicon lookup from a rule guess.
- `STOPWORDS` (115 entries) with `NEGATION` held **separately**. A stopword list that
  deletes `نہیں` inverts every sentiment label.
- `docs/CORPUS.md` — every claim measured against 84,581 BBC Urdu articles (44.7M tokens):
  **one article in eleven** carries a substituted Arabic codepoint; the stopword list
  covers **41.4%** of tokens; transliteration round-trips **44.7%** with the default and
  **61.2%** with `insert_short_vowels=False`.
- `scripts/measure_corpus.py` — reproduces all of it on any corpus.
- `py.typed`, so type checkers see the annotations instead of returning `Any`.

### Fixed

- Urdu fed to the Roman→Urdu direction was returned untouched but labelled `rules`, as
  though the rule engine had resolved it, with `lexicon_coverage` reading `0.0`. Tokens
  already in Urdu script now report `already-urdu` and are excluded from the coverage
  denominator.
- `pytest` failed from a fresh clone with `ModuleNotFoundError`. CI installed the package
  first, so CI was green while every clone was broken.
- The README documented `pip install urdu-nlp-toolkit`, which returned 404.

[0.2.0]: https://github.com/hammasbuilds/urdunlp/releases/tag/v0.2.0
[0.1.0]: https://github.com/hammasbuilds/urdunlp/releases/tag/v0.1.0
