<h1 align="center">urdunlp</h1>
<p align="center"><i>Urdu and Roman Urdu text processing, scored against people. Pure Python, zero dependencies, no model downloads.</i></p>

<p align="center">
  <a href="https://github.com/hammasbuilds/urdunlp/actions/workflows/ci.yml"><img src="https://github.com/hammasbuilds/urdunlp/actions/workflows/ci.yml/badge.svg" alt="ci"></a>
  <a href="https://pypi.org/project/urdunlp/"><img src="https://img.shields.io/pypi/v/urdunlp" alt="pypi"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/dependencies-zero-success" alt="deps">
  <a href="https://github.com/hammasbuilds/urdunlp/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="license"></a>
</p>

Normalise Urdu text (including text copied out of PDFs), transliterate Roman Urdu to
Urdu script and back, tell Urdu from the ten other languages written in the same
script, find the English inside Roman Urdu, read numbers written as words (ڈیڑھ لاکھ,
`sawa do crore`), tokenise, stem and remove stopwords. Every accuracy figure below
was measured on data the library was not built from.

## Install

```bash
pip install urdunlp
```

Until 0.2.0 is on PyPI, install from source:
`pip install git+https://github.com/hammasbuilds/urdunlp`.

Python 3.10 or newer, any OS. No dependencies, and the three small statistical tables
it needs (3.4 MB) ship inside the wheel - nothing is downloaded at runtime. The package
is typed (`py.typed`), so mypy and pyright see every annotation.

On a Windows console that is not UTF-8 (cp1252 and friends), `print()` of Urdu text
from your own script raises `UnicodeEncodeError` - that is Python, not urdunlp. Set
`PYTHONIOENCODING=utf-8` (or run `python -X utf8`); the `urdunlp` command does this itself.

## Quickstart

```python
from urdunlp import (
    normalize,
    transliterate_to_urdu,
    transliterate_to_roman,
    transliterate_with_confidence,
    words,
    remove_stopwords,
)

normalize("كتاب") == normalize("کتاب")  # True: Arabic kaf/yeh unified with Urdu
normalize("ﻛﺘﺎﺏ")  # 'کتاب': presentation forms from a PDF

transliterate_to_urdu("main theek hoon")  # 'میں ٹھیک ہوں'
transliterate_to_urdu("ye kitab hai?")  # 'یہ کتاب ہے؟'
transliterate_to_urdu("is ke baad taur par haasil")  # 'اس کے بعد طور پر حاصل'
transliterate_to_urdu("yaar aaj bohttt garmi hai 😭")  # 'یار آج بہت گرمی ہے 😭'
transliterate_to_urdu("WhatsApp pe msg kr do")  # 'WhatsApp پے میسج کر دو'
transliterate_to_urdu("kal meeting cancel ho gayi", keep_english=True)
# 'کل meeting cancel ہو گئی'

transliterate_to_roman("میں خوش ہوں 😀")  # 'main khush hoon 😀'
transliterate_to_roman("وزیراعظم شہباز شریف")  # 'wazir-e-azam shahbaz sharif'

words("کیا، واقعی؟", keep_punctuation=True)  # ['کیا', '،', 'واقعی', '؟']
remove_stopwords(words("یہ اچھا نہیں ہے"))  # ['اچھا', 'نہیں'] - negation is kept
```

Roman Urdu has no standard spelling, so transliteration is a best guess, and the
library says how each word was resolved:

```python
r = transliterate_with_confidence("mera naam Ali hai")
r.text  # 'میرا نام علی ہے'
r.sources  # [('mera', 'lexicon'), ('naam', 'vocabulary'), ('Ali', 'vocabulary'), ('hai', 'lexicon')]
```

`lexicon` is a curated word, `vocabulary` a real Urdu word chosen by a noisy-channel
model over 60,638 words (decoding the sentence as a whole, so `us ne kaha ke` gives
کہا **کہ**), `rules` a letter-by-letter guess for a word it has never seen, and
`english` a word kept in Latin script. `rule_share` and `lexicon_coverage` summarise
them.

```python
from urdunlp import (
    identify_language,
    tag_roman_tokens,
    roman_key,
    parse_number,
    parse_ordinal,
    find_numbers,
    format_number,
    stem,
)

identify_language("هي ڪتاب منهنجو آهي").name  # 'Sindhi' - same script, different language
identify_language("کتاب").short  # True: one word is a guess, and says so
tag_roman_tokens("kal meeting cancel ho gayi")
# [('kal', 'ur'), ('meeting', 'en'), ('cancel', 'en'), ('ho', 'ur'), ('gayi', 'ur')]
roman_key("nhi") == roman_key("naheen")  # True: both spell نہیں

parse_number("ڈیڑھ لاکھ")  # 150000
parse_number("sawa do crore")  # 22500000
parse_ordinal("teesra")  # 3
format_number(1234567)  # '12,34,567'
[(n.text, n.value) for n in find_numbers("mujhe 2 lakh chahiye aur paanch hazar bhi")]
# [('2 lakh', 200000), ('paanch hazar', 5000)]
stem("کتابوں")  # 'کتاب'
```

## What is in it

| | |
|---|---|
| `normalize`, `is_urdu`, `resolve_arabic_heh`, `remove_urls_and_mentions` | Arabic → Urdu letters (ي ك ه), Arabic-Indic → Urdu digits, PDF presentation forms and ligatures (ﻛﺘﺎﺏ, ﷲ), diacritics, tatweel, zero-width characters; optionally digits to 0-9 and Urdu punctuation to ASCII. |
| `words`, `sentences`, `fix_spacing`, `character_ngrams` | Tokenisation that keeps URLs, emails, @mentions and numbers like `2.5` whole, sentence splitting on ۔ ؟, repair of merged compounds (کردیا → کر دیا). |
| `transliterate_to_urdu`, `transliterate_with_confidence` | Roman Urdu → Urdu script with per-token provenance. Handles chat (`nahiii`, `bohttt`, `ok`, `plz`), izafat (`tehreek-e-insaf`), keeps brands (`iPhone`) and chat abbreviations (`lol`, `Rs`) in Latin, and writes `?` `,` `.` as ؟ ، ۔. `keep_english=True` leaves English in Latin. |
| `transliterate_to_roman` | Urdu → Roman Urdu spelled the way people spell it; `method="rules"` for the letter-by-letter mapping. |
| `roman_key`, `group_roman_variants` | Group Roman spellings by the Urdu word they spell. |
| `identify_language` | Which of eleven Perso-Arabic-script languages: Urdu, Punjabi (Shahmukhi), Saraiki, Sindhi, Pashto, Kashmiri, Persian, Arabic, Central Kurdish, Uyghur, South Azerbaijani. |
| `tag_roman_tokens` | Every token of Roman Urdu text tagged `ur`/`en` (words) or `num`/`punct`/`id`/`code`/`other`. |
| `parse_number`, `parse_ordinal`, `find_numbers`, `number_to_words`, `format_number` | Urdu and Roman Urdu number words both ways, with ڈیڑھ ڈھائی سوا ساڑھے پونے and lakh/crore; ordinals (تیسرا, `teesra`, `5th`); lakh-style grouping. |
| `stem`, `stem_tokens` | Rule-based suffix stripping for retrieval keys. |
| `STOPWORDS`, `NEGATION`, `is_stopword`, `remove_stopwords` | 159 function words, with negation held separately and kept by default. |

## Command line

Installing also gives an `urdunlp` command (or `python -m urdunlp`):

```bash
urdunlp to-urdu "mera naam ali hai"            # میرا نام علی ہے
urdunlp to-roman "میں ٹھیک ہوں"                  # main theek hoon
urdunlp langid "هي ڪتاب منهنجو آهي"              # sd  Sindhi  margin ...
urdunlp normalize "كتاب"                       # کتاب
urdunlp words "رابطہ: test@x.com"              # رابطہ | test@x.com
echo "kal milte hain" | urdunlp to-urdu        # standard input, line by line
urdunlp to-roman -i news.txt -o news-roman.txt # a file in, a UTF-8 file out
urdunlp to-urdu --help                         # --keep-english, --sources, --bom, ...
```

Input files and standard input may be UTF-8 (with or without a byte-order mark) or
UTF-16 (what Windows PowerShell 5.1's `>` writes). Anything else stops with one line
naming the file, line and byte; `--encoding cp1252` reads such a file. Output is UTF-8;
`--bom` adds the byte-order mark Excel and old Notepad look for.

**Windows PowerShell 5.1** (the one that ships with Windows; PowerShell 7 is fine)
sends text to programs as ASCII, which turns every Urdu letter into `?` before urdunlp
sees it, and reads their output in the console's code page, which garbles it. Either
set the session to UTF-8 once:

```powershell
$OutputEncoding = [Text.UTF8Encoding]::new()
[Console]::InputEncoding = [Console]::OutputEncoding = $OutputEncoding
```

or skip the pipe and let urdunlp read and write the files itself: `-i in.txt -o out.txt`.
urdunlp warns when its input arrives as question marks.

## Accuracy

Measured on data held out from everything the models were built from. Method, splits
and every number: [docs/CORPUS.md](https://github.com/hammasbuilds/urdunlp/blob/main/docs/CORPUS.md).

| | 0.1 | **0.2** | measured on |
|---|---:|---:|---|
| Roman → Urdu, word accuracy | 43.1% | **91.3%** | 52,087 words of hand-romanised test sentences ([Dakshina](https://github.com/google-research-datasets/dakshina)) |
| Urdu → Roman, spelled exactly as the annotator did | 28.6% | **54.5%** | the same sentences |
| Urdu → Roman, spelled as some annotator spelled that word | 41.4% | **78.2%** | the same sentences |
| Urdu → Roman → Urdu round trip | 42.0% | **93.9%** | 15,088 tokens of held-out sentences |
| Grouping spelling variants (`nahi`, `nhi`, `naheen`), B-cubed F1 | 0.577 | **0.831** | 10,517 test-lexicon spellings |
| Which of 11 Perso-Arabic languages, whole paragraph | — | **97.9%** | 1,254 test paragraphs |
| ... on 20 characters | — | **91.0%** | |
| English words found inside Roman Urdu (recall) | — | **87.3%** | synthetic code-mixed test sentences |
| Stemming, retrieval recall@10 | — | **+0.008** | 2,583 test queries, sign test p = 0.019 |

The weak numbers are in the table on purpose: the stemmer helps a little, language
identification guesses between neighbours on a single word, and `is_urdu` - a script
check - says yes to 99.9% of Persian.

**Speed.** `import urdunlp` loads nothing (about 0.1 s). Each model loads on the first
call that needs it: about 0.6 s of CPU for transliteration, 0.2 s for
`identify_language`, 0.1 s for `tag_roman_tokens`. Roman → Urdu then runs at about
1,700 words a second of CPU on 51,764 words of held-out Wikipedia sentences,
and faster on text that repeats its words (chat does); Urdu → Roman and
`identify_language` are much faster. For a large corpus, split it across processes
with `multiprocessing.Pool` - each process loads the model once. In a web server,
make one call at startup so the first request does not pay for the load.

## Input

One deliberately messy sentence: Arabic `ک` and `ی` rather than the Urdu forms, doubled
spaces, a URL and a mention.

```
میں  کل  لاہور  سے  آیا  ہوں۔ http://x.co @ali
```

## Output

`python demo.py`

```
INPUT
   ميں  كل  لاہور  سے  آيا  ہوں۔ http://x.co @ali
   Arabic letters in it: U+064A U+0643 U+064A

OUTPUT
   remove_urls_and_mentions ميں كل لاہور سے آيا ہوں۔
   normalize                میں کل لاہور سے آیا ہوں۔
   words                    میں کل لاہور سے آیا ہوں
   remove_stopwords         کل لاہور آیا
   transliterate_to_roman   main kal lahore se aaya hoon.

   Arabic letters left after normalize: 0
   6 tokens in, 3 content words out (3 stopwords removed)

   Roman -> Urdu, and which stage answered:
      mera naam Ali hai            -> میرا نام علی ہے    [lexicon vocabulary vocabulary lexicon]
      is ke baad taur par haasil   -> اس کے بعد طور پر حاصل    [vocabulary lexicon vocabulary vocabulary lexicon vocabulary]
      dekho http://x.co par        -> دیکھو http://x.co پر    [vocabulary identifier lexicon]

   Spelling variants, grouped by the Urdu word they spell:
      نہیں     nahi, nhi, naheen
      اچھا     acha, accha, achha

   English inside Roman Urdu:
      kal meeting cancel ho gayi   -> kal/ur meeting/en cancel/en ho/ur gayi/ur
      keep_english=True            -> کل meeting cancel ہو گئی

   Script is not language - every one of these passes is_urdu():
      یہ کتاب میری ہے اور میں اسے پڑھتا ہوں    -> Urdu     right  
      هذا الكتاب لي وأنا أقرأه                 -> Arabic   right  
      هي ڪتاب منهنجو آهي                       -> Sindhi   right  sd:ڪ
      دا کتاب زما دی او زه یې هره ماښام لولم   -> Pashto   right  ps:ښې ug:ې
      دا کتاب زما دی                           -> Punjabi (Shahmukhi) WRONG, it is Pashto - four words is too few  

   Inflected forms, stemmed to one retrieval key:
      کتاب کتابیں کتابوں لڑکا لڑکے لڑکیاں  -> کتاب کتاب کتاب لڑک لڑک لڑک

   Numbers, with the fractions English has no word for:
      ڈیڑھ لاکھ        = 1,50,000
      سوا دو کروڑ      = 2,25,00,000
      15 lakh          = 15,00,000
      sawa baara lakh  = 12,25,000
```

*Shown as text, not a screenshot: Urdu is a joining right-to-left script, and an image
renderer without HarfBuzz shaping produces disconnected letters in the wrong order.*

## Known limits

- **Roman → Urdu is ambiguous by nature.** `ke` is کے (*of*) or کہ (*that*); a bigram
  model gets it right after کہا, not where the deciding word is further back. On
  held-out sentences 8.7% of words still come out wrong.
- **The accuracy figures are for careful romanisation of encyclopaedia text.** Chat is
  shorter, drops more vowels and switches to English more; nobody has measured how much
  lower it scores. The word statistics are Wikipedia's, so a chat phrase can be read as
  an encyclopaedia one: `kya hal hai bhai` gives کیا حال ہی بھائی (*recently*); `haal`
  gets it right.
- **English inside Roman Urdu.** By default an English word is written the way Urdu
  writes it (`station` → اسٹیشن), and a word the vocabulary does not hold may be matched
  to the wrong Urdu word (`exam` → اقسام, `late` → لاتے). `keep_english=True` leaves the
  words `tag_roman_tokens` calls English in Latin; that tagger finds 87.3% of
  English words on synthetic test sentences and misses some in real chat
  (`kal meeting hai` tags `meeting` as Urdu). Unknown words that are clearly English
  (`recharge`) and mixed-case names (`iPhone`) are kept in Latin either way.
- **Language identification needs a sentence**: 97.9% on a paragraph, 91.0% on 20
  characters, 81.5% on ten, with the errors between Urdu, Punjabi and Saraiki. A guess
  on fewer than 20 letters has `short=True`; `margin` is not a confidence score.
- **Urdu → Roman is lossy.** س ص ث all give `s`. The round trip recovers most of it for
  words that exist, and none of it for names.
- **Compound splitting and the stopword list are fixed lists**, and the stemmer makes
  retrieval keys (لڑکا, لڑکی → لڑک), not lemmas.
- **The bundled data is CC BY-SA 4.0** (derived from Dakshina, Wikipedia and HotpotQA);
  the code is MIT.

## Development

```bash
git clone https://github.com/hammasbuilds/urdunlp
cd urdunlp
python demo.py                                  # nothing to install
pip install pytest && python -m pytest -q       # 578 tests
```

To check the headline number without downloading anything, run
`python scripts/quick_check.py`: it scores Roman → Urdu on a tenth of the Dakshina test
sentences, committed in `eval/` (364 sentences, 5,249 words), and prints 91.4% in about a
minute against 91.3% on the full split.

The tests use only what ships in the package. To reproduce the measurements, fetch the
evaluation data first (none of it is needed to use the library or run the tests):

```bash
python scripts/fetch_dakshina.py            # Roman Urdu, 34 MB of a 2 GB archive
python scripts/fetch_wikipedia_samples.py   # 1,500 paragraphs in each of 11 languages
python scripts/measure_translit.py          # transliteration, docs/CORPUS.md sections 7, 9, 14
python scripts/measure_langid.py            # sections 10 and 11 (needs extract_english.py)
```

How each part was built, and every problem found on the way:
[docs/DEVELOPMENT.md](https://github.com/hammasbuilds/urdunlp/blob/main/docs/DEVELOPMENT.md).
Changes by version: [CHANGELOG.md](https://github.com/hammasbuilds/urdunlp/blob/main/CHANGELOG.md).

## License

Code: MIT. Bundled model data (`src/urdunlp/data/`): derived from
[Dakshina](https://github.com/google-research-datasets/dakshina) (Roark et al., 2020),
Wikipedia and [HotpotQA](https://hotpotqa.github.io/), each CC BY-SA 4.0.
