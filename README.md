# urdunlp

A Python library for Urdu and Roman Urdu text. We built it for developers and researchers who handle Urdu in
search, chat, data cleaning or NLP pipelines: it normalises Urdu (including text copied out of PDFs), transliterates
Roman Urdu to Urdu script and back, tells Urdu from the ten other languages written in the same script, finds the
English inside Roman Urdu, reads numbers written as words (ڈیڑھ لاکھ, `sawa do crore`), tokenises, stems and removes
stopwords. Pure Python, zero dependencies, no model downloads, and every accuracy figure is measured on text people
wrote that the library was not built from.

## What it does

| | |
|---|---|
| `normalize`, `is_urdu`, `resolve_arabic_heh`, `remove_urls_and_mentions` | Arabic → Urdu letters (ي ك ه), Arabic-Indic → Urdu digits, PDF presentation forms and ligatures (ﻛﺘﺎﺏ, ﷲ), diacritics, tatweel, zero-width characters; optionally digits to 0-9 and Urdu punctuation to ASCII. |
| `words`, `sentences`, `fix_spacing`, `character_ngrams` | Tokenisation that keeps URLs, emails, @mentions and numbers like `2.5` whole, sentence splitting on ۔ ؟, repair of merged compounds (کردیا → کر دیا). |
| `transliterate_to_urdu`, `transliterate_with_confidence` | Roman Urdu → Urdu script with per-token provenance. Handles chat (`nahiii`, `bohttt`, `ok`, `plz`), izafat (`tehreek-e-insaf`), keeps brands (`iPhone`) and chat abbreviations (`lol`, `Rs`) in Latin, and writes `?` `,` `.` as ؟ ، ۔. `keep_english=True` leaves English in Latin. |
| `transliterate_to_roman` | Urdu → Roman Urdu spelled the way people spell it; `method="rules"` for the letter-by-letter mapping. |
| `roman_key`, `group_roman_variants` | Group Roman spellings by the Urdu word they spell. |
| `identify_language` | Which of eleven Perso-Arabic-script languages: Urdu, Punjabi (Shahmukhi), Saraiki, Sindhi, Pashto, Kashmiri, Persian, Arabic, Central Kurdish, Uyghur, South Azerbaijani. |
| `tag_roman_tokens` | Every token of Roman Urdu text tagged `ur`/`en` (words) or `num`/`punct`/`id`/`code`/`other`. |
| `parse_number`, `parse_ordinal`, `find_numbers`, `number_to_words`, `format_number` | Urdu and Roman Urdu number words both ways, with ڈیڑھ ڈھائی سوا ساڑھے پونے, counted fractions (تین چوتھائی, دو تہائی) and lakh/crore; the English number words Roman Urdu mixes in (`eight hundred`, `six thousand eight sau`) and the chat thousand (`15k`); ordinals (تیسرا, `teesra`, `5th`); lakh-style grouping. |
| `stem`, `stem_tokens` | Rule-based suffix stripping for retrieval keys. |
| `STOPWORDS`, `NEGATION`, `is_stopword`, `remove_stopwords` | 159 function words, with negation held separately and kept by default. |

## Screenshots

A small browser front end drives the package for demonstrations. It is a local demo and is **not part of
the package** — the package has no web surface and no runtime dependencies. Every result on these screens
is a live call into the library.

![Overview](https://raw.githubusercontent.com/hammasbuilds/urdunlp/main/docs/screenshots/00-overview.png)
*The front end as it opens: every feature is a tab, and the page states that nothing on it is precomputed.*

![Roman To Urdu](https://raw.githubusercontent.com/hammasbuilds/urdunlp/main/docs/screenshots/01-roman-to-urdu.png)
*Input: `kal meeting cancel ho gayi, plz 125 rupay bhej do` with "keep English in Latin" ticked. Output: کل meeting cancel ہو گئی، plz 125 روپے بھیج دو, and the stage that decided each token (lexicon, English, punctuation, passthrough, vocabulary).*

![Urdu To Roman](https://raw.githubusercontent.com/hammasbuilds/urdunlp/main/docs/screenshots/02-urdu-to-roman.png)
*Input: an Urdu sentence. Output: its romanisation, with the round trip back to Urdu script.*

![Language Id](https://raw.githubusercontent.com/hammasbuilds/urdunlp/main/docs/screenshots/03-language-id.png)
*Input: an Urdu paragraph. Output: the language picked out of eleven written in the same script, with the margin and the letters that are distinctive to particular languages.*

![Analyse](https://raw.githubusercontent.com/hammasbuilds/urdunlp/main/docs/screenshots/04-analyse.png)
*Input: Urdu text containing a number, a URL and a mention. Output: the normalised text, tokens, sentences, stems and the numbers found.*

![Numbers](https://raw.githubusercontent.com/hammasbuilds/urdunlp/main/docs/screenshots/05-numbers.png)
*Input: `320`. Output: the Urdu words for it, parsed back to 320, with the round trip checked.*

## Sample input / output

### Input

One deliberately messy sentence: Arabic `ک` and `ی` rather than the Urdu forms, doubled
spaces, a URL and a mention.

```
ميں  كل  لاہور  سے  آيا  ہوں۔ http://x.co @ali
```

### Output

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

## Results

Measured on data held out from everything the models were built from. Method, splits and every number:
[docs/CORPUS.md](https://github.com/hammasbuilds/urdunlp/blob/main/docs/CORPUS.md).

| | 0.1 | **0.2** | measured on |
|---|---:|---:|---|
| Roman → Urdu, word accuracy | 43.1% | **91.3%** | 52,087 words of the 3,632 of 4,945 hand-romanised test sentences whose token counts align ([Dakshina](https://github.com/google-research-datasets/dakshina)) |
| Urdu → Roman, spelled exactly as the annotator did | 28.6% | **54.5%** | the same sentences |
| Urdu → Roman, spelled as some annotator spelled that word | 41.4% | **78.2%** | the same sentences |
| Urdu → Roman → Urdu round trip | 42.0% | **93.8%** | 15,088 tokens of Dakshina's dev and test sentences |
| Grouping spelling variants (`nahi`, `nhi`, `naheen`), B-cubed F1 | 0.577 | **0.831** | 10,517 test-lexicon spellings |
| Which of 11 Perso-Arabic languages, whole paragraph | — | **99.1%** | 6,136 test paragraphs (Wikipedia, split by a hash of the text) |
| ... on 20 characters | — | **94.9%** | |
| English words found inside Roman Urdu (recall) | — | **86.9%** | Dakshina test sentences with 1-3 English words spliced in |
| Stemming, retrieval recall@10 | — | **+0.008** | 2,582 test queries, sign test p = 0.019 |

Word accuracy is over the 73% of test sentences whose Urdu and Roman token counts agree, which excludes the
hardest ones (izafat written as one token or three, merged compounds, inserted English). The language-ID figure
pools eight languages that are nearly separable on orthography alone (98.8–100% on whole paragraphs) with the three that share
nearly all of it:

| | test paragraphs | whole | 50 chars | 20 chars | 10 chars |
|---|---:|---:|---:|---:|---:|
| Urdu | 578 | 99.0% | 95.6% | 93.4% | 85.3% |
| Punjabi (Shahmukhi) | 537 | 96.7% | 92.9% | 85.7% | 72.6% |
| Saraiki | 549 | 96.9% | 93.1% | **80.3%** | **65.8%** |
| the other eight | 510–592 each | 98.8–100.0% | 98.9–100.0% | 96.1–99.8% | 87.2–98.2% |

Over Urdu, Punjabi and Saraiki alone the whole-paragraph accuracy is 97.5% (1,664 paragraphs). A guess on fewer than
20 letters is marked `short=True`, and `margin` is not a confidence score.

**Speed.** `import urdunlp` loads nothing (about 0.1 s); each model loads on first use (about 0.6 s for
transliteration). Roman → Urdu runs at 440-500 words a second of CPU on words it has not seen before (measured on
the 5,289 words of `eval/dakshina_test_sample.tsv`) and about 90,000 words a second on repeated text from warm
caches. For a large corpus, split it across processes; in a web server, make one call at startup.

## Tech stack

- Python 3.10+, standard library only; typed (`py.typed`, checked with `mypy --strict`)
- Bundled statistical tables (5.0 MB in the wheel): a noisy-channel transliteration model over a 60,638-word
  vocabulary with a bigram language model, a character n-gram language-ID model for 11 languages, an English
  lexicon for the code-mix tagger
- Measurement scripts use `pyarrow` to read the Wikipedia and HotpotQA parquets (not needed to use the library)
- pytest, ruff and mypy; CI on Python 3.10-3.14 plus an installed-wheel test job

## Run it

```bash
pip install git+https://github.com/hammasbuilds/urdunlp
```

Python 3.10 or newer, any OS. On a Windows console that is not UTF-8 (cp1252 and friends), `print()` of Urdu text
from your own script raises `UnicodeEncodeError`: set `PYTHONIOENCODING=utf-8` (or run `python -X utf8`); the
`urdunlp` command does this itself.

### Quickstart

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
identify_language("hello world").name  # None - not a Perso-Arabic script at all
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

### Command line

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

### Developing and reproducing the numbers

```bash
git clone https://github.com/hammasbuilds/urdunlp
cd urdunlp
python demo.py                                  # nothing to install
pip install pytest && python -m pytest -q       # the test suite uses only what ships in the package
python scripts/quick_check.py                   # Roman -> Urdu on every tenth Dakshina test sentence, ~10 s
```

`quick_check.py` scores the 364 committed sentences in `eval/` and prints 91.4% against 91.3% on the full split.
To reproduce every measurement, fetch the evaluation data first (none of it is needed to use the library):

```bash
python scripts/fetch_dakshina.py            # Roman Urdu, 34 MB of a 2 GB archive
python scripts/fetch_wikipedia_samples.py   # 1,500 paragraphs in each of 11 languages
python scripts/measure_translit.py          # transliteration
python scripts/measure_corpus.py data/dakshina/ur/romanized --every 11   # the round trip
python scripts/extract_english.py DIR       # English for the tagger; DIR = HotpotQA distractor parquets
python scripts/measure_langid.py            # language identification
```

`fetch_wikipedia_samples.py` draws random articles from the live Wikipedias, so a fresh fetch gives figures close to
the published ones, not equal to the decimal. The stemming row needs the first six row groups of the Urdu Wikipedia
parquet ([`wikimedia/wikipedia`, `20231101.ur`](https://huggingface.co/datasets/wikimedia/wikipedia/tree/main/20231101.ur)):
`python scripts/measure_stemmer.py train-00000-of-00001.parquet --row-groups 6`.

## Licence

Code: MIT. Bundled model data (`src/urdunlp/data/`): derived from
[Dakshina](https://github.com/google-research-datasets/dakshina) (Roark et al., 2020),
Wikipedia and [HotpotQA](https://hotpotqa.github.io/), each CC BY-SA 4.0.
