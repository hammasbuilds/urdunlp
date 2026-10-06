<h1 align="center">urdunlp</h1>
<p align="center"><i>Urdu and Roman Urdu text processing, scored against people. Pure Python, zero dependencies, no model downloads.</i></p>

<p align="center">
  <a href="https://github.com/hammasbuilds/urdunlp/actions/workflows/ci.yml"><img src="https://github.com/hammasbuilds/urdunlp/actions/workflows/ci.yml/badge.svg" alt="ci"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/dependencies-zero-success" alt="deps">
  <a href="https://github.com/hammasbuilds/urdunlp/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT%20code%20%2B%20CC%20BY--SA%20data-green" alt="license"></a>
</p>

Normalise Urdu text (including text copied out of PDFs), transliterate Roman Urdu to
Urdu script and back, tell Urdu from the ten other languages written in the same
script, find the English inside Roman Urdu, read numbers written as words (ڈیڑھ لاکھ,
`sawa do crore`), tokenise, stem and remove stopwords. Every accuracy figure below
was measured on data the library was not built from.

## Install

```bash
pip install git+https://github.com/hammasbuilds/urdunlp
```

PyPI release coming: `pip install urdunlp` will work once 0.2.0 is published.

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
| `parse_number`, `parse_ordinal`, `find_numbers`, `number_to_words`, `format_number` | Urdu and Roman Urdu number words both ways, with ڈیڑھ ڈھائی سوا ساڑھے پونے, counted fractions (تین چوتھائی, دو تہائی) and lakh/crore; ordinals (تیسرا, `teesra`, `5th`); lakh-style grouping. |
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
| Roman → Urdu, word accuracy | 43.1% | **91.3%** | 52,087 words of the 3,632 of 4,945 hand-romanised test sentences whose token counts align ([Dakshina](https://github.com/google-research-datasets/dakshina)) |
| Urdu → Roman, spelled exactly as the annotator did | 28.6% | **54.5%** | the same sentences |
| Urdu → Roman, spelled as some annotator spelled that word | 41.4% | **78.2%** | the same sentences |
| Urdu → Roman → Urdu round trip | 42.0% | **93.8%** | 15,088 tokens of Dakshina's dev and test sentences |
| Grouping spelling variants (`nahi`, `nhi`, `naheen`), B-cubed F1 | 0.577 | **0.831** | 10,517 test-lexicon spellings |
| Which of 11 Perso-Arabic languages, whole paragraph | — | **97.9%** | 1,254 test paragraphs — but see the note below: this pools an easy 8-way task with a hard 3-way one |
| ... on 20 characters | — | **91.0%** | |
| English words found inside Roman Urdu (recall) | — | **87.0%** | synthetic code-mixed test sentences |

**Two of these numbers have a denominator worth knowing.**

**Word accuracy is over the 73% of test sentences that can be aligned.** A sentence is
scored word by word only when its Urdu and Roman token counts agree, which is true of
3,632 of 4,945; the other 1,313 cannot be aligned without guessing which Roman token
belongs to which Urdu one. That exclusion is not random — counts disagree when an izafat
compound is romanised as one token or three, when compounds merge or split, when English
is inserted, or when an annotator joined words, which is to say on the sentences hardest
to transliterate. Treat 91.3% as word accuracy *on alignable sentences*, not on the test
set.

**97.9% pools an easy task with the hard one.** Eight of the eleven languages are
separable on orthography alone and score **exactly 100%** on whole paragraphs, over 63–80
test paragraphs each. The informative part is Urdu, Punjabi (Shahmukhi) and Saraiki, which
share nearly all their orthography:

| | test paragraphs | whole | 50 chars | 20 chars | 10 chars |
|---|---:|---:|---:|---:|---:|
| Urdu | 225 | 98.7% | 98.0% | 93.3% | 84.0% |
| Punjabi (Shahmukhi) | 224 | 97.3% | 94.2% | 85.7% | 70.1% |
| Saraiki | 229 | 92.6% | 88.9% | **75.1%** | **59.4%** |
| the other eight | 63–80 each | **100.0%** | 98.7–100% | 95.8–100% | 88.7–100% |

**Pooled over all eleven: 97.9%. Over Urdu/Punjabi/Saraiki alone: 96.2%** (678
paragraphs). So the headline is ~100% on an 8-way orthography question and 96.2% on the
three-way one, and the three-way one is where length hurts — Saraiki falls to 75.1% at 20
characters and 59.4% at 10, while seven of the other eight are still at 100% on 50. A rate
of exactly 100% over 70 paragraphs means the test set holds no hard case for those
languages, not that the model is perfect on them.

(Windows are centred slices, as `scripts/measure_langid.py` takes them; the middle of a
paragraph carries more signal than its opening. Taking the first N characters instead
scores Saraiki 11 points lower at 20.)

The gold labels are also *which Wikipedia edition a paragraph came from*. For Urdu,
Arabic, Pashto, Sindhi and Uyghur that is close enough to a language label. For Punjabi
(Shahmukhi) versus Saraiki it is not: whether Saraiki is a language or a Punjabi dialect
group is a live sociolinguistic question, so for that pair the label itself is editorial —
which is part of why it is the pair the model confuses.

| Stemming, retrieval recall@10 | — | **+0.008** | 2,582 test queries, sign test p = 0.019 |

The weak numbers are in the table on purpose: the stemmer helps a little, language
identification guesses between neighbours on a single word, and `is_urdu` - a script
check - says yes to 99.9% of Persian.

**Speed.** `import urdunlp` loads nothing (about 0.1 s). Each model loads on the first
call that needs it: about 0.6 s of CPU for transliteration, 0.2 s for
`identify_language`, 0.1 s for `tag_roman_tokens`. Roman → Urdu then runs at
**440-500 words a second** of CPU on words it has not seen before — measured on the
5,289 words of `eval/dakshina_test_sample.tsv` that ship with the repo, so you can
check it, and matching what `scripts/quick_check.py` prints. Throughput depends almost
entirely on word repetition rather than on length: running the same text a second time
gives about **90,000 words a second** from warm caches, and real chat sits between the
two because it reuses a small vocabulary. An earlier figure of 1,700-1,900 here was
measured on a different, more repetitive corpus and did not describe first-pass text.
`scripts/measure_translit.py` prints the figure for your machine; Urdu → Roman and
`identify_language` are much faster. For a large corpus, split it across processes
with `multiprocessing.Pool` - each process loads the model once. In a web server,
make one call at startup so the first request does not pay for the load.

## Input

One deliberately messy sentence: Arabic `ک` and `ی` rather than the Urdu forms, doubled
spaces, a URL and a mention.

```
ميں  كل  لاہور  سے  آيا  ہوں۔ http://x.co @ali
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
  lower it scores. The word statistics are Wikipedia's, so the word that follows can pull
  a token to the wrong homophone: `kya hal hai` is right (کیا حال ہے), but
  `kya hal hai bhai` gives کیا حال ہی بھائی — `hai` becomes ہی (*only*) rather than
  ہے (*is*), because of what comes after it. Spelling `hal` as `haal` changes nothing;
  both give حال.
- **English inside Roman Urdu.** By default an English word is written the way Urdu
  writes it (`station` → اسٹیشن), and a word the vocabulary does not hold may be matched
  to a different Urdu word that is spelled similarly (`cancel` → کونسل, which reads
  *council*). `keep_english=True` leaves the
  words `tag_roman_tokens` calls English in Latin; that tagger finds 87.0% of
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
pip install pytest && python -m pytest -q       # 831 tests
```

To check the headline number without downloading anything, run
`python scripts/quick_check.py`: it scores Roman → Urdu on a tenth of the Dakshina test
sentences, committed in `eval/` (364 sentences, 5,249 words), and prints 91.4% in about ten
seconds against 91.3% on the full split. Once the full split is fetched (below),
`python scripts/quick_check.py --rebuild` regenerates that sample from it, so you can
check it is every tenth sentence and not a hand-picked one.

The tests use only what ships in the package. To reproduce the measurements, fetch the
evaluation data first (none of it is needed to use the library or run the tests):

```bash
python scripts/fetch_dakshina.py            # Roman Urdu, 34 MB of a 2 GB archive
python scripts/fetch_wikipedia_samples.py   # 1,500 paragraphs in each of 11 languages
python scripts/measure_translit.py          # transliteration, docs/CORPUS.md sections 7, 9, 14
python scripts/measure_corpus.py data/dakshina/ur/romanized --every 11   # the round trip
python scripts/extract_english.py DIR       # English for the tagger; DIR = HotpotQA distractor parquets
python scripts/measure_langid.py            # sections 10 and 11
```

`fetch_wikipedia_samples.py` draws random articles from the live Wikipedias, so a fresh
fetch gives a different sample from the one the language-identification rows were
measured on: expect figures close to those, not equal to the decimal. Paragraphs are
split into train and test by a hash of their text, so a paragraph the bundled model was
built from can never land in your test set. `extract_english.py` needs `pyarrow` and
the `distractor` parquets of [`hotpotqa/hotpot_qa`](https://huggingface.co/datasets/hotpotqa/hotpot_qa).

The stemming row needs the first six row groups (6,000 articles) of the public Urdu
Wikipedia parquet, [`wikimedia/wikipedia`, `20231101.ur`](https://huggingface.co/datasets/wikimedia/wikipedia/tree/main/20231101.ur)
(168 MB), and `pyarrow` to read it:
`python scripts/measure_stemmer.py train-00000-of-00001.parquet --row-groups 6`.

How each part was built, and every problem found on the way:
[docs/DEVELOPMENT.md](https://github.com/hammasbuilds/urdunlp/blob/main/docs/DEVELOPMENT.md).
Changes by version: [CHANGELOG.md](https://github.com/hammasbuilds/urdunlp/blob/main/CHANGELOG.md).

## License

Code: MIT. Bundled model data (`src/urdunlp/data/`): derived from
[Dakshina](https://github.com/google-research-datasets/dakshina) (Roark et al., 2020),
Wikipedia and [HotpotQA](https://hotpotqa.github.io/), each CC BY-SA 4.0.
