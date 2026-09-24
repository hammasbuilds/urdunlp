# Measured: 84,581 articles, 10,000 hand-romanised sentences, eleven languages

[<- back to README](../README.md)

Every claim this toolkit makes was qualitative. Sections 1-6 measure the 0.1 claims
against [XL-Sum Urdu](https://huggingface.co/datasets/csebuetnlp/xlsum): **84,581 BBC
Urdu news articles, 206,887,475 characters, 44,698,779 tokens**.

Sections 7-13 measure what 0.2 added, against data held out from everything each part
was built from: human romanisations from Google's
[Dakshina](https://github.com/google-research-datasets/dakshina) dataset, Wikipedia in
eleven languages, and HotpotQA's English sentences. Every table there names its split.

Reproduce it on any corpus:

```bash
python scripts/measure_corpus.py <directory-of-txt-or-a-parquet> --json out.json
```

## The corpus

| | |
|---|---:|
| Articles | 84,581 |
| Characters | 206,887,475 |
| Tokens | 44,698,779 |
| Distinct normalised types | 353,358 |

---

## 1. The Arabic/Urdu codepoint problem is real, and it is common

The README claims scraped Urdu freely mixes Arabic codepoints with Urdu ones. It does.

**7,495 of 84,581 articles (8.9%)** contain at least one Arabic letter standing in for its
Urdu counterpart — in professionally edited BBC copy, not user-generated text.

| Codepoint | Should be | Occurrences |
|---|---|---:|
| `U+064A` ARABIC YEH | `U+06CC` FARSI YEH | **20,555** |
| `U+0643` ARABIC KAF | `U+06A9` KEHEH | **4,132** |
| `U+0647` ARABIC HEH | `U+06BE` or `U+06C1`, by context | 981 |
| `U+0649` ALEF MAKSURA | `U+06CC` FARSI YEH | 40 |
| `U+0629` TEH MARBUTA | `U+06C1` HEH GOAL | 20 |

The heh row is a late addition, and the reason it was missing is worth stating: this
script's substitution list and `normalize()`'s mapping table were written from the same
list of letters. The one absent from the table was absent from the measurement too, so the
figure published here was counting only the letters the code already handled. A
measurement that shares a blind spot with the code it measures cannot find the gap — it
returns a clean result for the part nobody looked at. The earlier figure was 8.8%.

`normalize()` fixes **24,046 tokens** that Unicode NFC leaves alone, because these are
genuinely distinct codepoints for distinct languages.

Nearly one article in eleven. Any exact-match lookup, vocabulary build or deduplication
over this corpus is silently wrong without normalisation.

### The obvious mapping for ARABIC HEH is wrong nine times in ten

The other four letters have one Urdu counterpart each. `ه` has two, and Urdu uses them for
different jobs:

| | | |
|---|---|---|
| `ہ` | `U+06C1` HEH GOAL | an ordinary *h* — نہ, اللہ |
| `ھ` | `U+06BE` DOACHASHMEE HE | aspiration — بھی, تھا, کھانا |

Which one a stray `ه` stands for is decided by the letter before it. That can be checked
rather than argued about: a token containing no `ه` is correctly spelled by definition, so
the corpus vocabulary adjudicates each candidate spelling. Of the 977 occurrences, 934
have at least one candidate the corpus recognises:

| rule | correct |
|---|---:|
| always `ہ` — what this toolkit's docstring promised | 84/934 = **9.0%** |
| `ھ` after an aspirable consonant, else `ہ` | 905/934 = **96.9%** |

The intuitive mapping is wrong more than nine times in ten, and the reason is mechanical:
a keyboard without `ھ` is a keyboard without *bh, ph, th, kh, gh*, so that is where the
substitution turns up. `بهی` outnumbers everything else in the list.

```
لکها   ->  لکھا   seen 12,909 times    vs  لکہا   seen 1
سنده   ->  سندھ   seen 13,399 times    vs  سندہ   seen 93
نه     ->  نہ     seen 75,285 times    vs  نھ     seen 1
```

The remaining 3.1% is not noise to be engineered away: بہار (spring) and بھار (weight)
differ only in this letter, and nothing context-free separates them. `resolve_arabic_heh`
is exported so the heuristic can be applied, inspected or switched off, rather than
presented as a rule.

## 2. Normalisation moves 0.59% of tokens, and merges 14,611 types

| | |
|---|---:|
| Raw whitespace tokens | 44,682,626 |
| Changed by `normalize()` | **263,833** (0.59%) |
| Types written more than one way | **14,611** |
| Tokens moved onto the majority spelling | 210,906 |

0.59% sounds small until you notice it is concentrated: 14,611 distinct words appear in two
or more spellings, and 210,906 token instances are the minority form. Those are exactly the
words a vocabulary fragments on.

## 3. 115 stopwords cover 41.4% of running text

The README calls the list "closed-class vocabulary — pronouns, postpositions, auxiliaries —
which is where most tokens in real text actually are". Measured:

| | |
|---|---:|
| Stopword list size | 115 |
| Entries that appear in the corpus | **114 of 115** |
| Token hits | 18,492,484 |
| **Share of all tokens** | **41.4%** |

114 of 115 entries earn their place. The ten most frequent types in 44.7M tokens:

| | Count | | Count |
|---|---:|---|---:|
| کے | 2,047,014 | اور | 830,808 |
| میں | 1,472,477 | کہ | 770,042 |
| کی | 1,327,136 | نے | 758,610 |
| ہے | 1,093,615 | کا | 713,233 |
| سے | 893,078 | کو | 670,246 |

Every one is a function word. That is the claim, measured.

## 4. Transliteration is lossy — and the default setting is the worse one

Round-tripping Urdu → Roman → Urdu on **457,380 sampled tokens** (every 97th, spread across
the whole corpus, 21,766 distinct types):

| `insert_short_vowels` | Exact round-trip |
|---|---:|
| `True` **(the default)** | **44.7%** |
| `False` | **61.2%** |

**The default round-trips 16.5 points worse than turning it off.** That is not a bug in the
round-trip — it is the cost of the default being right for its actual job. Urdu does not
write short vowels, so `صرف` maps to the consonants `srf`, which no English reader can
pronounce. Inserting them gives `saraf`, which is readable and is what Roman Urdu users
type. But every inserted vowel comes back as an alef:

| Urdu | Roman (default) | Back | Roman (bare) | Back |
|---|---|---|---|---|
| کتاب | `katab` | کاتاب ✗ | `ktab` | کتاب ✓ |
| پاکستان | `pakasatan` | پاکاساتان ✗ | `pakstan` | پاکستان ✓ |
| مشکل | `mashakal` | ماشاکال ✗ | `mshkl` | مشکل ✓ |
| تقریبا | `taqariba` | تاقاریبا ✗ | `tqriba` | تقریبا ✓ |
| امریکی | `amariki` | اماریکی ✗ | `amriki` | امریکی ✓ |

So: use the default when a human reads the output, and `insert_short_vowels=False` when
something has to convert it back.

### Three losses no setting recovers

The remaining 38.8% is not the vowel setting. Urdu distinguishes letters that Roman spells
identically, so the map back can only pick one:

| Urdu | Bare roman | Back | What collapsed |
|---|---|---|---|
| صرف | `srf` | سرف | ص and س are both `s` |
| حسن | `hsn` | ہسن | ح, ہ and ھ are all `h` |
| بیماریاں | `bimarian` | بیماریان | ں nasalisation is a plain `n` |
| اختلافات | `akhtlafat` | اکھتلافات | خ and کھ are both `kh` |

`لاہور` → `lahor` → `لاہور` round-trips under **both** settings — no ambiguous letter, no
inserted vowel. That is the control: without it, the rows above could be describing a
transliterator that never round-trips anything.

These are properties of the two writing systems, not of the implementation. A round-trip
figure of 100% would mean the transliteration was not doing its job.

> **0.2: half of that paragraph was wrong.** The information is gone from the Roman
> string, but not from the language. `srf` is not a word; صرف is the real word whose
> romanisation it most likely is, and the vocabulary stage (section 7) finds it. On
> 15,098 tokens sampled from 9,759 held-out Dakshina sentences:
>
> | `insert_short_vowels` | 0.1 pipeline (lexicon, rules) | 0.2 pipeline (+ vocabulary) |
> |---|---:|---:|
> | `True` (default) | 42.0% | **90.5%** |
> | `False` | 61.8% | **92.4%** |
>
> The 0.1 column reproduces the BBC figures above on a second corpus (44.7% and 61.2%),
> so the method holds. With the vocabulary, the 18-point penalty for the readable
> default almost disappears. What still fails is mostly words the vocabulary does not
> hold - names and rare loanwords - where the rules above still apply.

*(The first draft of this table used `صرف` as a vowel example. It is not one — it fails
both ways because of the sibilant collapse. The test that pins these examples caught it.)*

## 5. The lexicon is small on purpose, and the README's own example shows why

249 lexicon entries and 55 grapheme rules. The lexicon deliberately covers closed-class
vocabulary — the words rules cannot disambiguate — and not proper nouns:

```python
>>> transliterate_with_confidence("mera naam Ali hai")
Transliteration(text='میرا نام الی ہے',
                sources=[('mera', 'lexicon'), ('naam', 'rules'),
                         ('Ali', 'rules'), ('hai', 'lexicon')])
```

`Ali` resolves by rule to `الی`, not the conventional `علی`, because ع is unwritable in
Roman and no lexicon covers names. `lexicon_coverage` reports `0.5` rather than hiding it.

An earlier version of this page, and the README, showed `علی` in that output. That was
wrong: it is what the reader expects, not what the code returns.

> **0.2:** it now returns `علی`, and this time it is what the code does. The vocabulary
> stage (section 7) prefers the real word that contains ع over the rule-built `الی`,
> and `sources` reports `vocabulary` for it. `use_vocabulary=False` still gives `الی`.

---

## 6. Merged compounds cluster at the end of a clause, where the old matcher could not see them

`fix_spacing` repairs 19 compounds that Urdu typists routinely write without the space.
It used to find them by splitting the text on whitespace, which attaches any adjacent
punctuation to the token — so `کردیا` matched and `کردیا۔` did not.

Over the full corpus, counting each compound as a word span rather than a whitespace
chunk:

| | count |
|---|---:|
| occurrences present | 60,305 |
| found by whitespace splitting | 48,582 |
| **missed** | **11,723 (19.4%)** |

The loss is not spread evenly, and that is what identifies the cause. These compounds are
verb + auxiliary. The *completive* ones end a clause, so a sentence mark sits against
them; the *progressive* ones continue it:

| compound | | occurrences | missed | |
|---|---|---:|---:|---:|
| `کردیں` | did | 1,069 | 381 | **35.6%** |
| `ہوگئے` | became | 12,156 | 4,161 | **34.2%** |
| `ہوگیا` | became | 6,428 | 1,792 | 27.9% |
| `کردیا` | did | 10,291 | 1,898 | 18.4% |
| `آرہا` | is coming | 821 | 48 | 5.8% |
| `جارہا` | is going | 3,588 | 50 | 1.4% |
| `کررہے` | are doing | 3,151 | 23 | **0.7%** |

A fiftyfold difference in miss rate between `کردیں` and `کررہے` is not noise in the
matcher — it is clause position, and it is why the bug was invisible to a unit test
written from a single example.

Matching word spans also stopped the function rewriting text it was not asked to touch:
`" ".join(text.split())` collapsed every newline, indent and double space in the input as
a side effect of inserting one.

---

## 7. Transliteration, scored against people: 43.1% → 87.0%

The section below this one used to say that measuring transliteration correctness "needs
human-checked pairs, which do not exist for Urdu at any useful scale". They exist.
Dakshina (Roark et al., LREC 2020) had native speakers romanise Urdu Wikipedia two ways:
a **word lexicon** listing every spelling annotators wrote for each word, with how many
wrote it, and **~10,000 whole sentences** romanised by hand.

```bash
python scripts/fetch_dakshina.py        # 34 MB, taken from a 2 GB tar by byte range
python scripts/measure_translit.py
```

**Word accuracy on hand-romanised sentences** — Dakshina's own dev/test split. A sentence
is scored word by word when its Urdu and Roman token counts agree (3,632 of 4,945 test
sentences); the rest cannot be aligned without guessing. Correct means equal after
`normalize` on both sides.

| | dev (51,764 words) | **test (52,087 words)** |
|---|---:|---:|
| 0.1: lexicon, then rules | 43.3% | **43.1%** |
| 0.2: lexicon, then vocabulary, then rules | 87.4% | **87.0%** |

Where the 0.2 answers come from, on test:

| stage | share of words | right |
|---|---:|---:|
| curated lexicon | 34.1% | 95.4% |
| vocabulary (new) | 65.3% | 83.2% |
| rules | 0.5% | 2.9% |

The 0.1 figure is worth dwelling on. Over all 148,591 word pairs in Dakshina's aligned
file, **two thirds of running text fell through to the rules, and the rules got 16.5% of
it right.** Two mechanical errors dominated — a word-initial vowel with no carrier (`is` →
یس instead of اس, `aik` → ےک instead of ایک) and short vowels written out that Urdu does
not write (`jis` → جیس instead of جس, `karne` → کارنے instead of کرنے).

**What the vocabulary stage is.** A noisy channel: choose the Urdu word *u* that maximises
P(*u*) · P(roman | *u*). P(*u*) is the word's frequency in Urdu Wikipedia. P(roman | *u*)
comes from a letter-emission model — each Urdu letter emits 0-4 Roman characters,
conditioned on its position and on whether the next letter is a vowel letter — trained
with EM on the 106,260 attested pairs of Dakshina's **training** lexicon. Candidates are
retrieved by a coarse consonant key both scripts map to, with `h` dropped entirely and the
letters Roman cannot tell apart (س ص ث ش, ت ط ٹ, ز ذ ض ظ ج) in one class. The gold word's
key is among the Roman word's keys for **98.3%** of test spellings — that is the ceiling.

**Word accuracy on the lexicon** — every spelling weighted by how many annotators wrote it:

| | train (106,260) | dev (10,424) | test (10,517) |
|---|---:|---:|---:|
| 0.1 | 10.7% | 10.0% | 11.8% |
| 0.2 | 64.9% | 59.1% | 58.1% |

Far lower than on sentences, and expected to be: the lexicon samples words across the
vocabulary, so the rare words that the frequency prior helps least are over-represented
compared with running text. The train/test gap is 6.8 points — the emissions were fitted
to the training lexicon, the vocabulary was not.

**What was chosen on dev, and how.** Vocabulary size (words seen ≥ 2, 3 or 5 times:
84,783 / 60,638 / 42,498 words) moved dev sentence accuracy by less than 0.3 points, so
the smallest was kept. The prior weight: 0.5 gave 86.2%, 1.0 gave 87.0% — plain Bayes
won. Conditioning emissions on the next letter improved the training log-likelihood
after six EM iterations from −786,736 to −752,577 (−711,603 after ten) and dev accuracy by
only 0.3 points: the remaining errors are not in the channel. Each extra letter class in the key (`g` as ج for *germany*, `s` as ز, `c` as
س, `m` as ن) was kept only because removing it cost dev accuracy; `z` as س and `d` as ت
changed nothing and were dropped.

**The remaining errors**, from a dev run with the earlier prior weight of 0.5 (6,890 of
51,764 words wrong):

| kind | words |
|---|---:|
| the vocabulary picked a different real word | 4,811 |
| the right word is not in the vocabulary | 1,344 |
| the curated lexicon was wrong in context | 735 |

The largest single error is context the model cannot see: `ke` is کے (*of*) or کہ
(*that*), and the lexicon says کے — wrong 223 times on dev. `number` and `november` both
key to N-M-P-R, and word frequency alone cannot separate them. A word-bigram model would
address both; it would also add megabytes to a package whose premise is that it has none.

## 8. Dakshina's held-out sentences are not held out

Dakshina's romanised sentences come from its held-out Wikipedia partition. The
vocabulary is counted from its training partition. Checked rather than assumed:
**456 of 4,879 dev and 460 of 4,880 test sentences — 9.4% — also occur verbatim in the
training partition**, 7,348 lines in all counting repeats. A vocabulary counted over those
lines would have seen the evaluation sentences.

`count_vocabulary.py --exclude` drops every such line before counting, and the shipped
vocabulary is the one built that way. Separately, 42 sentences sit in *both* Dakshina's
dev and test files; they are dropped from dev so validation and test never share a
sentence.

## 9. Spelling variants: `roman_key` against two baselines

Dakshina's annotators wrote آئینی seven ways: `aaeeni`, `aaini`, `aayinee`, `aayini`,
`aayiny`, `ainey`, `aini`. `roman_key` groups spellings by the Urdu word they resolve to.
Scored as clustering over the 10,517 test-lexicon spellings, where the gold cluster is
the Urdu word each one spells (B-cubed):

| key | precision | recall | F1 |
|---|---:|---:|---:|
| exact lowercase spelling | 0.985 | 0.232 | 0.375 |
| consonant skeleton | 0.338 | 0.940 | 0.497 |
| 0.1 transliteration (rules) | 0.977 | 0.409 | 0.577 |
| **`roman_key`** | 0.917 | 0.756 | **0.829** |

Exact matching is not 1.000 precise because some spellings genuinely belong to two words.
The skeleton finds almost every variant and merges everything else with them. Resolving
to a real word is what buys both.

## 10. Script is not language

`is_urdu` checks the script. Eleven Wikipedias are written in it. On 1,500 random article
intros per language, it says yes to:

| language | `is_urdu` | | language | `is_urdu` |
|---|---:|---|---|---:|
| Urdu | 94.2% | | Persian | 99.9% |
| Punjabi (Shahmukhi) | 99.5% | | Arabic | 99.5% |
| Saraiki | 99.5% | | Central Kurdish | 99.4% |
| Sindhi | 97.1% | | Uyghur | 98.8% |
| Pashto | 99.6% | | South Azerbaijani | 98.2% |
| Kashmiri | 99.6% | | | |

It accepts Urdu *least* often — Urdu Wikipedia's bot-written stubs carry Latin-script
names in parentheses. A script check cannot identify Urdu, and no letter can either: the
letters people think of as Urdu's own (ٹ ڈ ڑ ں ے) are shared with Punjabi, Saraiki and
Kashmiri. The distinctive letters `build_langid_models.py` found automatically belong to
the *other* languages — Sindhi ڪ ٻ ڻ, Pashto ښ ځ ډ, Central Kurdish ڵ ێ, Uyghur ۇ ۋ.

`identify_language` is a character 1-3-gram naive Bayes model, one table per language,
trained on 80% of each sample (split by paragraph hash, 80/15/5). Accuracy on the test
paragraphs — 771 of them, 57-80 per language, so each per-language figure is ±3-5 points:

| text | val (2,543) | **test (771)** |
|---|---:|---:|
| whole paragraph | 99.0% | **99.1%** |
| 50 characters | 98.0% | **98.1%** |
| 20 characters | 92.9% | **94.3%** |
| 10 characters | 83.8% | **84.2%** |

The errors are almost all between the three closest languages. At 20 characters, Saraiki
is right 73.7% of the time (mostly read as Punjabi) and **Urdu 85.1%** — mostly read as
Punjabi too. Arabic, Central Kurdish and Uyghur are 100% at 20 characters: they have
letters or spellings nobody else uses. At 10 characters Urdu drops to 70.1%, read as
Persian as often as Punjabi.

**`margin` is a warning light, not a probability.** On 20-character test windows, the 29
guesses with a margin under 0.1 were right 55% of the time — but most errors (31 of 44)
came with a margin above it. Naive Bayes is confidently wrong when two languages share
every word in a short window. An earlier docstring claimed almost every error had a margin
below 0.05; the measurement said otherwise, and the docstring was changed.

Accuracy per character count is the honest summary: give it a sentence and it is
reliable; give it a word and it is guessing between neighbours.

## 11. English inside Roman Urdu

`tag_roman_tokens` labels each word `ur` or `en`. Each language gets a word-frequency table
interpolated with a Witten-Bell character 4-gram model, and a two-state Viterbi pass
smooths over the sentence. The most frequent words are the ambiguous ones — `is`, `to`,
`me`, `the` are words in both — so context is doing real work.

English frequencies come from HotpotQA's English Wikipedia sentences. Roman Urdu has no
running text to count, so its frequencies are **estimated**: each Urdu word's corpus
frequency is spread across the romanisations annotators wrote for it, so `ke` inherits
the frequency of کے. On dev, the first version used character shape alone and found 60%
of English words at a 5.8% false-alarm rate; adding the frequency tables raised recall to
87% at 3.6%.

Test results (chosen on Dakshina dev and HotpotQA validation):

| set | measure | test |
|---|---|---:|
| Dakshina romanised Urdu, 86,343 words | tagged `en` | 4.0% |
| HotpotQA English, 188,522 words | tagged `ur` | 1.6% |
| synthetic code-mix (1-3 English words spliced in) | token accuracy | 95.0% |
| synthetic code-mix | English recall | 87.3% |

**The 4.0% overstates the error, and by how much was checked by hand.** Dakshina's
annotators left names and English words in English spelling. Of the 40 most frequently
flagged "Urdu" words: **26 are English** words or names spelled the English way (*the,
of, county, website, degree, Germany, Robert, Scotland*), **6 are single-letter
initials**, and **8 are real errors** on Urdu words — `o` (و), `is` (اس), `to` (تو), `ne`
(نے), `masjid`, `markazi`, `abdul`, `hindustani`. The function words among those are the
ones context could not rescue; the content words are Urdu words spelled in a way the
estimated Roman frequency table never saw.

The code-mix set is synthetic by construction: real code-switching happens at grammatical
boundaries, and a random splice does not. It measures whether an English span can be
found, not how people switch.

## 12. Stemming helps Urdu retrieval — a little

`scripts/measure_stemmer.py` scores `stem` on two retrieval tasks over 5,016 Urdu
Wikipedia articles with BM25, queries split by hash into validation (2,433) and test
(2,583). The setting was chosen on validation; test is reported with a sign test over the
queries that changed.

| task | plain words | stemmed | change | sign test |
|---|---:|---:|---:|---:|
| title → body | 0.4553 | 0.4634 | **+0.008** | p = 0.019 |
| lead sentence → rest of article | 0.6080 | 0.6128 | +0.005 | p = 0.32 |

The first task is nlp-lab project 23's benchmark, reused so the numbers compare. It
barely exercises a stemmer: titles are mostly names, and names do not inflect. The second
task exists because of that — a lead sentence is ordinary prose, full of the endings a
stemmer removes — and it moved less, not more. Stemming shrinks the index by 15% (126,502
→ 107,303 types) and gains less than a point. Project 23 found subword tokenisation worth
+0.039 on the same title task, five times as much.

Both levels (`light=True` and the default) and both minimum stem lengths were tried; the
default won validation on both tasks.

## 13. Numbers

There is no corpus measurement here, because the property that matters is exact: every
word from 0 to 99 is irregular in Urdu, so a wrong entry is a wrong number, not a lower
score. What was checked instead:

- **Every one of the hundred words is attested** in the Urdu Wikipedia counts at least
  three times. Two alternative spellings that were in the first draft (تیتیس, تیتالیس)
  were not, and were removed.
- **Round trip**: `parse_number(number_to_words(n)) == n` for every n below 200,000 and
  20,000 random n up to 10¹³.
- Two bugs the round trip could not catch, because `number_to_words` never produces the
  forms that trigger them — both found by writing tests from how people write numbers:
  پانچ سو تیس (530) was rejected as "two numbers in a row", and ایک ہزار کروڑ (a thousand
  crore) was read as 10,010,000,000.

---

## What this does not measure

**One corpus, one register, for sections 1-6.** XL-Sum Urdu is edited BBC news prose.
Social media, legal text, poetry and transcribed speech all differ, and the
Arabic-substitution rate is very likely *higher* in user-generated text than the 8.9%
measured in professional copy.

**Roman Urdu from Wikipedia, not from chat.** Dakshina's romanisations were written by
annotators transcribing encyclopaedia sentences. People texting write shorter words,
drop more vowels and switch to English more often. Section 7's 87.0% is a figure for
careful romanisation; typed chat will score lower, by an amount nobody has measured.

**Every language model trained and tested on Wikipedia.** Section 10's accuracies are
for encyclopaedia prose. Wikipedia in the smaller languages is heavy with bot-written
stubs, which makes paragraphs within one language unusually alike.

**No context.** Transliteration picks each word on its own, which is why `ke` is always
کے. That is the largest remaining error and the one this design cannot fix.
