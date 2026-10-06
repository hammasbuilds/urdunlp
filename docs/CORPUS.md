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

### Which corpus, exactly

`data/` is gitignored - it is 70 MB of other people's text - so until now every number in
this document rested on files no reader could see. Two small files, both committed, fix
what can be fixed:

**`data/FINGERPRINT.tsv`** records a hash, line count and character count for all 23
corpus files the published numbers were measured over. `python
scripts/corpus_fingerprint.py --check` answers the question that mattered and could not be
asked: *is the corpus on this disk the one these numbers describe?* A hash cannot rebuild
a corpus, but a mismatch tells you the figures do not describe what you have. The hashes
are over sorted lines, because paragraph order is not part of a corpus's identity here -
the train/val/test split is by content hash and n-gram counts are order-independent.

**`data/wiki/<code>.pages.tsv`** records the Wikipedia article id and title behind every
paragraph, and `fetch_wikipedia_samples.py --from-pages` re-fetches exactly those. This
was the real hole: the draw is the server's `generator=random`, so **no seed on our side
can reproduce it**, and before the ids were recorded a re-run produced a different corpus
every time. Verified on Central Kurdish: delete the corpus file, rebuild from the ids
alone, and the paragraph set is identical - and two rebuilds are byte-for-byte identical,
which is the property that matters.

**The corpus behind the currently shipped models predates the id recording**, so it is
pinned by fingerprint and cannot be re-fetched. That is stated rather than glossed: a
reader can confirm they have the right corpus, and cannot obtain it from scratch. How this
was discovered is the clearest illustration of the cost - `data/wiki/ur.txt` was grown
from 820k to 1.6M characters during a review, which left the shipped `langid.json.gz`
trained against a file that no longer existed on disk, and nothing in the repository could
have told anyone.

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

## 4. Round trips: what 0.1 lost, and what 0.2 recovers

*Written for 0.1, and the first half still describes 0.1: the letter-by-letter
Urdu → Roman rules, and a Roman → Urdu direction with no vocabulary. The 0.2 box at the
end of the section and section 14 have the current figures. In 0.2 the default setting
is no longer the worse one, and the round trip is 93.8%.*

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
> 15,088 tokens sampled from 9,759 held-out Dakshina sentences:
>
> | `insert_short_vowels` | 0.1 pipeline (lexicon, rules) | 0.2 pipeline (+ vocabulary) |
> |---|---:|---:|
> | `True` (default) | 42.0% | **90.9%** |
> | `False` | 61.7% | **92.7%** |
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

## 7. Transliteration, scored against people: 43.1% → 91.3%

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
| 0.1: lexicon, then rules | 43.4% | **43.1%** |
| 0.2, each word on its own (`use_context=False`) | 88.9% | **88.5%** |
| 0.2, each sentence decoded as a whole (default) | 92.0% | **91.3%** |

Where the 0.2 answers come from, on test:

| stage | share of words | right |
|---|---:|---:|
| curated lexicon | 33.6% | 97.0% |
| vocabulary | 66.1% | 88.8% |
| rules | 0.3% | 3.9% |

The 0.1 figure is worth dwelling on. Over all 148,591 word pairs in Dakshina's aligned
file, **two thirds of running text fell through to the rules, and the rules got 16.5% of
it right.** Two mechanical errors dominated — a word-initial vowel with no carrier (`is` →
یس instead of اس, `aik` → ےک instead of ایک) and short vowels written out that Urdu does
not write (`jis` → جیس instead of جس, `karne` → کارنے instead of کرنے).

### How it got there, one measured step at a time (dev sentences)

| step | dev |
|---|---:|
| 0.1: lexicon, then rules | 43.4% |
| + vocabulary: letter channel × word frequency, 42,498 words | 87.3% |
| + decode the sentence with a word bigram model | 89.8% |
| + 60,638 words instead of 42,498 | 90.0% |
| + each word's own attested spellings mixed into the channel | 91.1% |
| + the Arabic article moved across the word boundary; initials, titles, `o` | 91.8% |
| + کہ offered for `ki`, `ke`, `kay`, `keh` | 91.9% |
| + chat spellings (`pata`, `gaari`, `h`, ...), the merged future (`karunga`): 5 more words right, 91.94% → 91.95% | **92.0%** |

**The letter channel.** Choose the Urdu word *u* that maximises P(*u*) · P(roman | *u*).
P(roman | *u*) comes from a letter-emission model — each Urdu letter emits 0-4 Roman
characters, conditioned on its position and on whether the next letter is a vowel letter —
trained with EM on the 106,260 attested pairs of Dakshina's **training** lexicon.
Candidates are retrieved by a coarse consonant key both scripts map to, with `h` dropped
entirely and the letters Roman cannot tell apart (س ص ث ش, ت ط ٹ, ز ذ ض ظ ج) in one class.
The gold word's key is among the Roman word's keys for **98.3%** of test spellings — that
is the ceiling.

**Context.** After the channel, the right word was the top candidate 87.3% of the time and
among the top five 93.6% of the time — six points a context model could reach. The
largest single error was `ke`: کے (*of*) or کہ (*that*), which the curated lexicon always
read as کے, wrong 223 times on dev. The sentence is now decoded by Viterbi over each word's
five best candidates plus its lexicon entry, scoring P(roman | word) against a word bigram
model — interpolated absolute discounting over the same decontaminated Wikipedia text,
344,258 bigrams seen at least three times.

That alone did not fix `ke`. The letter channel rated `ke` as a spelling of کہ at
log-probability −6.6, because a final ہ is rarely typed as `e` — true of the letter, false
of this word, which annotators wrote as `ke`. **Mixing each word's own attested spellings**
into the channel (for the ~24,000 words of the training lexicon, backing off to the letter
model with weight κ) put کہ within reach, and the bigram model does the rest: after کہا
(*said*), کہ.

**Word accuracy on the lexicon** — every spelling weighted by how many annotators wrote it:

| | train (106,260) | dev (10,424) | test (10,517) |
|---|---:|---:|---:|
| 0.1 | 10.7% | 10.0% | 11.8% |
| 0.2 | 81.0% | 63.8% | 63.6% |

The 0.2 train figure is not comparable with the other two and is shown so that nobody
mistakes it for one: the training lexicon's own spellings are now inside the model, so a
training word is partly looked up. Dev and test words are disjoint from training words.
Both are far lower than on sentences, as expected — the lexicon samples words across the
vocabulary, so the rare words that the priors help least are over-represented compared
with running text.

**What was chosen on dev, and how.** Word by word, vocabulary size (84,783 / 60,638 /
42,498 words) moved dev accuracy by less than 0.3 points, and the prior weight 1.0 (plain
Bayes) beat 0.5 by 0.8. With context: bigram weight 0.3 → 88.6%, 0.6 → 89.8%, 1.0 → 89.5%;
bigrams kept from count 2 / 3 / 5 / 10 → 89.6 / 89.5 / 89.3 / 89.0% (at weight 0.45); κ 1 /
3 / 10 / 30 / 100 → 90.7 / 91.0 / 91.1 / 91.0 / 90.9%; five candidates per word (three lost
0.2, ten gained 0.05). Conditioning letter emissions on the next letter improved the
training log-likelihood after six EM iterations from −786,736 to −752,577 and dev accuracy
by only 0.3 points. Each extra letter class in the key (`g` as ج for *germany*, `s` as ز,
`c` as س, `m` as ن) was kept only because removing it cost dev accuracy; `z` as س and `d`
as ت changed nothing and were dropped.

**A regression found by the round trip.** Decoding a lone word as a one-word sentence
scores it by how often words *start* sentences, and the section 4 round trip — one word at
a time — fell from 90.8% to 88.2% when context was switched on. A single word now takes
the word-by-word path; there is no context to use.

**The Arabic article.** A new breakdown of the remaining dev errors put 1,565 under "the
right word is in the vocabulary but under a different key" - and most of those were one
pattern. Roman Urdu writes the article on the word before it (`abdul rehman`, `bainul
aqwami`, `darul uloom`); Urdu writes it on the word after (عبد الرحمن, بین الاقوامی, دار
العلوم). Neither half of the pair can be right word by word: there is no Urdu word for
`abdul`, and `rehman` is not الرحمن. That pattern alone was 441 dev errors. A pre-pass
now rewrites a pair (`abdul rehman` → `abd alrehman`) when a real ال-word exists for the
second half and the rewritten pair scores better by a margin (chosen on dev: margin 2 →
+0.54 points; −2 → +0.50; 8 → +0.45). `kabul`, `rasul` and `phool` end in -ul as well and
are left alone, because nothing scores better. Capital initials are spelled by letter name
(Dakshina: 115 of 127 capital single letters), a title's dot is dropped and no longer ends
the sentence, and lowercase `o` is the conjunction و (123 of 127).

**کہ.** `ki` and `ke` meaning کہ (*that*) were still 118 dev errors after context, because
the training lexicon barely attests those spellings for کہ, so it never reached the
decoder's candidates. Offering it for `ki`, `ke`, `kay` and `keh` at the best candidate's
emission, and letting the bigram model decide, added 0.18 points on dev; widening the list
to other function-word readings (`na` → نا, `ya` → یہ, `ki` → کے) cost up to 0.47.

**Tried and not kept: generating words the vocabulary does not have.** 1,052 dev words
have a gold spelling outside the 60,638-word vocabulary. Running the letter model in
reverse with an Urdu character model generated the right spelling for 18.1% of them
(34.8% in its top five), against 6.7% for the rules. As an extra candidate in the decoder
it cost 0.31 points overall: an invented word that looks plausible wins against real
ones too often, and there were not enough out-of-vocabulary words for it to pay that back.

**What is left** — 8.7% of test words. On dev, of the words still wrong after context was
added (measured before the attested spellings, at 90.0%):
2,053 had the right word among the five candidates and context chose another, 2,049 had
it in the vocabulary but outside the five, and 1,095 had it outside the vocabulary
altogether — English words and names spelled the English way (*Neptune*, *Texas*,
*paradise*) and Arabic phrases (*sallallahu alaihi wasallam*). The package is 3 MB rather
than 1 MB because of the bigram table; that is the price of the 3.8 points context and
the attested spellings add.

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
| **`roman_key`** | 0.932 | 0.750 | **0.831** |

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

`identify_language` is a character 1-5-gram naive Bayes model, one table per language,
trained on 80% of each sample (split by paragraph hash, 80/15/5). Urdu, Punjabi and
Saraiki - the three that get confused - have 5,000 paragraphs each; the other eight have
1,500.

| text | first version, same test | val (4,109) | **test (1,254)** | Urdu, test |
|---|---:|---:|---:|---:|
| whole paragraph | 96.4% | 97.4% | **97.9%** | 98.7% |
| 50 characters | 94.6% | 95.9% | **96.6%** | 98.0% |
| 20 characters | 89.8% | 90.4% | **91.0%** | 93.3% |
| 10 characters | 81.5% | 81.2% | **81.5%** | 84.0% |

*First version: 1-3-grams, 1,500 paragraphs per language.*

**These numbers are lower than the ones this page first published, and the model is
better.** The first version was scored on 771 test paragraphs - 57 to 80 per language -
and reported 99.1% on a paragraph. Growing the three closest languages to 5,000 paragraphs
grew their test sets as well, and the first version, unchanged, scored 96.4% on the larger
set. Punjabi fell from 95.8% to 89.3%: the small test set had been kind to it. On the
larger set every change below is a gain.

- **1-3 → 1-5-grams.** On validation windows 4-grams added 1.7 points at 20 characters and
  5-grams 3.3, unpruned; 6-grams added 0.3 more for another 4.5 MB. All 5-grams are
  5.4 MB, so each language keeps its 20,000 most frequent - a top-K cut beat a
  minimum-count cut of the same size.
- **1,500 → 5,000 paragraphs of Urdu, Punjabi and Saraiki.** On the same enlarged test set:
  paragraph 96.4 → 97.9%, 50 characters 94.6 → 96.6%, 20 characters 89.8 → 91.0%, 10
  characters level. Punjabi on a paragraph: 89.3 → 97.3%. The cost is Urdu on short
  windows: at 10 characters on test it went from 87.1% to 84.0%, because a stronger
  Punjabi model claims more of them.
- **A prior for Urdu, tried and dropped.** Adding 2, 4 or 8 nats to Urdu's score moved its
  10-character validation accuracy from 87.5% to 88.6%, 89.8% and 92.2% - and took the same
  or more from Punjabi, lowering overall accuracy each time. There is no free point here.

The errors are almost all between the three closest languages. At 20 characters, Saraiki
is right 75.1% of the time (mostly read as Punjabi), Punjabi 85.7% and Urdu 93.3%. Arabic,
Central Kurdish, Sindhi and Uyghur are 100% at 20 characters: they have letters or
spellings nobody else uses.

**`margin` is not confidence.** With 1-3-grams, the 29 test windows (of 771, at 20
characters) with a margin under 0.1 were right 55% of the time, and most errors still had
a margin above it. With 5-grams the scores spread further: on the enlarged test set 1,239
of 1,254 windows have a margin above 0.1, and so do 102 of the 113 errors. Naive Bayes is
confidently wrong when two languages share every word in a short window. An earlier
docstring claimed almost every error had a margin below 0.05; the measurement said
otherwise, twice, and the docstring now says to read length, not margin.

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
| Dakshina romanised Urdu, 86,343 words | tagged `en` | 3.8% |
| ...of those, the 68,902 the Roman Urdu lexicon attests | tagged `en` | **1.9%** |
| HotpotQA English, 188,522 words | tagged `ur` | 1.8% |
| synthetic code-mix (1-3 English words spliced in) | token accuracy | 95.1% |
| synthetic code-mix | English recall | 87.0% |

**Why there are two rows for the same measurement.** The 3.8% is an upper bound, and
roughly half of it is the reference rather than the tagger. Decomposing the 3,314 tokens
it counts on test, against the shipped unigram tables:

| | share |
|---|---:|
| in the English table and **absent from the Roman Urdu lexicon** | 45.6% |
| in both — a genuine homograph (`the`, `in`, `of`, `is`, `a`, `new`, `film`, `school`) | 36.8% |
| in neither | 17.5% |

Dakshina's romanised side comes from Wikipedia and really does contain `county` (61),
`website` (36), `germany` (34), `carolina` (30), `scotland` (23) — and its gold label for
every token on that side is `ur`, by construction. A token with no Roman-Urdu lexical
evidence at all is a reference-label problem, so the first row includes cases where
tagging `en` was *right*. The second row restricts the same count to tokens the lexicon
attests, which is the closest thing here to the tagger's own precision. Both are
published; neither alone is honest.

**A real defect underneath, recorded and not fixed.** The two tables are not symmetric.
`masjid` is in the English table (5) and absent from the Roman Urdu one, as are `markazi`
(5), `abdul` (34), `ahmed` (62), `hyderabad` (32) and `afghanistan` (112). English
frequencies come from HotpotQA's Wikipedia prose, which is full of South Asian proper
nouns; the Roman Urdu side is capped at 30,000 Dakshina types that do not include them, and
only 3,088 of the 30,000 English entries appear in it at all. So `masjid` *cannot* be
tagged `ur` by the unigram term — it is not a word the model knows in Urdu. That is a
directional modelling problem rather than the irreducible homograph one, and fixing it
means changing what goes into the tables, not the inference.

*Re-measured with `python scripts/measure_langid.py` after the last chat-lexicon fixes
(single-letter chat words such as `h` and `g` are now Urdu); an earlier draft showed
4.0%, 1.6%, 95.0% and 87.3%.*

**The 3.8% overstates the error, and by how much was checked by hand.** Dakshina's
annotators left names and English words in English spelling. Of the 40 most frequently
flagged "Urdu" words: **27 are English** words or names spelled the English way (*the,
of, county, website, degree, Germany, Robert, Scotland*), **6 are single-letter
initials**, and **7 are real errors** on Urdu words — `is` (اس), `to` (تو), `ne`
(نے), `masjid`, `markazi`, `abdul`, `hindustani`. The function words among those are the
ones context could not rescue; the content words are Urdu words spelled in a way the
estimated Roman frequency table never saw.

**Tried, and not kept.** The Roman Urdu frequency table covers only words Dakshina's
annotators spelled. Adding a generated spelling for every other word in the 60,638-word
vocabulary - 34,973 more spellings, weighted by frequency - moved the dev false-English
rate from 3.63% to between 3.61% and 3.74%, and English recall by under half a point. The
remaining false alarms are mostly English, as the hand count above shows, so there was
little left for coverage to fix. It added 71 seconds to the build and nothing to the
result, so it was not shipped.

The code-mix set is synthetic by construction: real code-switching happens at grammatical
boundaries, and a random splice does not. It measures whether an English span can be
found, not how people switch.

## 12. Stemming helps Urdu retrieval — a little

`scripts/measure_stemmer.py` scores `stem` on two retrieval tasks over 5,009 Urdu
Wikipedia articles with BM25, queries split by hash into validation (2,427) and test
(2,582). The articles are the first six row groups (6,000 articles, of which 5,009 have
60 tokens of body) of the public `wikimedia/wikipedia` parquet, config `20231101.ur`:

    python scripts/measure_stemmer.py train-00000-of-00001.parquet --row-groups 6

Test is reported for the shipped default, `stem(word)` (full suffix list, minimum stem
3), with a sign test over the queries that changed.

| task | plain words | stemmed | change | sign test |
|---|---:|---:|---:|---:|
| title → body | 0.4566 | 0.4648 | **+0.008** | p = 0.019 |
| lead sentence → rest of article | 0.6090 | 0.6126 | +0.004 | p = 0.47 |

*Re-measured on the current tokeniser. An earlier draft (5,016 articles, 2,583 test
queries) gave 0.4553 → 0.4634 and 0.6080 → 0.6128. The tokeniser changed after that run
(`words` keeps numbers like `2.5` whole and merges more split verbs) and seven articles
now fall under the 60-token floor; the gain from stemming on the title task is unchanged.*

The first task is nlp-lab project 23's benchmark, reused so the numbers compare. It
barely exercises a stemmer: titles are mostly names, and names do not inflect. The second
task exists because of that — a lead sentence is ordinary prose, full of the endings a
stemmer removes — and it moved less, not more. Stemming shrinks the index by 15% (127,284
→ 108,066 types) and gains less than a point. Project 23 found subword tokenisation worth
+0.039 on the same title task, five times as much.

Both levels (`light=True` and the default) and both minimum stem lengths were tried. The
default won validation on the lead-sentence task and tied `light=True` on the title task
(0.4454 each to four places); `light=True` would score +0.011 on title test (p = 0.002)
and +0.001 on lead test. That is one tie on validation, not a reason to change the
default, so the table reports what `stem()` does.

**Tried and not kept: a vocabulary-checked stemmer**, which strips a suffix only when what
is left - or it plus ا, ی, ہ or نا - is one of the 60,638 known words. The idea was that
the small gain came from over-stripping. It did not (measured on the earlier tokeniser,
against the earlier figures; the experiment is not in the scripts): title retrieval 0.4447 validation /
0.4638 test against the plain stemmer's 0.4455 / 0.4634, lead sentence 0.5991 / 0.6124
against 0.6004 / 0.6128. Stemming is simply worth little to this kind of retrieval.

## 13. Numbers

There is no corpus measurement here, because the property that matters is exact: every
word from 0 to 99 is irregular in Urdu, so a wrong entry is a wrong number, not a lower
score. What was checked instead:

- **Every one of the hundred words is attested** in the Urdu Wikipedia counts at least
  three times. Two alternative spellings that were in the first draft (تیتیس, تیتالیس)
  were not, and were removed.
- **Round trip**: `parse_number(number_to_words(n)) == n` for every n below 200,000 and
  20,000 random n up to 10¹³.
- **Ordinals** (`parse_ordinal`, and `ordinal=True` in `find_numbers`): every form
  accepted was checked against the corpus counts first. The regular endings واں and ویں
  are attested on every regular cardinal checked - 5, 7, 8, 10 to 20, 25, 30 and 50 -
  and on ہزار and لاکھ (بیسویں 321
  times, پانچویں 432); the ending -وی was left out because ہزاروی is mostly the surname
  Hazarvi. پہلے is "before" far more often than "first" (11,584 occurrences), so on its
  own it is not reported as an ordinal - nor are دوسرا/دوسری/دوسرے, which usually mean
  "other".
- Two bugs the round trip could not catch, because `number_to_words` never produces the
  forms that trigger them — both found by writing tests from how people write numbers:
  پانچ سو تیس (530) was rejected as "two numbers in a row", and ایک ہزار کروڑ (a thousand
  crore) was read as 10,010,000,000.

---

## 14. Urdu → Roman, scored against people

`transliterate_to_roman` had only ever been measured by round trip, which says whether the
Roman can be read back, not whether a person would write it. Dakshina answers the second
question: a spelling is right if an annotator wrote exactly it.

| | dev | **test** |
|---|---:|---:|
| **lexicon words** (10,424 / 10,517) spelled as some annotator spelled them | | |
| 0.1 rules, short vowels inserted | 33.1% | **33.4%** |
| 0.1 rules, literal | 22.2% | 23.7% |
| 0.2 learned | 54.5% | **54.3%** |
| **sentence words** (51,764 / 52,087) spelled exactly as that annotator did | | |
| 0.1 rules, short vowels inserted | 29.1% | **28.6%** |
| 0.2 learned | 55.0% | **54.5%** |
| **sentence words** spelled as any annotator spelled that word | | |
| 0.1 rules, short vowels inserted | 41.5% | **41.4%** |
| 0.2 learned | 77.6% | **78.2%** |

The learned speller takes, in order: the curated lexicon's spelling (میں is `main`, not
the rules' `min`); the commonest spelling annotators wrote, for the ~24,000 words of the
training lexicon; and otherwise a spelling generated from the same letter emissions the
Roman → Urdu direction uses - a beam over each letter's likeliest Roman strings, rescored
by the full P(roman | urdu) and a Roman Urdu character model. The character model's weight
was chosen on dev: 0 → 45.8% of dev lexicon words, 0.3 → 53.9%, 0.6 → 52.7%, 1.0 → 48.3%.
Dev and test lexicon words are disjoint from training-lexicon words, so the lexicon rows
measure the generator, not the lookup.

**It also round-trips better.** Converting 15,088 held-out tokens to Roman and back:

| Roman spelling used | back to Urdu |
|---|---:|
| 0.1 rules, short vowels inserted | 90.9% |
| 0.1 rules, literal | 92.7% |
| **0.2 learned** | **93.8%** |

An earlier draft of this table said 94.3%, from an older run that was not repeated when
the decoder changed afterwards. Re-measured for the release it was 93.9%, and after the
last chat-lexicon fixes (`JazakAllah`, `acha g`, `yr`) it is 93.8% (0.9378). To re-run it:
`python scripts/measure_corpus.py data/dakshina/ur/romanized --every 11` - that directory's
two `.txt` files are Dakshina's dev and test sentences, 9,759 of them. The last round
of changes (chat spellings, the merged future, punctuation) is not the cause: the code
before them scores the same on the same tokens, and only one distinct token in the
sample changed its Roman spelling (پتہ is now `pata`). The token count moved too, from 15,098
to 15,088, because `words` now keeps a number like `2.5` whole, and every eleventh token
is sampled.

**A second small drift, from the same cause.** The word-accuracy, spelled-exactly and
spelled-as-any figures above moved by 0.1-0.6 points between drafts of this page (test
word accuracy 91.2% → 91.3%, sentence-exact 54.7% → 54.5%, sentence-any 77.9% → 78.2%,
lexicon 54.1% → 54.3%) when `python scripts/measure_translit.py` was re-run for this
release against the current candidate-scoring code. None of it changes a conclusion this
page draws - the ordering of every comparison and every "X points" delta in the prose
above still holds - so the numbers were updated in place rather than argued over.

0.1 documented a trade-off: insert short vowels for a reader, leave them out for a
machine. The learned spelling beats both at both, because it is the spelling the
Roman → Urdu model was trained to read.

Found on the way: the rules dropped every Urdu digit. They kept only ASCII they did not
recognise, and ۱۲۳ is not ASCII.

## What this does not measure

**One corpus, one register, for sections 1-6.** XL-Sum Urdu is edited BBC news prose.
Social media, legal text, poetry and transcribed speech all differ, and the
Arabic-substitution rate is very likely *higher* in user-generated text than the 8.9%
measured in professional copy.

**Roman Urdu from Wikipedia, not from chat.** Dakshina's romanisations were written by
annotators transcribing encyclopaedia sentences. People texting write shorter words,
drop more vowels and switch to English more often. Section 7's 91.3% is a figure for
careful romanisation; typed chat will score lower, by an amount nobody has measured.

**Every language model trained and tested on Wikipedia.** Section 10's accuracies are
for encyclopaedia prose. Wikipedia in the smaller languages is heavy with bot-written
stubs, which makes paragraphs within one language unusually alike.

**One word of context.** The transliterator reads each word with the word before it and
no further. `ke` after کہا is کہ; `ke` whose deciding word is three words back is still a
guess, and `sher` is شیر or شعر whatever precedes it.
