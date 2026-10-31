# identify_language next to two general-purpose detectors

The task is Urdu against the ten languages it is **actually** confused with: Shahmukhi
Punjabi, Saraiki, Sindhi, Pashto, Kashmiri, Persian, Arabic, Sorani Kurdish, Uyghur and
South Azerbaijani. All ten are written in the Perso-Arabic script. Telling Urdu from
English is not the problem. Telling it from Saraiki is.

The baselines are what a project reaches for when it needs language ID:

- **`langdetect` 1.0.9**, the port of Google's language-detection library, 55 languages.
- **fastText `lid.176.ftz` 0.9.2**, 176 languages, the usual off-the-shelf choice.

Measured 2026-10-29 on the **test** split of the 132,000-paragraph Wikipedia corpus -
**6,136 paragraphs over 11 languages** - whole and cut to 50, 20 and 10 characters. Both
sides are scored on the same rows in the same run, and this tool's figure reproduces the
0.9925 published in the README from a separate script, which is the check that the harness
scores the same way. Every number is in
[`langid-comparison.json`](langid-comparison.json), written by
`scripts/compare_langid.py`; `tests/test_langid_comparison.py` checks this page against it.

## The first number is not accuracy, it is coverage

A detector with no label for Saraiki cannot be right about Saraiki, and scoring it as
wrong does not mean much either. So coverage comes first, because it decides how to read
everything below:

| | labels in the model | labels for this task | no label for |
|---|---:|---:|---|
| **urdunlp** | 11 | **11 of 11** | — |
| fastText `lid.176` | 176 | 9 of 11 | **Saraiki, Kashmiri** |
| `langdetect` | 55 | **3 of 11** | Punjabi, Saraiki, Sindhi, Pashto, Kashmiri, Sorani, Uyghur, South Azerbaijani |

`langdetect` has a label for three of the eleven. fastText covers nine and still misses
**Saraiki and Kashmiri** - two of the three languages Urdu is most confusable with.

## Accuracy

| | whole | 50 chars | 20 chars | 10 chars |
|---|---:|---:|---:|---:|
| **urdunlp** | **0.9925** | **0.9798** | **0.9260** | **0.7909** |
| fastText, all 11 languages | 0.7140 | 0.6410 | 0.5106 | 0.4379 |
| fastText, where it has a label | 0.8731 | 0.7838 | 0.6244 | 0.5355 |
| `langdetect`, all 11 languages | 0.2699 | 0.2678 | 0.2518 | 0.2226 |
| `langdetect`, where it has a label | 0.9982 | 0.9904 | 0.9313 | 0.8234 |

Read the two `langdetect` rows together. On the three languages it knows it is **excellent**
- 0.9982 on whole paragraphs, better than this tool - and across the task it is 0.2699,
because eight of the eleven classes are not in its label set. That is not a flaw in
`langdetect`. It is the whole argument for a task-specific model: the general tool is very
good at a different, easier question.

Where fastText *does* have the label it still trails, 0.8731 against 0.9925, and the gap
widens as the text gets shorter - 0.5355 against 0.7909 at ten characters.

## What a general detector says instead

This is the part that matters for anyone processing Pakistani web text. For the languages
it has no label for, `langdetect` does not abstain - it answers:

| true language | `langdetect` says | of |
|---|---|---:|
| Shahmukhi Punjabi | **`ur` 523**, `ar` 12, `fa` 2 | 537 |
| Saraiki | **`ur` 543**, `fa` 4, `ar` 2 | 549 |
| Kashmiri | **`ur` 568**, `fa` 1 | 569 |
| Sindhi | `fa` 329, `ur` 162, `ar` 19 | 510 |
| Pashto | `fa` 560, `ar` 15, `ur` 3 | 578 |

**Punjabi, Saraiki and Kashmiri are labelled Urdu 97%, 99% and 99.8% of the time.** A
pipeline that filters "Urdu" with `langdetect` collects three other languages almost in
full, and nothing in its output says so.

fastText, with nine labels, spreads the two it lacks across its nearest neighbours:

| true language | fastText says | of |
|---|---|---:|
| Saraiki | `pnb` 401, `ur` 140, `fa` 8 | 549 |
| Kashmiri | `ur` 310, `pnb` 186, `fa` 73 | 569 |

## What this does not show

- **The baselines are not Urdu tools and do not claim to be.** `langdetect` advertises 55
  languages and delivers them; neither model was trained for this task, and "no label for
  Saraiki" is a design decision about scope, not an error.
- **The corpus is Wikipedia, and it is this tool's own training distribution.** The model
  was built on the train split of the same corpus, so it has a home-ground advantage that
  the baselines do not. The split is by paragraph with a fixed seed and the test rows were
  never trained on, but the *domain* match is real and it flatters this tool. A test set
  from a different domain would be a better comparison and does not exist yet.
- **The corpus is not in the repository** (132,000 paragraphs, gitignored).
  `scripts/fetch_wikipedia_samples.py` rebuilds it and `data/wiki/` holds the committed
  1,500-per-language sample, on which the same comparison gives 0.9825 / 0.680 / 0.297.
- **Short-text accuracy is a property of the cut, not of real short text.** Ten characters
  of a Wikipedia paragraph is not a ten-character query; it is a truncation, and real short
  text has different statistics.
- **One run, one seed, no confidence intervals.** The per-language counts above are single
  numbers over 500-580 paragraphs each; nothing here is a significance claim.
