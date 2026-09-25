"""Urdu stopwords.

Function words: pronouns, postpositions, auxiliaries, conjunctions. Removing them
helps bag-of-words models and retrieval, and **hurts** anything that depends on
syntax or negation - sentiment especially, where نہیں carries the entire meaning.

So `NEGATION` is kept as a separate set and excluded from `STOPWORDS` by default.
A stopword list that silently deletes negation is a bug that looks like a feature.
"""

from __future__ import annotations

from .normalize import _require_str, _require_words, normalize

# Deliberately conservative: content words are never included, because a stopword
# list is far more damaging when it is too aggressive than when it is too small.
_BASE = """
میں مجھے مجھ میرا میری میرے ہم ہمیں ہمارا ہماری ہمارے
تم تمہیں تمہارا تمہاری تمہارے تو تیرا تیری آپ
وہ اس اسے اسکا اسکی اس کا اس کی ان انہیں انکا ان کا ان کی
یہ ہے ہیں ہوں ہو ہوا ہوئی ہوئے ہوتا ہوتی ہوتے ہونا
تھا تھی تھے تھیں گا گی گے رہا رہی رہے
کا کی کے کو سے پر میں تک لیے لئے ساتھ بغیر بارے وجہ
اور یا لیکن مگر پھر بھی ہی تو کہ اگر جو جب تب چونکہ
کیا کیوں کیسے کہاں کب کون کتنا کتنی کتنے کس
ایک دو کچھ سب کوئی ہر بہت زیادہ کم تھوڑا
کر کرنا کرتا کرتی کرتے کیا گیا گئی گئے
اب یہاں وہاں ابھی پہلے بعد دوران اوپر نیچے اندر باہر
"""

# Kept out of STOPWORDS on purpose - see the module docstring.
NEGATION = frozenset({"نہیں", "نہ", "مت", "بغیر", "بنا"})

STOPWORDS = frozenset(normalize(w) for w in _BASE.split() if w.strip())


def is_stopword(word: str, *, include_negation: bool = False) -> bool:
    _require_str(word, "is_stopword")
    word = normalize(word)
    if word in NEGATION:
        return include_negation
    return word in STOPWORDS


def remove_stopwords(tokens: list[str], *, include_negation: bool = False) -> list[str]:
    """Filter a token list.

    `include_negation=True` removes negation too. Do that only for topic modelling
    or retrieval - never for sentiment, where it inverts the label.
    """
    _require_words(tokens, "remove_stopwords")
    return [t for t in tokens if not is_stopword(t, include_negation=include_negation)]
