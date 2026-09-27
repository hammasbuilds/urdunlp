"""Transliteration between Roman Urdu and Urdu script.

Roman Urdu is how most Pakistanis actually type Urdu - in messages, comments, reviews
and support tickets. It is also close to unprocessable by standard NLP tooling,
because it has **no standard orthography**. One word has many spellings:

    نہیں  ->  nahi, nahin, nhi, nahee, naheen
    ہے    ->  hai, hay, he, h
    کیا   ->  kya, kia, kaya

and the same Roman string can map to different Urdu words:

    khana  ->  کھانا (food / to eat)   or   خانہ (compartment, in compounds)
    sher   ->  شیر (lion)              or   شعر (couplet)

So this module does **not** pretend to be a solved problem. It works in three stages
and reports which one produced each answer:

  1. **Lexicon.** A curated map of high-frequency words, covering the closed-class
     vocabulary - pronouns, postpositions, auxiliaries, common verbs - which is where
     most tokens in real text actually are.
  2. **Vocabulary.** A noisy-channel search over 60,638 real Urdu words: the Urdu word
     most likely to have been typed as this Roman string, weighing how people spell
     it against how likely the word is after the word before it. This is where `baad`
     finds بعد, `taur` finds طور and `ali` finds علی - letters Roman cannot write,
     recovered because the word that contains them exists and the rule-built one does
     not - and where `ke` after `kaha` becomes کہ rather than کے. See `_channel.py`.
  3. **Rules.** Longest-match grapheme substitution for whatever is left - mostly
     names and rare loanwords the vocabulary has never seen.

Scored against 52,087 words of hand-romanised Urdu Wikipedia sentences held out
from everything the model was built from (Dakshina's test split), the first and
last stages alone - lexicon, then rules - get 43.1% of words exactly right. With
the vocabulary stage choosing each word on its own, 88.5%; choosing the sentence
as a whole, 91.2%. The numbers, and how they were measured, are in docs/CORPUS.md.

`transliterate_to_urdu` returns the text; `transliterate_with_confidence` returns the
same thing plus which stage handled each token, so a caller can decide whether to
trust it. Hiding that distinction would be the dishonest design.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from . import _channel
from .normalize import IDENTIFIER, ZWJ, ZWNJ, _is_urdu_letter, _require_str, normalize

# --- Stage 1: lexicon -------------------------------------------------------------
# High-frequency Roman Urdu, mapped to correct Urdu script. Deliberately weighted
# towards closed-class words, which dominate token counts and are the most irregular.
LEXICON: dict[str, str] = {
    # pronouns
    "main": "میں",
    "mein": "میں",
    "mai": "میں",
    "me": "میں",
    "tum": "تم",
    "tu": "تو",
    "ap": "آپ",
    "aap": "آپ",
    "hum": "ہم",
    "hm": "ہم",
    "wo": "وہ",
    "woh": "وہ",
    "ye": "یہ",
    "yeh": "یہ",
    "mera": "میرا",
    "meri": "میری",
    "mere": "میرے",
    "tumhara": "تمہارا",
    "tumhari": "تمہاری",
    "apka": "آپ کا",
    "apki": "آپ کی",
    "hamara": "ہمارا",
    "hamari": "ہماری",
    "uska": "اس کا",
    "uski": "اس کی",
    "unka": "ان کا",
    # auxiliaries and copulas
    "hai": "ہے",
    "hay": "ہے",
    "h": "ہے",  # chat: `kya hal h`
    "he": "ہے",
    "hain": "ہیں",
    "hn": "ہیں",
    "ho": "ہو",
    "hoon": "ہوں",
    "hun": "ہوں",
    "hu": "ہوں",
    "tha": "تھا",
    "thi": "تھی",
    "thay": "تھے",
    "the": "تھے",
    "hoga": "ہوگا",
    "hogi": "ہوگی",
    "honge": "ہوں گے",
    "raha": "رہا",
    "rahi": "رہی",
    "rahe": "رہے",
    "gaya": "گیا",
    "gayi": "گئی",
    "gaye": "گئے",
    "diya": "دیا",
    "liya": "لیا",
    "kiya": "کیا",
    "kia": "کیا",
    # negation and question words
    "nahi": "نہیں",
    "nahin": "نہیں",
    "nhi": "نہیں",
    "naheen": "نہیں",
    "na": "نہ",
    "mat": "مت",
    "bina": "بغیر",
    "kya": "کیا",
    "kyun": "کیوں",
    "kyu": "کیوں",
    "kiyun": "کیوں",
    "kaise": "کیسے",
    "kse": "کیسے",
    "kaisa": "کیسا",
    "kaisi": "کیسی",
    "kahan": "کہاں",
    "kahaan": "کہاں",
    "kab": "کب",
    "kaun": "کون",
    "kon": "کون",
    "kitna": "کتنا",
    "kitni": "کتنی",
    "kitne": "کتنے",
    # postpositions and particles
    "ka": "کا",
    "ki": "کی",
    "ke": "کے",
    "ko": "کو",
    "se": "سے",
    "par": "پر",
    "pe": "پے",
    "tak": "تک",
    "liye": "لیے",
    "lye": "لیے",
    "sath": "ساتھ",
    "saath": "ساتھ",
    "bhi": "بھی",
    "hi": "ہی",
    "aur": "اور",
    "or": "اور",
    "ya": "یا",
    "o": "و",  # the conjunction in zabt o nazm; Dakshina: 123 of 127 lowercase o
    "agar": "اگر",
    "to": "تو",
    "tou": "تو",
    "lekin": "لیکن",
    "magar": "مگر",
    "phir": "پھر",
    "fir": "پھر",
    "abhi": "ابھی",
    "ab": "اب",
    "jab": "جب",
    "tab": "تب",
    # common verbs
    "karna": "کرنا",
    "karta": "کرتا",
    "karti": "کرتی",
    "karte": "کرتے",
    "kar": "کر",
    "karo": "کرو",
    "kro": "کرو",
    "karen": "کریں",
    "hona": "ہونا",
    "jana": "جانا",
    "jata": "جاتا",
    "jati": "جاتی",
    "ana": "آنا",
    "aana": "آنا",
    "ata": "آتا",
    "aata": "آتا",
    "dena": "دینا",
    "lena": "لینا",
    "kehna": "کہنا",
    "kaha": "کہا",
    "dekhna": "دیکھنا",
    "dekha": "دیکھا",
    "sunna": "سننا",
    "suna": "سنا",
    "khana": "کھانا",
    "peena": "پینا",
    "sona": "سونا",
    "chalna": "چلنا",
    "milna": "ملنا",
    "mila": "ملا",
    "bolna": "بولنا",
    "likhna": "لکھنا",
    "parhna": "پڑھنا",
    "samajhna": "سمجھنا",
    "samajh": "سمجھ",
    "chahiye": "چاہیے",
    "chahta": "چاہتا",
    "chahti": "چاہتی",
    "sakta": "سکتا",
    "sakti": "سکتی",
    "sakte": "سکتے",
    # adjectives and adverbs
    "acha": "اچھا",
    "accha": "اچھا",
    "achi": "اچھی",
    "ache": "اچھے",
    "bura": "برا",
    "buri": "بری",
    "theek": "ٹھیک",
    "thik": "ٹھیک",
    "bohat": "بہت",
    "bahut": "بہت",
    "buhat": "بہت",
    "bht": "بہت",
    "bhot": "بہت",
    "thora": "تھوڑا",
    "thori": "تھوڑی",
    "zyada": "زیادہ",
    "ziada": "زیادہ",
    "kam": "کم",
    "bara": "بڑا",
    "bari": "بڑی",
    "chota": "چھوٹا",
    "choti": "چھوٹی",
    "naya": "نیا",
    "nayi": "نئی",
    "purana": "پرانا",
    "khush": "خوش",
    "udaas": "اداس",
    "sach": "سچ",
    "jhoot": "جھوٹ",
    "jaldi": "جلدی",
    "dheere": "دھیرے",
    "aaram": "آرام",
    # nouns
    "ghar": "گھر",
    "kaam": "کام",
    "waqt": "وقت",
    "wqt": "وقت",
    "din": "دن",
    "raat": "رات",
    "subah": "صبح",
    "sham": "شام",
    "aaj": "آج",
    "aj": "آج",
    "kal": "کل",
    "parso": "پرسوں",
    "saal": "سال",
    "mahina": "مہینہ",
    "hafta": "ہفتہ",
    "pani": "پانی",
    "paani": "پانی",
    "roti": "روٹی",
    "chai": "چائے",
    "paisa": "پیسہ",
    "paise": "پیسے",
    "rupay": "روپے",
    "dost": "دوست",
    "bhai": "بھائی",
    "behen": "بہن",
    "ammi": "امی",
    "abbu": "ابو",
    "walid": "والد",
    "walida": "والدہ",
    "beta": "بیٹا",
    "beti": "بیٹی",
    "dil": "دل",
    "jaan": "جان",
    "pyar": "پیار",
    "muhabbat": "محبت",
    "zindagi": "زندگی",
    "khushi": "خوشی",
    "gham": "غم",
    "kitab": "کتاب",
    "school": "اسکول",
    "university": "یونیورسٹی",
    "sarak": "سڑک",
    "gari": "گاڑی",
    # Chat spellings. Each was checked on Dakshina's dev sentences before it went in:
    # none of them changed a single dev word except `pata` (+5), and without them
    # `meri gaari` gave میری غار - annotators once wrote غار (cave) as `gaari`.
    "gaari": "گاڑی",
    "gaadi": "گاڑی",
    "gadi": "گاڑی",
    "pata": "پتہ",
    "shehar": "شہر",
    "gaon": "گاؤں",
    "mulk": "ملک",
    "duniya": "دنیا",
    "log": "لوگ",
    "admi": "آدمی",
    "aurat": "عورت",
    # titles, written abbreviated in Roman and in full in Urdu
    "dr": "ڈاکٹر",
    "prof": "پروفیسر",
    "mr": "مسٹر",
    "mrs": "مسز",
    # greetings and fixed phrases
    "salam": "سلام",
    "assalam": "السلام",
    "walaikum": "وعلیکم",
    "shukriya": "شکریہ",
    "meherbani": "مہربانی",
    "khuda": "خدا",
    "allah": "اللہ",
    "inshallah": "ان شاء اللہ",
    "mashallah": "ماشاء اللہ",
    "hafiz": "حافظ",
    "khudahafiz": "خدا حافظ",
    "mubarak": "مبارک",
    "maaf": "معاف",
    "sorry": "معذرت",
    # place names
    "pakistan": "پاکستان",
    "lahore": "لاہور",
    "karachi": "کراچی",
    "islamabad": "اسلام آباد",
    "peshawar": "پشاور",
    "quetta": "کوئٹہ",
    "multan": "ملتان",
    "faisalabad": "فیصل آباد",
    "punjab": "پنجاب",
    "sindh": "سندھ",
    "urdu": "اردو",
}

# --- Stage 2: rules ---------------------------------------------------------------
# Ordered longest-first, so digraphs win over the single letters inside them: `kh`
# must be tried before `k`, or "khana" becomes ک+ہ instead of کھ.
#
# Aspirated consonants use do-chashmi he (ھ U+06BE), which is the correct letter for
# aspiration and a different codepoint from the he that appears as a standalone
# letter (ہ U+06C1). Getting this wrong is the single most common mistake in
# machine-produced Urdu.
RULES: list[tuple[str, str]] = [
    # trigraphs
    ("sch", "سچ"),
    ("tch", "چ"),
    # aspirated consonants
    ("bh", "بھ"),
    ("ph", "پھ"),
    ("th", "تھ"),
    ("jh", "جھ"),
    ("chh", "چھ"),
    ("dh", "دھ"),
    ("rh", "رھ"),
    ("kh", "کھ"),
    ("gh", "گھ"),
    ("lh", "لھ"),
    ("mh", "مھ"),
    ("nh", "نھ"),
    # digraph consonants
    ("ch", "چ"),
    ("sh", "ش"),
    ("zh", "ژ"),
    ("ng", "نگ"),
    ("ny", "نی"),
    # long vowels before short ones
    ("aa", "ا"),
    ("ee", "ی"),
    ("ii", "ی"),
    ("oo", "و"),
    ("uu", "و"),
    ("ai", "ے"),
    ("ay", "ے"),
    ("ei", "ی"),
    ("au", "و"),
    ("ou", "و"),
    # single consonants
    ("b", "ب"),
    ("p", "پ"),
    ("t", "ت"),
    ("j", "ج"),
    ("d", "د"),
    ("r", "ر"),
    ("z", "ز"),
    ("s", "س"),
    ("f", "ف"),
    ("q", "ق"),
    ("k", "ک"),
    ("g", "گ"),
    ("l", "ل"),
    ("m", "م"),
    ("n", "ن"),
    ("v", "و"),
    ("w", "و"),
    ("h", "ہ"),
    ("y", "ی"),
    ("c", "ک"),
    ("x", "کس"),
    # short vowels
    ("a", "ا"),
    ("e", "ے"),
    ("i", "ی"),
    ("o", "و"),
    ("u", "و"),
]

# Urdu -> Roman. Lossy by nature: Urdu has several letters that share one Roman
# sound (س ص ث all -> s), and that merge cannot be undone.
URDU_TO_ROMAN: dict[str, str] = {
    "ا": "a",
    "آ": "aa",
    "ب": "b",
    "پ": "p",
    "ت": "t",
    "ٹ": "t",
    "ث": "s",
    "ج": "j",
    "چ": "ch",
    "ح": "h",
    "خ": "kh",
    "د": "d",
    "ڈ": "d",
    "ذ": "z",
    "ر": "r",
    "ڑ": "r",
    "ز": "z",
    "ژ": "zh",
    "س": "s",
    "ش": "sh",
    "ص": "s",
    "ض": "z",
    "ط": "t",
    "ظ": "z",
    "ع": "a",
    "غ": "gh",
    "ف": "f",
    "ق": "q",
    "ک": "k",
    "گ": "g",
    "ل": "l",
    "م": "m",
    "ن": "n",
    "ں": "n",
    "و": "o",
    "ہ": "h",
    "ھ": "h",
    "ۂ": "h",
    "ۃ": "h",
    "ء": "",
    "ی": "i",
    "ے": "e",
    "ئ": "y",
    "ؤ": "o",
    "أ": "a",
}

_SORTED_RULES = sorted(RULES, key=lambda r: -len(r[0]))

# One pass over the input, keeping *everything* - whitespace included - so the
# output can be rebuilt with the input's own spacing. The first version joined the
# converted tokens with single spaces, which split `2.5` into `2. 5`, `3:30` into
# `3: 30` and `0300-1234567` into `0300- 1234567`, glued `hai :)` into `ہے:)` and
# threw away every line break.
#
# Identifiers come first because they must survive untouched: split into `http`,
# `://`, `x`, `co`, a URL is transliterated into `ہتتپ:// کس. کو`, which no longer
# resolves. Ordinary English words are NOT identifiers - `lahore` -> لاہور is the
# function working. Only spans whose *syntax* marks them as identifiers, numbers or
# codes are protected.
_TOKEN = re.compile(
    rf"""
      (?P<space>\s+)
    | (?P<identifier>{IDENTIFIER})           # URL, email, @mention, #hashtag
    | (?P<dotted>(?:[A-Z]\.)+[A-Z]\b\.?)    # U.S.A, U.N. - an acronym, spelled
    | (?P<mixed>[A-Za-z0-9]*(?:[A-Za-z][0-9]|[0-9][A-Za-z])[A-Za-z0-9]*)  # 5th, mp3, A1
    | (?P<number>\d+(?:[.,:/-]\d+)*%?)      # 2.5  12,34,567  3:30  25-12-2024  10%
    | (?P<word>[A-Za-z]+(?![^\W\d_]))        # plain Latin, not the start of café
    | (?P<letters>[^\W\d_]+)                 # letters in any other script, or café whole
    | (?P<other>.)                           # one punctuation mark or symbol
    """,
    re.VERBOSE | re.DOTALL,
)

# Urdu reads most acronyms letter by letter: Dakshina's annotators wrote TV as ٹی وی
# 25 times out of 25, BBC as بی بی سی, FBI as ایف بی آئی. The ones pronounced as
# words - FIFA فیفا, FATA فاٹا, UNESCO یونیسکو - have at least two vowels, which is the
# rule used: an all-capital word of 2-6 letters with at most one vowel is spelled.
LETTER_NAMES: dict[str, str] = {
    "A": "اے", "B": "بی", "C": "سی", "D": "ڈی", "E": "ای", "F": "ایف", "G": "جی",
    "H": "ایچ", "I": "آئی", "J": "جے", "K": "کے", "L": "ایل", "M": "ایم", "N": "این",
    "O": "او", "P": "پی", "Q": "کیو", "R": "آر", "S": "ایس", "T": "ٹی", "U": "یو",
    "V": "وی", "W": "ڈبلیو", "X": "ایکس", "Y": "وائی", "Z": "زیڈ",
}  # fmt: skip


_TITLES = frozenset({"dr", "prof", "mr", "mrs"})


def _ends_context(token: str, kind: str) -> bool:
    """Whether a non-word token ends the sentence a Roman word is decoded in."""
    if kind in ("abbreviation-dot", "title"):
        return False
    if kind == "space":
        return "\n" in token
    if kind in ("passthrough", "punctuation"):
        return any(c in _SENTENCE_END for c in token)
    return True


def _is_acronym(word: str, shouting: bool = False) -> bool:
    # A single capital is an initial: Dakshina's annotators wrote C as سی 20 times,
    # L as ایل 18, A as اے 12 - letter names for about 115 of 127 capitals. A single
    # lowercase letter is not: `o` is the conjunction و, 123 times.
    if not (1 <= len(word) <= 6 and word.isupper() and sum(c in "AEIOU" for c in word) <= 1):
        return False
    if len(word) == 1:
        return True
    # Chat in capitals is shouting, not acronyms: `KYA HAAL HAI` came out
    # کے وائی اے حال ہے and `main NHI jaunga` spelled NHI letter by letter. A word
    # the curated lexicon knows is never an acronym, and in text written entirely
    # in capitals nothing is, unless it is dotted (U.S.A).
    return not shouting and word.lower() not in LEXICON


def _is_shouting(text: str) -> bool:
    """Whether the Latin text is written in capitals throughout, like `KYA HAAL HAI`.

    Two or more words of two letters or more, all upper case, one of them a word
    the curated lexicon knows - so a lone `BBC` or `BBC TV` is still acronyms.
    """
    long_words = [w for w in _LATIN_WORD.findall(text) if len(w) > 1]
    return (
        len(long_words) >= 2
        and all(w.isupper() for w in long_words)
        and any(w.lower() in LEXICON for w in long_words)
    )


_LATIN_WORD = re.compile(r"[A-Za-z]+")


# Any character in the Arabic/Urdu blocks. Used to spot text that is already in
# Urdu script, which the Roman->Urdu direction must not claim to have converted.
_IS_URDU_SCRIPT = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")

# Punctuation that ends a sentence, and so ends the context a word is chosen in.
_SENTENCE_END = frozenset(".?!۔؟")


# Latin punctuation -> the Urdu mark, for `urdu_punctuation`. The inverse of what
# transliterate_to_roman does with ؟ ، ؛ ۔.
_ASCII_TO_URDU_PUNCT = {"?": "؟", ",": "،", ";": "؛", ".": "۔"}

# What may stand before a converted mark: a word that was, or already is, Urdu.
_WORD_KINDS = frozenset({"roman", "already-urdu", "title", "acronym"})


def _is_urdu_punctuation_slot(text: str, match: re.Match[str], plan: list[tuple[str, str]]) -> bool:
    """Whether a punctuation token ends a word and so belongs in Urdu script.

    Converting every `.` would turn `...` into ۔۔۔ and an emoticon's `;)` into ؛),
    so a mark is converted only straight after a word, and only when a space, a
    closing quote or bracket, or the end of the text follows it.
    """
    token = match.group()
    if token not in _ASCII_TO_URDU_PUNCT or not plan or plan[-1][1] not in _WORD_KINDS:
        return False
    after = text[match.end() : match.end() + 1]
    return after == "" or after.isspace() or after in "\"')]}"


@dataclass
class Transliteration:
    text: str
    # Per token: "lexicon" (trusted), "vocabulary" (a real Urdu word, chosen by the
    # noisy channel), "rules" (best effort), "acronym" (spelled by letter names),
    # "english" (kept in Latin script because `keep_english` was set and the token
    # was tagged English), "passthrough" (numbers, punctuation, codes like `5th`,
    # other scripts - emitted unchanged), "punctuation" (`?` `,` `;` `.` after a word,
    # written ؟ ، ؛ ۔), "identifier" (a URL, email, @mention or #hashtag, emitted
    # verbatim) or "already-urdu" (the token was not Roman at all).
    # Whitespace is kept in the output and not listed here.
    sources: list[tuple[str, str]]

    @property
    def lexicon_coverage(self) -> float:
        """Share of Roman words resolved by the lexicon rather than guessed by rule.

        Tokens already in Urdu script are excluded from the denominator. They are not
        words the lexicon failed on - there was nothing to look up - and counting them
        made a wrong-direction call report 0.0 coverage, which reads as a confident bad
        answer rather than as "this input was not Roman Urdu".
        """
        words = [s for t, s in self.sources if t.isalpha() and s != "already-urdu"]
        if not words:
            return 0.0
        return round(sum(1 for s in words if s == "lexicon") / len(words), 4)

    @property
    def rule_share(self) -> float:
        """Share of Roman words that neither the lexicon nor the vocabulary resolved.

        These are the guesses. On held-out hand-romanised Wikipedia sentences 0.3%
        of words end up here, and 3.9% of those come out right, so a high value
        means the text is full of names or words this library has never seen.
        """
        words = [s for t, s in self.sources if t.isalpha() and s != "already-urdu"]
        if not words:
            return 0.0
        return round(sum(1 for s in words if s == "rules") / len(words), 4)

    @property
    def already_urdu_share(self) -> float:
        """Share of alphabetic tokens that were already Urdu script.

        A non-zero value on input you believed was Roman Urdu means the text is mixed,
        or the call is in the wrong direction.
        """
        words = [s for t, s in self.sources if t.isalpha()]
        if not words:
            return 0.0
        return round(sum(1 for s in words if s == "already-urdu") / len(words), 4)


def _apply_rules(token: str) -> str:
    out: list[str] = []
    i = 0
    lowered = token.lower()
    while i < len(lowered):
        for roman, urdu in _SORTED_RULES:
            if lowered.startswith(roman, i):
                out.append(urdu)
                i += len(roman)
                break
        else:
            out.append(lowered[i])
            i += 1
    return "".join(out)


def transliterate_with_confidence(
    text: str,
    *,
    use_vocabulary: bool = True,
    use_context: bool = True,
    keep_english: bool = False,
    urdu_punctuation: bool = True,
) -> Transliteration:
    """Roman Urdu to Urdu script, reporting how each token was resolved.

    `use_vocabulary=False` skips the noisy-channel stage and gives the 0.1 behaviour:
    lexicon, then rules. It is faster and far less accurate.

    `use_context=False` resolves each word on its own instead of decoding the
    sentence with a word-bigram model. Faster, and 2.7 points less accurate on
    held-out sentences (88.5% against 91.2%); the curated lexicon then always wins,
    so `ke` is always کے, never کہ. A single word on its own is always resolved
    this way - with no neighbours there is no context to use.

    `keep_english=True` leaves tokens that `tag_roman_tokens` labels English in Latin
    script instead of transliterating them, and reports them as `english`. Off by
    default: Urdu writes English loanwords in Urdu script (کالج, اسٹیشن), and the
    vocabulary stage usually finds that spelling.

    `urdu_punctuation=True` writes `?` `,` `;` and a full stop that follows a word
    as ؟ ، ؛ and ۔, which is how Urdu is punctuated - `ye kitab hai?` gives
    یہ کتاب ہے؟. Only a mark that ends a word and is followed by a space or the
    end of the text is converted, so `2.5`, `3:30`, `...` and `:)` are untouched.
    Reported as `punctuation`. `False` leaves every mark as typed.
    """
    # First decide what every token is, then transliterate. Two passes, because
    # `keep_english` has to tag the Roman words *as this function sees them*: tagging
    # the raw text instead counted the letters inside a URL as words, so one URL
    # shifted every later English tag onto the wrong word.
    _require_str(text, "transliterate_with_confidence")
    plan: list[tuple[str, str]] = []
    shouting = _is_shouting(text)
    for match in _TOKEN.finditer(text):
        token, group = match.group(), match.lastgroup
        if group == "space":
            plan.append((token, "space"))
        elif group == "identifier":
            plan.append((token, "identifier"))
        elif group == "word":
            plan.append((token, "acronym" if _is_acronym(token, shouting) else "roman"))
        elif group == "dotted":
            plan.append((token, "acronym"))
        elif group == "letters" and _IS_URDU_SCRIPT.search(token):
            # Already in Urdu script: this direction has nothing to do.
            #
            # Without this the token fell through to _apply_rules, which matches
            # only Latin graphemes and so returned it unchanged - correct output
            # labelled `rules`, as though the rule engine had resolved it. Feeding
            # Urdu to the Roman->Urdu direction by mistake then produced a result
            # that looked transliterated, with `lexicon_coverage` reading 0.0,
            # which says "guessed badly" rather than "wrong direction".
            plan.append((token, "already-urdu"))
        elif (
            token == "."
            and plan
            and (
                plan[-1][1] == "acronym"
                or (plan[-1][1] == "roman" and plan[-1][0].lower() in _TITLES)
            )
        ):
            # The dot of `Dr.` or `C.`: Urdu writes ڈاکٹر and سی without one, and it
            # does not end a sentence - it used to, and cut the context mid-name.
            # With its dot a title is certain, so it is not left to the decoder,
            # which read `Mr. Ali` as میر علی - Mir Ali is a common name.
            if plan[-1][1] == "roman":
                plan[-1] = (plan[-1][0], "title")
            plan.append((token, "abbreviation-dot"))
        elif urdu_punctuation and _is_urdu_punctuation_slot(text, match, plan):
            plan.append((token, "punctuation"))
        else:
            # Numbers, codes like 5th, punctuation, and letters of any script this
            # function does not convert (é, Devanagari): emitted unchanged.
            plan.append((token, "passthrough"))

    english: set[int] = set()
    if keep_english:
        from .langid import _tagger

        roman_positions = [i for i, (_, kind) in enumerate(plan) if kind == "roman"]
        tags = _tagger().tag([plan[i][0] for i in roman_positions])
        english = {i for i, tag in zip(roman_positions, tags, strict=True) if tag == "en"}

    # With context, each run of Roman words is decoded as a sequence, so a word can
    # be chosen for the word before it: کہ after کہا, کے before بعد. A run ends at
    # anything that is not a Roman word to transliterate, except spaces, commas and
    # numbers, which do not end a thought. Sentence-final punctuation and line breaks
    # always end one.
    decided: dict[int, tuple[str, str]] = {}
    if use_vocabulary and use_context:
        run: list[int] = []

        def flush() -> None:
            # A lone word has no context to use, and decoding it as a one-word
            # sentence scores it by how often words *start* sentences - which cost
            # 2.6 points of round-trip accuracy, word by word, until it was measured.
            # One word goes through the word-by-word path below instead.
            if len(run) == 1:
                run.clear()
            if run:
                romans = [plan[i][0].lower() for i in run]
                for i, choice in zip(run, _channel.channel().decode(romans, LEXICON), strict=True):
                    decided[i] = choice
                run.clear()

        for position, (token, kind) in enumerate(plan):
            if kind == "roman" and position not in english:
                run.append(position)
            elif not _ends_context(token, kind):
                continue
            else:
                flush()
        flush()

    pieces: list[str] = []
    sources: list[tuple[str, str]] = []
    for position, (token, kind) in enumerate(plan):
        if kind == "space":
            pieces.append(token)
            continue
        if kind == "abbreviation-dot":
            sources.append((token, "passthrough"))
            continue
        if kind == "title":
            pieces.append(LEXICON[token.lower()])
            sources.append((token, "lexicon"))
            continue
        if kind == "punctuation":
            pieces.append(_ASCII_TO_URDU_PUNCT[token])
            sources.append((token, "punctuation"))
            continue
        if kind == "acronym":
            pieces.append(" ".join(LETTER_NAMES[c] for c in token if c in LETTER_NAMES))
            sources.append((token, "acronym"))
            continue
        if kind != "roman":
            pieces.append(token)
            sources.append((token, kind))
            continue
        if position in english:
            pieces.append(token)
            sources.append((token, "english"))
            continue
        if position in decided:
            urdu, source = decided[position]
            pieces.append(urdu)
            sources.append((token, source))
            continue
        lowered = token.lower()
        if lowered in LEXICON:
            pieces.append(LEXICON[lowered])
            sources.append((token, "lexicon"))
            continue
        found = _channel.resolve(lowered) if use_vocabulary else None
        if found:
            pieces.append(found)
            sources.append((token, "vocabulary"))
        else:
            pieces.append(_apply_rules(token))
            sources.append((token, "rules"))

    # The input's own whitespace is in `pieces`, so joining with nothing reproduces
    # its spacing and line breaks exactly around the converted words.
    return Transliteration(text="".join(pieces), sources=sources)


def transliterate_to_urdu(
    text: str,
    *,
    use_vocabulary: bool = True,
    use_context: bool = True,
    keep_english: bool = False,
    urdu_punctuation: bool = True,
) -> str:
    """Roman Urdu to Urdu script.

    URLs, emails, @mentions and #hashtags are passed through unchanged. Ordinary
    English words are transliterated the way Urdu writes them - `station` gives
    اسٹیشن - unless `keep_english` is set. See `transliterate_with_confidence`.
    """
    _require_str(text, "transliterate_to_urdu")
    return transliterate_with_confidence(
        text,
        use_vocabulary=use_vocabulary,
        use_context=use_context,
        keep_english=keep_english,
        urdu_punctuation=urdu_punctuation,
    ).text


_ROMAN_VOWELS = set("aeiou")
# Letters that attach to the preceding consonant rather than standing alone.
_ASPIRATION = {"ھ", "ء"}
_URDU_PUNCT_TO_ASCII = {"،": ",", "؛": ";", "؟": "?", "۔": ".", "٪": "%", "٭": "*"}


def transliterate_to_roman(
    text: str, *, method: str = "learned", insert_short_vowels: bool = True
) -> str:
    """Urdu script to Roman Urdu, spelled the way people spell it.

    >>> transliterate_to_roman("میں ٹھیک ہوں")
    'main theek hoon'

    `method="learned"` (the default) spells each word the way Urdu speakers romanise
    it: the curated lexicon for function words, the most common spelling annotators
    wrote for the ~24,000 words Dakshina covers, and otherwise a spelling generated
    from the same letter model that reads Roman Urdu. Scored against people on
    held-out Dakshina words, it matches a human spelling far more often than the
    rules below - and it also converts back to Urdu more often. docs/CORPUS.md has
    both numbers.

    `method="rules"` is the 0.1 letter-by-letter mapping. Urdu does not write short
    vowels, so a literal mapping gives consonant runs - جملہ becomes `jmlh`; with
    `insert_short_vowels` an `a` goes between adjacent consonants, giving `jamlah`.
    `insert_short_vowels=False` always means the literal mapping.

    Either way it is lossy (س ص ث are all `s`), digits come out as ASCII, Urdu
    punctuation as its ASCII equivalent and whitespace is kept. Anything with no
    Roman form - emoji, symbols, letters of other scripts - is kept unchanged:
    `میں خوش ہوں 😀` gives `main khush hoon 😀`. Only what belongs to an Urdu
    word and cannot be written in Roman - a stray diacritic, a zero-width
    non-joiner - is dropped.
    """
    _require_str(text, "transliterate_to_roman")
    if method not in ("learned", "rules"):
        raise ValueError(f"method must be 'learned' or 'rules', not {method!r}")
    text = normalize(text, normalize_digits=True, collapse_whitespace=False)
    if method == "rules" or not insert_short_vowels:
        return _rules_to_roman(text, insert_short_vowels)

    model = _channel.channel()
    out: list[str] = []
    for match in _URDU_OR_OTHER.finditer(text):
        token = match.group()
        if _IS_URDU_SCRIPT.search(token) and token.isalpha():
            spelled = _CURATED_ROMAN.get(token) or model.romanize(token)
            out.append(spelled or _rules_to_roman(token, True))
        else:
            previous = text[match.start() - 1] if match.start() else ""
            out.append(_rules_to_roman(token, True, previous))
    return "".join(out)


# Runs of letters, and everything else one character at a time.
_URDU_OR_OTHER = re.compile(r"[^\W\d_]+|.", re.DOTALL)

# Urdu word -> the Roman spelling the curated lexicon lists first for it: میں is
# `main`, not the rules' `min`. Only single words, and only real spellings.
_CURATED_ROMAN: dict[str, str] = {}
for _roman, _urdu in LEXICON.items():
    _key = normalize(_urdu)
    if _roman.isalpha() and " " not in _key and _key not in _CURATED_ROMAN:
        _CURATED_ROMAN[_key] = _roman


def _rules_to_roman(text: str, insert_short_vowels: bool, previous: str = "") -> str:
    """The 0.1 letter-by-letter mapping, over already-normalised text.

    `previous` is the character before `text`, when it is a piece of a longer string.
    """
    out: list[str] = []
    for char in text:
        before, previous = previous, char
        if char in URDU_TO_ROMAN:
            piece = URDU_TO_ROMAN[char]
            if (
                insert_short_vowels
                and piece
                # ھ marks aspiration on the letter before it - کھ is one sound, not
                # two. A vowel inserted here turns کھانا into "kahana".
                and char not in _ASPIRATION
                and out
                and out[-1]
                and out[-1][-1].isalpha()
                and out[-1][-1] not in _ROMAN_VOWELS
                and piece[0] not in _ROMAN_VOWELS
            ):
                out.append("a")
            out.append(piece)
        elif char in _URDU_PUNCT_TO_ASCII:
            out.append(_URDU_PUNCT_TO_ASCII[char])
        elif char.isascii():
            out.append(char)
        elif char.isspace():
            out.append(" ")  # a non-ASCII space (U+00A0, U+3000) keeps its place
        elif not _is_urdu_mark(char, before):
            # An emoji, a symbol (★ ✓ © ₨), a letter of another script: kept as it
            # is. This branch used to drop everything, so `میں خوش ہوں 😀` came out
            # `main khush hoon ` - a deleted emoji is lost meaning in the very text
            # (chat, reviews) that Roman Urdu is written in.
            out.append(char)

    return "".join(out)


def _is_urdu_mark(char: str, previous: str) -> bool:
    """A character with no Roman form that belongs to the Urdu word around it.

    Arabic-script combining marks (a diacritic normalize did not strip, the
    takhallus sign) and the zero-width non-joiner Urdu uses inside compounds. A
    zero-width joiner is dropped after an Urdu letter and kept anywhere else,
    because between two emoji it is part of the emoji (👨‍👩‍👧).
    """
    if char == ZWNJ:
        return True
    if char == ZWJ:
        return _is_urdu_letter(previous)
    return unicodedata.category(char).startswith("M") and _is_urdu_letter(char)
