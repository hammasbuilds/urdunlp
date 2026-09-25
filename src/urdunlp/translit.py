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
the vocabulary stage choosing each word on its own, 88.4%; choosing the sentence
as a whole, 90.7%. The numbers, and how they were measured, are in docs/CORPUS.md.

`transliterate_to_urdu` returns the text; `transliterate_with_confidence` returns the
same thing plus which stage handled each token, so a caller can decide whether to
trust it. Hiding that distinction would be the dishonest design.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import _channel
from .normalize import _require_str, normalize

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
    "kam2": "کام",
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
    "shehar": "شہر",
    "gaon": "گاؤں",
    "mulk": "ملک",
    "duniya": "دنیا",
    "log": "لوگ",
    "admi": "آدمی",
    "aurat": "عورت",
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
    r"""
      (?P<space>\s+)
    | (?P<identifier>
          https?://\S+                       # http(s) URL
        | www\.\S+                           # bare www URL
        | [\w.+-]+@[\w-]+\.[\w.-]*\w         # email
        | [@#]\w+                            # mention or hashtag
      )
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


def _ends_context(token: str, kind: str) -> bool:
    """Whether a non-word token ends the sentence a Roman word is decoded in."""
    if kind == "space":
        return "\n" in token
    if kind == "passthrough":
        return any(c in _SENTENCE_END for c in token)
    return True


def _is_acronym(word: str) -> bool:
    return 2 <= len(word) <= 6 and word.isupper() and sum(c in "AEIOU" for c in word) <= 1


# Any character in the Arabic/Urdu blocks. Used to spot text that is already in
# Urdu script, which the Roman->Urdu direction must not claim to have converted.
_IS_URDU_SCRIPT = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")

# Punctuation that ends a sentence, and so ends the context a word is chosen in.
_SENTENCE_END = frozenset(".?!۔؟")


@dataclass
class Transliteration:
    text: str
    # Per token: "lexicon" (trusted), "vocabulary" (a real Urdu word, chosen by the
    # noisy channel), "rules" (best effort), "acronym" (spelled by letter names),
    # "english" (kept in Latin script because `keep_english` was set and the token
    # was tagged English), "passthrough" (numbers, punctuation, codes like `5th`,
    # other scripts - emitted unchanged), "identifier" (a URL, email, @mention or
    # #hashtag, emitted verbatim) or "already-urdu" (the token was not Roman at all).
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

        These are the guesses. On held-out hand-romanised Wikipedia sentences 0.4%
        of words end up here, and 3.5% of those come out right, so a high value
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
) -> Transliteration:
    """Roman Urdu to Urdu script, reporting how each token was resolved.

    `use_vocabulary=False` skips the noisy-channel stage and gives the 0.1 behaviour:
    lexicon, then rules. It is faster and far less accurate.

    `use_context=False` resolves each word on its own instead of decoding the
    sentence with a word-bigram model. Faster, and 2.3 points less accurate on
    held-out sentences (88.4% against 90.7%); the curated lexicon then always wins,
    so `ke` is always کے, never کہ. A single word on its own is always resolved
    this way - with no neighbours there is no context to use.

    `keep_english=True` leaves tokens that `tag_roman_tokens` labels English in Latin
    script instead of transliterating them, and reports them as `english`. Off by
    default: Urdu writes English loanwords in Urdu script (کالج, اسٹیشن), and the
    vocabulary stage usually finds that spelling.
    """
    # First decide what every token is, then transliterate. Two passes, because
    # `keep_english` has to tag the Roman words *as this function sees them*: tagging
    # the raw text instead counted the letters inside a URL as words, so one URL
    # shifted every later English tag onto the wrong word.
    _require_str(text, "transliterate_with_confidence")
    plan: list[tuple[str, str]] = []
    for match in _TOKEN.finditer(text):
        token, group = match.group(), match.lastgroup
        if group == "space":
            plan.append((token, "space"))
        elif group == "identifier":
            plan.append((token, "identifier"))
        elif group == "word":
            plan.append((token, "acronym" if _is_acronym(token) else "roman"))
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
) -> str:
    """Roman Urdu to Urdu script.

    URLs, emails, @mentions and #hashtags are passed through unchanged. Ordinary
    English words are transliterated the way Urdu writes them - `station` gives
    اسٹیشن - unless `keep_english` is set. See `transliterate_with_confidence`.
    """
    _require_str(text, "transliterate_to_urdu")
    return transliterate_with_confidence(
        text, use_vocabulary=use_vocabulary, use_context=use_context, keep_english=keep_english
    ).text


_ROMAN_VOWELS = set("aeiou")
# Letters that attach to the preceding consonant rather than standing alone.
_ASPIRATION = {"ھ", "ء"}
_URDU_PUNCT_TO_ASCII = {"،": ",", "؛": ";", "؟": "?", "۔": "."}


def transliterate_to_roman(text: str, *, insert_short_vowels: bool = True) -> str:
    """Urdu script to Roman Urdu.

    Lossy and one-way: several Urdu letters share a Roman sound (س ص ث all give `s`),
    and that merge cannot be undone.

    Urdu does not write short vowels, so a literal character mapping produces
    consonant runs - جملہ becomes `jmlh`, which no Roman Urdu reader would write.
    With `insert_short_vowels` an `a` is placed between adjacent consonants, giving
    `jamlah`. That is a heuristic, not a pronunciation model: it is right far more
    often than it is wrong, and it is off with one flag when you need the raw mapping.
    """
    _require_str(text, "transliterate_to_roman")
    text = normalize(text)
    out: list[str] = []

    for char in text:
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
        elif char.isspace() or char.isascii():
            out.append(char)
        # Anything else - a stray mark - is dropped rather than emitted as noise.

    return re.sub(r"\s+", " ", "".join(out)).strip()
