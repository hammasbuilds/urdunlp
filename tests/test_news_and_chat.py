"""News compounds and chat Roman Urdu: every case audit round 2 reported, pinned.

The accuracy figures are measured on hand-romanised encyclopaedia sentences, which
have almost no compounds written with izafat and no chat at all. These tests are the
part of the behaviour those figures cannot see.
"""

import pytest

from urdunlp import (
    normalize,
    transliterate_to_roman,
    transliterate_to_urdu,
    transliterate_with_confidence,
)
from urdunlp._channel import channel, covers

# --- Urdu -> Roman: nothing may be dropped -----------------------------------------

# Words and names from the front page of an Urdu newspaper. Each must come back from
# Roman as the same Urdu - the Roman spelling must carry the whole word.
_FRONT_PAGE = """حکومت وزیراعظم صدر عدالت سپریم کورٹ الیکشن انتخابات پارلیمنٹ قومی اسمبلی
سینیٹ فوج پولیس معیشت مہنگائی بجٹ ٹیکس روپے ڈالر اسٹیٹ بینک شرح سود کرکٹ میچ ٹیم
کھلاڑی بارش سیلاب زلزلہ ہلاک زخمی دھماکہ اپوزیشن مسلم لیگ پیپلز پارٹی بلوچستان
سندھ پنجاب کراچی لاہور راولپنڈی کشمیر بھارت چین افغانستان ایران سعودی عرب وزارت
ترجمان بیان اجلاس فیصلہ مذاکرات معاہدہ سرمایہ کاری برآمدات درآمدات عمران خان نواز
شریف شہباز آصف زرداری بلاول بھٹو مریم چیف جسٹس آرمی کورونا وائرس ویکسین ہسپتال
تعلیم طلبہ یونیورسٹی امتحان نتائج موسم گرمی درجہ حرارت پٹرول بجلی گیس قیمت اضافہ
کمی شہری عوام احتجاج مظاہرہ گرفتار مقدمہ ضمانت سزا"""
NEWS = _FRONT_PAGE.split()


def test_news_words_survive_the_round_trip():
    back = [normalize(transliterate_to_urdu(transliterate_to_roman(w))) for w in NEWS]
    kept = sum(b == normalize(w) for b, w in zip(back, NEWS, strict=True))
    assert kept / len(NEWS) >= 0.95, [
        (w, b) for w, b in zip(NEWS, back, strict=True) if b != normalize(w)
    ]


def test_no_news_word_loses_a_consonant():
    for word in NEWS:
        assert covers(word, transliterate_to_roman(word).replace("-", "")), word


@pytest.mark.parametrize(
    ("urdu", "roman"),
    [
        # was `wazir` - minister - and came back as وزیر
        ("وزیراعظم", "wazir-e-azam"),
        ("وزیر اعظم", "wazir-e-azam"),
        ("وزیرِاعظم", "wazir-e-azam"),  # with the izafat diacritic
        ("وزیراعلیٰ پنجاب", "wazir-e-aala punjab"),
        ("قائداعظم", "quaid-e-azam"),
        ("تحریک انصاف", "tehreek-e-insaf"),
        ("زندہ باد", "zindabad"),  # was `jinda baad`
        ("پاکستان زندہ باد", "pakistan zindabad"),
        ("زندہ", "zinda"),  # `jinda` and `zinda` tied; alphabetical order chose jinda
        ("بین الاقوامی", "bainul aqwami"),
        ("السلام علیکم", "assalam-o-alaikum"),
        ("ان شاء اللہ", "inshallah"),
    ],
)
def test_compounds_and_fixed_phrases(urdu, roman):
    assert transliterate_to_roman(urdu) == roman


@pytest.mark.parametrize(
    "roman", ["wazir-e-azam", "tehreek-e-insaf", "quaid-e-azam", "zindabad", "inshallah"]
)
def test_phrase_spellings_read_back(roman):
    """Every spelling the phrase table writes, transliterate_to_urdu reads back."""
    urdu = transliterate_to_urdu(roman)
    assert transliterate_to_roman(urdu) == roman


def test_the_learned_speller_never_drops_a_consonant():
    """Over every word annotators spelled, the chosen spelling covers the word - or
    is empty, and the caller falls back to the letter rules."""
    model = channel()
    for urdu in list(model.attested)[:3000]:
        spelled = model.romanize(urdu)
        assert spelled == "" or covers(urdu, spelled), (urdu, spelled)


def test_covers():
    assert covers("وزیراعظم", "waziraazam")
    assert not covers("وزیراعظم", "wazir")  # the ظ and م of اعظم are missing
    assert covers("سورۃ", "surat")  # an extra consonant is allowed: ۃ written t
    assert covers("اسٹیشن", "station")


# --- Roman -> Urdu: izafat ----------------------------------------------------------


@pytest.mark.parametrize(
    ("roman", "urdu"),
    [
        ("wazir-e-azam ne kaha", "وزیر اعظم نے کہا"),  # was وزیر-ے-اعظم
        ("tehreek-e-insaf", "تحریک انصاف"),
        ("sheesha-e-dil", "شیشۂ دل"),  # after ہ the izafat is ۂ
        ("dunya-e-islam", "دنیائے اسلام"),  # after ا it is ئے
        ("zabt-o-nazm", "ضبط و نظم"),  # was ضبط-و-نظم
    ],
)
def test_izafat(roman, urdu):
    assert transliterate_to_urdu(roman) == urdu


# --- Roman -> Urdu: chat ------------------------------------------------------------


@pytest.mark.parametrize(
    ("roman", "urdu"),
    [
        ("bohttt", "بہت"),  # was بہتات, abundance
        ("bohtttt", "بہت"),
        ("nahiii", "نہیں"),  # was نہی
        ("sachiii", "سچی"),  # was سچائی, truth
        ("yaaaar", "یار"),
        ("kyaaa baat hai", "کیا بات ہے"),
        ("yaar aaj bohttt garmi hai", "یار آج بہت گرمی ہے"),
        ("ok thanks", "اوکے تھینکس"),  # was اوک تھنگز: context overruled the lexicon
        ("Assalam o alaikum! kaise ho", "السلام علیکم! کیسے ہو"),  # was السلام و علیکم
        ("walaikum assalam", "وعلیکم السلام"),
    ],
)
def test_chat_spellings(roman, urdu):
    assert transliterate_to_urdu(roman) == urdu


@pytest.mark.parametrize(
    ("roman", "urdu"),
    [
        ("iPhone 15 Pro", "iPhone 15 پرو"),  # was افیون (opium) 15 پرو
        ("WhatsApp pe msg kr do", "WhatsApp پے میسج کر دو"),  # was وہاتساپپ ... مسکگ
        ("lol ye kya hai", "lol یہ کیا ہے"),  # was لال (red)
        ("Rs 500 ka recharge", "Rs 500 کا recharge"),  # was راس 500 کا رےچارگے
        ("OMG kya baat hai", "OMG کیا بات ہے"),
        ("YouTube pe dekho", "YouTube پے دیکھو"),
    ],
)
def test_brands_abbreviations_and_unknown_english_stay_latin(roman, urdu):
    assert transliterate_to_urdu(roman) == urdu


def test_kept_latin_is_reported_as_english():
    sources = dict(transliterate_with_confidence("Rs 500 ka recharge, iPhone pe").sources)
    assert sources["Rs"] == sources["recharge"] == sources["iPhone"] == "english"


def test_unknown_non_english_words_still_go_to_the_rules():
    """Names spelled the Urdu way and keyboard noise are not kept as English."""
    assert dict(transliterate_with_confidence("Xqzvt aaya").sources)["Xqzvt"] == "rules"


def test_acronyms_are_still_spelled():
    assert transliterate_to_urdu("BBC TV") == "بی بی سی ٹی وی"


def test_keep_english_reads_stretched_words_and_urdu_function_words():
    # nahiii, sachiii and pe were left in Latin
    assert transliterate_to_urdu("nahiii yaar", keep_english=True) == "نہیں یار"
    assert transliterate_to_urdu("sachiii", keep_english=True) == "سچی"
    out = transliterate_to_urdu("WhatsApp pe msg kr do", keep_english=True)
    assert out.startswith("WhatsApp پے ")


def test_keep_english_keeps_the_pronoun_i():
    assert transliterate_to_urdu("I love you", keep_english=True) == "I love you"
    assert transliterate_to_urdu("I love Pakistan") == "آئی لو پاکستان"


# --- punctuation --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("roman", "urdu"),
    [
        ("kya haal hai??", "کیا حال ہے؟؟"),  # ?? stayed Latin while ? was converted
        ("tum kahan ho???", "تم کہاں ہو؟؟؟"),
        ("sach?! wow", "سچ؟! واؤ"),
        ("hai?😂", "ہے؟😂"),
        ("kya baat hai!!!", "کیا بات ہے!!!"),  # Urdu writes ! as English does
        ("ye kitab hai?", "یہ کتاب ہے؟"),
    ],
)
def test_runs_of_marks_are_converted_consistently(roman, urdu):
    assert transliterate_to_urdu(roman) == urdu


def test_punctuation_after_english_left_in_latin_is_not_converted():
    # was `Hello، how are you`
    assert transliterate_to_urdu("Hello, how are you", keep_english=True) == "Hello, how are you"
    assert transliterate_to_urdu("Rs, bas") == "Rs, بس"
    # ...and after an Urdu word it still is
    assert transliterate_to_urdu("main theek hoon, tum sunao").count("،") == 1


def test_latin_punctuation_setting_still_leaves_everything():
    assert transliterate_to_urdu("kya haal hai??", urdu_punctuation=False) == "کیا حال ہے??"
