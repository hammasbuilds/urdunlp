"""Text copied out of PDFs and old systems: presentation forms and Arabic-Indic digits.

Audit round 2 found that `normalize` left both alone. ﻛﺘﺎﺏ (four presentation-form
codepoints, the way a PDF text layer stores the word) looks exactly like کتاب and
compared unequal to it after normalisation; ﷲ stayed one ligature; and ٣ (Arabic-Indic)
and ۳ (Urdu) were different strings unless `normalize_digits=True`.
"""

import unicodedata

import pytest

from urdunlp import identify_language, normalize, transliterate_to_roman, words

# The same Urdu sentence twice: once in ordinary letters, once in the presentation
# forms a PDF text layer holds - initial, medial, final and isolated shapes.
PLAIN = "یہ کتاب میری ہے اور میں اسے پڑھتا ہوں"
PDF = "ﯾﮧ ﮐﺘﺎﺏ ﻣﯿﺮﯼ ﮨﮯ ﺍﻭﺭ ﻣﯿﮟ ﺍﺳﮯ ﭘﮍﮬﺘﺎ ﮨﻮﮞ"


def test_presentation_forms_normalise_to_the_letters():
    assert PDF != PLAIN
    assert normalize(PDF) == normalize(PLAIN)
    assert normalize("ﻛﺘﺎﺏ") == "کتاب"


@pytest.mark.parametrize(
    ("shape", "letter"),
    [
        ("ﮨ", "ہ"),  # heh goal, initial - not Arabic heh, not do-chashmi
        ("ﮬ", "ھ"),  # do-chashmi he, initial
        ("ﮐ", "ک"),  # keheh, initial - not Arabic kaf
        ("ﯾ", "ی"),  # farsi yeh, initial - not Arabic yeh
        ("ﮮ", "ے"),  # yeh barree, isolated
        ("ﭘ", "پ"),  # peh
        ("ﭨ", "ٹ"),  # tteh
        ("ﮈ", "ڈ"),  # ddal
        ("ﮌ", "ڑ"),  # rreh
        ("ﮔ", "گ"),  # gaf
        ("ﮞ", "ں"),  # noon ghunna
    ],
)
def test_urdu_specific_letters_survive(shape, letter):
    """NFKC over the presentation blocks must give the *Urdu* letter each shape is of."""
    assert normalize(shape) == letter


def test_the_allah_ligature_is_spelled_out():
    assert normalize("ﷲ") == "اللہ"
    assert transliterate_to_roman("ﷲ") == "allah"


def test_phrase_ligatures_are_left_alone():
    """ﷺ and the basmala are symbols in Urdu writing; expanding them would put a
    sentence where one character was."""
    for ligature in ("ﷺ", "ﷻ", "﷽"):
        assert normalize(ligature) == ligature


def test_isolated_diacritic_forms_leave_no_space_behind():
    # U+FE70 decomposes to a space plus the mark; neither belongs in the output
    assert normalize("کتابﹰ") == "کتاب"


def test_nothing_outside_the_presentation_blocks_is_touched():
    text = "ﬁ café ２ x² ℌ"  # a Latin ligature, a full-width digit, a superscript
    assert normalize(text) == unicodedata.normalize("NFC", text)


def test_arabic_indic_digits_become_urdu_digits():
    assert normalize("٣") == "۳"
    assert normalize("٠١٢٣٤٥٦٧٨٩") == "۰۱۲۳۴۵۶۷۸۹"
    assert normalize("٢٣") == normalize("۲۳")
    assert normalize("٢٣", normalize_digits=True) == "23"
    assert normalize("٣", unify_characters=False) == "٣"


def test_pdf_text_works_downstream():
    assert words(PDF) == words(PLAIN)
    assert transliterate_to_roman(PDF) == transliterate_to_roman(PLAIN)
    # identify_language read the shapes as unknown characters and said Azerbaijani
    assert identify_language(PDF).language == "ur"
