"""Tests for indextts/text_frontend_serve.py (docs/ROUTING.md §3).

Two things are being defended: parity with what training did to the corpus (numbers
verbalized in the target language, symbols expanded, Cyrillic cleaned) and the serve-time
rule that nothing about the *content* may raise or be rejected.
"""

from __future__ import annotations

import pytest

from indextts.text_frontend_serve import (
    NATIVE_ROUTE_LANGS,
    sanitize,
    spec_for,
    tokenizer_language_hint,
)

num2words = pytest.importorskip("num2words", reason="num2words is required at serve time")


# --------------------------------------------------------------------------------------
# Number verbalization, per language
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,lang,expected", [
    ("за 500 рублей", "ru", "пятьсот"),
    ("це коштує 3 гривні", "uk", "три"),
    ("гэта каштуе 7 еўра", "be", "сем"),
    ("Ich zahle 25 Euro", "de", "fünfundzwanzig"),
    ("Il y a 42 personnes", "fr", "quarante-deux"),
    ("Hay 15 libros", "es", "quince"),
    ("Ci sono 15 libri", "it", "quindici"),
    ("Existem 15 livros", "pt", "quinze"),
    ("Jest 15 książek", "pl", "piętnaście"),
    ("Je tam 15 knih", "cs", "patnáct"),
    ("Er zijn 15 boeken", "nl", "vijftien"),
    ("Det finns 15 böcker", "sv", "femton"),
])
def test_digits_become_words_in_the_target_language(text, lang, expected):
    out = sanitize(text, lang)
    assert expected in out
    assert not any(c.isdigit() for c in out)


def test_german_currency_expands_to_german_words():
    out = sanitize("Ich zahle 25 Euro und 50 €", "de")
    assert "fünfundzwanzig Euro" in out
    assert "fünfzig Euro" in out
    assert "€" not in out


def test_russian_currency_and_percent():
    out = sanitize("Скидка 20% и 500 ₽", "ru")
    assert "двадцать процентов" in out
    assert "пятьсот рублей" in out


def test_turkish_percent_precedes_and_number_is_split():
    out = sanitize("%50 indirim ve 1234 lira", "tr")
    assert out.startswith("yüzde elli")
    # num2words glues Turkish numerals; real orthography spaces them.
    assert "bin iki yüz otuz dört" in out
    assert "%" not in out


def test_french_decimal_uses_the_spoken_separator():
    out = sanitize("Il fait 12,5 degrés", "fr")
    assert "douze virgule cinq" in out


def test_russian_decimal_and_group_separator():
    out = sanitize("Цена 1 500,50 рублей", "ru")
    assert "тысяча пятьсот целых пять ноль" in out


def test_english_style_group_separator_for_non_comma_languages():
    # en is passthrough, so use the only decimal-point Latin language we rewrite: none --
    # instead check that a Cyrillic language reads "1.234,56" the European way.
    out = sanitize("Бюджет 1.234,56 евро", "ru")
    assert "тысяча двести тридцать четыре целых пять шесть" in out


def test_swedish_currency_code_needs_a_number_next_to_it():
    assert "kronor" in sanitize("Det kostar 500 kr", "sv")
    # "kr" inside an ordinary word must survive untouched.
    assert sanitize("Han skrek högt", "sv") == "Han skrek högt"


# --------------------------------------------------------------------------------------
# ba/tt: Russian verbalizer by design
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("lang,expect", [("ba", "егерме биш"), ("tt", "егерме биш")])
def test_ba_tt_digits_verbalize_natively(lang, expect):
    # Replaced the Russian num2words approximation with native Turkic numerals.
    out = sanitize("Миндә 25 китап", lang)
    assert expect in out.lower()
    assert "двадцать" not in out.lower()


def test_ba_tt_keep_their_own_tokenizer_hint():
    assert tokenizer_language_hint("ba") == "ba"
    assert tokenizer_language_hint("tt") == "tt"


# --------------------------------------------------------------------------------------
# Cyrillic cleanup
# --------------------------------------------------------------------------------------

def test_stress_marks_are_stripped():
    assert sanitize("за́мок и замо́к", "ru") == "замок и замок"


def test_latin_lookalikes_inside_cyrillic_words_are_folded():
    # Latin "i" typed mid-word -> Cyrillic "і" (U+0456), the letter the model was
    # trained on. Contributors do this constantly; the two glyphs are indistinguishable.
    out = sanitize("мiсто Кiїв", "uk")
    assert "i" not in out
    assert out == "місто Кіїв"


def test_genuine_latin_words_are_not_folded():
    out = sanitize("Я использую OK Google", "ru")
    assert "OK Google" in out


def test_typographic_quotes_and_dashes_are_folded():
    out = sanitize("«Привет» — сказал он, “да”", "ru")
    assert "“" not in out and "”" not in out
    assert "'да'" in out
    assert "«Привет»" in out            # guillemets are kept, they are real punctuation


# --------------------------------------------------------------------------------------
# Passthrough and serve-time relaxations
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("lang", sorted(NATIVE_ROUTE_LANGS))
@pytest.mark.parametrize("text", [
    "Hello, I have 123 apples - not 456!",
    "你好，我有123个苹果",
    "こんにちは、123個のりんご",
    "  weird   spacing  ",
])
def test_native_route_languages_pass_through_unchanged(lang, text):
    assert sanitize(text, lang) == text


def test_none_input_is_empty_string():
    assert sanitize(None, "ru") == ""
    assert sanitize(None, "en") == ""
    assert sanitize("", "de") == ""


@pytest.mark.parametrize("text", [
    "привет пойдем в macdonalds и возьмем big tasty за 500 рублей",
    "Das ist Привет auf Russisch",
    "Ceci est un test 東京 avec du kanji",
    "🤡🤡🤡 emoji only",
    "\x00\x01 control chars",
    "мешанина ҵ ҩ ӌ редких букв",
    "1e999999 and ٤٥٦ arabic-indic digits",
])
@pytest.mark.parametrize("lang", ["ru", "de", "tr", "ba", "sv"])
def test_sanitize_never_raises_on_content(text, lang):
    out = sanitize(text, lang)
    assert isinstance(out, str)


def test_cross_script_words_pass_through_unchanged():
    out = sanitize("привет пойдем в macdonalds и возьмем big tasty за 500 рублей", "ru")
    assert "macdonalds" in out
    assert "big tasty" in out
    assert "пятьсот рублей" in out


def test_unparseable_numbers_survive_instead_of_raising():
    huge = "9" * 400
    assert huge in sanitize(f"число {huge} конец", "ru")


# --------------------------------------------------------------------------------------
# Language codes
# --------------------------------------------------------------------------------------

def test_sv_and_sv_se_are_the_same_language():
    assert spec_for("sv") is spec_for("sv-SE")
    assert spec_for("sv-SE").code == "sv"
    assert tokenizer_language_hint("sv") == tokenizer_language_hint("sv-SE") == "sv"
    assert sanitize("Det kostar 500 kr", "sv-SE") == sanitize("Det kostar 500 kr", "sv")


def test_codes_are_case_insensitive():
    assert spec_for("RU").code == "ru"
    assert spec_for(" De ").code == "de"


@pytest.mark.parametrize("lang", [
    "en", "zh", "ja", "ru", "uk", "be", "ba", "tt",
    "de", "fr", "es", "it", "pt", "pl", "tr", "nl", "cs", "sv",
])
def test_every_api_language_has_a_spec_and_a_hint(lang):
    assert spec_for(lang).code == lang
    assert tokenizer_language_hint(lang) == lang


def test_unknown_language_is_an_api_error_not_a_content_error():
    with pytest.raises(KeyError):
        sanitize("text", "klingon")
    assert tokenizer_language_hint("klingon") is None


# ---------------------------------------------------------------------------------------
# Native Bashkir/Tatar numerals (replaced the Russian num2words approximation)
# ---------------------------------------------------------------------------------------

def test_tt_numbers_native():
    out = sanitize("бүген 500 сум", "tt")
    assert "биш йөз" in out.lower()
    assert "пятьсот" not in out.lower()          # the old Russian approximation is gone

def test_ba_numbers_native():
    out = sanitize("миңә 21 йәш", "ba")
    assert "егерме бер" in out.lower()

def test_tt_year_composition():
    assert "ике мең егерме дүрт" in sanitize("2024 елда", "tt").lower()

def test_ba_hundred_and_thousand_take_no_ber():
    low = sanitize("100 һәм 1000", "ba").lower()
    assert "йөҙ" in low and "мең" in low
    assert "бер йөҙ" not in low and "бер мең" not in low

def test_tt_zero():
    assert "ноль" in sanitize("0 булды", "tt").lower()
