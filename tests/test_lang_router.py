"""Tests for indextts/lang_router.py (docs/ROUTING.md §2).

The deterministic script gate is tested with a fake predictor so a failure points at the
gate and not at GlotLID. Everything that genuinely needs the model is marked
`integration` and skips when the 1.6 GB file is not on disk.
"""

from __future__ import annotations

import pytest

from indextts.lang_router import (
    ADAPTER_LANGS,
    API_LANGS,
    BASE_LANGS,
    LATIN_LANGS,
    LangRouter,
    normalize_code,
)


def router(model, **kw):
    return LangRouter(None, model=model, **kw)


# --------------------------------------------------------------------------------------
# Inventory / routing
# --------------------------------------------------------------------------------------

def test_language_inventory_is_the_documented_18():
    assert len(API_LANGS) == 18
    assert BASE_LANGS == {"en", "zh"}
    assert len(ADAPTER_LANGS) == 16
    assert "ja" in ADAPTER_LANGS and "sv" in ADAPTER_LANGS
    assert not (BASE_LANGS & ADAPTER_LANGS)


@pytest.mark.parametrize("lang", sorted(BASE_LANGS))
def test_route_base(lang):
    assert LangRouter.route(lang) == "base"


@pytest.mark.parametrize("lang", sorted(ADAPTER_LANGS))
def test_route_adapter(lang):
    assert LangRouter.route(lang) == "adapter"


def test_route_accepts_aliases_and_rejects_junk():
    assert LangRouter.route("sv-SE") == "adapter"
    assert LangRouter.route("EN") == "base"
    with pytest.raises(ValueError):
        LangRouter.route("klingon")


def test_normalize_code():
    assert normalize_code("RU") == "ru"
    assert normalize_code("{ru}") == "ru"
    assert normalize_code("sv-SE") == "sv"
    assert normalize_code("en_US") == "en"
    assert normalize_code("xx") is None
    assert normalize_code(None) is None


# --------------------------------------------------------------------------------------
# Forced codes
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("lang", API_LANGS)
def test_forced_code_wins_over_script(lang, no_model):
    # Text is unmistakably Russian; the forced code must still win, without any LID call.
    assert router(no_model).decide("Привет, как дела?", forced=lang) == lang


def test_forced_code_normalizes(no_model):
    r = router(no_model)
    assert r.decide("hello", forced="sv-SE") == "sv"
    assert r.decide("hello", forced=" {DE} ") == "de"


@pytest.mark.parametrize("bad", ["klingon", "xx", "e", "russian", "zz-ZZ", "ру"])
def test_forced_unknown_code_raises(bad, no_model):
    with pytest.raises(ValueError):
        router(no_model).decide("Привет", forced=bad)


def test_auto_tokens_mean_autodetect(no_model):
    # "auto"/""/None all fall through to detection -- kana decides without the model.
    for forced in (None, "", "auto", "AUTO", "none"):
        assert router(no_model).decide("これはテストです", forced=forced) == "ja"


# --------------------------------------------------------------------------------------
# Script gate: kana, han, Cyrillic markers
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "これはテストです",                 # hiragana
    "テストです",                       # katakana
    "東京に住んでいますか、はい",        # kanji + kana
    "Tokyoでテスト",                    # Latin + kana
])
def test_any_kana_is_japanese_without_lid(text, no_model):
    assert router(no_model).decide(text) == "ja"


def test_pure_kanji_asks_lid_restricted_to_ja_zh(fake_model):
    # Real GlotLID ranks other languages above ja/zh all the time; the restriction is an
    # argmax over the subset, so the first ja/zh label in the ranking wins.
    m = fake_model([("eng_Latn", 0.5), ("jpn_Jpan", 0.3), ("zho_Hans", 0.2)])
    assert router(m).decide("東京都水道局") == "ja"
    assert m.calls, "the han branch must consult the detector"

    m = fake_model([("zho_Hans", 0.6), ("jpn_Jpan", 0.4)])
    assert router(m).decide("東京都水道局") == "zh"


def test_pure_kanji_falls_back_to_zh_when_lid_offers_nothing(fake_model):
    m = fake_model([("eng_Latn", 0.9), ("deu_Latn", 0.1)])
    assert router(m).decide("東京都水道局") == "zh"


def test_glotlid_mandarin_label_maps_to_zh(fake_model):
    # GlotLID v3 has NO zho_* label: Mandarin is cmn_Hani. Getting this wrong sends every
    # Chinese sentence to ja (and therefore to the adapter view).
    m = fake_model([("cmn_Hani", 0.99), ("jpn_Jpan", 0.01)])
    assert router(m).decide("今天天气很好") == "zh"


@pytest.mark.parametrize("label", ["zho_Hans", "zho_Hant", "yue_Hani", "wuu_Hani", "lzh_Hani"])
def test_other_sinitic_labels_map_to_zh(label, fake_model):
    m = fake_model([(label, 0.9), ("jpn_Jpan", 0.1)])
    assert router(m).decide("臺灣的天氣很好") == "zh"


def test_han_script_fallback_never_leaks_into_other_branches(fake_model):
    # *_Hani -> zh only applies where zh is allowed, i.e. the han branch.
    m = fake_model([("yue_Hani", 0.9), ("rus_Cyrl", 0.1)])
    assert router(m).decide("текст без маркеров") == "ru"


@pytest.mark.parametrize("char,lang", [
    ("ҡ", "ba"), ("ғ", "ba"), ("ҙ", "ba"), ("ҫ", "ba"),
    ("җ", "tt"),
    ("ў", "be"),
    ("ї", "uk"), ("є", "uk"), ("ґ", "uk"),
])
def test_unique_cyrillic_markers_hard_override(char, lang, no_model):
    text = f"текст с буквой {char} внутри"
    assert router(no_model).decide(text) == lang
    # Uppercase must work too: the gate lowercases first.
    assert router(no_model).decide(text.upper()) == lang


@pytest.mark.parametrize("char", list("әөүңһі"))
def test_shared_cyrillic_markers_do_not_decide(char, fake_model):
    # These misrouted 865/1200 tt rows when they were allowed to hard-decide.
    m = fake_model([("rus_Cyrl", 0.8), ("tat_Cyrl", 0.2)])
    assert router(m).decide(f"текст {char} текст") == "ru"
    assert m.calls, "shared markers must fall through to the detector"


def test_cyrillic_branch_is_restricted_to_cyrillic_languages(fake_model):
    m = fake_model([("eng_Latn", 0.7), ("bak_Cyrl", 0.2), ("rus_Cyrl", 0.1)])
    assert router(m).decide("текст без маркеров") == "ba"


def test_cyrillic_branch_defaults_to_ru(fake_model):
    m = fake_model([("eng_Latn", 0.99)])
    assert router(m).decide("текст без маркеров") == "ru"


def test_cyrillic_wins_over_han_when_both_present(fake_model):
    m = fake_model([("rus_Cyrl", 0.9), ("zho_Hans", 0.1)])
    assert router(m).decide("иероглиф 東京 в тексте") == "ru"


# --------------------------------------------------------------------------------------
# Latin branch + confidence floor
# --------------------------------------------------------------------------------------

def test_latin_branch_restricted_argmax(fake_model):
    m = fake_model([("nno_Latn", 0.6), ("swe_Latn", 0.35), ("eng_Latn", 0.05)])
    assert router(m).decide("Det var en gang en katt") == "sv"


def test_low_confidence_latin_falls_back_to_message_dominant(fake_model):
    # The floor is off by default (0.0, measured optimum); enable it to test the mechanism.
    m = fake_model([("deu_Latn", 0.12), ("nld_Latn", 0.11)])
    r = router(m, latin_confidence_threshold=0.3)
    assert r.decide("big tasty", message_dominant="ru") == "ru"
    assert r.decide("big tasty", message_dominant="sv-SE") == "sv"


def test_low_confidence_latin_falls_back_to_en(fake_model):
    m = fake_model([("deu_Latn", 0.12), ("nld_Latn", 0.11)])
    r = router(m, latin_confidence_threshold=0.3)
    assert r.decide("big tasty") == "en"
    # An unusable message_dominant must not leak through.
    assert r.decide("big tasty", message_dominant="klingon") == "en"


def test_default_threshold_trusts_the_model(fake_model):
    # Default floor 0.0: GlotLID's argmax wins even at low confidence (78% right when
    # unsure, measured) — no silent en-fallback.
    m = fake_model([("deu_Latn", 0.12), ("nld_Latn", 0.11)])
    assert router(m).decide("big tasty") == "de"


def test_confidence_threshold_is_configurable(fake_model):
    ranking = [("deu_Latn", 0.25), ("eng_Latn", 0.2)]
    assert router(fake_model(ranking)).decide("Ein Test") == "de"          # default 0.0
    assert router(fake_model(ranking), latin_confidence_threshold=0.3).decide("Ein Test") == "en"


def test_high_confidence_latin_is_trusted_over_message_dominant(fake_model):
    m = fake_model([("deu_Latn", 0.95)])
    assert router(m).decide("Guten Morgen", message_dominant="ru") == "de"


# --------------------------------------------------------------------------------------
# Junk guards (docs/ROUTING.md §5.2)
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["", "   ", "?!?!", "...", "🤡🤡", "— — —", "123 456", None])
def test_junk_returns_none(text, no_model):
    assert router(no_model).decide(text) is None


def test_zxx_top_label_on_wordless_text_is_junk(fake_model):
    # Reached only when the text has letters in one of our scripts but no word chars --
    # kept as a belt-and-braces guard per §2.7.
    m = fake_model([("zxx_Zyyy", 0.9), ("eng_Latn", 0.1)])
    r = router(m)
    assert r.decide("!!!") is None            # no branch even runs
    assert r.decide("abc") == "en"            # word chars present -> not junk


# --------------------------------------------------------------------------------------
# Loading behaviour
# --------------------------------------------------------------------------------------

def test_lazy_mode_does_not_load_until_needed():
    r = LangRouter("/definitely/not/here/model.bin", preload=False)
    assert r._model is None
    assert r.decide("これは") == "ja"          # script gate needs no model
    with pytest.raises(Exception):
        r.decide("hello")                      # Latin branch does -> load fails loudly


def test_injected_model_is_used_verbatim(fake_model):
    m = fake_model([("eng_Latn", 0.9)])
    assert router(m).model is m


def test_newlines_are_stripped_before_predict(fake_model):
    m = fake_model([("eng_Latn", 0.9)])
    router(m).decide("hello\nworld")
    assert "\n" not in m.calls[0][0]


# --------------------------------------------------------------------------------------
# Integration: the real GlotLID model
# --------------------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.parametrize("text,lang", [
    ("привет пойдем в macdonalds и возьмем big tasty за 500 рублей", "ru"),
    ("Привет, как твои дела сегодня вечером?", "ru"),
    ("Привіт, як твої справи сьогодні ввечері?", "uk"),
    ("Прывітанне, як твае справы сёння ўвечары?", "be"),
    ("Һаумыһығыҙ, хәлдәрегеҙ нисек?", "ba"),
    ("Исәнмесез, хәлләрегез ничек?", "tt"),
])
def test_real_model_cyrillic(glotlid_router, text, lang):
    assert glotlid_router.decide(text) == lang


@pytest.mark.integration
@pytest.mark.parametrize("text,lang", [
    ("The weather is lovely today and the garden looks wonderful.", "en"),
    ("Das Wetter ist heute wunderschön und der Garten sieht toll aus.", "de"),
    ("Le temps est magnifique aujourd'hui et le jardin est superbe.", "fr"),
    ("Hace un tiempo maravilloso hoy y el jardín se ve estupendo.", "es"),
    ("Il tempo è splendido oggi e il giardino sembra meraviglioso.", "it"),
    ("O tempo está maravilhoso hoje e o jardim está lindo.", "pt"),
    ("Pogoda jest dziś cudowna, a ogród wygląda wspaniale.", "pl"),
    ("Hava bugün çok güzel ve bahçe harika görünüyor.", "tr"),
    ("Het weer is vandaag heerlijk en de tuin ziet er prachtig uit.", "nl"),
    ("Počasí je dnes nádherné a zahrada vypadá skvěle.", "cs"),
    ("Vädret är underbart idag och trädgården ser fantastisk ut.", "sv"),
])
def test_real_model_latin(glotlid_router, text, lang):
    assert glotlid_router.decide(text) == lang


@pytest.mark.integration
@pytest.mark.parametrize("text,lang", [
    ("今日はいい天気ですね", "ja"),
    ("東京都水道局の職員募集", "ja"),          # pure kanji, still Japanese
    ("今天天气很好，我们去公园散步吧", "zh"),
])
def test_real_model_cjk(glotlid_router, text, lang):
    assert glotlid_router.decide(text) == lang


@pytest.mark.integration
@pytest.mark.parametrize("text", ["?!?!", "🤡🤡", "", "   ", "###"])
def test_real_model_junk(glotlid_router, text):
    assert glotlid_router.decide(text) is None


@pytest.mark.integration
def test_real_model_routes_views(glotlid_router):
    assert glotlid_router.route(glotlid_router.decide("How are you doing today?")) == "base"
    assert glotlid_router.route(glotlid_router.decide("Как у тебя сегодня дела?")) == "adapter"
