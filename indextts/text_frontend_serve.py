"""Serve-time text conditioning, one function per language (docs/ROUTING.md §3).

Port of `data_prep/text_frontend.py` from the index-finetune repo -- the module that
conditioned every training line. Serving MUST reproduce it or the model sees text at
inference that it never saw in training.

Why it exists: the bundled front-end auto-routes. `TextNormalizer.use_chinese()` returns
True for any text with no ASCII letters, so *every* non-Latin script (Russian, Ukrainian,
Bashkir, ...) is fed to the **Chinese** normalizer, which turns digits into Chinese
numerals and dashes into 减 ("minus"). Latin-script languages fare no better: they route
to the English normalizer, which verbalizes German numbers in English. The routing is also
unstable -- "Я использую OK Google" has ASCII letters so it routes to EN, while
"Я использую Гугл" routes to ZH. Same language, different pipeline.

The fix (verified at `index-tts-training/indextts/utils/front.py:158-199`):
`normalize(text, language=<code>)` falls through to `_basic_cleanup()` for ANY code that
is not ja/zh/en. `_basic_cleanup` is inert -- NFKC, whitespace collapse, and a small
punctuation map. It does not verbalize numbers and does not mangle dashes. So:

    1. :func:`sanitize` does ALL language-specific work, here, where it is visible.
    2. The tokenizer is then always called with :func:`tokenizer_language_hint`, which
       forces the inert path. Deterministic, no surprises.
    3. en / zh / ja keep their native routes -- they are the base model's own languages
       and `ja` has a real `normalize_japanese`. Do not disturb them.

Differences from the data_prep original, all of them serve-time relaxations from
docs/ROUTING.md §3 (behaviour for the 15 non-native languages is otherwise identical):

* **Nothing rejects.** `charset_issues()` / `is_clean()` are NOT ported: cross-script
  words pass through unchanged (viewers code-switch -- "big tasty" inside a Russian
  sentence is real input, not a corpus defect) and unknown characters pass through
  instead of failing a check. Training-time rejection stays in data_prep.
* **en/zh/ja are a strict passthrough** -- the text is returned byte-identical, not even
  NFKC'd, because their native normalizer routes already do that and the base view must
  stay bit-identical to today's production.
* **"sv" is the canonical Swedish code** (the API code); "sv-SE" (the training-side
  code) is accepted as an alias.

ba/tt digits verbalize through NATIVE Bashkir/Tatar numerals (`_TURKIC_NUMERALS`) —
num2words has neither language, so they get a small built-in verbalizer instead of the
earlier Russian approximation. 0.00% of validated CV17 lines in either language contain
a digit, so training never exercised numbers in ba/tt context either way.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

try:
    from num2words import num2words
except ImportError:  # keeps the module importable (and passthrough-correct) without it
    num2words = None


# --------------------------------------------------------------------------------------
# Language table
# --------------------------------------------------------------------------------------

# Native Bashkir/Tatar numerals — num2words has neither, and reading Bashkir numbers in
# Russian was an approximation nobody asked for. Turkic numbers compose by plain
# juxtaposition (2024 = "ике мең егерме дүрт"), so units+tens+scales cover everything;
# 100/1000 take no leading "бер". Sources: standard literary Tatar/Bashkir numerals —
# worth a native speaker's read before anyone ships marketing copy through it.
_TURKIC_NUMERALS = {
    "tt": {
        "zero": "ноль", "minus": "минус",
        "units": ["", "бер", "ике", "өч", "дүрт", "биш", "алты", "җиде", "сигез", "тугыз"],
        "tens": ["", "ун", "егерме", "утыз", "кырык", "илле", "алтмыш", "җитмеш",
                 "сиксән", "туксан"],
        "hundred": "йөз", "scales": [(10 ** 9, "миллиард"), (10 ** 6, "миллион"), (1000, "мең")],
    },
    "ba": {
        "zero": "ноль", "minus": "минус",
        "units": ["", "бер", "ике", "өс", "дүрт", "биш", "алты", "ете", "һигеҙ", "туғыҙ"],
        "tens": ["", "ун", "егерме", "утыҙ", "ҡырҡ", "илле", "алтмыш", "етмеш",
                 "һикһән", "туҡһан"],
        "hundred": "йөҙ", "scales": [(10 ** 9, "миллиард"), (10 ** 6, "миллион"), (1000, "мең")],
    },
}


def _turkic_num_to_words(n: int, table: dict) -> str:
    if n == 0:
        return table["zero"]
    if n < 0:
        return f"{table['minus']} {_turkic_num_to_words(-n, table)}"

    def under_thousand(k: int) -> str:
        parts = []
        h, rem = divmod(k, 100)
        if h:
            parts.append(f"{table['units'][h]} {table['hundred']}".strip()
                         if h > 1 else table["hundred"])
        t, u = divmod(rem, 10)
        if t:
            parts.append(table["tens"][t])
        if u:
            parts.append(table["units"][u])
        return " ".join(parts)

    parts = []
    for value, word in table["scales"]:
        count, n = divmod(n, value)
        if count:
            prefix = under_thousand(count) if (count > 1 or value > 1000) else ""
            parts.append(f"{prefix} {word}".strip())
    if n:
        parts.append(under_thousand(n))
    return " ".join(parts)


def _number_to_words(n: int, spec) -> str:
    """Native tt/ba verbalizer first; num2words for every other language."""
    table = _TURKIC_NUMERALS.get(spec.code)
    if table is not None and abs(n) < 10 ** 12:
        return _turkic_num_to_words(n, table)
    return num2words(n, lang=spec.num2words_lang)


@dataclass(frozen=True)
class LangSpec:
    code: str
    script: str                 # "latin" | "cyrillic" | "cjk"
    num2words_lang: Optional[str]   # None => no verbalizer available
    route: str                  # language= value handed to TextNormalizer.normalize()
    percent_word: Optional[str] = None
    currency: dict = field(default_factory=dict)
    # num2words_lang is an approximation rather than the real thing (ba/tt -> ru)
    num_is_approximate: bool = False
    decimal_word: str = ""      # spoken decimal separator ("virgule", "запятая", ...)
    percent_precedes: bool = False  # Turkish writes "%50" and says "yüzde elli"


_SPECS = {
    "ru": LangSpec("ru", "cyrillic", "ru", "ru", "процентов", {"₽": "рублей", "$": "долларов", "€": "евро"},
                   decimal_word="целых"),
    "uk": LangSpec("uk", "cyrillic", "uk", "uk", "відсотків", {"₴": "гривень", "$": "доларів", "€": "євро"},
                   decimal_word="кома"),
    "be": LangSpec("be", "cyrillic", "be", "be", "працэнтаў", {"$": "долараў", "€": "еўра"},
                   decimal_word="коска"),
    "ba": LangSpec("ba", "cyrillic", "ru", "ba", "процент", {"₽": "һум"},
                   decimal_word="бөтөн"),
    "tt": LangSpec("tt", "cyrillic", "ru", "tt", "процент", {"₽": "сум"},
                   decimal_word="бөтен"),
    "de": LangSpec("de", "latin", "de", "de", "Prozent", {"€": "Euro", "$": "Dollar", "£": "Pfund"},
                   decimal_word="Komma"),
    "fr": LangSpec("fr", "latin", "fr", "fr", "pour cent", {"€": "euros", "$": "dollars", "£": "livres"},
                   decimal_word="virgule"),
    "es": LangSpec("es", "latin", "es", "es", "por ciento", {"€": "euros", "$": "dólares", "£": "libras"},
                   decimal_word="coma"),
    "it": LangSpec("it", "latin", "it", "it", "per cento", {"€": "euro", "$": "dollari", "£": "sterline"},
                   decimal_word="virgola"),
    "pt": LangSpec("pt", "latin", "pt", "pt", "por cento", {"€": "euros", "$": "dólares", "R$": "reais"},
                   decimal_word="vírgula"),
    "pl": LangSpec("pl", "latin", "pl", "pl", "procent", {"zł": "złotych", "€": "euro", "$": "dolarów"},
                   decimal_word="przecinek"),
    "cs": LangSpec("cs", "latin", "cs", "cs", "procent", {"Kč": "korun", "€": "eur", "$": "dolarů"},
                   decimal_word="celá"),
    "nl": LangSpec("nl", "latin", "nl", "nl", "procent", {"€": "euro", "$": "dollar"},
                   decimal_word="komma"),
    "tr": LangSpec("tr", "latin", "tr", "tr", "yüzde", {"₺": "lira", "€": "euro", "$": "dolar"},
                   decimal_word="virgül", percent_precedes=True),
    "sv": LangSpec("sv", "latin", "sv", "sv", "procent", {"kr": "kronor", "€": "euro", "$": "dollar"},
                   decimal_word="komma"),
    # Base-model languages: native routes, no rewriting by us.
    "en": LangSpec("en", "latin", "en", "en"),
    "zh": LangSpec("zh", "cjk", None, "zh"),
    "ja": LangSpec("ja", "cjk", "ja", "ja"),
}

#: Codes whose native normalizer route we must not disturb.
NATIVE_ROUTE_LANGS = {"en", "zh", "ja"}

#: Caller spellings folded onto canonical codes. "sv-SE" is what data_prep used.
_ALIASES = {"sv-se": "sv", "sv_se": "sv", "zh-cn": "zh", "zh-hans": "zh", "zh-hant": "zh", "jp": "ja"}


def spec_for(lang: str) -> LangSpec:
    """LangSpec for a language code (case-insensitive, "sv-SE" == "sv").

    Raises:
        KeyError: the code is not one of the 18. This is a programming/API error, never
            a content error -- :func:`sanitize` never raises on the *text*.
    """
    key = (lang or "").strip()
    if key in _SPECS:
        return _SPECS[key]
    low = key.lower().replace("_", "-")
    if low in _ALIASES:
        return _SPECS[_ALIASES[low]]
    for k, v in _SPECS.items():
        if k.lower() == low:
            return v
    raise KeyError(f"No LangSpec for language {lang!r}. Add one before using it.")


def tokenizer_language_hint(lang: str) -> Optional[str]:
    """The value to pass as `language=` to TextNormalizer.normalize()/tokenize()/encode().

    Never let this auto-detect: that is the unstable route described in the module
    docstring. Training and serving must both call this. For the 15 non-native languages
    the returned code is one the normalizer does not know, which is the point -- it lands
    in the inert `_basic_cleanup` branch. en/zh/ja return their own codes and keep their
    native routes.

    Returns ``None`` for an unknown code, i.e. "no hint, let the front-end decide" --
    a serve-time caller should not crash a request over this.
    """
    try:
        return spec_for(lang).route
    except KeyError:
        return None


# --------------------------------------------------------------------------------------
# Character classification
# --------------------------------------------------------------------------------------

# NFKC leaves typographic quotes alone, but the front-end's char_rep_map folds them to
# "'" anyway. Doing it here instead keeps our text and what the model actually sees in
# agreement -- otherwise French diverges on the apostrophe in "l'homme" alone.
_QUOTE_FOLD = {
    "“": "'", "”": "'", "„": "'", "‟": "'",
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "ʻ": "'", "ʼ": "'", "ʿ": "'", "ʾ": "'", "`": "'", "´": "'",
}
_DASH_FOLD = {c: "-" for c in "‐‑‒⁃−·"}
# Combining acute/grave: Russian and Ukrainian stress marks. They are a reading aid,
# never part of the orthography, and they would otherwise mint junk BPE pieces.
_COMBINING_STRESS = "̀́"

# Per-language Cyrillic alphabets, used to decide whether a confusable fold has a
# legitimate target. Bashkir/Tatar extras come from the measured char inventory (STATE 9-C).
_CYRILLIC_ALPHABET = {
    "ru": set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя"),
    "uk": set("абвгдежзиклмнопрстуфхцчшщьюяієїґ"),
    "be": set("абвгдежзйклмнопрстуфхцчшыьэюяіў"),
    "ba": set("абвгдеёжзийклмнопрстуфхцчшщъыьэюяәөүғҡңҙҫһ"),
    "tt": set("абвгдеёжзийклмнопрстуфхцчшщъыьэюяәөүҗңһ"),
}
# Visually identical Latin -> Cyrillic. Applied ONLY inside words that are otherwise
# Cyrillic, so a real Latin word ("OK Google") is left alone -- at serve time it simply
# passes through as itself.
_CONFUSABLE = {
    "a": "а", "A": "А", "B": "В", "c": "с", "C": "С", "e": "е", "E": "Е",
    "H": "Н", "i": "і", "K": "К", "M": "М", "o": "о", "O": "О", "p": "р",
    "P": "Р", "T": "Т", "x": "х", "X": "Х", "y": "у", "Y": "У", "k": "к",
}
_WORD_RE = re.compile(r"\S+")


def _char_script(ch: str) -> Optional[str]:
    """"LATIN"/"CYRILLIC"/"CJK"/... for letters; None for non-letters."""
    if not ch.isalpha():
        return None
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return "UNKNOWN"
    first = name.split(" ")[0]
    if first in ("CJK", "HIRAGANA", "KATAKANA"):
        return "CJK"
    return first


def _fold_confusables(text: str, lang: str) -> str:
    """Latin lookalikes inside Cyrillic words -> their Cyrillic twins.

    Contributors type a Latin "i" in Ukrainian words constantly (185 of 8000 sampled
    training lines) and the same habit shows up in user input. Left alone it is a
    different token from "і" and the model has never seen it there. Folding only fires
    when the word is already mostly Cyrillic, so genuine Latin words ("Google") survive
    untouched.
    """
    alphabet = _CYRILLIC_ALPHABET.get(lang)
    if alphabet is None:
        return text

    def fix(m: re.Match) -> str:
        w = m.group()
        has_cyr = any(_char_script(c) == "CYRILLIC" for c in w)
        if not has_cyr:
            return w
        latin = [c for c in w if _char_script(c) == "LATIN"]
        if not latin or not all(
            c in _CONFUSABLE and _CONFUSABLE[c].lower() in alphabet for c in latin
        ):
            return w
        return "".join(_CONFUSABLE.get(c, c) for c in w)

    return _WORD_RE.sub(fix, text)


# --------------------------------------------------------------------------------------
# Number / symbol verbalization
# --------------------------------------------------------------------------------------

# Digit runs that may carry group separators: "1 234", "1.234,56", "1,234.56".
# \s covers the ASCII space; NFKC has already folded NBSP/NNBSP down to it.
_DIGIT_GROUP_RE = re.compile(r"\d(?:[\d.,]|\s(?=\d))*\d|\d")
_WS_RE = re.compile(r"\s+")

# num2words' Turkish returns one glued word ("binikiyuzotuzdort"); real Turkish
# orthography spaces the components. Longest-first so "yuz" cannot eat "yirmi".
_TR_PARTS = ["milyar", "milyon", "bin", "yüz", "doksan", "seksen", "yetmiş", "altmış",
             "elli", "kırk", "otuz", "yirmi", "on", "dokuz", "sekiz", "yedi", "altı",
             "beş", "dört", "üç", "iki", "bir", "sıfır"]
_TR_SPLIT_RE = re.compile("(" + "|".join(_TR_PARTS) + ")")

# Languages that write "1.234,56" rather than "1,234.56".
_DECIMAL_COMMA_LATIN = {"de", "fr", "es", "it", "pt", "pl", "cs", "nl", "tr", "sv"}


def _split_turkish(word: str) -> str:
    return _WS_RE.sub(" ", _TR_SPLIT_RE.sub(r" \1 ", word)).strip()


def _strip_group_separators(token: str, decimal_comma: bool) -> tuple[str, Optional[str]]:
    """Split "1.234,56" (or "1,234.56") into an integer string and a decimal tail."""
    t = _WS_RE.sub("", token)
    dec_sep, grp_sep = (",", ".") if decimal_comma else (".", ",")
    head, sep, tail = t.rpartition(dec_sep)
    if sep and tail.isdigit() and 1 <= len(tail) <= 3 and head:
        return head.replace(grp_sep, "").replace(dec_sep, ""), tail
    return t.replace(grp_sep, "").replace(dec_sep, ""), None


def verbalize_numbers(text: str, spec: LangSpec) -> str:
    """Digits -> words, in the *target* language.

    Without this the auto-route verbalizes German numbers in English and Russian numbers
    in Mandarin. Any number num2words cannot handle is left as-is rather than raising --
    a digit that survives is a mispronunciation, an exception is a failed request.
    """
    if spec.code not in _TURKIC_NUMERALS and (num2words is None or spec.num2words_lang is None):
        return text
    decimal_comma = spec.script == "cyrillic" or spec.code in _DECIMAL_COMMA_LATIN

    def repl(m: re.Match) -> str:
        raw = m.group()
        if not any(c.isdigit() for c in raw):
            return raw
        whole, frac = _strip_group_separators(raw, decimal_comma)
        if not whole.isdigit():
            return raw
        try:
            out = _number_to_words(int(whole), spec)
            if frac is not None:
                # Read the fractional part digit by digit and name the separator;
                # num2words' own decimal handling differs per language and dropping
                # the separator turns "1500,50" into "one thousand five hundred five zero".
                digits = [_number_to_words(int(d), spec) for d in frac]
                out = f"{out} {spec.decimal_word} {' '.join(digits)}".replace("  ", " ")
            if spec.code == "tr":
                out = _split_turkish(out)
            return out
        except Exception:
            return raw

    return _DIGIT_GROUP_RE.sub(repl, text)


def expand_symbols(text: str, spec: LangSpec) -> str:
    """Percent and currency -> words. Multi-char symbols first so "R$" beats "$"."""
    if spec.percent_word:
        # Turkish writes the sign before the number ("%50" -> "yüzde elli"); everywhere
        # else it follows. Either way the word needs whitespace on both sides, or it
        # fuses with the number once that is verbalized.
        if spec.percent_precedes:
            text = re.sub(r"%\s*", " " + spec.percent_word + " ", text)
        else:
            text = re.sub(r"\s*%", " " + spec.percent_word + " ", text)
    for sym in sorted(spec.currency, key=len, reverse=True):
        if sym not in text:
            continue
        word = " " + spec.currency[sym] + " "
        if any(c.isalpha() for c in sym):
            # Alphabetic codes are substrings of ordinary words -- Swedish "kr" turns
            # "skrek" into "s kronor ek" if replaced blindly. Require a word boundary
            # AND an adjacent number, which is the only context a currency code appears
            # in anyway.
            pat = (r"(?<![^\W\d_])" + re.escape(sym) + r"(?![^\W\d_])")
            text = re.sub(rf"(?<=\d)\s*{pat}", word, text)
            text = re.sub(rf"{pat}\s*(?=\d)", word, text)
        else:
            text = text.replace(sym, word)
    return text


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------

def sanitize(text: str, lang: str) -> str:
    """Full conditioning for `lang`, matching what training did to the corpus.

    en/zh/ja are returned UNCHANGED: their native normalizer routes are the base model's
    own and must not change (docs/ROUTING.md §3).

    This never raises on content. Cross-script words, emoji, unknown symbols and
    unparseable numbers all pass through -- the caller's §5 guards decide what to do with
    a sentence, not this function.

    Args:
        text: one sentence, as split by the caller.
        lang: an API language code (":func:`spec_for`" accepts "sv-SE" too).

    Returns:
        The conditioned text; ``""`` for empty input.

    Raises:
        KeyError: `lang` is not a known language code (an API-level error, not a content
            one -- validate the code before calling, or use
            :func:`tokenizer_language_hint` which degrades to ``None``).
    """
    spec = spec_for(lang)
    if spec.code in NATIVE_ROUTE_LANGS:
        return text or ""

    t = unicodedata.normalize("NFKC", text or "").strip()
    t = t.replace(" ", " ").replace("　", " ")
    t = "".join(_QUOTE_FOLD.get(c, _DASH_FOLD.get(c, c)) for c in t
                if c not in _COMBINING_STRESS)
    if spec.script == "cyrillic":
        t = _fold_confusables(t, spec.code)
    t = expand_symbols(t, spec)
    t = verbalize_numbers(t, spec)
    # Collapse whitespace and tidy punctuation spacing introduced by the expansions.
    t = _WS_RE.sub(" ", t)
    t = re.sub(r"\s+([,.!?;:…])", r"\1", t)
    t = re.sub(r"([«(])\s+", r"\1", t)
    t = re.sub(r"\s+([»)])", r"\1", t)
    return t.strip()


__all__ = [
    "LangSpec", "spec_for", "tokenizer_language_hint", "sanitize",
    "verbalize_numbers", "expand_symbols", "NATIVE_ROUTE_LANGS",
]
