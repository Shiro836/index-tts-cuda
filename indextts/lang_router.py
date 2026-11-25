"""Per-sentence language decision and view routing (docs/ROUTING.md §2).

Two model views live in one process: the PRISTINE base (en, zh) and the LoRA-merged
adapter (the 16 new languages). Every sentence has to pick one, and the pick has to be
right far more often than a bare LID call would be -- so the decision is a deterministic
script gate first, with GlotLID consulted only on the genuinely ambiguous branches and
always RESTRICTED to that branch's label set.

Measured on the 21k-row benchmark in the index-finetune repo (`eval/lid/`), pipeline
totals: GlotLID v3 **98.70%** > NLLB-lid218e 98.36% > OpenLID 97.56% > lid.176 96.50% >
lingua 91.15% (lingua has no Bashkir/Tatar at all). The reference implementation of the
gate that produced those numbers is `eval/lid/score_detectors.py:script_route`; the
regexes and the marker sets below are carried over from it verbatim. Do not "improve"
them without re-running that benchmark.

Two rules are load-bearing and were both paid for in misroutes:

1. Han without kana does NOT mean Chinese. Pure-kanji Japanese sentences exist and a
   hard han->zh rule mislabeled 108/1200 ja rows. GlotLID decides {ja, zh} instead.
2. Only UNIQUE Cyrillic marker letters may hard-decide. The shared ones (`ә ө ү ң һ`
   for ba/tt, `і` for uk/be) misrouted 865/1200 tt rows when they were allowed to.

Asymmetry rule (docs/ROUTING.md §2.6): bias every tie toward base/en. The base tokenizer
has ZERO Cyrillic, so new-script text sent to the base view comes out as `<unk>` garbage,
whereas Latin text sent to the base view merely renders with base-EN pronunciation.

Usage::

    router = LangRouter("/path/to/glotlid/model.bin")
    lang = router.decide("Привет, это тест")     # -> "ru"
    view = router.route(lang)                     # -> "adapter"

`decide()` returns ``None`` for junk (no letters at all, or GlotLID says `zxx`/`und` on
wordless input); the caller drops that sentence per docs/ROUTING.md §5.2.

Re-measured with this implementation on the same 21k rows (`latin_confidence_threshold=0`,
i.e. the exact algorithm the reference scored): 98.70% total / 99.68% full / 99.41%
informal / 96.94% short3 / 99.33% code-switched -- identical to the recorded numbers. See
:class:`LangRouter` for what the confidence floor costs on top of that.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# Language inventory
# --------------------------------------------------------------------------------------

#: Every code the API accepts, canonical two-letter form.
API_LANGS: Tuple[str, ...] = (
    "en", "zh", "ja", "ru", "uk", "be", "ba", "tt",
    "de", "fr", "es", "it", "pt", "pl", "tr", "nl", "cs", "sv",
)

#: Languages served by the PRISTINE base view (original tokenizer, 12001-row embedding).
BASE_LANGS = frozenset({"en", "zh"})
#: Languages served by the LoRA-merged adapter view (extended tokenizer, 16239 rows).
ADAPTER_LANGS = frozenset(set(API_LANGS) - BASE_LANGS)

#: Branch label sets. Order is documentation only -- restriction is an argmax over the set.
LATIN_LANGS: Tuple[str, ...] = ("en", "de", "fr", "es", "it", "pt", "pl", "tr", "nl", "cs", "sv")
CYRILLIC_LANGS: Tuple[str, ...] = ("ru", "uk", "be", "ba", "tt")
HAN_LANGS: Tuple[str, ...] = ("ja", "zh")

#: Aliases accepted from callers (BCP-47-ish tags, the training-side "sv-SE").
_ALIASES = {"sv-se": "sv", "zh-cn": "zh", "zh-hans": "zh", "zh-hant": "zh", "jp": "ja"}

#: Values of the API `language` field that mean "no forcing".
AUTO_TOKENS = frozenset({"", "auto", "none", "detect"})

# GlotLID v3 emits FLORES-style labels.
#
# CAREFUL: GlotLID v3 has NO `zho_*` label at all -- Mandarin is `cmn_Hani`, and the other
# Sinitic varieties are `yue_Hani`, `wuu_Hani`, `lzh_Hani`, ... (verified against the
# model's own 2102-label inventory). `eval/lid/score_detectors.py` uses the NLLB/FLORES
# spelling `zho_Hans` for every fasttext candidate, which simply never matches for GlotLID;
# that went unnoticed because the 21k benchmark contains no Chinese rows. Left uncorrected
# it sends EVERY Chinese sentence to `ja` -- i.e. to the adapter view. `zho_Hans/Hant` are
# kept below as forward-compat for models that do use them.
_FLORES_LABELS = {
    "eng_Latn": "en", "deu_Latn": "de", "fra_Latn": "fr", "spa_Latn": "es",
    "ita_Latn": "it", "por_Latn": "pt", "pol_Latn": "pl", "tur_Latn": "tr",
    "nld_Latn": "nl", "ces_Latn": "cs", "swe_Latn": "sv", "rus_Cyrl": "ru",
    "ukr_Cyrl": "uk", "bel_Cyrl": "be", "bak_Cyrl": "ba", "tat_Cyrl": "tt",
    "jpn_Jpan": "ja",
    "cmn_Hani": "zh", "zho_Hans": "zh", "zho_Hant": "zh",
}
#: `__label__xxx_Yyyy` -> API code.
LABEL_TO_LANG = {f"__label__{label}": code for label, code in _FLORES_LABELS.items()}

#: Any other Han-script label (`yue_Hani`, `wuu_Hani`, `lzh_Hani`, ...) is a Sinitic
#: variety: written Chinese, which the base view reads. Only consulted on the han branch,
#: where "zh" is in the allowed set -- it can never leak into the Latin/Cyrillic branches.
_HAN_SCRIPT_SUFFIX = "_Hani"

#: GlotLID's own "this is not language" labels (`zxx_*` = no linguistic content,
#: `und_*` = undetermined). Used only as a junk guard, never as a decision.
_NON_LANGUAGE_PREFIXES = ("__label__zxx", "__label__und")

# --------------------------------------------------------------------------------------
# Script gate (verbatim from eval/lid/score_detectors.py -- these ranges produced 98.70%)
# --------------------------------------------------------------------------------------

# Hiragana (U+3040-309F) + katakana (U+30A0-30FF), exactly as measured, plus halfwidth
# katakana (U+FF66-FF9D) which is Japanese-only too and reaches us un-NFKC'd from the API.
# Deliberately NOT the CJK Symbols and Punctuation block: 、。「」 are Chinese punctuation
# as much as Japanese.
RE_KANA = re.compile(r"[぀-ヿｦ-ﾝ]")
RE_HAN = re.compile(r"[一-鿿]")
RE_CYR = re.compile(r"[Ѐ-ӿ]")
RE_LAT = re.compile(r"[a-zA-Z]")
_RE_WORD_CHAR = re.compile(r"\w", re.UNICODE)

# UNIQUE marker chars only -- hard overrides. Shared chars (ә ө ү ң һ for ba/tt, і for
# uk/be) must NOT decide; see the module docstring.
BA_ONLY = frozenset("ҡғҙҫ")
TT_ONLY = frozenset("җ")
BE_ONLY = frozenset("ў")
UK_ONLY = frozenset("їєґ")


def normalize_code(lang: str) -> Optional[str]:
    """Canonicalize a caller-supplied language code, or ``None`` if it is not one of ours.

    Accepts case variations, region subtags ("sv-SE", "en_US"), the Go side's `{ru}`
    tag form, and the aliases in :data:`_ALIASES`. Returns ``None`` for anything else --
    callers decide whether that is a 400 or a ValueError.
    """
    if lang is None:
        return None
    key = str(lang).strip().strip("{}").strip().lower().replace("_", "-")
    if not key:
        return None
    if key in _ALIASES:
        return _ALIASES[key]
    if key in API_LANGS:
        return key
    head = key.split("-")[0]
    return head if head in API_LANGS else None


class LangRouter:
    """Decide a sentence's language and which model view serves it.

    The GlotLID model is ~1.6 GB on disk and takes seconds to load, so it is loaded
    EAGERLY in ``__init__`` (this runs at server boot, where a slow constructor is free
    and a slow first request is not). Pass ``preload=False`` to defer to the first
    :meth:`decide`, or inject ``model=`` to bypass loading entirely -- the only thing
    used is ``model.predict(text, k=-1) -> (labels, probs)``, which makes fakes trivial.

    Args:
        lid_model_path: path to GlotLID `model.bin` (`cis-lmu/glotlid`). May be ``None``
            when ``model`` is injected.
        latin_confidence_threshold: below this, the Latin branch falls back to the
            message-level dominant language (else `en`) instead of trusting GlotLID
            (docs/ROUTING.md §2.5). MEASURED on the 21k benchmark, which passes no
            message_dominant and so sees only the `en` half of the rule: the floor fires
            on 373 Latin rows (1.8% of the set) and GlotLID's low-confidence guess was
            right on 78.3% of them, so the floor costs total accuracy -- 98.70% at 0.0
            (exact parity with the measured pipeline), 98.44% at 0.02, 98.15% at 0.1,
            97.58% at 0.3. Default is 0.0: both failure modes are mild (a Latin sentence
            misrouted to base gets an EN accent; a misrouted short EN fragment gets a
            European accent), and 78% means the model's guess beats the en-fallback even
            when unsure. Raise it if production listening says otherwise.
        preload: load the model in the constructor (default) instead of lazily.
        model: pre-built/fake predictor, mainly for tests.
    """

    def __init__(
        self,
        lid_model_path: Optional[str],
        latin_confidence_threshold: float = 0.0,
        preload: bool = True,
        model=None,
    ) -> None:
        self.lid_model_path = lid_model_path
        self.latin_confidence_threshold = float(latin_confidence_threshold)
        self._model = model
        if model is None and preload:
            self._model = self._load_model()

    # -- model ---------------------------------------------------------------------

    def _load_model(self):
        """Load GlotLID through `fasttext-predict`.

        The plain `fasttext` wheels do not build on py>=3.10; `fasttext-predict` ships a
        predict-only binding with the same `load_model`/`predict` surface.
        """
        if not self.lid_model_path:
            raise ValueError("LangRouter needs a GlotLID model path (or an injected model)")
        import fasttext

        return fasttext.load_model(str(self.lid_model_path))

    @property
    def model(self):
        """The GlotLID predictor, loading it on first use if construction deferred it."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _predict(self, text: str) -> Tuple[Sequence[str], Sequence[float]]:
        """Full ranked prediction. fasttext rejects newlines, so they are spaces here.

        Predicts on LOWERCASED text: ALL-CAPS input wrecks GlotLID ("STOP! DON'T TOUCH
        THAT WIRE" scores eng at 0.82 cased vs 0.999 lowercased, and short caps shouts
        misroute entirely), while the 21k benchmark measured lowercasing as free —
        99.41% on the lowercased variant vs 99.68% cased.
        """
        labels, probs = self.model.predict(text.replace("\n", " ").lower(), k=-1)
        return labels, probs

    @staticmethod
    def _restrict(
        labels: Sequence[str], probs: Sequence[float], allowed: Iterable[str]
    ) -> Tuple[Optional[str], float]:
        """Argmax over the allowed subset: the first ranked label inside it wins.

        This *is* the restriction the 98.70% was measured with -- not a re-softmax, not a
        threshold on the unrestricted top. Labels outside the branch are simply skipped.
        """
        want = set(allowed)
        han_is_zh = "zh" in want
        for label, prob in zip(labels, probs):
            code = LABEL_TO_LANG.get(label)
            if code is None and han_is_zh and label.endswith(_HAN_SCRIPT_SUFFIX):
                code = "zh"
            if code is not None and code in want:
                return code, float(prob)
        return None, 0.0

    # -- decision ------------------------------------------------------------------

    def decide(
        self,
        text: str,
        forced: Optional[str] = None,
        message_dominant: Optional[str] = None,
    ) -> Optional[str]:
        """Language of one sentence, or ``None`` if it carries no language at all.

        Args:
            text: the sentence, as the caller split it.
            forced: API `language` field. A code from :data:`API_LANGS` wins outright;
                ``None`` or an auto token (see :data:`AUTO_TOKENS`) means auto-detect.
            message_dominant: language another sentence of the same message already
                decided, used only to break low-confidence Latin ties.

        Returns:
            An API code, or ``None`` for junk (punctuation/emoji-only input, or GlotLID
            reporting `zxx`/`und` on wordless text). ``None`` means DROP the sentence.

        Raises:
            ValueError: `forced` is a non-empty code that is not one of the 18.
        """
        if forced is not None and str(forced).strip().lower() not in AUTO_TOKENS:
            code = normalize_code(forced)
            if code is None:
                raise ValueError(
                    f"Unknown language {forced!r}; expected one of {', '.join(API_LANGS)}"
                )
            return code

        t = text or ""
        low = t.lower()

        # 1. Any kana is decisive: no other language uses it.
        if RE_KANA.search(low):
            return "ja"

        has_cyr = RE_CYR.search(low)

        # 2. Han without kana (and without Cyrillic): ja vs zh is GlotLID's call.
        if RE_HAN.search(low) and not has_cyr:
            return self._detect(t, HAN_LANGS, default="zh")

        # 3. Unique Cyrillic markers hard-override; 4. the rest go to GlotLID.
        #    Marker order matches the reference implementation.
        if has_cyr:
            chars = set(low)
            if chars & TT_ONLY:
                return "tt"
            if chars & BA_ONLY:
                return "ba"
            if chars & BE_ONLY:
                return "be"
            if chars & UK_ONLY:
                return "uk"
            return self._detect(t, CYRILLIC_LANGS, default="ru")

        # 5. Pure Latin: GlotLID restricted to the 11 Latin languages, with a confidence
        #    floor because 3-word fragments are frequently undecidable between them.
        if RE_LAT.search(low):
            fallback = normalize_code(message_dominant) or "en"
            return self._detect(
                t, LATIN_LANGS, default=fallback, min_confidence=self.latin_confidence_threshold
            )

        # 6. No letter in any script we serve: punctuation/emoji/digits-only, or a script
        #    no view can tokenize (Greek, Arabic, ...). Either way §5.2 drops the sentence
        #    -- sending it to the base view would only produce <unk> garbage.
        return None

    def _detect(
        self,
        text: str,
        allowed: Sequence[str],
        default: Optional[str],
        min_confidence: float = 0.0,
    ) -> Optional[str]:
        """One GlotLID call: junk guard on the unrestricted top, then restricted argmax."""
        labels, probs = self._predict(text)
        if labels and labels[0].startswith(_NON_LANGUAGE_PREFIXES) and not _RE_WORD_CHAR.search(text):
            return None
        lang, conf = self._restrict(labels, probs, allowed)
        if lang is None or conf < min_confidence:
            return default
        return lang

    # -- routing -------------------------------------------------------------------

    @staticmethod
    def route(lang: str) -> str:
        """`"base"` for en/zh, `"adapter"` for the 16 new languages.

        Raises:
            ValueError: unknown language code.
        """
        code = normalize_code(lang)
        if code is None:
            raise ValueError(f"Unknown language {lang!r}; cannot route it")
        return "base" if code in BASE_LANGS else "adapter"


__all__ = [
    "API_LANGS", "BASE_LANGS", "ADAPTER_LANGS",
    "LATIN_LANGS", "CYRILLIC_LANGS", "HAN_LANGS",
    "LABEL_TO_LANG", "AUTO_TOKENS",
    "BA_ONLY", "TT_ONLY", "BE_ONLY", "UK_ONLY",
    "normalize_code", "LangRouter",
]
