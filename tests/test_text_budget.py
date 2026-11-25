"""The GPT text budget guard in _plan_sentences (infer_vllm_v2.py).

A chunk that encodes past `max_text_tokens` indexes off the end of the text
position embedding; the resulting device-side assert poisons the CUDA context for
the life of the process, so over-budget chunks must be re-split, never shipped.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm", reason="infer_vllm_v2 imports vLLM at module level")

from indextts.infer_vllm_v2 import IndexTTS2


class FakeTokenizer:
    def decode(self, ids):
        return " ".join(str(i) for i in ids)


def view(max_text_tokens=600):
    return SimpleNamespace(
        name="base",
        model=SimpleNamespace(max_text_tokens=max_text_tokens),
        tokenizer=FakeTokenizer(),
    )


def fit(token_ids, max_text_tokens=600, per_sentence=120, orig="text"):
    return IndexTTS2._fit_text_budget(None, view(max_text_tokens), orig, token_ids, per_sentence)


def test_chunk_within_budget_is_untouched():
    ids = list(range(120))
    assert fit(ids) == [("text", ids)]


def test_over_budget_chunk_is_split_to_the_sentence_budget():
    ids = list(range(1560))
    pieces = fit(ids)
    assert [len(frag) for _, frag in pieces] == [120] * 13
    assert [i for _, frag in pieces for i in frag] == ids


def test_every_piece_fits_the_gpt_budget():
    for total in (601, 1000, 5000):
        assert all(len(frag) <= 600 for _, frag in fit(list(range(total))))


def test_pieces_never_exceed_the_gpt_budget_when_the_sentence_budget_is_larger():
    pieces = fit(list(range(1200)), per_sentence=900)
    assert [len(frag) for _, frag in pieces] == [600, 600]


def test_piece_text_falls_back_to_the_original_when_decode_is_empty():
    v = view()
    v.tokenizer = SimpleNamespace(decode=lambda ids: "   ")
    pieces = IndexTTS2._fit_text_budget(None, v, "orig", list(range(700)), 120)
    assert all(text == "orig" for text, _ in pieces)


def test_missing_budget_disables_the_guard():
    v = view()
    v.model = SimpleNamespace()
    ids = list(range(5000))
    assert IndexTTS2._fit_text_budget(None, v, "text", ids, 120) == [("text", ids)]
