"""Shared fixtures: the fake GlotLID predictor and the real one (opt-in, slow)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Allow `python -m pytest tests/` from anywhere in the checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Real GlotLID weights (1.6 GB). Override with GLOTLID_MODEL for another checkout.
GLOTLID_PATH = os.environ.get(
    "GLOTLID_MODEL", "/home/forsen/repos/index-finetune/vllm_models/glotlid/model.bin"
)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: needs the real 1.6 GB GlotLID model on disk"
    )


class FakeFastText:
    """Stand-in for `fasttext.load_model()`'s object.

    `predict(text, k=-1)` returns `(labels, probs)` ranked descending, exactly like the
    real thing. Configure with a list of `(label, prob)` pairs, or a callable taking the
    text and returning such a list; every call is recorded in `.calls`.
    """

    def __init__(self, ranking):
        self.ranking = ranking
        self.calls = []

    def predict(self, text, k=-1):
        self.calls.append((text, k))
        ranking = self.ranking(text) if callable(self.ranking) else self.ranking
        labels = tuple(f"__label__{label}" for label, _ in ranking)
        probs = tuple(float(p) for _, p in ranking)
        return labels, probs


class ExplodingFastText:
    """Predictor that fails the test if the script gate ever consults it."""

    calls = ()

    def predict(self, text, k=-1):  # pragma: no cover - only reached on a bug
        raise AssertionError(f"LID must not be consulted for {text!r}")


@pytest.fixture
def fake_model():
    return FakeFastText


@pytest.fixture
def no_model():
    return ExplodingFastText()


@pytest.fixture(scope="session")
def glotlid_router():
    """A LangRouter on the real model; skips the test if the weights are absent."""
    if not Path(GLOTLID_PATH).is_file():
        pytest.skip(f"GlotLID model not found at {GLOTLID_PATH}")
    pytest.importorskip("fasttext", reason="fasttext-predict is not installed")
    from indextts.lang_router import LangRouter

    return LangRouter(GLOTLID_PATH)
