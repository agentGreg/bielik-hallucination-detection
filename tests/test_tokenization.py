from bielik_hallu.dataset import candidates
from transformers import AutoTokenizer
from bielik_hallu import config
from bielik_hallu.dataset.tokenization import token_length, tokenization_metadata

import pytest


def test_each_condition_has_42_entities():
    # n=42 per condition (Douglas Adams easter egg), matching the new domains.
    assert len(candidates.KNOWN) == 42
    assert len(candidates.UNKNOWN_REAL) == 42
    assert len(candidates.FABRICATED) == 42


def test_no_overlap_known_fabricated():
    assert set(candidates.KNOWN).isdisjoint(candidates.FABRICATED)


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(config.MODEL_ID)


def test_token_length_positive(tok):
    assert token_length(tok, "Robert Lewandowski") > 0


def test_metadata_has_key(tok):
    md = tokenization_metadata(tok, "Iga Świątek")
    assert md["n_tokens_entity"] == token_length(tok, "Iga Świątek")
