"""Tests for the E6/E7 domain runner's dataset-building path (no GPU)."""
import pandas as pd
import pytest

from bielik_hallu import config
from bielik_hallu.dataset.candidates_domains import DOMAINS
from run_domains import NEW_DOMAINS, build_domain_labeled


class FakeTokenizer:
    """Whitespace tokenizer sufficient for token_length / tokenization_metadata."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


@pytest.mark.parametrize("domain", sorted(NEW_DOMAINS))
def test_build_domain_labeled_schema_and_counts(domain, tmp_path):
    out = tmp_path / domain / "labeled.parquet"
    build_domain_labeled(domain, FakeTokenizer(), out)
    df = pd.read_parquet(out)

    assert list(df.columns) == [
        "entity", "condition", "prompt", "label_hallucination", "n_tokens_entity"]
    # 3 conditions x 42 entities.
    assert len(df) == 3 * 42
    assert df["condition"].value_counts().to_dict() == {
        "KNOWN": 42, "UNKNOWN_REAL": 42, "FABRICATED": 42}


@pytest.mark.parametrize("domain", sorted(NEW_DOMAINS))
def test_build_domain_labeled_labels(domain, tmp_path):
    out = tmp_path / domain / "labeled.parquet"
    build_domain_labeled(domain, FakeTokenizer(), out)
    df = pd.read_parquet(out)

    known = df[df["condition"] == "KNOWN"]
    other = df[df["condition"] != "KNOWN"]
    assert (known["label_hallucination"] == 0).all()
    assert (other["label_hallucination"] == 1).all()


@pytest.mark.parametrize("domain", sorted(NEW_DOMAINS))
def test_build_domain_labeled_prompt_uses_domain_template(domain, tmp_path):
    out = tmp_path / domain / "labeled.parquet"
    build_domain_labeled(domain, FakeTokenizer(), out)
    df = pd.read_parquet(out)

    template = DOMAINS[domain]["prompt_template"]
    for _, row in df.iterrows():
        assert row["prompt"] == template.format(entity=row["entity"])
    if domain == "cities":
        assert df["prompt"].str.startswith("Czym jest ").all()
    else:
        assert df["prompt"].str.startswith("Kim jest ").all()


def test_build_domain_labeled_n_tokens_entity(tmp_path):
    out = tmp_path / "writers" / "labeled.parquet"
    build_domain_labeled("writers", FakeTokenizer(), out)
    df = pd.read_parquet(out)
    # FakeTokenizer splits on whitespace, so n_tokens == word count of entity.
    for _, row in df.iterrows():
        assert row["n_tokens_entity"] == len(row["entity"].split())


def test_conditions_constant_matches_expectation():
    # build_domain_labeled iterates config.CONDITIONS; guard the order/content.
    assert set(config.CONDITIONS) == {"KNOWN", "UNKNOWN_REAL", "FABRICATED"}
