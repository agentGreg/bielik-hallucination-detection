import numpy as np
import pytest

from bielik_hallu.extract.extended import (ATTN_PER_HEAD_COLUMNS,
                                           attention_entropy,
                                           attention_entropy_per_head,
                                           effective_rank, logit_lens_entropy)


# --- effective rank ---

def test_effective_rank_identity_is_n():
    assert effective_rank(np.eye(4)) == pytest.approx(4.0)


def test_effective_rank_rank_one_is_one():
    m = np.outer([1.0, 2.0, 3.0], [4.0, 5.0])
    assert effective_rank(m) == pytest.approx(1.0)


def test_effective_rank_zero_matrix_is_nan():
    assert np.isnan(effective_rank(np.zeros((3, 3))))


def test_effective_rank_scale_invariant():
    m = np.random.default_rng(0).normal(size=(6, 4))
    assert effective_rank(10.0 * m) == pytest.approx(effective_rank(m))


def test_effective_rank_bounded_by_min_dim():
    m = np.random.default_rng(1).normal(size=(5, 3))
    er = effective_rank(m)
    assert 1.0 <= er <= 3.0


# --- attention entropy ---

def test_attention_entropy_uniform():
    raw, norm = attention_entropy(np.ones(8) / 8)
    assert raw == pytest.approx(np.log(8))
    assert norm == pytest.approx(1.0)


def test_attention_entropy_one_hot_is_zero():
    raw, norm = attention_entropy(np.array([0.0, 1.0, 0.0, 0.0]))
    assert raw == pytest.approx(0.0)
    assert norm == pytest.approx(0.0)


def test_attention_entropy_single_key():
    raw, norm = attention_entropy(np.array([1.0]))
    assert raw == pytest.approx(0.0)
    assert norm == 0.0


def test_attention_entropy_renormalizes_drift():
    # Low-precision attention rows may not sum exactly to 1.
    raw, norm = attention_entropy(np.ones(4) * 0.2501)
    assert raw == pytest.approx(np.log(4))
    assert norm == pytest.approx(1.0)


def test_attention_entropy_all_zero_is_nan():
    raw, norm = attention_entropy(np.zeros(4))
    assert np.isnan(raw) and np.isnan(norm)


# --- per-head attention entropy (E7) ---

def _synthetic_attn(n_heads: int, seq: int) -> np.ndarray:
    """A (heads, seq, seq) lower-triangular, row-normalized attention tensor.

    Each query row attends uniformly over its causally visible keys, matching
    the autoregressive mask the real model enforces.
    """
    attn = np.zeros((n_heads, seq, seq), dtype=np.float64)
    for h in range(n_heads):
        for q in range(seq):
            attn[h, q, : q + 1] = 1.0 / (q + 1)
    return attn


def test_per_head_returns_one_tuple_per_head():
    attn = _synthetic_attn(n_heads=5, seq=4)
    per_head = attention_entropy_per_head(attn, query_idx=3)
    assert len(per_head) == 5
    assert all(len(t) == 2 for t in per_head)


def test_per_head_uniform_causal_row_matches_log_visible_keys():
    # Query index 3 sees keys 0..3 -> 4 visible, uniform -> raw = log(4), norm = 1.
    attn = _synthetic_attn(n_heads=3, seq=6)
    for raw, norm in attention_entropy_per_head(attn, query_idx=3):
        assert raw == pytest.approx(np.log(4))
        assert norm == pytest.approx(1.0)


def test_per_head_first_query_is_single_key_zero_entropy():
    # Query index 0 sees only key 0 -> forced distribution -> raw = 0, norm = 0.
    attn = _synthetic_attn(n_heads=4, seq=5)
    for raw, norm in attention_entropy_per_head(attn, query_idx=0):
        assert raw == pytest.approx(0.0)
        assert norm == 0.0


def test_per_head_mean_equals_layer_mean_convention():
    # The per-layer attn_entropy_mean is defined as the mean over these
    # per-head raw values; verify that identity on a heterogeneous tensor.
    rng = np.random.default_rng(7)
    seq, n_heads = 5, 6
    attn = np.zeros((n_heads, seq, seq), dtype=np.float64)
    for h in range(n_heads):
        for q in range(seq):
            w = rng.random(q + 1) + 1e-3
            attn[h, q, : q + 1] = w / w.sum()
    idx = 4
    per_head = attention_entropy_per_head(attn, query_idx=idx)
    expected_mean = np.mean([attention_entropy(attn[h, idx, : idx + 1])[0]
                             for h in range(n_heads)])
    assert np.mean([raw for raw, _ in per_head]) == pytest.approx(expected_mean)


def test_per_head_distinct_heads_give_distinct_entropies():
    # One-hot head vs uniform head must yield different entropies at same query.
    seq = 4
    attn = np.zeros((2, seq, seq), dtype=np.float64)
    attn[0, 3, :4] = [1.0, 0.0, 0.0, 0.0]   # peaked
    attn[1, 3, :4] = 0.25                    # uniform
    per_head = attention_entropy_per_head(attn, query_idx=3)
    assert per_head[0][0] == pytest.approx(0.0)
    assert per_head[1][0] == pytest.approx(np.log(4))


def test_attn_per_head_columns_schema():
    assert ATTN_PER_HEAD_COLUMNS == (
        "entity", "condition", "layer", "head", "point",
        "attn_entropy", "attn_entropy_norm")


# --- logit-lens entropy ---

def test_logit_lens_entropy_uniform_logits_is_log_vocab():
    assert logit_lens_entropy(np.zeros(100)) == pytest.approx(np.log(100))


def test_logit_lens_entropy_peaked_logits_is_zero():
    z = np.zeros(100)
    z[0] = 1000.0
    assert logit_lens_entropy(z) == pytest.approx(0.0, abs=1e-6)


def test_logit_lens_entropy_shift_invariant():
    z = np.random.default_rng(2).normal(size=50)
    assert logit_lens_entropy(z + 123.0) == pytest.approx(logit_lens_entropy(z))


def test_logit_lens_entropy_stable_for_large_logits():
    # Max-subtraction must prevent overflow.
    z = np.array([1e4, 1e4, -1e4])
    assert logit_lens_entropy(z) == pytest.approx(np.log(2))
