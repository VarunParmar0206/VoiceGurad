"""Tests for the FeatureResult -> model-input adapter (Phase 6)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.unit.ml_helpers import make_feature_result
from voiceguard.ml.errors import EmbeddingContractError
from voiceguard.ml.models import CNNLSTMAttention, pool_length
from voiceguard.voice.embedding import (
    batch_inputs,
    embed_batch,
    embed_model_input,
    prepare_input,
)


def test_prepare_input_components() -> None:
    feat = make_feature_result(t=44, seed=3)
    model_input = prepare_input(feat)
    # layout is always the fixed (n_mels, max_frames) axis
    assert model_input.mel.shape == (1, 1, 80, 300)
    assert model_input.mel.dtype == torch.float32
    assert int(model_input.lengths[0]) == 44
    assert model_input.padded is True


def test_prepare_input_caps_at_max_frames() -> None:
    feat = make_feature_result(t=400, seed=3)  # 400 > MAX_EMBEDDING_FRAMES=300
    model_input = prepare_input(feat)
    assert model_input.mel.shape == (1, 1, 80, 300)
    assert int(model_input.lengths[0]) == 300
    assert model_input.truncated is True


def test_prepare_input_lengths_match_actual_frames() -> None:
    feat = make_feature_result(t=123, seed=7)
    model_input = prepare_input(feat)
    assert int(model_input.lengths[0]) == min(123, 300)


def test_prepare_input_pads_with_zeros() -> None:
    feat = make_feature_result(t=10, seed=7)
    model_input = prepare_input(feat)
    padded_cols = model_input.mel[0, 0, :, 10:]
    assert torch.count_nonzero(padded_cols) == 0


def test_batch_inputs_common_layout() -> None:
    a_bi = prepare_input(make_feature_result(t=32, seed=1))
    b_bi = prepare_input(make_feature_result(t=64, seed=2))
    batch = batch_inputs([a_bi, b_bi])
    assert set(int(x) for x in batch.lengths.tolist()) == {32, 64}
    assert int(batch.mel.shape[0]) == 2
    assert batch.mel.shape[2] == 80
    assert int(batch.mel.shape[3]) == 300  # fixed frame axis


def test_batch_inputs_rejects_mixed_layout() -> None:
    a_bi = prepare_input(make_feature_result(t=64, seed=1))
    b_bi = prepare_input(make_feature_result(t=64, seed=2), max_frames=100)
    with pytest.raises(EmbeddingContractError):
        batch_inputs([a_bi, b_bi])


def test_contract_mel_layout() -> None:
    feat = make_feature_result(t=32)
    feat.mel.log_mel = np.zeros((64, 32), dtype=np.float32)  # wrong bands
    with pytest.raises(EmbeddingContractError):
        prepare_input(feat)


def test_contract_finite_values() -> None:
    feat = make_feature_result(t=32)
    feat.mel.log_mel[0, 1] = np.nan
    with pytest.raises(EmbeddingContractError):
        prepare_input(feat)


def test_embed_model_input_shape() -> None:
    torch.manual_seed(0)
    model = CNNLSTMAttention().eval()
    model_input = prepare_input(make_feature_result(t=48, seed=5))
    embedding = embed_model_input(model_input, model, batch_index=0)
    assert embedding.shape == (256,)
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)


def test_embed_batch_shape() -> None:
    torch.manual_seed(0)
    model = CNNLSTMAttention().eval()
    results = [
        make_feature_result(t=t, seed=s) for s, t in zip(range(3), (20, 60, 120), strict=True)
    ]
    embs = embed_batch(results, model)
    assert embs.shape == (3, 256)
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_model_contract_violations_consistent() -> None:
    feat = make_feature_result(t=4, seed=5)
    model_input = prepare_input(feat)
    assert int(model_input.lengths[0]) == 4
    # even the shortest viable input maps to a non-zero pooled length
    assert pool_length(pool_length(4)) == 1
