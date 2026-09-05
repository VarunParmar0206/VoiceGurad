"""Tests for the CNN-LSTM-Attention speaker-embedding model (Phase 6)."""

from __future__ import annotations

import pytest
import torch

from voiceguard.ml.errors import ModelError
from voiceguard.ml.models import CNNLSTMAttention, pool_length


@pytest.fixture(scope="module")
def model() -> CNNLSTMAttention:
    torch.manual_seed(0)
    return CNNLSTMAttention().eval()


def test_pool_length_math() -> None:
    # MaxPool2d(kernel=2, stride=2) valid-mode length: (L-2)//2 + 1, min 1
    assert pool_length(8) == 4
    assert pool_length(44) == 22
    assert pool_length(3) == 1
    assert pool_length(4, kernel=3, stride=1) == 2


def test_pool_length_minimum() -> None:
    assert pool_length(1) == 1
    assert pool_length(2) == 1


def test_forward_output_shape_and_norm(model: CNNLSTMAttention) -> None:
    mel = torch.randn(4, 1, 80, 256)
    out = model(mel)
    assert out.shape == (4, 256)
    assert torch.allclose(out.norm(dim=1), torch.ones(4), atol=1e-5)


def test_forward_accepts_valid_lengths(model: CNNLSTMAttention) -> None:
    mel = torch.randn(3, 1, 80, 100)
    lengths = torch.tensor([100, 64, 20], dtype=torch.long)
    out = model(mel, lengths)
    assert out.shape == (3, 256)
    assert torch.isfinite(out).all()


def test_forward_rejects_invalid_shapes(model: CNNLSTMAttention) -> None:
    with pytest.raises(ModelError):
        model(torch.randn(3, 2, 80, 64))  # wrong channel dim
    with pytest.raises(ModelError):
        model(torch.randn(3, 1, 64, 64))  # wrong mel bands
    with pytest.raises(ModelError):
        model(torch.randn(3, 1, 80, 64), torch.tensor([64, 9999, 8], dtype=torch.long))
    with pytest.raises(ModelError):
        model(torch.randn(3, 1, 80, 64), torch.tensor([32, 64], dtype=torch.long))


def test_forward_minimal_time_axis(model: CNNLSTMAttention) -> None:
    # A 1-frame input cannot survive two 2x MaxPool layers.
    with pytest.raises(ModelError):
        model(torch.randn(1, 1, 80, 1))
    # T=4 -> pooled length 1 still yields a finite embedding.
    out = model(torch.randn(1, 1, 80, 4))
    assert out.shape == (1, 256)
    assert torch.isfinite(out).all()


def test_forward_rejects_non_finite(model: CNNLSTMAttention) -> None:
    mel = torch.randn(2, 1, 80, 64)
    mel[0, 0, 0, 0] = float("inf")
    with pytest.raises(ModelError):
        model(mel)


def test_attention_weights_sum_to_one(model: CNNLSTMAttention) -> None:
    mel = torch.randn(4, 1, 80, 120)
    lengths = torch.tensor([120, 80, 40, 16], dtype=torch.long)
    alpha = model.attention_weights(mel, lengths)
    assert alpha.shape == (4, 30)  # 120 -> pooled -> 30
    assert torch.allclose(alpha.sum(dim=1), torch.ones(4), atol=1e-5)
    assert (alpha >= 0).all()


def test_attention_zeroes_padded_frames(model: CNNLSTMAttention) -> None:
    mel = torch.randn(2, 1, 80, 64)
    lengths = torch.tensor([64, 16], dtype=torch.long)
    alpha = model.attention_weights(mel, lengths)
    pooled = pool_length(pool_length(16))  # 16 -> 8 -> 4
    assert (alpha[1, pooled:] < 1e-6).all()


def test_cnn_lstm_attention_deterministic() -> None:
    torch.manual_seed(123)
    a = CNNLSTMAttention().eval()
    torch.manual_seed(123)
    b = CNNLSTMAttention().eval()
    mel = torch.randn(2, 1, 80, 96)
    assert torch.allclose(a(mel), b(mel), atol=1e-6)


def test_num_params_sane() -> None:
    torch.manual_seed(0)
    with torch.no_grad():
        n = sum(p.numel() for p in CNNLSTMAttention().parameters())
    assert 2_000_000 < n < 5_000_000  # bounded parametric budget
