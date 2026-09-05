"""Tests for the training scaffold (losses, dataset, trainer, checkpoints)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from voiceguard.ml.errors import TrainingError
from voiceguard.ml.models import CNNLSTMAttention
from voiceguard.ml.training import (
    ArcFaceLoss,
    SyntheticSpeakerDataset,
    TrainingConfig,
    TripletLoss,
    collate_speaker_batch,
    deserialize_model_state,
    load_checkpoint,
    run_training,
    save_checkpoint,
    serialize_model_state,
    set_seed,
)


@pytest.fixture(scope="module")
def model() -> CNNLSTMAttention:
    set_seed(0)
    return CNNLSTMAttention()


def test_triplet_loss_sane(model: CNNLSTMAttention) -> None:
    set_seed(1)
    a = model(torch.randn(6, 1, 80, 64))
    loss = TripletLoss(margin=0.2)
    value = loss(a, torch.tensor([0, 0, 1, 1, 2, 2]))
    assert torch.isfinite(value)
    assert value.item() >= 0.0
    # identical anchor/positive/purposely allows zero found -> falls back to zero
    val2 = loss(a, torch.full((6,), 0))
    assert val2.item() == 0.0


def test_arcface_loss_sane(model: CNNLSTMAttention) -> None:
    set_seed(2)
    emb = torch.nn.functional.normalize(torch.randn(8, 64))
    loss = ArcFaceLoss(n_classes=4, embedding_dim=64, s=30.0, m=0.3)
    labels = torch.arange(4).repeat(2)
    value = loss(emb, labels)
    assert torch.isfinite(value)
    assert 0.0 <= value.item() < 20.0


def test_synthetic_dataset_shapes() -> None:
    set_seed(3)
    ds = SyntheticSpeakerDataset(
        n_speakers=4, n_utterances=6, n_mels=80, max_frames=64, seed=3
    )
    assert len(ds) == 24
    mel, length, label = ds[0]
    assert isinstance(mel, np.ndarray)
    assert mel.shape[0] == 80
    assert 32 <= mel.shape[1] <= 64  # length jitter within min/max frames
    assert int(length) == mel.shape[1]
    assert label == 0


def test_collate_speaker_batch_pads_max_length() -> None:
    ds = SyntheticSpeakerDataset(n_speakers=2, n_utterances=4, max_frames=64, seed=4)
    batch = [ds[i] for i in range(len(ds) - 2)]
    mels, lengths, labels = collate_speaker_batch(batch)
    assert mels.shape[0] == len(batch)
    assert mels.shape[1] == 1
    assert mels.shape[2] == 80
    assert int(mels.shape[3]) == int(lengths.max())
    assert torch.all(lengths <= 64)


def test_train_one_epoch_runs_and_reduces_loss() -> None:
    set_seed(5)
    cfg = TrainingConfig(
        seed=5, batch_size=8, epochs=1, max_frames=64, n_speakers=8, n_utterances=10
    )
    metrics = run_training(CNNLSTMAttention(), cfg)
    assert set(metrics) == {"triplet", "arcface", "combined"}
    assert metrics["triplet"] >= 0.0
    assert np.isfinite(metrics["triplet"])
    assert metrics["arcface"] >= 0.0
    assert abs(metrics["combined"] - (metrics["triplet"] + metrics["arcface"])) < 1e-6


def test_checkpoint_round_trip(model: CNNLSTMAttention, tmp_path: Path) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ckpt_path = tmp_path / "model.ckpt"
    save_checkpoint(
        ckpt_path,
        model,
        optimizer,
        epoch=3,
        extra={"description": "phase6"},
    )
    loaded = load_checkpoint(ckpt_path)
    assert loaded["epoch"] == 3
    assert loaded["extra"]["description"] == "phase6"
    # loading restores weights: forward behaves identically
    loaded_model = CNNLSTMAttention()
    loaded_model.load_state_dict(loaded["model_state"])
    loaded_model.eval()
    mel = torch.randn(2, 1, 80, 64)
    assert torch.allclose(model.eval()(mel), loaded_model(mel), atol=1e-6)


def test_checkpoint_rejects_corrupt_file(tmp_path: Path) -> None:
    p = tmp_path / "bad.ckpt"
    p.write_bytes(b"not-a-checkpoint")
    with pytest.raises(TrainingError):
        load_checkpoint(p)


def test_state_bytes_round_trip(model: CNNLSTMAttention) -> None:
    blob = serialize_model_state(model)
    assert isinstance(blob, bytes)
    assert len(blob) > 0
    state = deserialize_model_state(blob)
    assert set(state.keys()) == set(model.state_dict().keys())
    assert sum(v.numel() for v in state.values()) == sum(
        v.numel() for v in model.state_dict().values()
    )


def test_set_seed_restores_determinism() -> None:
    set_seed(7)
    a = torch.randn(5)
    set_seed(7)
    b = torch.randn(5)
    assert torch.allclose(a, b)
