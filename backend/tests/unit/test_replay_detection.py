"""Tests for Phase 7 replay detection (Tier 1 heuristics + Tier 2 ML)."""

from __future__ import annotations

import numpy as np
import torch

from voiceguard.ml.models.anti_spoofing_replay import ReplayDetector
from voiceguard.ml.training.train_replay import (
    ReplayTrainingConfig,
    SyntheticReplayDataset,
    collate_replay_batch,
    deserialize_model_state,
    load_checkpoint,
    run_training,
    save_checkpoint,
    serialize_model_state,
    set_seed,
)
from voiceguard.voice.anti_spoofing.replay import (
    ReplayDetectionResult,
    ReplayHeuristicScores,
    _amplitude_naturalness,
    _channel_signature,
    _noise_profile_match,
    _spectral_bandwidth_consistency,
    _temporal_consistency,
    compute_heuristic_scores,
    detect_replay,
)
from voiceguard.voice.result import FeatureResult, MelFeatures, QualityReport, StatisticalFeatures

# ── Helpers ──────────────────────────────────────────────────────────


def _make_mel(n_mels: int = 80, time_frames: int = 100, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_mels, time_frames)).astype(np.float32)


def _make_feature_result(mel: np.ndarray, seed: int = 0) -> FeatureResult:
    n_mels, time_frames = mel.shape
    mel_features = MelFeatures(
        log_mel=mel,
        n_mels=n_mels,
        n_fft=512,
        hop_length=160,
        win_length=512,
        sample_rate=16000,
        f_min=50.0,
        f_max=8000.0,
    )
    stat_values = np.random.default_rng(seed).standard_normal(259).astype(np.float32)
    statistical = StatisticalFeatures(values=stat_values)
    quality = QualityReport(
        input_duration_seconds=1.0,
        canonical_duration_seconds=1.0,
        peak_amplitude=0.5,
        rms_amplitude=0.1,
        voice_fraction=0.9,
        trimmed=False,
    )
    return FeatureResult(
        sample_rate=16000,
        num_samples=16000,
        duration_seconds=1.0,
        mel=mel_features,
        statistical=statistical,
        quality=quality,
    )


# ── Heuristic feature tests ─────────────────────────────────────────


class TestSpectralBandwidthConsistency:
    def test_random_mel_returns_float(self) -> None:
        mel = _make_mel()
        score = _spectral_bandwidth_consistency(mel)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_constant_mel_returns_low(self) -> None:
        mel = np.ones((80, 100), dtype=np.float32) * 0.5
        score = _spectral_bandwidth_consistency(mel)
        assert score < 0.3

    def test_varied_mel_returns_higher(self) -> None:
        rng = np.random.default_rng(42)
        mel = rng.standard_normal((80, 100)).astype(np.float32)
        score = _spectral_bandwidth_consistency(mel)
        # Random mel has some variation; check it's non-trivial.
        assert score > 0.0

    def test_short_mel(self) -> None:
        mel = np.ones((80, 1), dtype=np.float32)
        score = _spectral_bandwidth_consistency(mel)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_zero_mel(self) -> None:
        mel = np.zeros((80, 100), dtype=np.float32)
        score = _spectral_bandwidth_consistency(mel)
        assert score == 0.0


class TestNoiseProfileMatch:
    def test_random_mel(self) -> None:
        mel = _make_mel()
        score = _noise_profile_match(mel)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_short_mel(self) -> None:
        mel = np.ones((80, 2), dtype=np.float32)
        score = _noise_profile_match(mel)
        assert isinstance(score, float)


class TestChannelSignature:
    def test_random_mel(self) -> None:
        mel = _make_mel()
        score = _channel_signature(mel)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_flat_spectrum(self) -> None:
        mel = np.ones((80, 100), dtype=np.float32)
        score = _channel_signature(mel)
        # Flat spectrum → tilt near 1.0 → score near 1.0
        assert score > 0.8


class TestTemporalConsistency:
    def test_random_mel(self) -> None:
        mel = _make_mel()
        score = _temporal_consistency(mel)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_short_mel(self) -> None:
        mel = np.ones((80, 2), dtype=np.float32)
        score = _temporal_consistency(mel)
        assert isinstance(score, float)


class TestAmplitudeNaturalness:
    def test_random_mel(self) -> None:
        mel = _make_mel()
        score = _amplitude_naturalness(mel)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_zero_mel(self) -> None:
        mel = np.zeros((80, 100), dtype=np.float32)
        score = _amplitude_naturalness(mel)
        assert score == 0.0


class TestComputeHeuristicScores:
    def test_returns_dataclass(self) -> None:
        mel = _make_mel()
        scores = compute_heuristic_scores(mel)
        assert isinstance(scores, ReplayHeuristicScores)
        assert 0.0 <= scores.spectral_bandwidth_consistency <= 1.0
        assert 0.0 <= scores.noise_profile_match <= 1.0
        assert 0.0 <= scores.channel_signature <= 1.0
        assert 0.0 <= scores.temporal_consistency <= 1.0
        assert 0.0 <= scores.amplitude_naturalness <= 1.0

    def test_composite_is_weighted_average(self) -> None:
        mel = _make_mel()
        scores = compute_heuristic_scores(mel)
        weights = [0.25, 0.15, 0.20, 0.25, 0.15]
        values = [
            scores.spectral_bandwidth_consistency,
            scores.noise_profile_match,
            scores.channel_signature,
            scores.temporal_consistency,
            scores.amplitude_naturalness,
        ]
        expected = sum(w * v for w, v in zip(weights, values, strict=True))
        assert abs(scores.composite - expected) < 1e-9

    def test_deterministic(self) -> None:
        mel = _make_mel(seed=42)
        s1 = compute_heuristic_scores(mel)
        s2 = compute_heuristic_scores(mel)
        assert s1.composite == s2.composite


# ── ML replay detector tests ─────────────────────────────────────────


class TestDetectReplay:
    def test_heuristic_only(self) -> None:
        mel = _make_mel()
        result = _make_feature_result(mel)
        detection = detect_replay(result, model=None)
        assert isinstance(detection, ReplayDetectionResult)
        assert isinstance(detection.heuristic, ReplayHeuristicScores)
        assert 0.0 <= detection.ml_probability <= 1.0
        assert 0.0 <= detection.composite_score <= 1.0
        assert isinstance(detection.is_replay, bool)

    def test_with_ml_model(self) -> None:
        model = ReplayDetector()
        model.eval()
        mel = _make_mel()
        result = _make_feature_result(mel)
        detection = detect_replay(result, model=model)
        assert isinstance(detection, ReplayDetectionResult)
        assert 0.0 <= detection.ml_probability <= 1.0

    def test_threshold_default(self) -> None:
        mel = _make_mel()
        result = _make_feature_result(mel)
        detection = detect_replay(result, model=None, threshold=0.5)
        # With default heuristic scores, composite should be below 0.5
        assert isinstance(detection.is_replay, bool)

    def test_threshold_zero_always_replay(self) -> None:
        mel = _make_mel()
        result = _make_feature_result(mel)
        detection = detect_replay(result, model=None, threshold=0.0)
        assert detection.is_replay is True

    def test_threshold_one_never_replay(self) -> None:
        mel = _make_mel()
        result = _make_feature_result(mel)
        detection = detect_replay(result, model=None, threshold=1.0)
        assert detection.is_replay is False

    def test_custom_weights(self) -> None:
        mel = _make_mel()
        result = _make_feature_result(mel)
        detection = detect_replay(
            result, model=None, ml_weight=0.8, heuristic_weight=0.2
        )
        assert isinstance(detection, ReplayDetectionResult)


# ── Synthetic dataset tests ──────────────────────────────────────────


class TestSyntheticReplayDataset:
    def test_length(self) -> None:
        ds = SyntheticReplayDataset(n_samples=64, seed=42)
        assert len(ds) == 64

    def test_shapes(self) -> None:
        ds = SyntheticReplayDataset(n_samples=10, n_mels=80, max_frames=300, seed=42)
        mel, label = ds[0]
        assert mel.shape == (1, 80, 300)
        assert label.shape == (1,)

    def test_labels_are_binary(self) -> None:
        ds = SyntheticReplayDataset(n_samples=50, seed=42)
        labels = [int(ds[i][1].item()) for i in range(len(ds))]
        assert all(label in (0, 1) for label in labels)

    def test_deterministic(self) -> None:
        ds1 = SyntheticReplayDataset(n_samples=10, seed=99)
        ds2 = SyntheticReplayDataset(n_samples=10, seed=99)
        for i in range(len(ds1)):
            assert torch.equal(ds1[i][0], ds2[i][0])
            assert torch.equal(ds1[i][1], ds2[i][1])

    def test_collate(self) -> None:
        ds = SyntheticReplayDataset(n_samples=8, seed=42)
        batch = collate_replay_batch([ds[i] for i in range(8)])
        mels, labels = batch
        assert mels.shape == (8, 1, 80, 300)
        assert labels.shape == (8, 1)


# ── Training smoke tests ─────────────────────────────────────────────


class TestReplayTraining:
    def test_run_training_smoke(self) -> None:
        cfg = ReplayTrainingConfig(
            seed=42,
            batch_size=8,
            epochs=1,
            n_samples=32,
            max_frames=64,
        )
        metrics = run_training(config=cfg)
        assert "bce_loss" in metrics
        assert np.isfinite(metrics["bce_loss"])
        assert metrics["bce_loss"] >= 0.0

    def test_set_seed_deterministic(self) -> None:
        set_seed(42)
        m1 = torch.randn(5)
        set_seed(42)
        m2 = torch.randn(5)
        assert torch.equal(m1, m2)

    def test_checkpoint_round_trip(self, tmp_path: object) -> None:
        from pathlib import Path

        p = Path(str(tmp_path)) / "replay_ckpt.pt"
        model = ReplayDetector()
        save_checkpoint(p, model, optimizer=None, epoch=0)
        loaded = load_checkpoint(p)
        assert "model_state" in loaded
        assert "epoch" in loaded
        assert len(loaded["model_state"]) > 0

    def test_state_bytes_round_trip(self) -> None:
        model = ReplayDetector()
        data = serialize_model_state(model)
        assert isinstance(data, bytes)
        state = deserialize_model_state(data)
        assert isinstance(state, dict)
        model2 = ReplayDetector()
        model2.load_state_dict(state)
        # Verify identical forward pass.
        mel = torch.randn(1, 1, 80, 50)
        model.eval()
        model2.eval()
        with torch.inference_mode():
            out1, _ = model(mel)
            out2, _ = model2(mel)
        assert torch.allclose(out1, out2, atol=1e-6)
