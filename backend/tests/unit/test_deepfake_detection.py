"""Tests for Phase 8 deepfake detection (Tier 1 heuristics + Tier 2 ML)."""

from __future__ import annotations

import numpy as np
import torch

from voiceguard.ml.models.anti_spoofing_deepfake import DeepfakeDetector
from voiceguard.ml.training.train_deepfake import (
    DeepfakeTrainingConfig,
    SyntheticDeepfakeDataset,
    collate_deepfake_batch,
    deserialize_model_state,
    load_checkpoint,
    run_training,
    save_checkpoint,
    serialize_model_state,
    set_seed,
)
from voiceguard.voice.anti_spoofing.deepfake import (
    DeepfakeDetectionResult,
    DeepfakeHeuristicScores,
    _cepstral_flatness,
    _frame_boundary_discontinuity,
    _high_frequency_periodicity,
    _spectral_smoothness,
    _temporal_regularity,
    compute_heuristic_scores,
    detect_deepfake,
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


class TestFrameBoundaryDiscontinuity:
    def test_random_mel_returns_float(self) -> None:
        mel = _make_mel()
        score = _frame_boundary_discontinuity(mel)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_smooth_mel_returns_low(self) -> None:
        """Smooth mel (small diffs) should have low discontinuity score."""
        mel = np.ones((80, 100), dtype=np.float32) * 0.5
        score = _frame_boundary_discontinuity(mel)
        assert score < 0.5

    def test_discontinuous_mel_returns_higher(self) -> None:
        """Mel with large frame-to-frame jumps should score higher."""
        mel = np.zeros((80, 100), dtype=np.float32)
        # Create large discontinuities.
        mel[:, 0] = 10.0
        mel[:, 50] = -10.0
        score = _frame_boundary_discontinuity(mel)
        smooth_score = _frame_boundary_discontinuity(
            np.ones((80, 100), dtype=np.float32) * 0.5
        )
        assert score > smooth_score

    def test_short_mel(self) -> None:
        mel = np.ones((80, 2), dtype=np.float32)
        score = _frame_boundary_discontinuity(mel)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestHighFrequencyPeriodicity:
    def test_random_mel(self) -> None:
        mel = _make_mel()
        score = _high_frequency_periodicity(mel)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_periodic_mel_returns_higher(self) -> None:
        """Mel with repeating pattern in high bands should score higher."""
        mel = np.zeros((80, 100), dtype=np.float32)
        # Create a repeating pattern in high-frequency bins.
        pattern = np.sin(np.linspace(0, 8 * np.pi, 16)).astype(np.float32)
        for i in range(60, 80):
            for j in range(0, 100, 16):
                end = min(j + 16, 100)
                mel[i, j:end] = pattern[: end - j]
        score = _high_frequency_periodicity(mel)
        assert score > 0.3

    def test_short_mel(self) -> None:
        mel = np.ones((80, 5), dtype=np.float32)
        score = _high_frequency_periodicity(mel)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestSpectralSmoothness:
    def test_random_mel(self) -> None:
        mel = _make_mel()
        score = _spectral_smoothness(mel)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_low_variance_mel_returns_higher(self) -> None:
        """Mel with low but non-zero spectral variance should score higher."""
        rng = np.random.default_rng(42)
        # High-variance mel.
        high_var_mel = rng.standard_normal((80, 100)).astype(np.float32)
        # Low-variance mel (small perturbations around a constant).
        low_var_mel = (np.ones((80, 100), dtype=np.float32) * 0.5
                       + rng.standard_normal((80, 100)).astype(np.float32) * 0.01)
        high_score = _spectral_smoothness(high_var_mel)
        low_score = _spectral_smoothness(low_var_mel)
        assert low_score > high_score

    def test_short_mel(self) -> None:
        mel = np.ones((80, 2), dtype=np.float32)
        score = _spectral_smoothness(mel)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestTemporalRegularity:
    def test_random_mel(self) -> None:
        mel = _make_mel()
        score = _temporal_regularity(mel)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_constant_mel_returns_higher(self) -> None:
        """Mel with very regular frame-to-frame diffs should score higher."""
        rng = np.random.default_rng(42)
        # High-variance mel (irregular diffs).
        irregular_mel = rng.standard_normal((80, 100)).astype(np.float32)
        # Low-variation mel (very regular small changes).
        regular_mel = (np.ones((80, 100), dtype=np.float32) * 0.5
                       + np.linspace(0, 0.1, 100).astype(np.float32))
        irregular_score = _temporal_regularity(irregular_mel)
        regular_score = _temporal_regularity(regular_mel)
        assert regular_score > irregular_score

    def test_short_mel(self) -> None:
        mel = np.ones((80, 3), dtype=np.float32)
        score = _temporal_regularity(mel)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestCepstralFlatness:
    def test_random_mel(self) -> None:
        mel = _make_mel()
        score = _cepstral_flatness(mel)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_flat_mel_returns_higher(self) -> None:
        """Mel with uniform frame-wise variance should score higher."""
        rng = np.random.default_rng(42)
        # Variable mel: each frame has very different spectral variance.
        variable_mel = np.zeros((80, 100), dtype=np.float32)
        for i in range(100):
            # Alternate between very high and very low variance frames.
            if i % 2 == 0:
                variable_mel[:, i] = rng.standard_normal(80).astype(np.float32) * 5.0
            else:
                variable_mel[:, i] = rng.standard_normal(80).astype(np.float32) * 0.01
        # Uniform mel: all frames have similar variance.
        uniform_mel = (np.ones((80, 100), dtype=np.float32) * 0.5
                       + rng.standard_normal((80, 100)).astype(np.float32) * 0.3)
        variable_score = _cepstral_flatness(variable_mel)
        uniform_score = _cepstral_flatness(uniform_mel)
        assert uniform_score > variable_score

    def test_short_mel(self) -> None:
        mel = np.ones((80, 2), dtype=np.float32)
        score = _cepstral_flatness(mel)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestComputeHeuristicScores:
    def test_returns_dataclass(self) -> None:
        mel = _make_mel()
        scores = compute_heuristic_scores(mel)
        assert isinstance(scores, DeepfakeHeuristicScores)
        assert 0.0 <= scores.frame_boundary_discontinuity <= 1.0
        assert 0.0 <= scores.high_frequency_periodicity <= 1.0
        assert 0.0 <= scores.spectral_smoothness <= 1.0
        assert 0.0 <= scores.temporal_regularity <= 1.0
        assert 0.0 <= scores.cepstral_flatness <= 1.0

    def test_composite_is_weighted_average(self) -> None:
        mel = _make_mel()
        scores = compute_heuristic_scores(mel)
        weights = [0.25, 0.20, 0.20, 0.20, 0.15]
        values = [
            scores.frame_boundary_discontinuity,
            scores.high_frequency_periodicity,
            scores.spectral_smoothness,
            scores.temporal_regularity,
            scores.cepstral_flatness,
        ]
        expected = sum(w * v for w, v in zip(weights, values, strict=True))
        assert abs(scores.composite - expected) < 1e-9

    def test_deterministic(self) -> None:
        mel = _make_mel(seed=42)
        s1 = compute_heuristic_scores(mel)
        s2 = compute_heuristic_scores(mel)
        assert s1.composite == s2.composite


# ── ML deepfake detector tests ──────────────────────────────────────


class TestDetectDeepfake:
    def test_heuristic_only(self) -> None:
        mel = _make_mel()
        result = _make_feature_result(mel)
        detection = detect_deepfake(result, model=None)
        assert isinstance(detection, DeepfakeDetectionResult)
        assert isinstance(detection.heuristic, DeepfakeHeuristicScores)
        assert 0.0 <= detection.ml_probability <= 1.0
        assert 0.0 <= detection.composite_score <= 1.0
        assert isinstance(detection.is_synthetic, bool)

    def test_with_ml_model(self) -> None:
        model = DeepfakeDetector()
        model.eval()
        mel = _make_mel()
        result = _make_feature_result(mel)
        detection = detect_deepfake(result, model=model)
        assert isinstance(detection, DeepfakeDetectionResult)
        assert 0.0 <= detection.ml_probability <= 1.0

    def test_threshold_zero_always_synthetic(self) -> None:
        mel = _make_mel()
        result = _make_feature_result(mel)
        detection = detect_deepfake(result, model=None, threshold=0.0)
        assert detection.is_synthetic is True

    def test_threshold_one_never_synthetic(self) -> None:
        mel = _make_mel()
        result = _make_feature_result(mel)
        detection = detect_deepfake(result, model=None, threshold=1.0)
        assert detection.is_synthetic is False

    def test_custom_weights(self) -> None:
        mel = _make_mel()
        result = _make_feature_result(mel)
        detection = detect_deepfake(
            result, model=None, ml_weight=0.8, heuristic_weight=0.2
        )
        assert isinstance(detection, DeepfakeDetectionResult)

    def test_synthetic_mel_scores_higher_than_natural(self) -> None:
        """Synthetic-pattern mel should score higher than random mel."""
        rng = np.random.default_rng(42)
        natural_mel = rng.standard_normal((80, 100)).astype(np.float32)

        # Create a mel with deepfake-like artifacts.
        synthetic_mel = rng.standard_normal((80, 100)).astype(np.float32)
        # Add frame boundary discontinuities.
        for i in range(0, 100, 8):
            if i + 1 < 100:
                synthetic_mel[:, i] += rng.uniform(0.5, 1.0, 80).astype(np.float32)
        # Add periodicity (copy first 8 columns to later positions).
        pattern = synthetic_mel[:, 0:8].copy()
        for i in range(8, 100, 8):
            end = min(i + 8, 100)
            synthetic_mel[:, i:end] = pattern[:, : end - i]

        natural_result = _make_feature_result(natural_mel)
        synthetic_result = _make_feature_result(synthetic_mel)
        natural_det = detect_deepfake(natural_result, model=None, threshold=0.5)
        synthetic_det = detect_deepfake(synthetic_result, model=None, threshold=0.5)
        # Heuristic composite should be higher for synthetic.
        assert synthetic_det.heuristic.composite >= natural_det.heuristic.composite


# ── Synthetic dataset tests ──────────────────────────────────────────


class TestSyntheticDeepfakeDataset:
    def test_length(self) -> None:
        ds = SyntheticDeepfakeDataset(n_samples=64, seed=42)
        assert len(ds) == 64

    def test_shapes(self) -> None:
        ds = SyntheticDeepfakeDataset(n_samples=10, n_mels=80, max_frames=300, seed=42)
        mel, label = ds[0]
        assert mel.shape == (1, 80, 300)
        assert label.shape == (1,)

    def test_labels_are_binary(self) -> None:
        ds = SyntheticDeepfakeDataset(n_samples=50, seed=42)
        labels = [int(ds[i][1].item()) for i in range(len(ds))]
        assert all(label in (0, 1) for label in labels)

    def test_deterministic(self) -> None:
        ds1 = SyntheticDeepfakeDataset(n_samples=10, seed=99)
        ds2 = SyntheticDeepfakeDataset(n_samples=10, seed=99)
        for i in range(len(ds1)):
            assert torch.equal(ds1[i][0], ds2[i][0])
            assert torch.equal(ds1[i][1], ds2[i][1])

    def test_collate(self) -> None:
        ds = SyntheticDeepfakeDataset(n_samples=8, seed=42)
        batch = collate_deepfake_batch([ds[i] for i in range(8)])
        mels, labels = batch
        assert mels.shape == (8, 1, 80, 300)
        assert labels.shape == (8, 1)


# ── Training smoke tests ─────────────────────────────────────────────


class TestDeepfakeTraining:
    def test_run_training_smoke(self) -> None:
        cfg = DeepfakeTrainingConfig(
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

        p = Path(str(tmp_path)) / "deepfake_ckpt.pt"
        model = DeepfakeDetector()
        save_checkpoint(p, model, optimizer=None, epoch=0)
        loaded = load_checkpoint(p)
        assert "model_state" in loaded
        assert "epoch" in loaded
        assert len(loaded["model_state"]) > 0

    def test_state_bytes_round_trip(self) -> None:
        model = DeepfakeDetector()
        data = serialize_model_state(model)
        assert isinstance(data, bytes)
        state = deserialize_model_state(data)
        assert isinstance(state, dict)
        model2 = DeepfakeDetector()
        model2.load_state_dict(state)
        # Verify identical forward pass.
        mel = torch.randn(1, 1, 80, 50)
        model.eval()
        model2.eval()
        with torch.inference_mode():
            out1, _ = model(mel)
            out2, _ = model2(mel)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_synthetic_dataset_has_both_classes(self) -> None:
        """Verify dataset actually produces both bona fide and synthetic."""
        ds = SyntheticDeepfakeDataset(n_samples=100, seed=42)
        labels = [int(ds[i][1].item()) for i in range(len(ds))]
        assert 0 in labels, "No bona fide samples generated"
        assert 1 in labels, "No synthetic samples generated"
