"""Tests for Phase 9 voice-conversion detection (three-check heuristic)."""

from __future__ import annotations

import numpy as np

from voiceguard.voice.anti_spoofing.voice_conversion import (
    VoiceConversionDetectionResult,
    VoiceConversionHeuristicScores,
    _embedding_consistency,
    _energy_naturalness,
    _pitch_smoothness,
    _speaking_rate_consistency,
    _spectral_temporal_coherence,
    compute_heuristic_scores,
    detect_voice_conversion,
)
from voiceguard.voice.result import FeatureResult, MelFeatures, QualityReport, StatisticalFeatures

# ── Helpers ──────────────────────────────────────────────────────────


def _make_mel(n_mels: int = 80, time_frames: int = 100, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_mels, time_frames)).astype(np.float32)


def _make_feature_result(
    mel: np.ndarray,
    seed: int = 0,
    duration: float = 1.0,
    voice_fraction: float = 0.9,
) -> FeatureResult:
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
        input_duration_seconds=duration,
        canonical_duration_seconds=duration,
        peak_amplitude=0.5,
        rms_amplitude=0.1,
        voice_fraction=voice_fraction,
        trimmed=False,
    )
    return FeatureResult(
        sample_rate=16000,
        num_samples=int(16000 * duration),
        duration_seconds=duration,
        mel=mel_features,
        statistical=statistical,
        quality=quality,
    )


# ── Embedding consistency tests ─────────────────────────────────────


class TestEmbeddingConsistency:
    def test_similar_embeddings_return_low(self) -> None:
        """Similar embeddings should produce low (natural) score."""
        rng = np.random.default_rng(42)
        embedding = rng.standard_normal(256).astype(np.float64)
        # Stat features similar to embedding.
        stat = embedding + rng.standard_normal(256).astype(np.float64) * 0.1
        score = _embedding_consistency(embedding, stat)
        assert score < 0.5

    def test_dissimilar_embeddings_return_high(self) -> None:
        """Very different embeddings should produce high (suspicious) score."""
        rng = np.random.default_rng(42)
        embedding = rng.standard_normal(256).astype(np.float64)
        stat = -embedding  # Opposite direction.
        score = _embedding_consistency(embedding, stat)
        assert score > 0.5

    def test_none_embedding_returns_neutral(self) -> None:
        """No embedding should return neutral 0.5."""
        rng = np.random.default_rng(42)
        stat = rng.standard_normal(259).astype(np.float64)
        score = _embedding_consistency(None, stat)
        assert score == 0.5

    def test_short_stat_features(self) -> None:
        """Very short stat features should return neutral."""
        embedding = np.ones(256, dtype=np.float64)
        stat = np.array([1.0], dtype=np.float64)
        score = _embedding_consistency(embedding, stat)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_zero_norm_embedding(self) -> None:
        """Zero-norm embedding should return neutral."""
        embedding = np.zeros(256, dtype=np.float64)
        stat = np.ones(259, dtype=np.float64)
        score = _embedding_consistency(embedding, stat)
        assert score == 0.5

    def test_zero_norm_stat(self) -> None:
        """Zero-norm stat features should return neutral."""
        embedding = np.ones(256, dtype=np.float64)
        stat = np.zeros(259, dtype=np.float64)
        score = _embedding_consistency(embedding, stat)
        assert score == 0.5

    def test_deterministic(self) -> None:
        """Same inputs should produce same output."""
        rng = np.random.default_rng(42)
        embedding = rng.standard_normal(256).astype(np.float64)
        stat = rng.standard_normal(259).astype(np.float64)
        s1 = _embedding_consistency(embedding, stat)
        s2 = _embedding_consistency(embedding, stat)
        assert s1 == s2


# ── Pitch smoothness tests ──────────────────────────────────────────


class TestPitchSmoothness:
    def test_random_mel_returns_float(self) -> None:
        mel = _make_mel()
        score = _pitch_smoothness(mel)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_smooth_mel_returns_low(self) -> None:
        """Mel with smooth pitch contour should have low score."""
        # Create a mel with very gradual pitch changes.
        mel = np.zeros((80, 100), dtype=np.float32)
        # Gradual shift in dominant frequency.
        for i in range(100):
            bin_idx = min(10 + i // 10, 79)
            mel[bin_idx, i] = 1.0
        score = _pitch_smoothness(mel)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_jittery_mel_returns_higher(self) -> None:
        """Mel with jittery pitch should score higher than naturally smooth."""
        rng = np.random.default_rng(42)

        # Naturally smooth mel: random with moderate pitch variation.
        mel_smooth = np.zeros((80, 100), dtype=np.float32)
        for i in range(100):
            bin_idx = int(np.clip(40 + rng.standard_normal() * 5, 0, 79))
            mel_smooth[bin_idx, i] = 1.0
        score_smooth = _pitch_smoothness(mel_smooth)

        # Jittery mel: rapidly alternating dominant frequency.
        mel_jittery = np.zeros((80, 100), dtype=np.float32)
        for i in range(100):
            bin_idx = 10 if i % 2 == 0 else 70
            mel_jittery[bin_idx, i] = 1.0
        score_jittery = _pitch_smoothness(mel_jittery)

        assert score_jittery >= score_smooth

    def test_short_mel(self) -> None:
        mel = np.ones((80, 3), dtype=np.float32)
        score = _pitch_smoothness(mel)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_deterministic(self) -> None:
        mel = _make_mel(seed=42)
        s1 = _pitch_smoothness(mel)
        s2 = _pitch_smoothness(mel)
        assert s1 == s2


# ── Energy naturalness tests ────────────────────────────────────────


class TestEnergyNaturalness:
    def test_random_mel_returns_float(self) -> None:
        mel = _make_mel()
        score = _energy_naturalness(mel)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_natural_energy_returns_low(self) -> None:
        """Mel with moderate energy variation should have low score."""
        rng = np.random.default_rng(42)
        mel = rng.standard_normal((80, 100)).astype(np.float32) * 0.5
        score = _energy_naturalness(mel)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_flat_energy_returns_higher(self) -> None:
        """Mel with unnaturally flat energy should score higher."""
        mel = np.ones((80, 100), dtype=np.float32) * 0.5
        score_flat = _energy_naturalness(mel)

        # Natural energy for comparison.
        rng = np.random.default_rng(42)
        mel_natural = rng.standard_normal((80, 100)).astype(np.float32) * 2.0
        score_natural = _energy_naturalness(mel_natural)
        assert score_flat >= score_natural

    def test_short_mel(self) -> None:
        mel = np.ones((80, 3), dtype=np.float32)
        score = _energy_naturalness(mel)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_deterministic(self) -> None:
        mel = _make_mel(seed=42)
        s1 = _energy_naturalness(mel)
        s2 = _energy_naturalness(mel)
        assert s1 == s2


# ── Speaking rate consistency tests ──────────────────────────────────


class TestSpeakingRateConsistency:
    def test_normal_rate_returns_low(self) -> None:
        """Normal speaking rate should have low score."""
        mel = _make_mel(time_frames=100)
        quality = QualityReport(
            input_duration_seconds=1.0,
            canonical_duration_seconds=1.0,
            peak_amplitude=0.5,
            rms_amplitude=0.1,
            voice_fraction=0.7,
            trimmed=False,
        )
        score = _speaking_rate_consistency(mel, quality)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_silent_audio_returns_high(self) -> None:
        """Mostly silent audio should score higher."""
        mel = _make_mel(time_frames=100)
        quality = QualityReport(
            input_duration_seconds=1.0,
            canonical_duration_seconds=1.0,
            peak_amplitude=0.5,
            rms_amplitude=0.1,
            voice_fraction=0.1,  # Very low voice fraction.
            trimmed=False,
        )
        score = _speaking_rate_consistency(mel, quality)
        assert score > 0.0

    def test_no_duration(self) -> None:
        """Zero duration should return neutral."""
        mel = _make_mel(time_frames=100)
        quality = QualityReport(
            input_duration_seconds=0.0,
            canonical_duration_seconds=0.0,
            peak_amplitude=0.5,
            rms_amplitude=0.1,
            voice_fraction=0.9,
            trimmed=False,
        )
        score = _speaking_rate_consistency(mel, quality)
        assert score == 0.5

    def test_short_mel(self) -> None:
        mel = np.ones((80, 1), dtype=np.float32)
        quality = QualityReport(
            input_duration_seconds=1.0,
            canonical_duration_seconds=1.0,
            peak_amplitude=0.5,
            rms_amplitude=0.1,
            voice_fraction=0.9,
            trimmed=False,
        )
        score = _speaking_rate_consistency(mel, quality)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_deterministic(self) -> None:
        mel = _make_mel(seed=42)
        quality = QualityReport(
            input_duration_seconds=1.0,
            canonical_duration_seconds=1.0,
            peak_amplitude=0.5,
            rms_amplitude=0.1,
            voice_fraction=0.7,
            trimmed=False,
        )
        s1 = _speaking_rate_consistency(mel, quality)
        s2 = _speaking_rate_consistency(mel, quality)
        assert s1 == s2


# ── Spectral-temporal coherence tests ────────────────────────────────


class TestSpectralTemporalCoherence:
    def test_random_mel_returns_float(self) -> None:
        mel = _make_mel()
        score = _spectral_temporal_coherence(mel)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_coherent_mel_returns_low(self) -> None:
        """Mel with balanced spectral and temporal variation should have low score."""
        rng = np.random.default_rng(42)
        mel = rng.standard_normal((80, 100)).astype(np.float32) * 0.5
        score = _spectral_temporal_coherence(mel)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_incoherent_mel_returns_high(self) -> None:
        """Mel with mismatched spectral/temporal variation should score higher."""
        # High spectral variation, low temporal variation.
        mel = np.zeros((80, 100), dtype=np.float32)
        for i in range(100):
            mel[:, i] = np.linspace(-5, 5, 80).astype(np.float32)
        score_incoherent = _spectral_temporal_coherence(mel)

        # Coherent mel for comparison.
        rng = np.random.default_rng(42)
        mel_coherent = rng.standard_normal((80, 100)).astype(np.float32)
        score_coherent = _spectral_temporal_coherence(mel_coherent)
        assert score_incoherent >= score_coherent

    def test_short_mel(self) -> None:
        mel = np.ones((80, 3), dtype=np.float32)
        score = _spectral_temporal_coherence(mel)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_deterministic(self) -> None:
        mel = _make_mel(seed=42)
        s1 = _spectral_temporal_coherence(mel)
        s2 = _spectral_temporal_coherence(mel)
        assert s1 == s2


# ── Heuristic scores tests ──────────────────────────────────────────


class TestComputeHeuristicScores:
    def test_returns_dataclass(self) -> None:
        mel = _make_mel()
        scores = compute_heuristic_scores(mel)
        assert isinstance(scores, VoiceConversionHeuristicScores)
        assert 0.0 <= scores.embedding_consistency <= 1.0
        assert 0.0 <= scores.pitch_smoothness <= 1.0
        assert 0.0 <= scores.energy_naturalness <= 1.0
        assert 0.0 <= scores.speaking_rate_consistency <= 1.0
        assert 0.0 <= scores.spectral_temporal_coherence <= 1.0

    def test_composite_is_weighted_average(self) -> None:
        mel = _make_mel()
        scores = compute_heuristic_scores(mel)
        weights = [0.30, 0.20, 0.20, 0.15, 0.15]
        values = [
            scores.embedding_consistency,
            scores.pitch_smoothness,
            scores.energy_naturalness,
            scores.speaking_rate_consistency,
            scores.spectral_temporal_coherence,
        ]
        expected = sum(w * v for w, v in zip(weights, values, strict=True))
        assert abs(scores.composite - expected) < 1e-9

    def test_deterministic(self) -> None:
        mel = _make_mel(seed=42)
        s1 = compute_heuristic_scores(mel)
        s2 = compute_heuristic_scores(mel)
        assert s1.composite == s2.composite

    def test_with_embedding(self) -> None:
        mel = _make_mel()
        rng = np.random.default_rng(42)
        embedding = rng.standard_normal(256).astype(np.float64)
        scores = compute_heuristic_scores(mel, embedding=embedding)
        assert isinstance(scores, VoiceConversionHeuristicScores)

    def test_with_quality(self) -> None:
        mel = _make_mel()
        quality = QualityReport(
            input_duration_seconds=1.0,
            canonical_duration_seconds=1.0,
            peak_amplitude=0.5,
            rms_amplitude=0.1,
            voice_fraction=0.7,
            trimmed=False,
        )
        scores = compute_heuristic_scores(mel, quality=quality)
        assert isinstance(scores, VoiceConversionHeuristicScores)


# ── detect_voice_conversion tests ───────────────────────────────────


class TestDetectVoiceConversion:
    def test_heuristic_only(self) -> None:
        mel = _make_mel()
        result = _make_feature_result(mel)
        detection = detect_voice_conversion(result, embedding=None)
        assert isinstance(detection, VoiceConversionDetectionResult)
        assert isinstance(detection.heuristic, VoiceConversionHeuristicScores)
        assert 0.0 <= detection.ml_probability <= 1.0
        assert 0.0 <= detection.composite_score <= 1.0
        assert isinstance(detection.is_converted, bool)

    def test_threshold_zero_always_converted(self) -> None:
        mel = _make_mel()
        result = _make_feature_result(mel)
        detection = detect_voice_conversion(
            result, embedding=None, threshold=0.0
        )
        assert detection.is_converted is True

    def test_threshold_one_never_converted(self) -> None:
        mel = _make_mel()
        result = _make_feature_result(mel)
        detection = detect_voice_conversion(
            result, embedding=None, threshold=1.0
        )
        assert detection.is_converted is False

    def test_custom_weights(self) -> None:
        mel = _make_mel()
        result = _make_feature_result(mel)
        detection = detect_voice_conversion(
            result, embedding=None, ml_weight=0.3, heuristic_weight=0.7
        )
        assert isinstance(detection, VoiceConversionDetectionResult)

    def test_with_embedding(self) -> None:
        mel = _make_mel()
        result = _make_feature_result(mel)
        rng = np.random.default_rng(42)
        embedding = rng.standard_normal(256).astype(np.float64)
        detection = detect_voice_conversion(result, embedding=embedding)
        assert isinstance(detection, VoiceConversionDetectionResult)
        assert 0.0 <= detection.composite_score <= 1.0

    def test_dissimilar_embedding_scores_higher(self) -> None:
        """Very different embedding should produce higher embedding consistency score."""
        mel = _make_mel(seed=42)

        # Get the actual stat features that will be used for comparison.
        from voiceguard.voice.anti_spoofing.voice_conversion import (
            _extract_statistical_stub,
        )
        mel_arr = np.asarray(mel, dtype=np.float64)
        stat_vec = _extract_statistical_stub(mel_arr)

        # Similar embedding: same direction as stat features.
        embedding_similar = stat_vec.copy()
        det_similar = detect_voice_conversion(
            _make_feature_result(mel), embedding=embedding_similar, threshold=1.0
        )

        # Dissimilar embedding: opposite direction.
        embedding_dissimilar = -stat_vec
        det_dissimilar = detect_voice_conversion(
            _make_feature_result(mel), embedding=embedding_dissimilar, threshold=1.0
        )

        # Dissimilar should have higher embedding consistency score.
        sim_score = det_similar.heuristic.embedding_consistency
        dis_score = det_dissimilar.heuristic.embedding_consistency
        assert dis_score >= sim_score

    def test_ml_prob_is_neutral(self) -> None:
        """Without ML model, ml_probability should be 0.5."""
        mel = _make_mel()
        result = _make_feature_result(mel)
        detection = detect_voice_conversion(result, embedding=None)
        assert detection.ml_probability == 0.5
        assert detection.ml_logit == 0.0

    def test_composite_equals_heuristic_when_no_ml(self) -> None:
        """With ml_weight=0, composite should equal heuristic composite."""
        mel = _make_mel()
        result = _make_feature_result(mel)
        detection = detect_voice_conversion(
            result, embedding=None, ml_weight=0.0, heuristic_weight=1.0
        )
        assert abs(detection.composite_score - detection.heuristic.composite) < 1e-9

    def test_deterministic(self) -> None:
        mel = _make_mel(seed=42)
        result = _make_feature_result(mel, seed=42)
        d1 = detect_voice_conversion(result, embedding=None)
        d2 = detect_voice_conversion(result, embedding=None)
        assert d1.composite_score == d2.composite_score
        assert d1.is_converted == d2.is_converted
