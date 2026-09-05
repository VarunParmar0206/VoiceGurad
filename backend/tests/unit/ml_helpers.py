"""Deterministic helpers for Phase 6 speaker-verification tests."""

from __future__ import annotations

import numpy as np

from voiceguard.voice.result import (
    FeatureResult,
    MelFeatures,
    QualityReport,
    StatisticalFeatures,
)


def normalized(x: np.ndarray) -> np.ndarray:
    """L2-normalize a vector (or each row of a matrix)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return x / max(np.linalg.norm(x), 1e-12)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def speaker_cluster(
    dim: int = 256,
    n_enroll: int = 10,
    noise: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Return (centroid, enrollment members, impostors).

    Members cluster tightly around ``centroid``; impostors are unrelated unit
    vectors.  All vectors are L2-normalized so cosine/manifold assumptions hold.
    """
    rng = np.random.default_rng(seed)
    centroid = normalized(rng.normal(size=dim))
    members = [
        normalized(centroid + rng.normal(scale=noise, size=dim)) for _ in range(n_enroll)
    ]
    impostor_rng = np.random.default_rng(seed + 10_000)
    impostors = [normalized(impostor_rng.normal(size=dim)) for _ in range(4)]
    return centroid, members, impostors


def make_statistical(seed: int = 1) -> np.ndarray:
    """259-dim statistical feature vector matching STATISTICAL_FEATURE_DIM."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=259).astype(np.float64)


def make_feature_result(t: int = 32, seed: int = 0) -> FeatureResult:
    """A valid Phase 5 FeatureResult with a ``(80, T)`` log-mel spectrogram."""
    rng = np.random.default_rng(seed)
    log_mel = rng.standard_normal((80, t)).astype(np.float32)
    duration = t * 0.01
    return FeatureResult(
        sample_rate=16_000,
        num_samples=int(t * 160),
        duration_seconds=duration,
        mel=MelFeatures(
            log_mel=log_mel,
            n_mels=80,
            n_fft=512,
            hop_length=160,
            win_length=400,
            sample_rate=16_000,
            f_min=20.0,
            f_max=8000.0,
        ),
        statistical=StatisticalFeatures(values=make_statistical(seed=seed)),
        quality=QualityReport(
            input_duration_seconds=duration,
            canonical_duration_seconds=duration,
            peak_amplitude=0.7,
            rms_amplitude=0.1,
            voice_fraction=0.9,
            trimmed=False,
        ),
    )
