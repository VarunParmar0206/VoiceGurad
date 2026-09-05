"""Tests for evaluation utilities (threshold/DET/EER/TAR@FAR) on synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from voiceguard.ml.errors import TrainingError
from voiceguard.ml.training import (
    det_curve,
    eer,
    pairwise_similarity,
    tar_at_far,
    threshold_analysis,
)


def test_pairwise_similarity_shape_and_range() -> None:
    p = np.random.default_rng(0).normal(size=(3, 8))
    e = np.random.default_rng(1).normal(size=(5, 8))
    sim = pairwise_similarity(p, e)
    assert sim.shape == (3, 5)
    assert (sim >= -1.0).all() and (sim <= 1.0).all()


def test_pairwise_similarity_identical_rows() -> None:
    x = np.random.default_rng(2).normal(size=(4, 6))
    sim = pairwise_similarity(x, x)
    assert np.allclose(np.diag(sim), 1.0, atol=1e-9)


def test_pairwise_similarity_rejects_mismatch() -> None:
    with pytest.raises(TrainingError):
        pairwise_similarity(np.ones((2, 4)), np.ones((2, 5)))


def test_threshold_analysis_monotonic() -> None:
    genuine = np.array([0.9, 0.85, 0.8])
    impostor = np.array([0.4, 0.2, 0.1])
    rows = threshold_analysis(genuine, impostor)
    fars = [far for _, far, _ in rows]
    frrs = [frr for _, _, frr in rows]
    # raising the threshold must never increase FAR or decrease FRR
    assert all(b <= a for a, b in zip(fars, fars[1:], strict=False))
    assert all(b >= a for a, b in zip(frrs, frrs[1:], strict=False))


def test_separated_scores_low_eer() -> None:
    genuine = np.random.default_rng(0).normal(0.9, 0.05, size=200)
    impostor = np.random.default_rng(1).normal(0.2, 0.05, size=200)
    rate = eer(genuine, impostor)
    assert 0.0 <= rate <= 0.5
    assert rate < 0.001  # well-separated distributions


def test_eer_identical_distributions() -> None:
    # Indistinguishable score classes yield an EER of 0.5.
    assert eer(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 0.5


def test_det_curve_sorting() -> None:
    g = np.array([0.9, 0.8, 0.7])
    i = np.array([0.3, 0.2, 0.1])
    fars, frrs, thresholds = det_curve(g, i)
    assert len(fars) == len(frrs) == len(thresholds)
    assert all(a <= b for a, b in zip(fars, fars[1:], strict=False))  # ascending FAR


def test_tar_at_far_monotonic_budget() -> None:
    genuine = np.random.default_rng(2).normal(0.95, 0.02, size=500)
    impostor = np.random.default_rng(3).normal(0.1, 0.05, size=500)
    tight_t, tight_tar = tar_at_far(genuine, impostor, far_target=0.01)
    loose_t, loose_tar = tar_at_far(genuine, impostor, far_target=0.10)
    assert float(np.mean(impostor >= tight_t)) <= 0.01
    assert float(np.mean(impostor >= loose_t)) <= 0.10
    assert loose_t <= tight_t  # looser budget => looser threshold
    assert loose_tar >= tight_tar  # and no less TAR
    assert tight_tar > 0.9


def test_tar_at_far_validates_target() -> None:
    with pytest.raises(TrainingError):
        tar_at_far(np.array([0.5]), np.array([0.1]), far_target=1.5)


def test_validate_scores_rejects_bad_input() -> None:
    with pytest.raises(TrainingError):
        threshold_analysis(np.array([np.nan]), np.array([0.1]))
    with pytest.raises(TrainingError):
        threshold_analysis(np.array([0.5, 0.6]), np.array([]))
