"""Tests for speaker enrollment + verification scoring (Phase 6)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.unit.ml_helpers import speaker_cluster
from voiceguard.ml.errors import EnrollmentError, VerificationError
from voiceguard.voice.verification import (
    AttemptLimiter,
    DiagonalGaussianMixture,
    EnrollmentSample,
    enroll,
    verify,
)


def samples_from(members: list[np.ndarray], dim: int = 256) -> list[EnrollmentSample]:
    stats = np.arange(dim, dtype=np.float64)  # shared quirky "feature vector"
    return [EnrollmentSample(m, stats) for m in members]


def test_enroll_builds_expected_profile() -> None:
    _, members, _ = speaker_cluster(noise=0.05, n_enroll=10, seed=1)
    prof = enroll(
        samples_from(members),
        user_id="u1",
        embedding_dim=256,
        statistical_dim=256,
    )
    assert prof.user_id == "u1"
    assert prof.embedding_dim == 256
    assert prof.n_samples == 10
    assert prof.space == "raw"
    assert prof.model_version == "v1.0"
    assert np.isclose(np.linalg.norm(prof.centroid), 1.0, atol=1e-9)
    assert prof.precision is not None
    assert prof.gmm is not None
    assert prof.adaptive_threshold > 0.0


def test_enroll_rejects_too_few_samples() -> None:
    _, members, _ = speaker_cluster(n_enroll=10, seed=2)
    with pytest.raises(EnrollmentError):
        enroll(
            samples_from(members[:4]),
            user_id="u1",
            embedding_dim=256,
            min_samples=5,
        )


def test_enroll_rejects_unormalized_embeddings() -> None:
    rng = np.random.default_rng(3)
    bad = [rng.normal(size=256) * 3.0 for _ in range(6)]  # not unit norm
    with pytest.raises(EnrollmentError):
        enroll(samples_from(bad), user_id="u1", embedding_dim=256)


def test_enroll_rejects_bad_shapes() -> None:
    _, members, _ = speaker_cluster(seed=4)
    with pytest.raises(EnrollmentError):
        enroll(samples_from(members), user_id="u1", embedding_dim=128)  # wrong dim


def test_verify_genuine_and_impostor_separate() -> None:
    _, members, impostors = speaker_cluster(noise=0.05, n_enroll=12, seed=5)
    prof = enroll(
        samples_from(members),
        user_id="u1",
        embedding_dim=256,
        statistical_dim=256,
    )
    assert verify(prof, members[0], statistical=np.arange(256)).status == "verified"
    imp = verify(prof, impostors[0], statistical=np.arange(256))
    assert imp.status == "rejected"
    assert verify(prof, members[0]).score > imp.score


def test_verify_confidence_band_soft_accept() -> None:
    # A wide confidence band turns a rejected probe into a soft_accept.
    _, members, impostors = speaker_cluster(noise=0.02, n_enroll=12, seed=6)
    prof = enroll(
        samples_from(members),
        user_id="u1",
        embedding_dim=256,
        statistical_dim=256,
    )
    rejected = verify(prof, impostors[0], statistical=np.arange(256))
    assert rejected.status == "rejected"
    assert rejected.score < prof.adaptive_threshold
    soft = verify(
        prof,
        impostors[0],
        statistical=np.arange(256),
        confidence_band=1.0,  # band >= threshold - score, always
    )
    assert soft.status == "soft_accept"


def test_verify_rejects_malformed_probe() -> None:
    _, members, _ = speaker_cluster(noise=0.05, seed=7)
    prof = enroll(samples_from(members), user_id="u1", embedding_dim=256)
    with pytest.raises(VerificationError):
        verify(prof, np.ones(300))  # wrong dim
    with pytest.raises(VerificationError):
        verify(prof, np.array([np.nan] * 256))  # non-finite


def test_gmm_fit_and_score() -> None:
    rng = np.random.default_rng(8)
    x = rng.normal(loc=1.0, size=(40, 8))
    gmm = DiagonalGaussianMixture(n_components=3).fit(x)
    assert gmm.fitted
    ll = gmm.score(x)
    assert np.isfinite(ll)
    assert gmm.score_samples(x).shape == (40,)


def test_attempt_limiter_lockout_and_escalation() -> None:
    limiter = AttemptLimiter(max_attempts=3, cooldowns=(30, 60, 300))
    now = 1000.0
    assert limiter.check("u9", now=now).allowed
    for _ in range(3):
        limiter.record("u9", success=False, now=now)
    locked = limiter.check("u9", now=now)
    assert not locked.allowed
    assert locked.retry_after_seconds > 0
    assert limiter.check("u9", now=now + 31.0).allowed  # after 30s cooldown
    for _ in range(3):
        limiter.record("u9", success=False, now=now + 100.0)
    assert limiter.check("u9", now=now + 100.0).retry_after_seconds >= 60  # escalated


def test_attempt_limiter_success_resets() -> None:
    limiter = AttemptLimiter(max_attempts=2, cooldowns=(30,))
    now = 0.0
    limiter.record("u1", success=False, now=now)
    limiter.record("u1", success=False, now=now)
    assert not limiter.check("u1", now=now + 5.0).allowed
    limiter.record("u1", success=True, now=now + 30.0)
    assert limiter.check("u1", now=now + 30.0).allowed
    limiter.record("u1", success=False, now=now + 30.0)
    assert limiter.check("u1", now=now + 31.0).allowed  # fresh cycle


def test_attempt_limiter_reset() -> None:
    limiter = AttemptLimiter(max_attempts=2)
    limiter.record("u1", success=False)
    limiter.reset("u1")
    assert limiter.check("u1").allowed
