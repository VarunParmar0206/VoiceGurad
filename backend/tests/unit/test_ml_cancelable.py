"""Tests for the cancelable (password + salt) template transform (Phase 6)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.unit.ml_helpers import normalized, speaker_cluster
from voiceguard.ml.errors import CancelableTransformError, VerificationError
from voiceguard.voice import cancelable
from voiceguard.voice.verification import (
    EnrollmentSample,
    enroll,
    export_template,
    import_template,
    verify,
)


def test_new_salt_unique() -> None:
    assert cancelable.new_salt() != cancelable.new_salt()
    assert len(cancelable.new_salt()) == 16


def test_derive_projection_deterministic_per_key() -> None:
    r1 = cancelable.derive_projection("pw", b"sa", embedding_dim=32)
    r2 = cancelable.derive_projection("pw", b"sa", embedding_dim=32)
    r3 = cancelable.derive_projection("pw2", b"sa", embedding_dim=32)
    r4 = cancelable.derive_projection("pw", b"sb", embedding_dim=32)
    assert np.allclose(r1, r2)
    assert not np.allclose(r1, r3)
    assert not np.allclose(r1, r4)


def test_derive_projection_is_orthogonal() -> None:
    r = cancelable.derive_projection("pw", b"sa", embedding_dim=64)
    err = np.abs(r.T @ r - np.eye(64)).max()
    assert err < 1e-10


def test_transform_batch_shape_and_norm() -> None:
    x = normalized(np.random.default_rng(0).normal(size=(5, 24)))
    t = cancelable.transform_batch(x, "pw", b"sa")
    assert t.shape == (5, 24)
    assert np.allclose(np.linalg.norm(t, axis=1), 1.0, atol=1e-9)


def test_transform_rejects_bad_input() -> None:
    with pytest.raises(CancelableTransformError):
        cancelable.transform_embedding(np.ones((3, 3)), "pw", b"sa")
    with pytest.raises(CancelableTransformError):
        cancelable.transform_embedding(np.array([1.0, np.inf]), "pw", b"sa")


def test_template_round_trip_is_invariant() -> None:
    # Raw-space and cancelable-space scores must match exactly (orthogonal
    # projection preserves cosine + Mahalanobis geometry).
    emb_dim = 64
    rng = np.random.default_rng(11)
    centroid = normalized(rng.normal(size=emb_dim))
    members = [
        normalized(centroid + rng.normal(scale=0.02, size=emb_dim)) for _ in range(10)
    ]
    samples = [EnrollmentSample(m, np.arange(emb_dim)) for m in members]
    prof = enroll(samples, user_id="u1", embedding_dim=emb_dim, statistical_dim=emb_dim)

    raw = verify(prof, members[0], statistical=np.arange(emb_dim))

    salt = cancelable.new_salt()
    password = "correct horse battery staple"
    blob = export_template(prof, password=password, salt=salt)
    cancelable_prof = import_template(blob)
    r = cancelable.derive_projection(password, salt, embedding_dim=emb_dim)
    transformed = verify(
        cancelable_prof,
        members[0],
        statistical=np.arange(emb_dim),
        transform_r=r,
    )
    assert cancelable_prof.space == "cancelable"
    assert np.isclose(transformed.score, raw.score, atol=1e-9)
    assert transformed.status == raw.status


def test_verify_requires_transform_for_cancelable() -> None:
    _, members, _ = speaker_cluster(seed=12)
    prof = enroll(samples_from(members), user_id="u1", embedding_dim=256)
    salt = cancelable.new_salt()
    blob = export_template(prof, password="pw", salt=salt)
    cancelable_prof = import_template(blob)
    with pytest.raises(VerificationError) as err:
        verify(cancelable_prof, members[0])
    assert "transform" in str(err.value)


def test_template_never_contains_raw_embeddings() -> None:
    _, members, _ = speaker_cluster(n_enroll=8, seed=13)
    prof = enroll(samples_from(members), user_id="u1", embedding_dim=256)
    blob = export_template(prof, password="pw", salt=b"sa")
    assert b"embeddings" not in blob
    cancelable_prof = import_template(blob)
    assert cancelable_prof.embeddings is None
    assert cancelable_prof.statistical is None


def test_wrong_password_breaks_score() -> None:
    _, members, _ = speaker_cluster(n_enroll=8, noise=0.05, seed=14)
    prof = enroll(samples_from(members), user_id="u1", embedding_dim=256)
    salt = cancelable.new_salt()
    blob = export_template(prof, password="right", salt=salt)
    cancelable_prof = import_template(blob)
    wrong_r = cancelable.derive_projection("wrong", salt, embedding_dim=256)
    res = verify(cancelable_prof, members[0], transform_r=wrong_r)
    assert res.status == "rejected"


def test_import_template_rejects_garbage() -> None:
    with pytest.raises(VerificationError):
        import_template(b"not a template")
    with pytest.raises(VerificationError):
        import_template(b"garbage!")
    import pickle

    fake = pickle.dumps({"type": "voiceguard.v2.speaker_profile", "format_version": "9"})
    with pytest.raises(VerificationError):
        import_template(fake)


def samples_from(members: list[np.ndarray], dim: int = 256) -> list[EnrollmentSample]:
    stats = np.arange(dim, dtype=np.float64)
    return [EnrollmentSample(m, stats) for m in members]
