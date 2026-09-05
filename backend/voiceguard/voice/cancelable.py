"""VoiceGuard V2 — Cancelable biometric transform (architecture §14).

A per-user random orthogonal projection ``R_u`` is derived deterministically
from the user's password and a random per-user salt:

    R_u = QR( Gaussian( PBKDF2-HMAC-SHA256(password, salt) ) )

and embeddings are stored only after transformation:

    template = normalize(R_u · embedding)

Why orthogonal?
    An orthogonal projection preserves the cosine similarity and Mahalanobis
    metric structure (``(R e)ᵀ (R e') = eᵀ e'``), so matching in the
    transformed space is identical to matching in the raw space — while the
    stored template no longer reveals the raw embedding.

Security properties (implemented + unit-tested)
- Deterministic: same password + salt + embedding → same template.
- Salt-sensitive: a different salt ⇒ a different projection.
- Password-sensitive: a different password ⇒ a different projection.
- Reversible-enrollment-safe: raw embeddings are never returned/stored;
  this module never logs inputs or outputs.

Password-change lifecycle
************************
The transform is bound to the user's password.  Changing the password
therefore invalidates every template derived from the previous password:
comparisons after rotation fail, which is the intended safe behaviour.
There is **no bypass** and no way to "recover" a template under a new
password.  The supported lifecycle is: after a password rotation the
application MUST require biometric re-enrollment (fresh salt + fresh
enrollment samples transformed under the new password).  Tests encode this
invariant: old-password templates fail, re-enrolled templates succeed.
"""

from __future__ import annotations

import hashlib
import secrets

import numpy as np
import numpy.typing as npt

from voiceguard.config import settings
from voiceguard.ml.errors import CancelableTransformError

FloatArray = npt.NDArray[np.float64]


def new_salt() -> bytes:
    """Return a fresh random per-user salt (configurable length, default 16B)."""
    return secrets.token_bytes(settings.CANCELABLE_SALT_BYTES)


def derive_projection(
    password: str,
    salt: bytes,
    *,
    embedding_dim: int,
    iterations: int | None = None,
    key_bytes: int | None = None,
) -> FloatArray:
    """Deterministically derive an ``(d, d)`` orthogonal projection matrix.

    Parameters
    ----------
    password : str
        The user's password (never logged).
    salt : bytes
        Per-user random salt (stored plaintext per architecture §14.3).
    embedding_dim : int
        Dimensionality of the embeddings to protect.
    iterations : int | None
        PBKDF2 iteration count (default from settings).
    key_bytes : int | None
        PBKDF2 derived-key length (default from settings).

    Returns
    -------
    FloatArray
        An orthogonal ``(d, d)`` matrix derived solely from the inputs.
    """
    if not isinstance(password, str) or not password:
        raise CancelableTransformError("a non-empty password is required")
    if not isinstance(salt, bytes) or not salt:
        raise CancelableTransformError("a non-empty salt is required")
    if embedding_dim < 1:
        raise CancelableTransformError("embedding_dim must be >= 1")

    iters = settings.CANCELABLE_PBKDF2_ITERATIONS if iterations is None else iterations
    key_len = settings.CANCELABLE_KEY_BYTES if key_bytes is None else key_bytes
    if iters < 1 or key_len < 1:
        raise CancelableTransformError("PBKDF2 iterations and key bytes must be >= 1")

    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, iters, dklen=key_len
    )
    entropy = np.frombuffer(dk, dtype=np.uint32).tolist()
    seed = np.random.SeedSequence(entropy)
    rng = np.random.default_rng(seed)
    gaussian = rng.standard_normal((embedding_dim, embedding_dim))
    q, _ = np.linalg.qr(gaussian)
    return q


def transform_embedding(
    embedding: npt.ArrayLike,
    password: str,
    salt: bytes,
    *,
    iterations: int | None = None,
) -> FloatArray:
    """Return ``normalize(R_u · embedding)`` (float64).

    Raises
    ------
    CancelableTransformError
        If the embedding is not a finite 1-D vector or the projection is
        degenerate.
    """
    vec = np.asarray(embedding, dtype=np.float64)
    _validate_embedding(vec)

    projection = derive_projection(
        password, salt, embedding_dim=int(vec.size), iterations=iterations
    )
    projected = projection @ vec
    norm = float(np.linalg.norm(projected))
    if not np.isfinite(norm) or norm == 0.0:
        raise CancelableTransformError("transformed embedding is degenerate")
    return projected / norm


def transform_batch(
    embeddings: npt.ArrayLike,
    password: str,
    salt: bytes,
    *,
    iterations: int | None = None,
) -> FloatArray:
    """Transform rows of an ``(N, d)`` embedding matrix (see transform_embedding)."""
    mat = np.asarray(embeddings, dtype=np.float64)
    if mat.ndim != 2 or mat.size == 0:
        raise CancelableTransformError("embeddings must be a non-empty (N, d) matrix")
    if not np.isfinite(mat).all():
        raise CancelableTransformError("embeddings contain non-finite values")

    projection = derive_projection(
        password, salt, embedding_dim=int(mat.shape[1]), iterations=iterations
    )
    projected = (projection @ mat.T).T
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms == 0.0):
        raise CancelableTransformError("some transformed embeddings are degenerate")
    return projected / norms


def _validate_embedding(vec: FloatArray) -> None:
    if vec.ndim != 1 or vec.size == 0:
        raise CancelableTransformError("embedding must be a non-empty 1-D vector")
    if not np.isfinite(vec).all():
        raise CancelableTransformError("embedding contains non-finite values")
