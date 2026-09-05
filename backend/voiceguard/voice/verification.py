"""VoiceGuard V2 — Speaker verification & scoring (architecture §5.2).

Implements the Phase 6 verification pipeline in pure numpy/scipy:

- enrollment with ``N >= min_samples`` embeddings (+ optional statistical
  vectors) into a :class:`EnrollmentProfile` (centroid, covariance/precision,
  per-user diagonal GMM, intra-class consistency, adaptive threshold);
- scoring of a probe embedding: centroid cosine, max/mean/min cosine over the
  retained enrollment set, Mahalanobis distance, GMM/background log-likelihood
  ratio, weighted composite score;
- single adaptive threshold + confidence-band decision (no graduated
  thresholds — the V1 mechanism is removed);
- in-memory attempt limiting with escalating cooldown (Redis-backed
  integration is deferred to a later phase);
- template export/import that stores only the **cancelable-transformed**
  centroid and encrypted-at-rest companions (never raw embeddings/vectors).

Calibration status
******************
:class:`DiagonalGaussianMixture`, the background/UBM interface, the composite
weights, and the threshold are **architectural scaffolding**.  They are
exercised with synthetic data only.  No production EER/FAR/FRR/TAR threshold,
no calibrated score weights, and no claim of real-world speaker separation is
made anywhere in this module.  Real calibration is a future step that
requires a genuine speaker dataset.
"""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, NamedTuple, cast

import numpy as np
import numpy.typing as npt

from voiceguard.config import settings
from voiceguard.ml.errors import EnrollmentError, VerificationError
from voiceguard.voice.cancelable import derive_projection, transform_embedding

FloatArray = npt.NDArray[np.float64]
MixtureParams = dict[str, object]

_EMPTY_TEMPLATE_MSG = "malformed speaker template payload"


class EnrollmentSample(NamedTuple):
    """One enrollment utterance: speaker embedding + optional statistics.

    The embedding must be L2-normalized (the CNN-LSTM-Attention output is).
    ``statistical`` is the 259-dim Path B vector used for the per-user GMM.
    """

    embedding: FloatArray
    statistical: FloatArray | None = None


# ── Diagonal Gaussian Mixture (Phase 6 scaffolding, synthetic-only) ────


class DiagonalGaussianMixture:
    """Deterministic diagonal-covariance Gaussian mixture (EM).

    Pure numpy/scipy implementation used for per-user GMM scoring and the
    background (UBM) interface.  Fitted component count is clamped to the
    number of training rows; results are deterministic for the same input.
    """

    def __init__(
        self,
        n_components: int = 8,
        *,
        max_iter: int = 100,
        tol: float = 1e-4,
        variances_floor: float = 1e-6,
        weights_floor: float = 1e-6,
    ) -> None:
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        self.n_components = int(n_components)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.variances_floor = float(variances_floor)
        self.weights_floor = float(weights_floor)
        self.weights_: FloatArray | None = None
        self.means_: FloatArray | None = None
        self.variances_: FloatArray | None = None
        self.feature_dimension_: int = 0

    @property
    def fitted(self) -> bool:
        return self.weights_ is not None

    def fit(self, features: npt.ArrayLike) -> DiagonalGaussianMixture:
        x = np.asarray(features, dtype=np.float64)
        n, d = x.shape
        if n < 1 or d < 1:
            raise ValueError("fit requires a non-empty (n, d) feature matrix")
        if not np.isfinite(x).all():
            raise ValueError("features contain non-finite values")

        k = min(self.n_components, n)
        self.feature_dimension_ = d
        weights = np.full(k, 1.0 / k)
        means = x[np.arange(k) % n].copy()
        global_var = np.var(x, axis=0)
        variances = np.tile(
            np.maximum(global_var, self.variances_floor), (k, 1)
        )

        prev_ll = -np.inf
        for _ in range(self.max_iter):
            log_resp = self._estimate_log_prob(
                features=x, weights=weights, means=means, variances=variances
            )
            log_like = _logsumexp(log_resp, axis=1)
            ll = float(np.mean(log_like))
            if ll - prev_ll < self.tol:
                break
            prev_ll = ll

            resp = np.exp(log_resp - log_like[:, None])
            nk = resp.sum(axis=0)
            nk[nk <= 0.0] = 1.0
            weights = np.maximum(nk / n, self.weights_floor)
            weights = weights / weights.sum()
            means = (resp.T @ x) / nk[:, None]
            for c in range(k):
                diff = x - means[c]
                variances[c] = (resp[:, c][:, None] * diff**2).sum(axis=0) / nk[c]
            variances = np.maximum(variances, self.variances_floor)

        self.weights_ = weights
        self.means_ = means
        self.variances_ = variances
        return self

    def score(self, features: npt.ArrayLike) -> float:
        return float(np.mean(self.score_samples(features)))

    def score_samples(self, features: npt.ArrayLike) -> FloatArray:
        """Per-row log-likelihood ``(n,)``."""
        if (
            not self.fitted
            or self.weights_ is None
            or self.means_ is None
            or self.variances_ is None
        ):
            raise VerificationError("GMM has not been fitted")
        x = np.asarray(features, dtype=np.float64)
        if x.shape[1] != self.feature_dimension_:
            raise VerificationError(
                f"GMM expects {self.feature_dimension_} dims; got {x.shape[1]}"
            )
        weights: FloatArray = self.weights_
        means: FloatArray = self.means_
        variances: FloatArray = self.variances_
        log_resp = self._estimate_log_prob(
            features=x,
            weights=weights,
            means=means,
            variances=variances,
        )
        return _logsumexp(log_resp, axis=1)

    def to_params(self) -> MixtureParams:
        if not self.fitted:
            raise VerificationError("GMM has not been fitted")
        return {
            "n_components": self.n_components,
            "fitted_components": int(self.weights_.shape[0]),  # type: ignore[union-attr]
            "feature_dimension": self.feature_dimension_,
            "weights": self.weights_,
            "means": self.means_,
            "variances": self.variances_,
        }

    @classmethod
    def from_params(cls, params: MixtureParams) -> DiagonalGaussianMixture:
        try:
            n_components = int(cast("int | float | str", params["n_components"]))
            weights = np.asarray(params["weights"], dtype=np.float64)
            means = np.asarray(params["means"], dtype=np.float64)
            variances = np.asarray(params["variances"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise VerificationError(_EMPTY_TEMPLATE_MSG) from exc
        if not (weights.ndim == 1 and means.ndim == 2 and variances.shape == means.shape):
            raise VerificationError(_EMPTY_TEMPLATE_MSG)
        if not (
            np.isfinite(weights).all()
            and np.isfinite(means).all()
            and np.isfinite(variances).all()
        ):
            raise VerificationError(_EMPTY_TEMPLATE_MSG)
        model = cls(n_components=n_components)
        model.feature_dimension_ = int(means.shape[1])
        model.weights_ = weights
        model.means_ = means
        model.variances_ = variances
        return model

    @staticmethod
    def _estimate_log_prob(
        features: FloatArray,
        *,
        weights: FloatArray,
        means: FloatArray,
        variances: FloatArray,
    ) -> FloatArray:
        k = means.shape[0]
        d = features.shape[1]
        log_like = np.empty((features.shape[0], k))
        for c in range(k):
            var = np.maximum(variances[c], 1e-12)
            diff = features - means[c]
            log_gauss = -0.5 * (d * np.log(2.0 * np.pi) + np.log(var).sum())
            log_gauss -= 0.5 * np.sum((diff * diff) / var, axis=1)
            log_like[:, c] = log_gauss + np.log(np.maximum(weights[c], 1e-12))
        return log_like


def fit_background(features: npt.ArrayLike, *, n_components: int = 8) -> DiagonalGaussianMixture:
    """Fit a background/UBM mixture on population statistics.

    Scaffolding only — unvalidated until a real speaker dataset exists.
    """
    return DiagonalGaussianMixture(n_components=n_components).fit(features)


def log_likelihood_ratio(user_log_likelihood: float, background_log_likelihood: float) -> float:
    """User-model log-likelihood minus background log-likelihood."""
    return float(user_log_likelihood) - float(background_log_likelihood)


def _logsumexp(a: FloatArray, axis: int) -> FloatArray:
    m = a.max(axis=axis, keepdims=True)
    return m.squeeze(axis) + np.log(np.exp(a - m).sum(axis=axis))


# ── Enrollment profile ────────────────────────────────────────────────


@dataclass
class EnrollmentProfile:
    """Geometric + statistical summary of an enrollment.

    ``centroid``/``precision`` live in one of two spaces:

    - ``space == "raw"`` (fresh enrollment): raw, in-memory only.
    - ``space == "cancelable"`` (loaded from a template): the cancelable
      centroid ``R_u · c`` with ``precision_c = R_u P R_uᵀ``.  Because the
      projection is orthogonal, cosine and Mahalanobis scores are identical
      to the raw-space values.

    Raw enrollment embeddings/statistics are retained **in memory only**
    (fields ``embeddings``/``statistical``) and are NEVER serialized; the
    cancelable template never contains them.
    """

    user_id: str
    embedding_dim: int
    statistical_dim: int | None
    n_samples: int
    model_version: str
    centroid: FloatArray
    precision: FloatArray | None
    gmm: DiagonalGaussianMixture | None
    intra_consistency: float
    adaptive_threshold: float
    space: Literal["raw", "cancelable"] = "raw"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    # In-memory only — never serialized.
    embeddings: FloatArray | None = None
    statistical: FloatArray | None = None


def enroll(
    samples: Sequence[EnrollmentSample],
    *,
    user_id: str,
    embedding_dim: int,
    statistical_dim: int | None = None,
    min_samples: int | None = None,
    gmm_components: int | None = None,
    base_threshold: float | None = None,
    adaptive_factor: float | None = None,
    threshold_lower: float | None = None,
    threshold_upper: float | None = None,
    model_version: str | None = None,
) -> EnrollmentProfile:
    """Build an ``EnrollmentProfile`` from ``N >= min_samples`` samples.

    Raises
    ------
    EnrollmentError
        If the samples are too few, mis-shaped, non-finite, or not
        L2-normalized, or if statistical vectors are inconsistent.
    """
    if embedding_dim < 1:
        raise EnrollmentError("embedding_dim must be >= 1")
    min_n = settings.ENROLLMENT_MIN_SAMPLES if min_samples is None else min_samples
    if min_n < 1:
        raise EnrollmentError("min_samples must be >= 1")

    samples_list = list(samples)
    if not samples_list:
        raise EnrollmentError("no enrollment samples provided")
    if len(samples_list) < min_n:
        raise EnrollmentError(
            f"enrollment requires at least {min_n} samples; got {len(samples_list)}"
        )
    if statistical_dim is not None and statistical_dim < 1:
        raise EnrollmentError("statistical_dim must be >= 1")

    embeddings, statistical = _stack_samples(samples_list, embedding_dim, statistical_dim)

    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-2):
        raise EnrollmentError("enrollment embeddings must be L2-normalized")
    # The speaker reference is the *direction* of the mean: the unit centroid.
    # Unit normalization keeps the cancelable transform (an orthogonal
    # projection of unit vectors) metric-invariant, so cosine and Mahalanobis
    # scores are identical in raw and transformed space.
    centroid = embeddings.mean(axis=0)
    centroid = centroid / np.maximum(np.linalg.norm(centroid), 1e-12)

    # Out-of-subspace Mahalanobis metric (stable for N << d).
    #
    # The empirical covariance of N unit vectors in d=256 dims is severely
    # rank-deficient.  Naive inverse/pinv estimators blow up: tiny eigenvalues
    # get inverted to enormous precision values, so any probe's squared
    # Mahalanobis distance explodes (10^4..10^6) and the s_md component
    # collapses to ~0 even for genuine speakers.
    #
    # Instead we measure the probe's displacement OUTSIDE the speaker subspace
    # spanned by the enrollment's within-speaker variation:
    #
    #     span  = top min(N-1, d) principal directions of (X - centroid)
    #     P     = (I - V V^T) / sigma^2        (deflated ridge-free precision)
    #     md2   = (probe - centroid)^T P (probe - centroid)
    #
    # A genuine probe lies in the span, so md2 ~ 0 and s_md ~ 1; an impostor
    # has a large out-of-span residual, so md2 is large and s_md -> 0.
    # Because P = (I - V V^T)/sigma^2 commutes with rotation in form, the
    # cancelable transform maps it exactly as R R^T folds into I - (RV)(RV)^T,
    # so raw- and template-space Mahalanobis scores are identical.
    cov = np.cov(embeddings, rowvar=False)
    cov = np.asarray(cov, dtype=np.float64)
    sigma2 = max(float(np.mean(np.diag(cov))), 1e-10)

    centered = embeddings - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    k_span = min(embeddings.shape[0] - 1, embedding_dim)
    span_dirs = vt[:k_span].T  # (d, k) within-speaker variation directions
    precision = (np.eye(embedding_dim) - span_dirs @ span_dirs.T) / sigma2

    intra = _pairwise_consistency(embeddings)
    thr = _adaptive_threshold(
        intra_consistency=intra,
        base_threshold=(
            settings.VERIFICATION_THRESHOLD if base_threshold is None else base_threshold
        ),
        adaptive_factor=(
            settings.ENROLLMENT_ADAPTIVE_FACTOR if adaptive_factor is None else adaptive_factor
        ),
        lower=settings.ENROLLMENT_THRESHOLD_LOWER if threshold_lower is None else threshold_lower,
        upper=settings.ENROLLMENT_THRESHOLD_UPPER if threshold_upper is None else threshold_upper,
    )

    gmm: DiagonalGaussianMixture | None = None
    if statistical is not None and (gmm_components is None or gmm_components > 0):
        n_comp = settings.GMM_N_COMPONENTS if gmm_components is None else gmm_components
        gmm = DiagonalGaussianMixture(n_components=n_comp).fit(statistical)

    return EnrollmentProfile(
        user_id=user_id,
        embedding_dim=embedding_dim,
        statistical_dim=statistical_dim,
        n_samples=len(samples_list),
        model_version=settings.MODEL_VERSION if model_version is None else model_version,
        centroid=centroid,
        precision=precision,
        gmm=gmm,
        intra_consistency=intra,
        adaptive_threshold=thr,
        embeddings=embeddings,
        statistical=statistical,
    )


def _stack_samples(
    samples: list[EnrollmentSample],
    embedding_dim: int,
    statistical_dim: int | None,
) -> tuple[FloatArray, FloatArray | None]:
    embeddings = np.stack([np.asarray(s.embedding, dtype=np.float64) for s in samples])
    if embeddings.ndim != 2 or embeddings.shape[1] != embedding_dim:
        raise EnrollmentError(
            f"embeddings must have shape ({len(samples)}, {embedding_dim})"
        )
    if not np.isfinite(embeddings).all():
        raise EnrollmentError("enrollment embeddings contain non-finite values")

    have_stat = [s.statistical is not None for s in samples]
    if any(have_stat) and not all(have_stat):
        raise EnrollmentError("statistical vectors must be present for ALL or NONE samples")

    statistical: FloatArray | None = None
    if all(have_stat):
        stat = np.stack([np.asarray(s.statistical, dtype=np.float64) for s in samples])
        if statistical_dim is not None and stat.shape[1] != statistical_dim:
            raise EnrollmentError(
                f"statistical vectors must have {statistical_dim} dims; got {stat.shape[1]}"
            )
        if not np.isfinite(stat).all():
            raise EnrollmentError("statistical vectors contain non-finite values")
        statistical = stat
    return embeddings, statistical


def _pairwise_consistency(embeddings: FloatArray) -> float:
    norm = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    sims = (norm @ norm.T) + 1.0  # (cos + 1)/2 * 2 gives [0,2] *2
    iu = np.triu_indices(embeddings.shape[0], k=1)
    if iu[0].size == 0:
        return 1.0
    return float(np.clip(sims[iu] / 2.0, 0.0, 1.0).mean())


def _adaptive_threshold(
    *,
    intra_consistency: float,
    base_threshold: float,
    adaptive_factor: float,
    lower: float,
    upper: float,
) -> float:
    """Per-user threshold (architecture §5.2).

    More consistent enrollments allow a slightly stricter threshold; noisy
    ones nudge it down.  Uncalibrated — bounds/weight are design defaults.
    """
    margin = adaptive_factor * (1.0 - float(np.clip(intra_consistency, 0.0, 1.0)))
    return float(np.clip(base_threshold + margin, lower, upper))


# ── Verification scoring ──────────────────────────────────────────────


@dataclass(frozen=True)
class VerificationScores:
    """Decomposed probe-vs-profile scores."""

    centroid_cosine: float
    max_cosine: float | None = None
    mean_cosine: float | None = None
    min_cosine: float | None = None
    mahalanobis_squared: float | None = None
    gmm_user_ll: float | None = None
    gmm_background_ll: float | None = None
    gmm_llr: float | None = None
    s_cos: float = 0.0
    s_md: float = 0.0
    s_gmm: float = 0.0
    composite: float = 0.0


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of a verification attempt.

    ``status`` is one of:

    - ``"verified"``   — score >= threshold
    - ``"soft_accept"`` — within the confidence band below threshold
    - ``"rejected"``   — below the confidence band
    """

    status: Literal["verified", "soft_accept", "rejected"]
    score: float
    threshold: float
    confidence_band: float
    reason: str
    scores: VerificationScores


def verify(
    profile: EnrollmentProfile,
    embedding: npt.ArrayLike,
    *,
    statistical: npt.ArrayLike | None = None,
    background: DiagonalGaussianMixture | None = None,
    confidence_band: float | None = None,
    transform_r: FloatArray | None = None,
    weights: tuple[float, float, float] | None = None,
) -> VerificationResult:
    """Score a probe against an enrollment profile and decide.

    Parameters
    ----------
    profile : EnrollmentProfile
    embedding : array-like
        ``(d,)`` probe embedding (L2-normalized, same space as the profile).
    statistical : array-like | None
        ``(s,)`` Path B vector for the per-user GMM.
    background : DiagonalGaussianMixture | None
        Optional UBM; when supplied together with ``statistical`` and a fitted
        profile GMM, an LLR component is included in the composite score.
    confidence_band : float | None
        Defaults to ``settings.CONFIDENCE_BAND`` (0.05).
    transform_r : (d, d) orthogonal matrix | None
        Required when ``profile.space == "cancelable"`` to transform the probe
        into the template's space; ignored for raw profiles.
    weights : tuple[float, float, float] | None
        (cosine, mahalanobis, gmm) composite weights; defaults to settings.

    Raises
    ------
    VerificationError
        For malformed probes or inconsistent profile/transform state.
    """
    probe = np.asarray(embedding, dtype=np.float64)
    if probe.ndim != 1 or probe.size != profile.embedding_dim:
        raise VerificationError(
            f"probe must be a 1-D vector of {profile.embedding_dim} dims; got {probe.shape}"
        )
    if not np.isfinite(probe).all():
        raise VerificationError("probe contains non-finite values")

    if profile.space == "cancelable":
        if transform_r is None:
            raise VerificationError(
                "profile is cancelable; a transform (transform_r) is required"
            )
        transform_r = np.asarray(transform_r, dtype=np.float64)
        if transform_r.shape != (profile.embedding_dim, profile.embedding_dim):
            raise VerificationError(
                f"transform_r must be ({profile.embedding_dim}, {profile.embedding_dim})"
            )
        probe_t = transform_r @ probe
        probe_t = probe_t / np.maximum(np.linalg.norm(probe_t), 1e-12)
    else:
        probe_t = probe

    band = settings.CONFIDENCE_BAND if confidence_band is None else confidence_band
    if band < 0.0:
        raise VerificationError("confidence_band must be >= 0")

    centroid_cos = _cosine(probe_t, profile.centroid)

    max_cos: float | None = None
    mean_cos: float | None = None
    min_cos: float | None = None
    if profile.embeddings is not None:
        sims = profile.embeddings @ probe_t
        sims = np.clip(sims / np.maximum(np.linalg.norm(probe_t), 1e-12), -1.0, 1.0)
        if sims.size:
            max_cos = float(sims.max())
            mean_cos = float(sims.mean())
            min_cos = float(sims.min())

    mahalanobis_sq: float | None = None
    s_md: float | None = None
    if profile.precision is not None:
        delta = probe_t - profile.centroid
        md2 = float(np.asarray(delta @ profile.precision @ delta).item())
        md2 = max(md2, 0.0)
        mahalanobis_sq = md2
        s_md = 1.0 / (1.0 + md2 / float(profile.embedding_dim))

    s_cos = (centroid_cos + 1.0) / 2.0

    gmm_user_ll: float | None = None
    gmm_background_ll: float | None = None
    gmm_llr: float | None = None
    s_gmm: float | None = None
    if statistical is not None and profile.gmm is not None:
        stats = np.asarray(statistical, dtype=np.float64)
        if stats.ndim != 1 or (
            profile.statistical_dim is not None and stats.size != profile.statistical_dim
        ):
            raise VerificationError("statistical vector shape does not match the profile")
        gmm_user_ll = float(profile.gmm.score_samples(stats[None, :])[0])
        if background is not None:
            gmm_background_ll = float(background.score_samples(stats[None, :])[0])
            gmm_llr = log_likelihood_ratio(gmm_user_ll, gmm_background_ll)
            s_gmm = _sigmoid(float(np.clip(gmm_llr, -30.0, 30.0)))

    w_cos, w_md, w_gmm = weights or (
        settings.COMPOSITE_WEIGHT_COSINE,
        settings.COMPOSITE_WEIGHT_MAHALANOBIS,
        settings.COMPOSITE_WEIGHT_GMM,
    )
    if not (w_cos >= 0.0 and w_md >= 0.0 and w_gmm >= 0.0 and (w_cos + w_md + w_gmm) > 0.0):
        raise VerificationError("composite weights must be non-negative and not all zero")

    present: list[tuple[float, float]] = []
    present.append((w_cos * s_cos, w_cos))
    if s_md is not None:
        present.append((w_md * s_md, w_md))
    if s_gmm is not None:
        present.append((w_gmm * s_gmm, w_gmm))
    if not present:
        raise VerificationError("no score components available for the composite")
    composite = float(sum(s for s, w in present) / max(sum(w for _, w in present), 1e-12))
    composite = float(np.clip(composite, 0.0, 1.0))

    threshold = profile.adaptive_threshold
    if composite >= threshold:
        status: Literal["verified", "soft_accept", "rejected"] = "verified"
        reason = "score at or above the adaptive threshold"
    elif composite >= threshold - band:
        status = "soft_accept"
        reason = "score within the confidence band below the threshold"
    else:
        status = "rejected"
        reason = "score below the confidence band"

    scores = VerificationScores(
        centroid_cosine=centroid_cos,
        max_cosine=max_cos,
        mean_cosine=mean_cos,
        min_cosine=min_cos,
        mahalanobis_squared=mahalanobis_sq,
        gmm_user_ll=gmm_user_ll,
        gmm_background_ll=gmm_background_ll,
        gmm_llr=gmm_llr,
        s_cos=float(s_cos),
        s_md=float(s_md) if s_md is not None else 0.0,
        s_gmm=float(s_gmm) if s_gmm is not None else 0.0,
        composite=composite,
    )
    return VerificationResult(
        status=status,
        score=composite,
        threshold=float(threshold),
        confidence_band=float(band),
        reason=reason,
        scores=scores,
    )


def _cosine(a: FloatArray, b: FloatArray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        raise VerificationError("degenerate (zero-norm) vectors in cosine scoring")
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        exp_neg = float(np.exp(-x))
        return float(1.0 / (1.0 + exp_neg))
    exp_pos = float(np.exp(x))
    return float(exp_pos / (1.0 + exp_pos))


# ── Template export/import (cancelable, never raw embeddings) ─────────


def _array_to_json(value: FloatArray) -> dict[str, object]:
    """Encode a numpy array as a JSON-safe base64 blob."""
    arr = np.asarray(value)
    return {
        "enc": "npy",
        "dtype": arr.dtype.str,
        "shape": list(arr.shape),
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def _array_from_json(value: object) -> FloatArray:
    """Decode a base64 numpy blob produced by :func:`_array_to_json`."""
    if not isinstance(value, dict) or value.get("enc") != "npy":
        raise VerificationError(_EMPTY_TEMPLATE_MSG)
    try:
        dtype = np.dtype(str(value["dtype"]))
        shape = tuple(int(dim) for dim in value["shape"])
        raw = base64.b64decode(str(value["data"]), validate=True)
        expected = int(np.prod(shape, dtype=int)) * dtype.itemsize
        if len(raw) != expected:
            raise VerificationError(_EMPTY_TEMPLATE_MSG)
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    except (TypeError, ValueError) as exc:
        raise VerificationError(_EMPTY_TEMPLATE_MSG) from exc


def _gmm_to_json(params: MixtureParams | None) -> dict[str, object] | None:
    if params is None:
        return None
    return {
        key: (_array_to_json(value) if isinstance(value, np.ndarray) else value)
        for key, value in params.items()
    }


def _gmm_from_json(params: object) -> MixtureParams | None:
    if params is None:
        return None
    if not isinstance(params, dict):
        raise VerificationError(_EMPTY_TEMPLATE_MSG)
    return {
        key: (_array_from_json(value) if isinstance(value, dict) else value)
        for key, value in params.items()
    }


def export_template(
    profile: EnrollmentProfile,
    *,
    password: str,
    salt: bytes,
    iterations: int | None = None,
) -> bytes:
    """Serialize an enrollment into a cancelable template (JSON bytes).

    The payload contains only the cancelable theta: the transformed centroid
    ``R_u · c``, transformed precision ``R_u P R_uᵀ``, GMM parameters, and
    non-biometric metadata.  Raw embeddings/feature vectors are never
    included.  The registry encrypts the result at rest.
    """
    if profile.space == "cancelable":
        raise VerificationError("template export only supports raw in-memory profiles")

    r = derive_projection(
        password, salt, embedding_dim=profile.embedding_dim, iterations=iterations
    )
    centroid_c = transform_embedding(profile.centroid, password, salt, iterations=iterations)
    precision_c: FloatArray | None = None
    if profile.precision is not None:
        precision_c = r @ profile.precision @ r.T

    payload: dict[str, object] = {
        "type": "voiceguard.v2.speaker_profile",
        "format_version": "1",
        "space": "cancelable",
        "model_version": profile.model_version,
        "embedding_dim": profile.embedding_dim,
        "statistical_dim": profile.statistical_dim,
        "created_at": profile.created_at,
        "n_samples": profile.n_samples,
        "intra_consistency": profile.intra_consistency,
        "adaptive_threshold": profile.adaptive_threshold,
        "centroid": _array_to_json(centroid_c.astype(np.float32)),
        "precision": _array_to_json(precision_c) if precision_c is not None else None,
        "gmm": _gmm_to_json(profile.gmm.to_params() if profile.gmm is not None else None),
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def import_template(data: bytes) -> EnrollmentProfile:
    """Rebuild an ``EnrollmentProfile`` from cancelable template bytes.

    The returned profile has ``space == "cancelable"``; ``verify`` then
    requires the caller-supplied ``transform_r`` (derived from the same
    password + salt) to transform the probe into template space.
    """
    try:
        payload: object = json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise VerificationError(_EMPTY_TEMPLATE_MSG) from exc
    if not isinstance(payload, dict):
        raise VerificationError(_EMPTY_TEMPLATE_MSG)
    if payload.get("type") != "voiceguard.v2.speaker_profile":
        raise VerificationError(_EMPTY_TEMPLATE_MSG)
    if payload.get("format_version") != "1":
        raise VerificationError(f"unsupported template format: {payload.get('format_version')!r}")
    if payload.get("space") != "cancelable":
        raise VerificationError("template is not in cancelable space")

    try:
        embedding_dim = int(payload["embedding_dim"])
        centroid = np.asarray(_array_from_json(payload["centroid"]), dtype=np.float64)
        precision = payload.get("precision")
        precision_arr: FloatArray | None = (
            _array_from_json(precision) if precision is not None else None
        )
        if centroid.shape != (embedding_dim,):
            raise VerificationError("template centroid has an invalid shape")
        if precision_arr is not None and precision_arr.shape != (embedding_dim, embedding_dim):
            raise VerificationError("template precision has an invalid shape")
        gmm_params = _gmm_from_json(payload.get("gmm"))
        gmm = (
            DiagonalGaussianMixture.from_params(gmm_params)
            if gmm_params is not None
            else None
        )
        statistical_dim_obj = payload.get("statistical_dim")
        statistical_dim = int(statistical_dim_obj) if statistical_dim_obj is not None else None
    except (KeyError, TypeError, ValueError) as exc:
        raise VerificationError(_EMPTY_TEMPLATE_MSG) from exc

    return EnrollmentProfile(
        user_id="",  # reassigned by the caller / service layer
        embedding_dim=embedding_dim,
        statistical_dim=statistical_dim,
        n_samples=int(payload["n_samples"]),
        model_version=str(payload["model_version"]),
        centroid=centroid,
        precision=precision_arr,
        gmm=gmm,
        intra_consistency=float(payload["intra_consistency"]),
        adaptive_threshold=float(payload["adaptive_threshold"]),
        space="cancelable",
        created_at=str(payload["created_at"]),
    )


# ── Attempt limiting (in-memory scaffold; Redis-backed later) ─────────


@dataclass
class AttemptDecision:
    """Whether an attempt is allowed and, if not, when to retry."""

    allowed: bool
    retry_after_seconds: int


class AttemptLimiter:
    """In-memory attempt limiting with escalating cooldown (§5.2).

    After ``max_attempts`` consecutive failures a cooldown starts; repeated
    lockout cycles escalate the cooldown (30s → 60s → 300s).  A successful
    attempt resets the counter.

    Note: this is a Phase 6 scaffold; production moves the state to Redis.
    """

    def __init__(
        self,
        *,
        max_attempts: int = 5,
        cooldowns: tuple[int, int, int] = (30, 60, 300),
    ) -> None:
        if max_attempts < 1:
            raise VerificationError("max_attempts must be >= 1")
        if not cooldowns or any(c < 0 for c in cooldowns):
            raise VerificationError("cooldowns must be non-empty and non-negative")
        self.max_attempts = int(max_attempts)
        self.cooldowns = tuple(int(c) for c in cooldowns)
        self._state: dict[str, tuple[int, float]] = {}

    def check(self, user_id: str, *, now: float | None = None) -> AttemptDecision:
        failures, cooldown_until = self._state.get(user_id, (0, 0.0))
        current = time.time() if now is None else now
        if current < cooldown_until:
            return AttemptDecision(False, max(1, int(np.ceil(cooldown_until - current))))
        return AttemptDecision(True, 0)

    def record(self, user_id: str, *, success: bool, now: float | None = None) -> None:
        current = time.time() if now is None else now
        failures, _ = self._state.get(user_id, (0, 0.0))
        if success:
            self._state.pop(user_id, None)
            return
        failures += 1
        cooldown_until = 0.0
        if failures % self.max_attempts == 0:
            index = min((failures // self.max_attempts) - 1, len(self.cooldowns) - 1)
            cooldown_until = current + self.cooldowns[index]
        self._state[user_id] = (failures, cooldown_until)

    def reset(self, user_id: str) -> None:
        self._state.pop(user_id, None)
