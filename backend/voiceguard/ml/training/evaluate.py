"""VoiceGuard V2 — Evaluation utilities (architecture §24.4).

Computes pairwise scores, EER, DET-curve data, TAR@FAR, and threshold
analysis on score arrays supplied by the caller.

Disclaimers
***********
These utilities are correct math helpers.  They may be applied to synthetic
data for tests, but any number produced from synthetic data is a **test-only
artifact** and must never be published as VoiceGuard biometric performance.
Real EER/TAR@FAR reporting requires a genuine speaker dataset (future work).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from voiceguard.ml.errors import TrainingError

FloatArray = npt.NDArray[np.float64]


def pairwise_similarity(probes: npt.ArrayLike, enrollment: npt.ArrayLike) -> FloatArray:
    """Cosine similarity matrix ``(M, N)`` between probe and enrollment rows.

    Score convention: higher = more similar.
    """
    p = np.asarray(probes, dtype=np.float64)
    e = np.asarray(enrollment, dtype=np.float64)
    if p.ndim != 2 or e.ndim != 2:
        raise TrainingError("probes/enrollment must be 2-D matrices")
    if p.shape[1] != e.shape[1]:
        raise TrainingError("probe and enrollment must share the feature dimension")
    if p.shape[0] == 0 or e.shape[0] == 0:
        raise TrainingError("probes/enrollment must be non-empty")

    pn = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)
    en = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
    return np.clip(pn @ en.T, -1.0, 1.0)


def _validate_scores(
    genuine: npt.ArrayLike, impostor: npt.ArrayLike
) -> tuple[FloatArray, FloatArray]:
    g = np.asarray(genuine, dtype=np.float64)
    i = np.asarray(impostor, dtype=np.float64)
    if g.ndim != 1 or i.ndim != 1:
        raise TrainingError("genuine/impostor scores must be 1-D")
    if g.size == 0 or i.size == 0:
        raise TrainingError("genuine/impostor score arrays must be non-empty")
    if not (np.isfinite(g).all() and np.isfinite(i).all()):
        raise TrainingError("score arrays contain non-finite values")
    return g, i


def threshold_analysis(
    genuine: npt.ArrayLike,
    impostor: npt.ArrayLike,
    thresholds: npt.ArrayLike | None = None,
) -> list[tuple[float, float, float]]:
    """For each threshold return ``(threshold, far, frr)``.

    ``far`` = fraction of impostor scores >= threshold (false acceptance).
    ``frr`` = fraction of genuine scores < threshold (false rejection).
    """
    g, i = _validate_scores(genuine, impostor)
    if thresholds is None:
        cand = np.unique(np.concatenate([g, i]))
        cand = np.concatenate([cand, [cand.min() - 1.0, cand.max() + 1.0]])
        cand = np.sort(cand)
    else:
        cand = np.asarray(thresholds, dtype=np.float64).ravel()
    results: list[tuple[float, float, float]] = []
    for t in cand:
        far = float(np.mean(i >= t))
        frr = float(np.mean(g < t))
        results.append((float(t), far, frr))
    return results


def det_curve(
    genuine: npt.ArrayLike,
    impostor: npt.ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """DET-curve data: ascending FAR, matching FRR, and thresholds."""
    analysis = threshold_analysis(genuine, impostor)
    fars = np.asarray([far for _, far, _ in analysis], dtype=np.float64)
    frrs = np.asarray([frr for _, _, frr in analysis], dtype=np.float64)
    thresholds = np.asarray([t for t, _, _ in analysis], dtype=np.float64)
    order = np.argsort(fars, kind="stable")
    return fars[order], frrs[order], thresholds[order]


def eer(genuine: npt.ArrayLike, impostor: npt.ArrayLike) -> float:
    """Equal-error rate: FAR == FRR, linearly interpolated between samples.

    Returns a value in [0.0, 1.0].  The exact result depends on thresholds,
    so this is a helper for supplied data — not a deployed-biometric claim.
    """
    analysis = threshold_analysis(genuine, impostor)
    if len(analysis) > 1:
        return _interpolate_eer(analysis)
    return 1.0


def _interpolate_eer(analysis: list[tuple[float, float, float]]) -> float:
    diffs = np.asarray([far - frr for _, far, frr in analysis], dtype=np.float64)
    for idx in range(len(diffs) - 1):
        d0, d1 = diffs[idx], diffs[idx + 1]
        if d0 == 0.0:
            _far, _frr = analysis[idx][1], analysis[idx][2]
            return float((_far + _frr) / 2.0)
        if d0 * d1 < 0.0:
            f0, f1 = analysis[idx][1], analysis[idx + 1][1]
            r0, r1 = analysis[idx][2], analysis[idx + 1][2]
            w = d0 / (d0 - d1)
            far = f0 + w * (f1 - f0)
            frr = r0 + w * (r1 - r0)
            return float((far + frr) / 2.0)
    # No sign change: one side always dominates.
    far, frr = diffs[-1], diffs[0]
    return float((far + frr) / 2.0)


def tar_at_far(
    genuine: npt.ArrayLike,
    impostor: npt.ArrayLike,
    far_target: float = 0.01,
) -> tuple[float, float]:
    """Return ``(threshold, TAR)`` for the FAR budget.

    Among thresholds whose false-acceptance rate is at most ``far_target``,
    the LOOSEST one is returned — that threshold attains the highest true
    accept rate (1 - FRR) while staying inside the FAR budget.
    """
    if not 0.0 <= far_target <= 1.0:
        raise TrainingError("far_target must be within [0, 1]")
    g, i = _validate_scores(genuine, impostor)
    cand = np.unique(np.concatenate([g, i]))
    cand = np.concatenate([cand, [cand.min() - 1.0, cand.max() + 1.0]])
    cand = np.sort(cand)
    chosen: float | None = None
    for t in cand:
        if float(np.mean(i >= t)) <= far_target:
            chosen = float(t)  # ascending order: first satisfying = loosest
            break
    if chosen is None:
        # The floor threshold (below every impostor score) always
        # satisfies any valid FAR budget, so this is unreachable.
        raise TrainingError("internal error: no threshold met the FAR budget")
    tar = float(np.mean(g >= chosen))
    return chosen, tar
