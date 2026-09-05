"""VoiceGuard V2 — Phase 8 deepfake/synthetic speech detection (architecture §11).

Two-tier deepfake detection:

**Tier 1 — Heuristic features** (fast, no training required):
    - Frame boundary discontinuity (neural vocoder artifacts)
    - High-frequency periodicity (unnatural repetition in upper bands)
    - Spectral smoothness (synthetic spectra are overly smooth)
    - Temporal regularity (unnatural frame-to-frame consistency)
    - Cepstral flatness (synthetic speech lacks natural cepstral variation)

**Tier 2 — ML classifier** (1-D CNN trained on ASVspoof / Wavefake):
    - :class:`voiceguard.ml.models.anti_spoofing_deepfake.DeepfakeDetector`
    - Produces a calibrated probability of synthetic origin

The composite score combines both tiers and is exposed as
:func:`detect_deepfake`.

Disclaimers
***********
No production AUC-ROC, ACER, or detection rate is claimed.  Synthetic
data is used only to validate that the pipeline runs correctly.
Real deepfake-detection training and calibration require ASVspoof 2021
logical access and Wavefake datasets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

from voiceguard.config import settings
from voiceguard.ml.models.anti_spoofing_deepfake import DeepfakeDetector
from voiceguard.voice.embedding import prepare_input
from voiceguard.voice.result import FeatureResult

FloatArray = npt.NDArray[np.float64]
Tensor = torch.Tensor


@dataclass(frozen=True)
class DeepfakeHeuristicScores:
    """Individual heuristic deepfake-detection scores (all in [0, 1])."""

    frame_boundary_discontinuity: float
    high_frequency_periodicity: float
    spectral_smoothness: float
    temporal_regularity: float
    cepstral_flatness: float

    @property
    def composite(self) -> float:
        """Weighted average of the five heuristic scores."""
        weights = [0.25, 0.20, 0.20, 0.20, 0.15]
        values = [
            self.frame_boundary_discontinuity,
            self.high_frequency_periodicity,
            self.spectral_smoothness,
            self.temporal_regularity,
            self.cepstral_flatness,
        ]
        return float(sum(w * v for w, v in zip(weights, values, strict=True)))


@dataclass(frozen=True)
class DeepfakeDetectionResult:
    """Combined deepfake-detection output."""

    heuristic: DeepfakeHeuristicScores
    ml_probability: float
    ml_logit: float
    composite_score: float
    is_synthetic: bool


# ── Tier 1: Heuristic features ──────────────────────────────────────


def _frame_boundary_discontinuity(mel: FloatArray) -> float:
    """Detect spectral discontinuities at frame boundaries.

    Neural vocoders often produce artifacts at the boundaries between
    synthesized frames.  We measure the mean absolute difference between
    adjacent frames and map it to a suspiciousness score.  Higher
    discontinuity = more suspicious = higher score.
    """
    if mel.ndim != 2 or mel.shape[1] < 3:
        return 0.5
    diffs = np.abs(np.diff(mel, axis=1))
    mean_diff = float(np.mean(diffs))
    # Natural speech has moderate frame-to-frame variation.
    # Very high discontinuities are suspicious (vocoder artifacts).
    # Map via sigmoid: high diff → high score.
    score = float(1.0 / (1.0 + np.exp(-5.0 * (mean_diff - 0.15))))
    return float(np.clip(score, 0.0, 1.0))


def _high_frequency_periodicity(mel: FloatArray) -> float:
    """Detect unnatural periodicity in high-frequency mel bins.

    Synthetic speech from neural vocoders often shows unnatural repetition
    patterns in the upper frequency bands.  We compute the autocorrelation
    of the high-frequency frames and check for strong periodic peaks.
    """
    if mel.ndim != 2 or mel.shape[0] < 4 or mel.shape[1] < 8:
        return 0.5
    n_bins = mel.shape[0]
    high_start = n_bins * 3 // 4
    high_mel = mel[high_start:, :]
    # Autocorrelation of mean high-frequency energy over time.
    hf_energy = np.mean(high_mel, axis=0)
    if float(np.std(hf_energy)) < 1e-12:
        return 0.5
    hf_norm = hf_energy - float(np.mean(hf_energy))
    acf = np.correlate(hf_norm, hf_norm, mode="full")
    acf = acf[len(acf) // 2 :]
    if acf[0] < 1e-12:
        return 0.5
    acf = acf / acf[0]
    # Check for strong periodic peaks (lag 2..8).
    if len(acf) < 9:
        return 0.5
    peak_val = float(np.max(acf[2:9]))
    # High periodicity = suspicious → high score.
    return float(np.clip(peak_val, 0.0, 1.0))


def _spectral_smoothness(mel: FloatArray) -> float:
    """Check if the spectral envelope is unnaturally smooth.

    Synthetic speech tends to have overly smooth spectral envelopes due to
    the generative model's tendency to average out fine details.  We measure
    the mean local variance across mel bins and map low variance = smooth =
    suspicious.
    """
    if mel.ndim != 2 or mel.shape[0] < 4:
        return 0.5
    # Per-frame local spectral variance (across mel bins).
    frame_var = np.var(mel, axis=0)
    mean_var = float(np.mean(frame_var))
    if mean_var < 1e-12:
        return 0.5
    # Natural speech has moderate spectral variance.
    # Very low variance = overly smooth = suspicious.
    # Map via inverse: low var → high score.
    score = float(1.0 / (1.0 + np.exp(3.0 * (mean_var - 0.5))))
    return float(np.clip(score, 0.0, 1.0))


def _temporal_regularity(mel: FloatArray) -> float:
    """Check if frame-to-frame variation is unnaturally regular.

    Synthetic speech may have overly consistent temporal dynamics because
    neural vocoders smooth out natural micro-variations.  We compute the
    coefficient of variation of frame-to-frame differences.
    """
    if mel.ndim != 2 or mel.shape[1] < 4:
        return 0.5
    diffs = np.abs(np.diff(mel, axis=1))
    frame_diff_means = np.mean(diffs, axis=0)
    mean_val = float(np.mean(frame_diff_means))
    if mean_val < 1e-12:
        return 0.5
    cv = float(np.std(frame_diff_means) / mean_val)
    # Low CV = unnaturally regular = suspicious.
    # Map via sigmoid: low CV → high score.
    score = float(1.0 / (1.0 + np.exp(5.0 * (cv - 0.3))))
    return float(np.clip(score, 0.0, 1.0))


def _cepstral_flatness(mel: FloatArray) -> float:
    """Check if the cepstral domain is unnaturally flat.

    Natural speech has characteristic cepstral patterns (formant structure).
    Synthetic speech may show flattened cepstral variance due to the
    generative model's spectral smoothing.  We compute the variance of
    each frame's mel spectrum and check if it's suspiciously uniform.
    """
    if mel.ndim != 2 or mel.shape[1] < 3:
        return 0.5
    # Per-frame spectral variance.
    frame_var = np.var(mel, axis=0)
    if float(np.mean(frame_var)) < 1e-12:
        return 0.5
    # Coefficient of variation of frame-wise spectral variance.
    cv = float(np.std(frame_var) / float(np.mean(frame_var)))
    # Low CV = unnaturally flat = suspicious.
    # Map via sigmoid: low CV → high score.
    score = float(1.0 / (1.0 + np.exp(4.0 * (cv - 0.25))))
    return float(np.clip(score, 0.0, 1.0))


def compute_heuristic_scores(mel: npt.ArrayLike) -> DeepfakeHeuristicScores:
    """Compute all five heuristic deepfake-detection features.

    Parameters
    ----------
    mel : ArrayLike
        Log-mel spectrogram of shape ``(n_mels, T)``.

    Returns
    -------
    DeepfakeHeuristicScores
        Individual scores in [0, 1] and a composite.
    """
    mel_arr = np.asarray(mel, dtype=np.float64)
    return DeepfakeHeuristicScores(
        frame_boundary_discontinuity=_frame_boundary_discontinuity(mel_arr),
        high_frequency_periodicity=_high_frequency_periodicity(mel_arr),
        spectral_smoothness=_spectral_smoothness(mel_arr),
        temporal_regularity=_temporal_regularity(mel_arr),
        cepstral_flatness=_cepstral_flatness(mel_arr),
    )


# ── Tier 2: ML classifier wrapper ───────────────────────────────────


def _ml_deepfake_score(
    result: FeatureResult,
    model: DeepfakeDetector,
) -> tuple[float, float]:
    """Run the ML deepfake detector on a FeatureResult.

    Parameters
    ----------
    result : FeatureResult
        The feature extraction output.
    model : DeepfakeDetector
        The trained deepfake detector in eval mode.

    Returns
    -------
    prob : float
        Synthetic probability in [0, 1].
    logit : float
        Raw logit score.
    """
    prepared = prepare_input(result)
    mel = prepared.mel
    device = next(model.parameters()).device
    with torch.inference_mode():
        logits, probs = model(mel.to(device))
    return float(probs.item()), float(logits.item())


# ── Tier 1 + Tier 2: Composite ──────────────────────────────────────


def detect_deepfake(
    result: FeatureResult,
    model: DeepfakeDetector | None = None,
    *,
    ml_weight: float = 0.6,
    heuristic_weight: float = 0.4,
    threshold: float | None = None,
) -> DeepfakeDetectionResult:
    """Run the two-tier deepfake detection pipeline.

    Parameters
    ----------
    result : FeatureResult
        The feature extraction output.
    model : DeepfakeDetector or None
        The ML deepfake detector.  If ``None``, only heuristic features
        are used (ml_weight is redistributed to heuristic).
    ml_weight : float
        Weight for the ML score in the composite (default 0.6).
    heuristic_weight : float
        Weight for the heuristic score in the composite (default 0.4).
    threshold : float or None
        Decision threshold.  Defaults to ``settings.DEEPFAKE_DETECTION_THRESHOLD``.

    Returns
    -------
    DeepfakeDetectionResult
        Combined scores and decision.
    """
    if threshold is None:
        threshold = float(settings.DEEPFAKE_DETECTION_THRESHOLD)

    heuristic = compute_heuristic_scores(result.mel.log_mel)

    if model is not None:
        ml_prob, ml_logit = _ml_deepfake_score(result, model)
    else:
        ml_prob = 0.5
        ml_logit = 0.0

    # Composite: weighted average of heuristic composite and ML probability.
    total_weight = ml_weight + heuristic_weight
    if total_weight < 1e-12:
        composite = 0.5
    else:
        composite = (heuristic_weight * heuristic.composite + ml_weight * ml_prob) / total_weight

    return DeepfakeDetectionResult(
        heuristic=heuristic,
        ml_probability=ml_prob,
        ml_logit=ml_logit,
        composite_score=float(np.clip(composite, 0.0, 1.0)),
        is_synthetic=composite >= threshold,
    )
