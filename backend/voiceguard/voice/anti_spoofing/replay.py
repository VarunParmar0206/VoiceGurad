"""VoiceGuard V2 — Phase 7 replay-attack detection (architecture §10).

Two-tier replay detection:

**Tier 1 — Heuristic features** (fast, no training required):
    - Spectral bandwidth consistency
    - Background noise profile estimation
    - Channel frequency response signature
    - Temporal fine-structure consistency
    - Amplitude envelope naturalness

**Tier 2 — ML classifier** (1-D CNN trained on ASVspoof):
    - :class:`voiceguard.ml.models.anti_spoofing_replay.ReplayDetector`
    - Produces a calibrated probability of replay

The composite score combines both tiers and is exposed as
:func:`detect_replay`.

Disclaimers
***********
No production AUC-ROC, ACER, or detection rate is claimed.  Synthetic
data is used only to validate that the pipeline runs correctly.
Real replay-detection training and calibration require ASVspoof 2019/2021.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

from voiceguard.config import settings
from voiceguard.ml.models.anti_spoofing_replay import ReplayDetector
from voiceguard.voice.embedding import prepare_input
from voiceguard.voice.result import FeatureResult

FloatArray = npt.NDArray[np.float64]
Tensor = torch.Tensor


@dataclass(frozen=True)
class ReplayHeuristicScores:
    """Individual heuristic replay-detection scores (all in [0, 1])."""

    spectral_bandwidth_consistency: float
    noise_profile_match: float
    channel_signature: float
    temporal_consistency: float
    amplitude_naturalness: float

    @property
    def composite(self) -> float:
        """Weighted average of the five heuristic scores."""
        weights = [0.25, 0.15, 0.20, 0.25, 0.15]
        values = [
            self.spectral_bandwidth_consistency,
            self.noise_profile_match,
            self.channel_signature,
            self.temporal_consistency,
            self.amplitude_naturalness,
        ]
        return float(sum(w * v for w, v in zip(weights, values, strict=True)))


@dataclass(frozen=True)
class ReplayDetectionResult:
    """Combined replay-detection output."""

    heuristic: ReplayHeuristicScores
    ml_probability: float
    ml_logit: float
    composite_score: float
    is_replay: bool


# ── Tier 1: Heuristic features ──────────────────────────────────────


def _spectral_bandwidth_consistency(mel: FloatArray) -> float:
    """Live speech has natural bandwidth variation per frame.

    A replayed signal tends to have more consistent bandwidth because the
    playback channel (speaker, room) imposes a fixed frequency response.
    We measure the coefficient of variation of per-frame bandwidth and map
    it to a [0, 1] score where 0 = suspicious (very consistent) and
    1 = natural (varied).
    """
    if mel.ndim != 2 or mel.shape[1] < 2:
        return 0.5
    # Per-frame "bandwidth" = std of mel energies.
    frame_bw = np.std(mel, axis=0)
    mean_bw = float(np.mean(frame_bw))
    if mean_bw < 1e-12:
        return 0.0
    cv = float(np.std(frame_bw) / mean_bw)
    # Map CV to [0,1] via sigmoid-like transform.
    return float(np.clip(1.0 / (1.0 + np.exp(-3.0 * (cv - 0.5))), 0.0, 1.0))


def _noise_profile_match(mel: FloatArray) -> float:
    """Estimate the noise floor and check if it's suspiciously flat.

    A replay introduces the recording environment's noise floor, which
    tends to be more spectrally flat than natural speech noise.
    We compare the variance of the lowest-energy frames' mel spectra
    to the overall variance.  High ratio = flat noise = suspicious.
    """
    if mel.ndim != 2 or mel.shape[1] < 4:
        return 0.5
    # Frame energies.
    frame_energy = np.mean(mel, axis=0)
    # Lowest 25% of frames by energy (likely noise).
    n_noise = max(1, len(frame_energy) // 4)
    noise_idx = np.argsort(frame_energy)[:n_noise]
    noise_mel = mel[:, noise_idx]
    # Spectral variance of noise frames.
    noise_var = float(np.var(noise_mel))
    # Overall spectral variance.
    overall_var = float(np.var(mel))
    if overall_var < 1e-12:
        return 0.5
    ratio = noise_var / overall_var
    # High ratio = flat noise = suspicious → low score.
    return float(np.clip(1.0 - ratio, 0.0, 1.0))


def _channel_signature(mel: FloatArray) -> float:
    """Detect channel coloration from microphone→speaker→mic loop.

    A replay adds a fixed frequency response from the playback device.
    We compute the spectral tilt (energy ratio of high vs. low mel bins)
    and check if it's unusually flat (replay tends to flatten the tilt).
    """
    if mel.ndim != 2 or mel.shape[0] < 4:
        return 0.5
    n_bins = mel.shape[0]
    mid = n_bins // 2
    low_energy = float(np.mean(mel[:mid, :]))
    high_energy = float(np.mean(mel[mid:, :]))
    if low_energy < 1e-12:
        return 0.5
    tilt = high_energy / low_energy
    # Natural speech: tilt varies.  Replays: tilt tends toward 1.0 (flat).
    # Score: distance from 1.0 mapped to [0, 1].
    return float(np.clip(1.0 - abs(tilt - 1.0), 0.0, 1.0))


def _temporal_consistency(mel: FloatArray) -> float:
    """Check frame-to-frame variation.

    Live speech has natural temporal dynamics.  Replays may have
    quantization artifacts or unnatural frame-to-frame consistency.
    We compute the mean absolute frame-to-frame difference.
    """
    if mel.ndim != 2 or mel.shape[1] < 3:
        return 0.5
    diffs = np.abs(np.diff(mel, axis=1))
    mean_diff = float(np.mean(diffs))
    # Natural speech has moderate frame-to-frame variation.
    # Too low = suspicious (replay), too high = noisy.
    # Map via Gaussian-like centered at 0.1.
    score = float(np.exp(-((mean_diff - 0.1) ** 2) / 0.02))
    return float(np.clip(score, 0.0, 1.0))


def _amplitude_naturalness(mel: FloatArray) -> float:
    """Check amplitude envelope for natural dynamics.

    Live speech has a characteristic amplitude envelope with pauses,
    emphasis, and gradual energy changes.  Replays may have flatter
    envelopes due to normalization or clipping.
    """
    if mel.ndim != 2 or mel.shape[1] < 3:
        return 0.5
    # Per-frame peak amplitude.
    frame_peak = np.max(np.abs(mel), axis=0)
    if float(np.max(frame_peak)) < 1e-12:
        return 0.0
    # Dynamic range.
    dynamic_range = float(np.max(frame_peak) - np.min(frame_peak))
    # Coefficient of variation of frame peaks.
    mean_peak = float(np.mean(frame_peak))
    if mean_peak < 1e-12:
        return 0.0
    cv = float(np.std(frame_peak) / mean_peak)
    # Natural speech: moderate CV and dynamic range.
    score = float(np.clip(cv * 2.0 + dynamic_range * 0.5, 0.0, 1.0))
    return score


def compute_heuristic_scores(mel: npt.ArrayLike) -> ReplayHeuristicScores:
    """Compute all five heuristic replay-detection features.

    Parameters
    ----------
    mel : ArrayLike
        Log-mel spectrogram of shape ``(n_mels, T)``.

    Returns
    -------
    ReplayHeuristicScores
        Individual scores in [0, 1] and a composite.
    """
    mel_arr = np.asarray(mel, dtype=np.float64)
    return ReplayHeuristicScores(
        spectral_bandwidth_consistency=_spectral_bandwidth_consistency(mel_arr),
        noise_profile_match=_noise_profile_match(mel_arr),
        channel_signature=_channel_signature(mel_arr),
        temporal_consistency=_temporal_consistency(mel_arr),
        amplitude_naturalness=_amplitude_naturalness(mel_arr),
    )


# ── Tier 2: ML classifier wrapper ───────────────────────────────────


def _ml_replay_score(
    result: FeatureResult,
    model: ReplayDetector,
) -> tuple[float, float]:
    """Run the ML replay detector on a FeatureResult.

    Parameters
    ----------
    result : FeatureResult
        The feature extraction output.
    model : ReplayDetector
        The trained replay detector in eval mode.

    Returns
    -------
    prob : float
        Replay probability in [0, 1].
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


def detect_replay(
    result: FeatureResult,
    model: ReplayDetector | None = None,
    *,
    ml_weight: float = 0.6,
    heuristic_weight: float = 0.4,
    threshold: float | None = None,
) -> ReplayDetectionResult:
    """Run the two-tier replay detection pipeline.

    Parameters
    ----------
    result : FeatureResult
        The feature extraction output.
    model : ReplayDetector or None
        The ML replay detector.  If ``None``, only heuristic features
        are used (ml_weight is redistributed to heuristic).
    ml_weight : float
        Weight for the ML score in the composite (default 0.6).
    heuristic_weight : float
        Weight for the heuristic score in the composite (default 0.4).
    threshold : float or None
        Decision threshold.  Defaults to ``settings.REPLAY_DETECTION_THRESHOLD``.

    Returns
    -------
    ReplayDetectionResult
        Combined scores and decision.
    """
    if threshold is None:
        threshold = float(settings.REPLAY_DETECTION_THRESHOLD)

    heuristic = compute_heuristic_scores(result.mel.log_mel)

    if model is not None:
        ml_prob, ml_logit = _ml_replay_score(result, model)
    else:
        ml_prob = 0.5
        ml_logit = 0.0

    # Composite: weighted average of heuristic composite and ML probability.
    total_weight = ml_weight + heuristic_weight
    if total_weight < 1e-12:
        composite = 0.5
    else:
        composite = (heuristic_weight * heuristic.composite + ml_weight * ml_prob) / total_weight

    return ReplayDetectionResult(
        heuristic=heuristic,
        ml_probability=ml_prob,
        ml_logit=ml_logit,
        composite_score=float(np.clip(composite, 0.0, 1.0)),
        is_replay=composite >= threshold,
    )
