"""VoiceGuard V2 — Phase 9 voice-conversion/mimicry detection (architecture §12).

Three-check voice-conversion detection:

**Check 1 — Embedding Consistency:**
    Compare CNN-LSTM-Attention speaker embedding with a parallel
    traditional feature vector (statistical features).  Converted
    speech may produce embeddings that disagree with traditional
    features because the VC system alters the voice identity while
    preserving linguistic content.

**Check 2 — Prosodic Naturalness:**
    - Pitch contour smoothness
    - Energy contour naturalness
    - Speaking rate consistency
    Unnatural prosody flags potential conversion.

**Check 3 — Spectral-Temporal Coherence:**
    Joint analysis of spectral envelope and temporal fine structure.
    Converted speech often shows inconsistencies between these domains
    because VC systems may smooth the spectral envelope while distorting
    temporal dynamics (or vice versa).

The composite score combines all three checks and is exposed as
:func:`detect_voice_conversion`.

Disclaimers
***********
No production AUC-ROC, ACER, or detection rate is claimed.  Synthetic
data is used only to validate that the pipeline runs correctly.
Real voice-conversion detection training and calibration require VC
system outputs (e.g., StarGAN-VC, OpenVoice) versus genuine speech.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from voiceguard.config import settings
from voiceguard.voice.result import FeatureResult, QualityReport

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class VoiceConversionHeuristicScores:
    """Individual heuristic voice-conversion-detection scores (all in [0, 1])."""

    embedding_consistency: float
    pitch_smoothness: float
    energy_naturalness: float
    speaking_rate_consistency: float
    spectral_temporal_coherence: float

    @property
    def composite(self) -> float:
        """Weighted average of the five heuristic scores."""
        weights = [0.30, 0.20, 0.20, 0.15, 0.15]
        values = [
            self.embedding_consistency,
            self.pitch_smoothness,
            self.energy_naturalness,
            self.speaking_rate_consistency,
            self.spectral_temporal_coherence,
        ]
        return float(sum(w * v for w, v in zip(weights, values, strict=True)))


@dataclass(frozen=True)
class VoiceConversionDetectionResult:
    """Combined voice-conversion-detection output."""

    heuristic: VoiceConversionHeuristicScores
    ml_probability: float
    ml_logit: float
    composite_score: float
    is_converted: bool


# ── Check 1: Embedding Consistency ──────────────────────────────────


def _embedding_consistency(
    embedding: FloatArray | None,
    stat_features: FloatArray,
) -> float:
    """Compare CNN-LSTM embedding with traditional feature vector.

    Converted speech may produce neural embeddings that disagree with
    traditional statistical features because the VC system alters voice
    identity while preserving linguistic content.  We compute the cosine
    distance between the L2-normalized CNN-LSTM embedding and a
    truncated, L2-normalized statistical feature vector.

    High distance = more suspicious = higher score.
    """
    if embedding is None or stat_features.size < 2:
        return 0.5

    # Truncate or pad statistical features to match embedding dimension.
    emb_dim = int(embedding.size)
    stat_dim = int(stat_features.size)
    if stat_dim >= emb_dim:
        stat_truncated = stat_features[:emb_dim].astype(np.float64)
    else:
        stat_truncated = np.zeros(emb_dim, dtype=np.float64)
        stat_truncated[:stat_dim] = stat_features.astype(np.float64)

    # L2-normalize both vectors.
    emb_norm = embedding.astype(np.float64)
    emb_len = float(np.linalg.norm(emb_norm))
    if emb_len < 1e-12:
        return 0.5
    emb_norm = emb_norm / emb_len

    stat_norm = stat_truncated
    stat_len = float(np.linalg.norm(stat_norm))
    if stat_len < 1e-12:
        return 0.5
    stat_norm = stat_norm / stat_len

    # Cosine distance = 1 - cosine similarity.
    cosine_sim = float(np.dot(emb_norm, stat_norm))
    cosine_dist = 1.0 - float(np.clip(cosine_sim, -1.0, 1.0))

    # Map via sigmoid: high distance → high score (suspicious).
    score = float(1.0 / (1.0 + np.exp(-10.0 * (cosine_dist - 0.5))))
    return float(np.clip(score, 0.0, 1.0))


# ── Check 2: Prosodic Naturalness ───────────────────────────────────


def _pitch_smoothness(mel: FloatArray) -> float:
    """Analyze pitch contour smoothness from mel-spectrogram.

    Natural speech has smooth, continuous pitch contours.  Converted
    speech may have jittery or unnaturally flat pitch due to the VC
    system's pitch transformation.  We estimate the dominant frequency
    per frame and measure its smoothness.
    """
    if mel.ndim != 2 or mel.shape[1] < 4:
        return 0.5

    # Estimate dominant frequency per frame (argmax of mel energy).
    dominant_bin = np.argmax(mel, axis=0).astype(np.float64)

    # Compute frame-to-frame pitch changes.
    pitch_diffs = np.abs(np.diff(dominant_bin))
    mean_diff = float(np.mean(pitch_diffs))
    if mean_diff < 1e-12:
        # Very flat pitch — could be natural (sustained note) or unnatural.
        # Return moderate score.
        return 0.5

    # Coefficient of variation of pitch changes.
    cv = float(np.std(pitch_diffs) / mean_diff) if mean_diff > 1e-12 else 0.0

    # Natural speech: moderate CV (0.3–1.0).
    # Very low CV = unnaturally smooth = suspicious.
    # Very high CV = unnaturally jittery = suspicious.
    # Map via Gaussian centered at 0.6 (natural CV).
    score = float(np.exp(-((cv - 0.6) ** 2) / 0.3))
    # Invert: low CV or high CV → high score (suspicious).
    suspiciousness = 1.0 - score
    return float(np.clip(suspiciousness, 0.0, 1.0))


def _energy_naturalness(mel: FloatArray) -> float:
    """Check if the energy contour has natural dynamics.

    Natural speech has characteristic energy patterns with pauses,
    emphasis, and gradual changes.  Converted speech may have flatter
    or more erratic energy contours due to the VC system's amplitude
    normalization or artifacts.
    """
    if mel.ndim != 2 or mel.shape[1] < 4:
        return 0.5

    # Energy envelope: mean mel energy per frame.
    energy = np.mean(mel, axis=0).astype(np.float64)
    mean_energy = float(np.mean(energy))
    if mean_energy < 1e-12:
        return 0.5

    # Coefficient of variation of energy.
    cv = float(np.std(energy) / mean_energy)

    # Natural speech: moderate CV (0.2–0.8) and moderate range.
    # Very low CV = unnaturally flat = suspicious.
    # Very high CV = unnaturally erratic = suspicious.
    score = float(np.exp(-((cv - 0.5) ** 2) / 0.2))
    suspiciousness = 1.0 - score
    return float(np.clip(suspiciousness, 0.0, 1.0))


def _speaking_rate_consistency(
    mel: FloatArray,
    quality: QualityReport,
) -> float:
    """Check if the speaking rate is within natural bounds.

    We estimate speaking rate from the mel-spectrogram's temporal
    structure and the quality report's duration/voice-fraction
    information.  Converted speech may have unnatural speaking rates
    because the VC system may stretch or compress time.
    """
    if mel.ndim != 2 or mel.shape[1] < 2:
        return 0.5

    duration = quality.canonical_duration_seconds
    if duration <= 0.0:
        return 0.5

    # Estimate speaking rate: frames per second.
    n_frames = mel.shape[1]
    frames_per_second = n_frames / duration

    # Voice fraction indicates how much of the audio is voiced.
    voice_frac = quality.voice_fraction

    # Natural speech: voice fraction 0.4–0.8, frames/sec 50–150.
    # Very low voice fraction = mostly silence = unusual.
    # Very high voice fraction = no pauses = unnatural.
    # Very high or low frames_per_second = unusual rate.
    rate_score = 0.0

    # Check voice fraction.
    if voice_frac < 0.2 or voice_frac > 0.95:
        rate_score += 0.3
    elif voice_frac < 0.3 or voice_frac > 0.85:
        rate_score += 0.15

    # Check frames per second (rate).
    if frames_per_second < 30 or frames_per_second > 200:
        rate_score += 0.3
    elif frames_per_second < 40 or frames_per_second > 160:
        rate_score += 0.15

    return float(np.clip(rate_score, 0.0, 1.0))


# ── Check 3: Spectral-Temporal Coherence ────────────────────────────


def _spectral_temporal_coherence(mel: FloatArray) -> float:
    """Check for inconsistencies between spectral and temporal domains.

    Natural speech has a consistent relationship between spectral
    envelope variation and temporal dynamics.  Converted speech often
    shows mismatches because VC systems may smooth the spectral
    envelope while distorting temporal dynamics (or vice versa).
    We measure the ratio of spectral variation to temporal variation
    and check if it falls within natural bounds.
    """
    if mel.ndim != 2 or mel.shape[0] < 4 or mel.shape[1] < 4:
        return 0.5

    # Spectral variation: variance of mel energies across frames.
    spectral_var = float(np.var(np.mean(mel, axis=0)))

    # Temporal variation: variance of frame-to-frame differences.
    frame_diffs = np.abs(np.diff(mel, axis=1))
    temporal_var = float(np.var(np.mean(frame_diffs, axis=0)))

    if spectral_var < 1e-12 or temporal_var < 1e-12:
        return 0.5

    # Ratio of spectral to temporal variation.
    ratio = spectral_var / temporal_var

    # Natural speech: ratio typically 0.5–5.0.
    # Very low ratio = spectral is smooth but temporal is rough = suspicious.
    # Very high ratio = spectral is rough but temporal is smooth = suspicious.
    # Map via Gaussian centered at 1.5 (natural ratio).
    score = float(np.exp(-((ratio - 1.5) ** 2) / 4.0))
    suspiciousness = 1.0 - score
    return float(np.clip(suspiciousness, 0.0, 1.0))


# ── Public API: compute heuristic scores ─────────────────────────────


def compute_heuristic_scores(
    mel: npt.ArrayLike,
    embedding: FloatArray | None = None,
    quality: QualityReport | None = None,
) -> VoiceConversionHeuristicScores:
    """Compute all five heuristic voice-conversion-detection features.

    Parameters
    ----------
    mel : ArrayLike
        Log-mel spectrogram of shape ``(n_mels, T)``.
    embedding : FloatArray or None
        CNN-LSTM-Attention speaker embedding, shape ``(256,)``.
        If ``None``, embedding consistency returns 0.5.
    quality : QualityReport or None
        Quality metadata from Phase 5 pipeline.
        If ``None``, speaking rate consistency returns 0.5.

    Returns
    -------
    VoiceConversionHeuristicScores
        Individual scores in [0, 1] and a composite.
    """
    mel_arr = np.asarray(mel, dtype=np.float64)
    stat_features = np.asarray(
        _extract_statistical_stub(mel_arr), dtype=np.float64
    )

    if quality is None:
        quality = QualityReport(
            input_duration_seconds=1.0,
            canonical_duration_seconds=1.0,
            peak_amplitude=0.5,
            rms_amplitude=0.1,
            voice_fraction=0.5,
            trimmed=False,
        )

    return VoiceConversionHeuristicScores(
        embedding_consistency=_embedding_consistency(
            embedding if embedding is not None else None,
            stat_features,
        ),
        pitch_smoothness=_pitch_smoothness(mel_arr),
        energy_naturalness=_energy_naturalness(mel_arr),
        speaking_rate_consistency=_speaking_rate_consistency(mel_arr, quality),
        spectral_temporal_coherence=_spectral_temporal_coherence(mel_arr),
    )


def _extract_statistical_stub(mel: FloatArray) -> FloatArray:
    """Extract a simple statistical feature vector from mel-spectrogram.

    This is a lightweight stand-in for the full 259-dim statistical
    feature vector from Phase 5.  It captures basic spectral statistics
    that can be compared against the CNN-LSTM embedding for embedding
    consistency checking.
    """
    n_bins, n_frames = mel.shape

    # Spectral statistics per frame.
    frame_mean = np.mean(mel, axis=0)  # (T,)
    frame_std = np.std(mel, axis=0)    # (T,)
    frame_max = np.max(mel, axis=0)    # (T,)

    # Global statistics.
    features = np.array(
        [
            float(np.mean(frame_mean)),
            float(np.std(frame_mean)),
            float(np.mean(frame_std)),
            float(np.std(frame_std)),
            float(np.mean(frame_max)),
            float(np.std(frame_max)),
            float(np.median(frame_mean)),
            float(np.median(frame_std)),
            # Spectral centroid per frame (mean of bin indices weighted by energy).
            float(np.mean(np.argmax(mel, axis=0).astype(np.float64))),
            # Spectral flatness.
            float(np.exp(np.mean(np.log(np.abs(mel) + 1e-12)))),
            # Temporal dynamics.
            float(np.mean(np.abs(np.diff(frame_mean)))),
            float(np.std(np.abs(np.diff(frame_mean)))),
            # Energy range.
            float(np.max(frame_mean) - np.min(frame_mean)),
            # Frame-to-frame correlation.
            float(
                np.corrcoef(frame_mean[:-1], frame_mean[1:])[0, 1]
                if n_frames > 1
                else 0.0
            ),
        ],
        dtype=np.float64,
    )

    # Pad to 256 dimensions for embedding comparison.
    result = np.zeros(256, dtype=np.float64)
    result[: min(features.size, 256)] = features[: min(features.size, 256)]
    return result


# ── Composite: detect_voice_conversion ──────────────────────────────


def detect_voice_conversion(
    result: FeatureResult,
    embedding: FloatArray | None = None,
    *,
    ml_weight: float = 0.0,
    heuristic_weight: float = 1.0,
    threshold: float | None = None,
) -> VoiceConversionDetectionResult:
    """Run the voice-conversion detection pipeline.

    Parameters
    ----------
    result : FeatureResult
        The feature extraction output.
    embedding : FloatArray or None
        CNN-LSTM-Attention speaker embedding, shape ``(256,)``.
        If ``None``, embedding consistency returns 0.5.
    ml_weight : float
        Weight for the ML score in the composite (default 0.0).
    heuristic_weight : float
        Weight for the heuristic score in the composite (default 1.0).
    threshold : float or None
        Decision threshold.  Defaults to
        ``settings.VOICE_CONVERSION_DETECTION_THRESHOLD``.

    Returns
    -------
    VoiceConversionDetectionResult
        Combined scores and decision.
    """
    if threshold is None:
        threshold = float(settings.VOICE_CONVERSION_DETECTION_THRESHOLD)

    heuristic = compute_heuristic_scores(
        result.mel.log_mel,
        embedding=embedding,
        quality=result.quality,
    )

    # No ML model for Phase 9 (heuristic-only).
    ml_prob = 0.5
    ml_logit = 0.0

    # Composite: weighted average of heuristic composite and ML probability.
    total_weight = ml_weight + heuristic_weight
    if total_weight < 1e-12:
        composite = 0.5
    else:
        composite = (
            heuristic_weight * heuristic.composite + ml_weight * ml_prob
        ) / total_weight

    return VoiceConversionDetectionResult(
        heuristic=heuristic,
        ml_probability=ml_prob,
        ml_logit=ml_logit,
        composite_score=float(np.clip(composite, 0.0, 1.0)),
        is_converted=composite >= threshold,
    )
