"""VoiceGuard V2 — Phase 5 canonicalization.

Deterministically converts a validated :class:`DecodedAudio` into the
canonical VoiceGuard representation:

  - mono (down-mix any channels)
  - target sample rate (configurable, default 16 kHz)
  - float32 waveform
  - peak-normalized to a configurable target amplitude (default -1 dBFS)
    with a hard clipping guard
  - finite and within [-1.0, 1.0]

All steps are deterministic for identical input.  No perceptual/enhancement
processing is applied here that could distort speaker characteristics.
"""

from __future__ import annotations

import numpy as np

from voiceguard.config import settings
from voiceguard.voice.errors import AudioSilenceError
from voiceguard.voice.result import CanonicalWaveform, DecodedAudio


def canonicalize(decoded: DecodedAudio) -> CanonicalWaveform:
    """Produce the canonical single-channel float32 waveform at target rate."""
    mono = _to_mono(decoded)
    resampled = _resample_to_target(mono, decoded.sample_rate)
    normalized = _normalize(resampled)
    _assert_finite_in_range(normalized)
    return CanonicalWaveform(
        samples=normalized.astype(np.float32, copy=False),
        sample_rate=settings.AUDIO_SAMPLE_RATE,
    )


def _to_mono(decoded: DecodedAudio) -> np.ndarray:
    """Down-mix to mono by averaging channels (deterministic)."""
    if decoded.channels == 1:
        return decoded.samples.copy()
    return np.mean(decoded.samples, axis=1, dtype=np.float64)


def _resample_to_target(
    mono: np.ndarray, source_rate: int
) -> np.ndarray:
    """Resample to target rate using a deterministic polyphase resampler."""
    from scipy.signal import resample_poly

    if source_rate == settings.AUDIO_SAMPLE_RATE:
        return mono
    from math import gcd

    g = gcd(int(source_rate), int(settings.AUDIO_SAMPLE_RATE))
    up = int(settings.AUDIO_SAMPLE_RATE) // g
    down = int(source_rate) // g
    return np.asarray(resample_poly(mono, up, down), dtype=np.float64)


def _normalize(mono: np.ndarray) -> np.ndarray:
    """Peak-normalize to the configured dBFS target with a clipping guard."""
    peak = float(np.max(np.abs(mono)))
    if peak <= 1e-12:
        raise AudioSilenceError("Cannot normalize: audio is (near) digital silence.")
    target_amp = _dbfs_to_linear(settings.PEAK_TARGET_DBFS)
    gain = target_amp / peak
    normalized = mono * gain
    # Clipping guard: never exceed unity magnitude, and reject if the raw
    # peak already clips our target (i.e. no headroom is possible).
    if float(np.max(np.abs(normalized))) > 1.0:
        raise AudioSilenceError("Normalization would clip the waveform.")
    return normalized


def _dbfs_to_linear(dbfs: float) -> float:
    """Convert decibels relative to full scale to a linear amplitude factor."""
    return float(10.0 ** (dbfs / 20.0))


def _assert_finite_in_range(mono: np.ndarray) -> None:
    if not np.isfinite(mono).all():
        raise AudioSilenceError(
            "Canonicalization produced non-finite values (NaN/Inf)."
        )
    lo = float(np.min(mono))
    hi = float(np.max(mono))
    if lo < -1.0 or hi > 1.0:
        raise AudioSilenceError(
            f"Canonicalization produced out-of-range amplitude [{lo:.3f}, {hi:.3f}]."
        )
