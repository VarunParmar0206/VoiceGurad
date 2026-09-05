"""VoiceGuard V2 — Phase 5 audio validation.

Validates a :class:`DecodedAudio` before canonicalization.  Checks structural
properties (channel count, sample rate) and value properties (finite, range,
duration, energy) and raises typed, domain-specific errors.
"""

from __future__ import annotations

import numpy as np

from voiceguard.config import settings
from voiceguard.voice.errors import (
    AudioDurationError,
    AudioNumericError,
    AudioSilenceError,
    AudioValidationError,
    UnsupportedAudioError,
)
from voiceguard.voice.result import DecodedAudio

_MIN_AMPLITUDE = -1.0
_MAX_AMPLITUDE = 1.0


def validate(decoded: DecodedAudio) -> DecodedAudio:
    """Validate a decoded sample; raise on the first detected problem."""
    _validate_layout(decoded)
    _validate_values(decoded)
    _validate_duration(decoded)
    _validate_energy(decoded)
    return decoded


def _validate_layout(decoded: DecodedAudio) -> None:
    if decoded.sample_rate != settings.AUDIO_SAMPLE_RATE and decoded.sample_rate <= 0:
        raise UnsupportedAudioError(
            f"Invalid sample rate {decoded.sample_rate}; must be positive."
        )
    if decoded.num_samples < 1:
        raise AudioValidationError("Audio is empty.")
    if decoded.channels < 1:
        raise UnsupportedAudioError(
            f"Unsupported channel count {decoded.channels}; expected mono."
        )


def _validate_values(decoded: DecodedAudio) -> None:
    if decoded.samples.size == 0:
        raise AudioValidationError("Audio is empty.")
    if not np.isfinite(decoded.samples).all():
        raise AudioNumericError(
            "Audio contains non-finite values (NaN/Inf); refusing to proceed."
        )
    lo = float(np.min(decoded.samples))
    hi = float(np.max(decoded.samples))
    if lo < _MIN_AMPLITUDE or hi > _MAX_AMPLITUDE:
        raise AudioNumericError(
            f"Audio amplitude out of range: [{lo:.3f}, {hi:.3f}]; "
            f"expected within [{_MIN_AMPLITUDE}, {_MAX_AMPLITUDE}]."
        )


def _validate_duration(decoded: DecodedAudio) -> None:
    duration = decoded.duration_seconds
    if duration < settings.MIN_AUDIO_DURATION_SECONDS:
        raise AudioValidationError(
            f"Audio too short: {duration:.3f}s (minimum "
            f"{settings.MIN_AUDIO_DURATION_SECONDS}s)."
        )
    if duration > settings.MAX_AUDIO_DURATION_SECONDS:
        raise AudioDurationError(
            f"Audio too long: {duration:.3f}s (maximum "
            f"{settings.MAX_AUDIO_DURATION_SECONDS}s)."
        )


def _validate_energy(decoded: DecodedAudio) -> None:
    rms = float(np.sqrt(np.mean(np.square(decoded.samples))))
    if rms == 0.0:
        raise AudioSilenceError("Audio is digital silence (zero energy).")
