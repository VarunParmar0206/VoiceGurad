"""VoiceGuard V2 — Phase 5 domain-specific audio errors.

These exceptions wrap low-level decoding / DSP failures into a small set of
typed errors so callers (and later phases) can handle failures without
depending on the underlying libraries (numpy, scipy, librosa).
"""

from __future__ import annotations


class AudioError(Exception):
    """Base class for all voice pipeline errors."""


class AudioValidationError(AudioError):
    """The supplied audio does not meet validation requirements.

    Raised for structural problems such as unsupported dtype, unsupported
    shape/channel layout, empty input, or out-of-range values that can be
    detected without decoding.
    """


class UnsupportedAudioError(AudioValidationError):
    """The input format or layout is not one VoiceGuard supports."""


class AudioDecodeError(AudioError):
    """The audio could not be decoded from the supplied bytes/path/file."""


class AudioDurationError(AudioValidationError):
    """The audio exceeds configured duration or sample-count bounds."""


class AudioNumericError(AudioValidationError):
    """The audio contains non-finite values (NaN/Inf) or violates the
    amplitude range contract."""


class AudioSilenceError(AudioValidationError):
    """The audio is (or becomes) effectively silent / below usable energy."""


class AudioFeatureError(AudioError):
    """Feature extraction produced an invalid or inconsistent result."""


class FeatureValidationError(AudioFeatureError):
    """The computed feature output failed a validity check (e.g. NaN/Inf)."""
