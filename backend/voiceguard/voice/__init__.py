"""VoiceGuard V2 — Phase 5 voice preprocessing & feature extraction.

Public entry points:
  - :func:`voiceguard.voice.process` —— run the full pipeline on any input
  - :mod:`voiceguard.voice.io` / :mod:`voiceguard.voice.validation` —— input
  - :mod:`voiceguard.voice.canonical` —— canonical 16 kHz mono float32
  - :mod:`voiceguard.voice.preprocessing` —— DC, pre-emphasis, VAD, framing
  - :mod:`voiceguard.voice.features` —— mel + statistical features
"""

from __future__ import annotations

from voiceguard.voice.canonical import canonicalize
from voiceguard.voice.errors import (
    AudioDecodeError,
    AudioDurationError,
    AudioError,
    AudioFeatureError,
    AudioNumericError,
    AudioSilenceError,
    AudioValidationError,
    FeatureValidationError,
    UnsupportedAudioError,
)
from voiceguard.voice.features import extract_features, extract_mel, extract_statistical
from voiceguard.voice.io import decode_array, decode_bytes, decode_source
from voiceguard.voice.pipeline import process
from voiceguard.voice.preprocessing import preprocess
from voiceguard.voice.result import (
    CanonicalWaveform,
    DecodedAudio,
    FeatureResult,
    MelFeatures,
    PreprocessedAudio,
    QualityReport,
    StatisticalFeatures,
    VoiceActivityResult,
)
from voiceguard.voice.validation import validate

__all__ = [
    "AudioDecodeError",
    "AudioDurationError",
    "AudioError",
    "AudioFeatureError",
    "AudioNumericError",
    "AudioSilenceError",
    "AudioValidationError",
    "CanonicalWaveform",
    "DecodedAudio",
    "FeatureResult",
    "FeatureValidationError",
    "MelFeatures",
    "PreprocessedAudio",
    "QualityReport",
    "StatisticalFeatures",
    "UnsupportedAudioError",
    "VoiceActivityResult",
    "canonicalize",
    "decode_array",
    "decode_bytes",
    "decode_source",
    "extract_features",
    "extract_mel",
    "extract_statistical",
    "preprocess",
    "process",
    "validate",
]
