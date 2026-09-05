"""VoiceGuard V2 — voice preprocessing & speaker-verification.

Public entry points (Phase 5):
  - :func:`voiceguard.voice.process` —— run the full pipeline on any input
  - :mod:`voiceguard.voice.io` / :mod:`voiceguard.voice.validation` —— input
  - :mod:`voiceguard.voice.canonical` —— canonical 16 kHz mono float32
  - :mod:`voiceguard.voice.preprocessing` —— DC, pre-emphasis, VAD, framing
  - :mod:`voiceguard.voice.features` —— mel + statistical features

Public entry points (Phase 6):
  - :mod:`voiceguard.voice.embedding` —— FeatureResult → model adapter
  - :mod:`voiceguard.voice.verification` —— enrollment + verification
  - :mod:`voiceguard.voice.cancelable` —— cancelable biometric transform

Public entry points (Phase 7):
  - :mod:`voiceguard.voice.anti_spoofing` —— replay detection (Tier 1 + Tier 2)

Public entry points (Phase 8):
  - :mod:`voiceguard.voice.anti_spoofing` —— deepfake detection (Tier 1 + Tier 2)

Public entry points (Phase 9):
  - :mod:`voiceguard.voice.anti_spoofing` —— voice-conversion detection (three-check heuristic)
"""

from __future__ import annotations

from voiceguard.voice.anti_spoofing import (
    DeepfakeDetectionResult,
    DeepfakeHeuristicScores,
    ReplayDetectionResult,
    ReplayHeuristicScores,
    VoiceConversionDetectionResult,
    VoiceConversionHeuristicScores,
    compute_deepfake_heuristic_scores,
    compute_heuristic_scores,
    compute_voice_conversion_heuristic_scores,
    detect_deepfake,
    detect_replay,
    detect_voice_conversion,
)
from voiceguard.voice.cancelable import (
    derive_projection,
    new_salt,
    transform_batch,
    transform_embedding,
)
from voiceguard.voice.canonical import canonicalize
from voiceguard.voice.embedding import (
    ModelInput,
    batch_inputs,
    embed_batch,
    embed_result,
    prepare_input,
)
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
from voiceguard.voice.verification import (
    AttemptDecision,
    AttemptLimiter,
    DiagonalGaussianMixture,
    EnrollmentProfile,
    EnrollmentSample,
    VerificationResult,
    VerificationScores,
    enroll,
    export_template,
    fit_background,
    import_template,
    log_likelihood_ratio,
    verify,
)

__all__ = [
    "AttemptDecision",
    "AttemptLimiter",
    "AudioDecodeError",
    "AudioDurationError",
    "AudioError",
    "AudioFeatureError",
    "AudioNumericError",
    "AudioSilenceError",
    "AudioValidationError",
    "CanonicalWaveform",
    "DecodedAudio",
    "DeepfakeDetectionResult",
    "DeepfakeHeuristicScores",
    "DiagonalGaussianMixture",
    "EnrollmentProfile",
    "EnrollmentSample",
    "FeatureResult",
    "FeatureValidationError",
    "MelFeatures",
    "ModelInput",
    "PreprocessedAudio",
    "QualityReport",
    "ReplayDetectionResult",
    "ReplayHeuristicScores",
    "StatisticalFeatures",
    "UnsupportedAudioError",
    "VerificationResult",
    "VerificationScores",
    "VoiceActivityResult",
    "VoiceConversionDetectionResult",
    "VoiceConversionHeuristicScores",
    "batch_inputs",
    "canonicalize",
    "compute_deepfake_heuristic_scores",
    "compute_heuristic_scores",
    "compute_voice_conversion_heuristic_scores",
    "decode_array",
    "decode_bytes",
    "decode_source",
    "derive_projection",
    "detect_deepfake",
    "detect_replay",
    "detect_voice_conversion",
    "embed_batch",
    "embed_result",
    "enroll",
    "export_template",
    "extract_features",
    "extract_mel",
    "extract_statistical",
    "fit_background",
    "import_template",
    "log_likelihood_ratio",
    "new_salt",
    "prepare_input",
    "preprocess",
    "process",
    "transform_batch",
    "transform_embedding",
    "validate",
    "verify",
]
