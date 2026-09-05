"""VoiceGuard V2 — Phase 5 processing pipeline.

Orchestrates the full Phase 5 flow:

    input -> decode -> validate -> canonicalize -> preprocess -> features

into a single :class:`FeatureResult`.  Callers (Phase 6 ML, API routes, or
shared client preprocessing) consume :class:`FeatureResult` without knowing
pipeline internals.  Raw waveforms are never retained or returned as part of
the result.
"""

from __future__ import annotations

import numpy as np

from voiceguard.voice.canonical import canonicalize
from voiceguard.voice.features import extract_mel, extract_statistical
from voiceguard.voice.io import decode_source
from voiceguard.voice.preprocessing import preprocess
from voiceguard.voice.result import (
    FeatureResult,
    QualityReport,
)
from voiceguard.voice.validation import validate


def process(source: object, *, sample_rate: int | None = None) -> FeatureResult:
    """Run the full Phase 5 pipeline on an arbitrary audio source.

    ``sample_rate`` must be supplied when ``source`` is a raw numpy array
    recorded at a rate other than the default canonical rate (16 kHz);
    byte/WAV sources carry their rate in the header.
    """
    decoded = decode_source(source, sample_rate=sample_rate)
    validate(decoded)

    canonical = canonicalize(decoded)
    preprocessed = preprocess(canonical)
    mel = extract_mel(preprocessed)
    statistical = extract_statistical(preprocessed)

    peak = (
        float(np.max(np.abs(canonical.samples)))
        if canonical.num_samples
        else 0.0
    )
    rms = (
        float(np.sqrt(np.mean(np.square(canonical.samples))))
        if canonical.num_samples
        else 0.0
    )

    warnings: list[str] = []
    if preprocessed.vad.was_trimmed:
        warnings.append("Leading/trailing silence trimmed during preprocessing.")
    if preprocessed.vad.voice_fraction < 0.5:
        warnings.append(
            "Low voice activity fraction "
            f"({preprocessed.vad.voice_fraction:.2f})."
        )

    quality = QualityReport(
        input_duration_seconds=decoded.duration_seconds,
        canonical_duration_seconds=canonical.duration_seconds,
        peak_amplitude=peak,
        rms_amplitude=rms,
        voice_fraction=preprocessed.vad.voice_fraction,
        trimmed=preprocessed.vad.was_trimmed,
        warnings=warnings,
    )

    return FeatureResult(
        sample_rate=canonical.sample_rate,
        num_samples=canonical.num_samples,
        duration_seconds=canonical.duration_seconds,
        mel=mel,
        statistical=statistical,
        quality=quality,
        meta={
            "feature_dtype": "float32",
            "mel_layout": f"(n_mels={mel.n_mels}, T={mel.time_frames})",
            "statistical_dim": statistical.dimension,
        },
    )
