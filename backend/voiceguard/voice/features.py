"""VoiceGuard V2 — Phase 5 feature extraction.

Implements the canonical mel-spectrogram (architecture §8.2 Path A) and the
per-utterance statistical/quality feature vector (architecture §8.2 Path B).

The DSP primitives live in :mod:`voiceguard.voice.dsp` (numpy/scipy only) so
feature extraction is deterministic and independent of librosa's runtime
numa/numba coupling.  All outputs are finite float32/float64 with documented
tensor layouts:

  - Mel:     ``log_mel`` shape ``(n_mels, T)``, float32
  - Stats:   1-D float32 vector, fixed known dimension
"""

from __future__ import annotations

import numpy as np

from voiceguard.config import settings
from voiceguard.voice import dsp
from voiceguard.voice.errors import FeatureValidationError
from voiceguard.voice.preprocessing import preprocess
from voiceguard.voice.result import (
    CanonicalWaveform,
    MelFeatures,
    PreprocessedAudio,
    StatisticalFeatures,
)


def extract_mel(preprocessed: PreprocessedAudio) -> MelFeatures:
    """Log-scaled mel spectrogram, shape ``(n_mels, T)``."""
    n_fft = int(settings.N_FFT)
    hop_length = int(settings.HOP_LENGTH)
    n_mels = int(settings.N_MELS)
    sr = preprocessed.sample_rate

    if preprocessed.num_frames < 1:
        raise FeatureValidationError("No frames available for mel extraction.")

    signal = preprocessed.samples.astype(np.float64, copy=False)
    power = dsp.stft_power(signal, sr, n_fft, hop_length)
    banks = dsp.mel_filterbank(n_mels, n_fft, sr, float(settings.F_MIN), float(settings.F_MAX))
    mel_power = banks @ power  # (n_mels, T)
    log_mel = np.log(mel_power + float(settings.MEL_FLOOR)).astype(np.float32)

    if not np.isfinite(log_mel).all():
        raise FeatureValidationError("Mel-spectrogram contains NaN/Inf.")
    if log_mel.ndim != 2 or log_mel.shape[0] != n_mels:
        raise FeatureValidationError(
            f"Unexpected mel shape {log_mel.shape}; expected ({n_mels}, T)."
        )

    return MelFeatures(
        log_mel=log_mel,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        sample_rate=sr,
        f_min=float(settings.F_MIN),
        f_max=float(settings.F_MAX),
    )


def extract_statistical(preprocessed: PreprocessedAudio) -> StatisticalFeatures:
    """One-dimensional statistical/quality feature vector (Path B)."""
    signal = preprocessed.samples.astype(np.float64, copy=False)
    sr = preprocessed.sample_rate
    n_mfcc = int(settings.N_MFCC)
    n_mel_mfcc = 128  # internal mel resolution used to derive MFCCs

    if signal.size < 2:
        raise FeatureValidationError("Signal too short for statistical features.")

    power = dsp.stft_power(signal, sr, int(settings.N_FFT), int(settings.HOP_LENGTH))
    freqs = np.linspace(0.0, sr / 2.0, power.shape[0], dtype=np.float64)

    # MFCC statistics with delta / delta-delta.
    mfcc = _mfcc(signal, sr, n_mfcc, n_mel_mfcc)
    delta = dsp.delta(mfcc)
    delta2 = dsp.delta(delta)
    mfcc_stats = _stack_stats(mfcc, delta, delta2)

    # Spectral statistics.
    centroid = dsp.spectral_centroid(power, freqs)
    rolloff = dsp.spectral_rolloff(power, freqs)
    bandwidth = dsp.spectral_bandwidth(power, freqs, centroid)
    contrast = dsp.spectral_contrast(
        power, freqs, float(settings.F_MIN), float(settings.F_MAX)
    )
    flatness = dsp.spectral_flatness(power)
    spectral = np.asarray(
        [
            float(np.mean(centroid)),
            float(np.mean(rolloff)),
            float(np.mean(bandwidth)),
            float(np.mean(contrast)),
            float(np.mean(flatness)),
        ],
        dtype=np.float64,
    )

    # Time-domain statistics.
    frame_len = int(sr * 0.020)
    hop_len = frame_len // 2
    zcr = dsp.zero_crossing_rate(signal, frame_len, hop_len)
    rms = dsp.rms(signal, frame_len, hop_len)
    zcr_stats = np.asarray([float(np.mean(zcr)), float(np.std(zcr))], dtype=np.float64)
    rms_stats = np.asarray([float(np.mean(rms)), float(np.std(rms))], dtype=np.float64)

    # Pitch (autocorrelation f0) statistics.
    f0, voiced = dsp.autocorrelation_f0(signal, sr)
    pitch_values = f0[voiced]
    if pitch_values.size:
        pitch_stats = np.asarray(
            [
                float(np.mean(pitch_values)),
                float(np.std(pitch_values)) if pitch_values.size > 1 else 0.0,
                float(np.max(pitch_values)) - float(np.min(pitch_values)),
                float(np.count_nonzero(voiced)) / float(voiced.size),
            ],
            dtype=np.float64,
        )
    else:
        pitch_stats = np.zeros(4, dtype=np.float64)

    # Formant statistics.
    formant_frames = dsp.formants(signal, sr, order=16, max_formants=3)
    nonzero_frames = formant_frames[formant_frames[:, 0] > 0.0]
    formant_stats = np.zeros(6, dtype=np.float64)
    if nonzero_frames.size:
        for j in range(3):
            col = nonzero_frames[:, j]
            if col.size:
                formant_stats[2 * j] = float(np.mean(col))
                formant_stats[2 * j + 1] = float(np.std(col)) if col.size > 1 else 0.0

    values = np.concatenate(
        [mfcc_stats, spectral, zcr_stats, pitch_stats, rms_stats, formant_stats]
    )
    if not np.isfinite(values).all():
        raise FeatureValidationError("Statistical feature vector contains NaN/Inf.")

    return StatisticalFeatures(
        values=values.astype(np.float32, copy=False),
        names=_statistical_names(n_mfcc),
    )


def extract_features(canonical: CanonicalWaveform) -> tuple[MelFeatures, StatisticalFeatures]:
    """Convenience: preprocess then extract both feature paths."""
    pp = preprocess(canonical)
    return extract_mel(pp), extract_statistical(pp)


def _mfcc(
    signal: np.ndarray, sr: int, n_mfcc: int, n_mel: int
) -> np.ndarray:
    """MFCC features, layout ``(n_mfcc, T)`` (standard librosa convention)."""
    n_fft = int(settings.N_FFT)
    hop = int(settings.HOP_LENGTH)
    power = dsp.stft_power(signal, sr, n_fft, hop)
    banks = dsp.mel_filterbank(
        n_mel, n_fft, sr, max(0.0, float(settings.F_MIN)), float(settings.F_MAX)
    )
    log_mel = np.log(banks @ power + float(settings.MEL_FLOOR))
    return dsp.dct_type2(log_mel, n_mfcc)


def _stack_stats(
    mfcc: np.ndarray, delta: np.ndarray, delta2: np.ndarray
) -> np.ndarray:
    """Concatenate mean/std of mfcc, delta, delta-delta (6 blocks of n_mfcc)."""
    return np.concatenate(
        [
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(delta, axis=1),
            np.std(delta, axis=1),
            np.mean(delta2, axis=1),
            np.std(delta2, axis=1),
        ]
    )


def _statistical_names(n_mfcc: int) -> list[str]:
    names: list[str] = []
    for prefix in (
        "mfcc_mean",
        "mfcc_std",
        "delta_mean",
        "delta_std",
        "delta2_mean",
        "delta2_std",
    ):
        names.extend(f"{prefix}_{i}" for i in range(n_mfcc))
    names.extend(
        [
            "spectral_centroid_mean",
            "spectral_rolloff_mean",
            "spectral_bandwidth_mean",
            "spectral_contrast_mean",
            "spectral_flatness_mean",
            "zcr_mean",
            "zcr_std",
            "pitch_mean",
            "pitch_std",
            "pitch_range",
            "pitch_voiced_fraction",
            "rms_mean",
            "rms_std",
            "f1_mean",
            "f1_std",
            "f2_mean",
            "f2_std",
            "f3_mean",
            "f3_std",
        ]
    )
    return names
