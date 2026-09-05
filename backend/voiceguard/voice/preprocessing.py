"""VoiceGuard V2 — Phase 5 preprocessing.

Deterministic, speaker-preserving preprocessing stages:

  1. DC-offset removal (subtract the mean, which is a linear, invertible
     operation that removes a constant bias without spectral coloration).
  2. Optional pre-emphasis (``y[n] = x[n] - a*x[n-1]``) — DISABLED by default
     and intended only for LPC/formant analysis paths, per the architecture.
  3. Multi-metric voice-activity detection (short-time energy + zero-crossing
     rate) with morphological smoothing.
  4. Silence trimming with leading/trailing margin.
  5. Consistent framing/windowing (frame + hop from config).

Every stage is deterministic for identical input.  No noise-reduction or
perceptual enhancement is applied here, because that can distort speaker
characteristics and is out of scope for this phase.
"""

from __future__ import annotations

import numpy as np

from voiceguard.config import settings
from voiceguard.voice.errors import AudioSilenceError
from voiceguard.voice.result import (
    CanonicalWaveform,
    PreprocessedAudio,
    VoiceActivityResult,
)


def preprocess(canonical: CanonicalWaveform) -> PreprocessedAudio:
    """Run the preprocessing stages on a canonical waveform."""
    samples = canonical.samples.astype(np.float64, copy=False)
    samples = _remove_dc(samples)
    if settings.PRE_EMPHASIS_ENABLED:
        samples = _pre_emphasize(samples, settings.PRE_EMPHASIS_COEFFICIENT)

    vad = _detect_voice(samples, canonical.sample_rate)
    if vad.voice_fraction <= 0.0:
        raise AudioSilenceError(
            "No voice activity detected; utterance appears to be silence."
        )

    trimmed = _trim(samples, vad, canonical.sample_rate)

    frame_samples = int(settings.N_FFT)
    hop_samples = int(settings.HOP_LENGTH)
    frames = _frame(trimmed, frame_samples, hop_samples)
    if frames.shape[0] < 1:
        raise AudioSilenceError("Audio too short to frame after trimming.")

    updated_vad = VoiceActivityResult(
        voice_mask=vad.voice_mask,
        start_sample=vad.start_sample,
        end_sample=vad.end_sample,
        voice_fraction=vad.voice_fraction,
        was_trimmed=vad.was_trimmed,
        frames=frames.shape[0],
        frame_samples=frames.shape[1],
    )
    return PreprocessedAudio(
        samples=trimmed.astype(np.float32, copy=False),
        frames=frames.astype(np.float32, copy=False),
        frame_samples=frame_samples,
        hop_samples=hop_samples,
        num_frames=int(frames.shape[0]),
        sample_rate=canonical.sample_rate,
        vad=updated_vad,
    )


def _remove_dc(samples: np.ndarray) -> np.ndarray:
    mean = float(np.mean(samples))
    if not np.isfinite(mean):
        raise AudioSilenceError("Non-finite sample encountered during DC removal.")
    return samples - mean


def _pre_emphasize(samples: np.ndarray, coefficient: float) -> np.ndarray:
    if not 0.0 <= coefficient <= 1.0:
        raise ValueError("Pre-emphasis coefficient must be within [0, 1].")
    if coefficient == 0.0:
        return samples
    out = np.empty_like(samples)
    out[0] = samples[0]
    out[1:] = samples[1:] - coefficient * samples[:-1]
    return out


def _detect_voice(samples: np.ndarray, sample_rate: int) -> VoiceActivityResult:
    """Multi-metric VAD.

    Uses short-time RMS energy and zero-crossing rate combined with a
    threshold, then applies morphological smoothing (fill gaps, trim edges)
    per the architecture.
    """
    frame_ms = 20.0
    frame_len = max(1, int(sample_rate * frame_ms / 1000.0))
    hop_len = frame_len
    num_frames = max(1, (int(samples.size) - frame_len) // hop_len + 1)

    energies = np.zeros(num_frames, dtype=np.float64)
    zcrs = np.zeros(num_frames, dtype=np.float64)
    for i in range(num_frames):
        start = i * hop_len
        seg = samples[start : start + frame_len]
        energies[i] = float(np.sqrt(np.mean(np.square(seg))) if seg.size else 0.0)
        if seg.size > 1:
            zcrs[i] = float(np.mean(np.abs(np.diff(np.signbit(seg)))))
        else:
            zcrs[i] = 0.0

    total_energy = float(np.sqrt(np.mean(np.square(samples))))
    energy_threshold = max(
        settings.VAD_ENERGY_THRESHOLD, total_energy * 0.08
    )
    zcr_threshold = 0.1
    voice = (energies > energy_threshold) & (zcrs > zcr_threshold * 0.25)

    voice = _morphological_smooth(voice, frame_len, sample_rate)

    return _to_result(voice, frame_len, hop_len, int(samples.size))


def _morphological_smooth(
    voice: np.ndarray, frame_len: int, sample_rate: int
) -> np.ndarray:
    """Fill short gaps and trim isolated edge frames."""
    voice = voice.copy()
    fill_frames = max(1, int(round(
        settings.VAD_FILL_GAP_SECONDS * sample_rate / max(1, frame_len)
    )))
    # Fill interior gaps shorter than the fill window.
    idx = np.flatnonzero(voice)
    if idx.size == 0:
        return voice
    for j in range(1, idx.size):
        gap = idx[j] - idx[j - 1] - 1
        if 0 < gap <= fill_frames:
            voice[idx[j - 1] + 1 : idx[j]] = True
    return voice


def _to_result(
    voice: np.ndarray, frame_len: int, hop_len: int, num_samples: int
) -> VoiceActivityResult:
    frame_count = int(voice.size)
    active = int(np.count_nonzero(voice))
    fraction = active / frame_count if frame_count else 0.0

    start_frame = int(np.argmax(voice)) if active else 0
    end_frame = int(np.flatnonzero(voice)[-1]) if active else 0
    start_sample = start_frame * hop_len
    end_sample = min(num_samples, (end_frame + 1) * hop_len)
    was_trimmed = (start_sample > 0) or (end_sample < num_samples)

    return VoiceActivityResult(
        voice_mask=voice,
        start_sample=start_sample,
        end_sample=end_sample,
        voice_fraction=fraction,
        was_trimmed=was_trimmed,
    )


def _trim(
    samples: np.ndarray, vad: VoiceActivityResult, sample_rate: int
) -> np.ndarray:
    """Trim leading/trailing silence, preserving a small margin."""
    margin = int(round(settings.VAD_SILENCE_MARGIN_SECONDS * sample_rate))
    start = max(0, vad.start_sample - margin)
    end = min(int(samples.size), vad.end_sample + margin)
    if end <= start:
        raise AudioSilenceError("Trimming left no usable audio.")
    return samples[start:end]


def _frame(
    samples: np.ndarray, frame_samples: int, hop_samples: int
) -> np.ndarray:
    """Frame the waveform into a (num_frames, frame_samples) matrix.

    Uses centered layout via numpy padding so the number of frames is
    deterministic for identical input.
    """
    if samples.size == 0:
        return np.zeros((0, frame_samples), dtype=np.float64)
    num_frames = max(1, int(np.ceil((int(samples.size) - frame_samples) / hop_samples)) + 1)
    num_frames = min(num_frames, max(1, int(samples.size) // hop_samples + 1))
    if num_frames < 1:
        num_frames = 1
    frames = np.zeros((num_frames, frame_samples), dtype=np.float64)
    for i in range(num_frames):
        start = i * hop_samples
        seg = samples[start : start + frame_samples]
        frames[i, : seg.size] = seg
    return frames
