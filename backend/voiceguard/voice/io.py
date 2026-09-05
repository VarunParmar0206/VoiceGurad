"""VoiceGuard V2 — Phase 5 audio input decoding.

Converts supported inputs into a :class:`DecodedAudio` at native sample rate
and channel count.  This module owns the decoding boundary: it enforces
memory/size limits *before* allocating output buffers and never stores raw
recordings to disk.
"""

from __future__ import annotations

import io as _io
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.io import wavfile

from voiceguard.config import settings
from voiceguard.voice.errors import (
    AudioDecodeError,
    AudioDurationError,
    UnsupportedAudioError,
)
from voiceguard.voice.result import DecodedAudio

# dtypes we can safely reinterpret to float without information loss beyond
# the source's precision.
_INT_DTYPES: frozenset[type[np.generic]] = frozenset(
    {np.int16, np.int32, np.uint8, np.int8, np.uint16}
)
_FLOAT_DTYPES: frozenset[type[np.generic]] = frozenset({np.float32, np.float64})

_MAX_SAMPLES_FALLBACK = 10_000_000  # ~10 min at 16 kHz, defensive ceiling


def _max_input_samples() -> int:
    """Upper bound on samples accepted per channel to bound memory.

    Uses the configured max duration when available, else a defensive
    absolute ceiling.  This prevents genuinely unbounded allocations.
    """
    max_samples = settings.MAX_AUDIO_SAMPLES or 0
    if max_samples and max_samples > 0:
        return max_samples
    configured = settings.MAX_AUDIO_DURATION_SECONDS * settings.AUDIO_SAMPLE_RATE
    return min(int(configured) or _MAX_SAMPLES_FALLBACK, _MAX_SAMPLES_FALLBACK)


def _ensure_finite_float(values: npt.NDArray[np.float64]) -> None:
    if not np.isfinite(values).all():
        raise AudioDecodeError("Audio contains non-finite values (NaN/Inf).")


def decode_bytes(data: bytes, *, sample_rate: int | None = None) -> DecodedAudio:
    """Decode raw audio bytes (currently WAV via a strict header parse).

    ``sample_rate`` may be supplied when the payload has no header (e.g. raw
    PCM); for WAV it is read from the header and the argument is ignored.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise UnsupportedAudioError(
            "decode_bytes expects bytes/bytearray; got "
            f"{type(data).__name__}."
        )
    if len(data) == 0:
        raise AudioDecodeError("Cannot decode empty audio payload.")

    try:
        rate, raw = wavfile.read(_io.BytesIO(bytes(data)))
    except Exception as exc:  # scipy raises several ValueError/TypeError/EOF
        raise AudioDecodeError(f"WAV decode failed: {exc}") from exc

    samples = _to_float_array(raw)
    channels = 1 if samples.ndim == 1 else int(samples.shape[1])
    decoded = DecodedAudio(
        samples=samples.astype(np.float32, copy=False),
        sample_rate=int(rate),
        channels=channels,
    )
    if decoded.duration_seconds > settings.MAX_AUDIO_DURATION_SECONDS:
        raise AudioDurationError(
            f"Audio duration {decoded.duration_seconds:.3f}s exceeds configured "
            f"maximum {settings.MAX_AUDIO_DURATION_SECONDS}s."
        )
    if decoded.num_samples > _max_input_samples():
        raise AudioDurationError(
            "Decoded sample count exceeds the configured safety bound."
        )
    return decoded


def decode_array(
    samples: npt.NDArray[np.generic],
    *,
    sample_rate: int = settings.AUDIO_SAMPLE_RATE,
) -> DecodedAudio:
    """Decode a raw numpy waveform into a :class:`DecodedAudio`.

    Supported dtypes: float32/float64, int16/int32/uint8/int8/uint16.
    1-D input is mono; 2-D input is ``(samples, channels)``.
    """
    if not isinstance(samples, np.ndarray):
        raise UnsupportedAudioError(
            f"Expected a numpy array, got {type(samples).__name__}."
        )
    if sample_rate <= 0:
        raise UnsupportedAudioError(f"Invalid sample rate: {sample_rate}.")
    if samples.size == 0 or samples.ndim == 0 or samples.ndim > 2:
        raise UnsupportedAudioError(
            "Audio must be a non-empty 1-D (mono) or 2-D (samples, channels) "
            f"array; got ndim={samples.ndim}, size={samples.size}."
        )

    if samples.dtype.type not in _INT_DTYPES | _FLOAT_DTYPES:
        raise UnsupportedAudioError(
            f"Unsupported dtype {samples.dtype}. Supported: "
            "float32/float64, int16/int32/uint8/int8/uint16."
        )
    if samples.dtype.type in _FLOAT_DTYPES:
        _ensure_finite_float(samples.astype(np.float64))

    if samples.ndim == 1:
        channels = 1
    else:
        channels = int(samples.shape[1])
        if channels < 1:
            raise UnsupportedAudioError("Audio must have at least one channel.")

    num_samples = int(samples.shape[0])
    if num_samples > _max_input_samples():
        raise AudioDurationError(
            "Array sample count exceeds the configured safety bound."
        )

    float_samples = _to_float_array(samples)
    decoded = DecodedAudio(
        samples=float_samples.astype(np.float32, copy=False),
        sample_rate=int(sample_rate),
        channels=channels,
    )
    if decoded.duration_seconds > settings.MAX_AUDIO_DURATION_SECONDS:
        raise AudioDurationError(
            f"Audio duration {decoded.duration_seconds:.3f}s exceeds configured "
            f"maximum {settings.MAX_AUDIO_DURATION_SECONDS}s."
        )
    return decoded


def decode_source(source: object, *, sample_rate: int | None = None) -> DecodedAudio:
    """Decode a flexible input source.

    Accepted sources:
      - numpy array -> :func:`decode_array`
      - bytes/bytearray -> :func:`decode_bytes`
      - ``pathlib.Path`` / ``str`` (WAV file path) -> read then decode
    """
    if isinstance(source, np.ndarray):
        return decode_array(source, sample_rate=sample_rate or settings.AUDIO_SAMPLE_RATE)
    if isinstance(source, (bytes, bytearray)):
        return decode_bytes(bytes(source), sample_rate=sample_rate)
    if isinstance(source, (str, Path)):
        path = Path(source)
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise AudioDecodeError(
                f"Could not read audio file {str(path)!r}: {exc}"
            ) from exc
        return decode_bytes(data, sample_rate=sample_rate)
    raise UnsupportedAudioError(
        f"Unsupported audio source type: {type(source).__name__}."
    )


def _to_float_array(samples: npt.NDArray[np.generic]) -> npt.NDArray[np.float64]:
    """Convert an integer/float waveform to float64 in the [-1.0, 1.0] range."""
    dtype = samples.dtype.type
    if dtype in _FLOAT_DTYPES:
        return samples.astype(np.float64)
    # Integer formats scale to [-1, 1]; normalize by the max magnitude.
    if dtype in (np.int16, np.int32):
        info = np.iinfo(dtype)
        return samples.astype(np.float64) / float(info.max)
    if dtype == np.uint8:
        return (samples.astype(np.float64) - 128.0) / 128.0
    if dtype == np.int8:
        return samples.astype(np.float64) / 128.0
    if dtype == np.uint16:
        info = np.iinfo(dtype)
        half = float(info.max) / 2.0
        return (samples.astype(np.float64) - half) / half
    raise UnsupportedAudioError(f"Unsupported dtype: {dtype}.")
