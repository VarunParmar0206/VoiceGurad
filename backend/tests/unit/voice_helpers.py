"""Shared synthetic-audio fixtures for Phase 5 voice tests.

These helpers generate deterministic, hardware-independent audio so the test
suite never touches a microphone or a real recording device.
"""

from __future__ import annotations

import io

import numpy as np
import numpy.typing as npt
from scipy.io import wavfile


def sine(
    frequency: float,
    duration: float,
    sample_rate: int = 16000,
    amplitude: float = 0.5,
    phase: float = 0.0,
) -> npt.NDArray[np.float32]:
    """A pure-tone waveform at the requested rate/duration."""
    n = max(1, int(round(duration * sample_rate)))
    t = np.arange(n) / sample_rate
    return (amplitude * np.sin(2.0 * np.pi * frequency * t + phase)).astype(
        np.float32
    )


def silence(duration: float, sample_rate: int = 16000) -> npt.NDArray[np.float32]:
    """An all-zero waveform."""
    n = max(1, int(round(duration * sample_rate)))
    return np.zeros(n, dtype=np.float32)


def tone_burst(
    duration: float,
    sample_rate: int = 16000,
    amp: float = 0.5,
    voice_fraction: float = 0.6,
    freq: float = 220.0,
) -> npt.NDArray[np.float32]:
    """A tone with leading/trailing silence (speech-like energy profile)."""
    sig = np.zeros(int(round(duration * sample_rate)), dtype=np.float64)
    start = int(round(duration * sample_rate * (1.0 - voice_fraction) / 2.0))
    seg = sine(freq, duration * voice_fraction, sample_rate, amplitude=amp)
    seg = seg.astype(np.float64)
    end = min(int(sig.size), start + int(seg.size))
    sig[start:end] = seg[: end - start]
    return sig.astype(np.float32)


def noise(
    duration: float,
    sample_rate: int = 16000,
    amplitude: float = 0.3,
    seed: int = 42,
) -> npt.NDArray[np.float32]:
    """Deterministic white noise."""
    n = max(1, int(round(duration * sample_rate)))
    rng = np.random.default_rng(seed)
    return (amplitude * rng.standard_normal(n)).astype(np.float32)


def wav_bytes(samples: npt.NDArray[np.float32 | np.float64], sample_rate: int) -> bytes:
    """Encode a waveform as 16-bit PCM WAV bytes (via scipy.io.wavfile)."""
    buf = io.BytesIO()
    pcm = np.clip(np.asarray(samples) * 32767.0, -32768, 32767).astype(np.int16)
    wavfile.write(buf, sample_rate, pcm)
    return buf.getvalue()
