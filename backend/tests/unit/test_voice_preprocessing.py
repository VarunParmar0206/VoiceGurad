"""VoiceGuard V2 — Phase 5 tests: preprocessing (DC, VAD, trimming, framing)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.unit.voice_helpers import silence, sine, tone_burst
from voiceguard.voice.canonical import canonicalize
from voiceguard.voice.errors import AudioSilenceError
from voiceguard.voice.io import decode_array
from voiceguard.voice.preprocessing import preprocess


def _canonical(x: np.ndarray) -> object:
    return canonicalize(decode_array(x, sample_rate=16000))


class TestPreprocessing:
    def test_removes_dc_offset(self) -> None:
        # Add a DC bias to a normal sine wave and verify it is removed.
        x = sine(220.0, 1.0).astype(np.float64) + 0.3
        pp = preprocess(_canonical(x.astype(np.float32)))
        assert abs(float(np.mean(pp.samples))) < 1e-3

    def test_no_dc_in_clean_signal(self) -> None:
        pp = preprocess(_canonical(sine(220.0, 1.0)))
        assert abs(float(np.mean(pp.samples))) < 1e-3

    def test_silence_rejected(self) -> None:
        with pytest.raises((AudioSilenceError, Exception)):
            preprocess(_canonical(silence(1.0)))

    def test_constant_signal_rejected_after_dc_removal(self) -> None:
        x = np.full(16000, 0.5, dtype=np.float32)
        with pytest.raises(AudioSilenceError):
            preprocess(_canonical(x))

    def test_framing_produces_expected_count(self) -> None:
        pp = preprocess(_canonical(sine(220.0, 1.0)))
        assert pp.frames.ndim == 2
        assert pp.frames.shape[1] == 512  # N_FFT
        assert pp.frames.shape[0] == pp.num_frames
        assert pp.num_frames > 0

    def test_deterministic_repeated_preprocessing(self) -> None:
        src = _canonical(tone_burst(2.0, voice_fraction=0.5))
        a = preprocess(src)
        b = preprocess(src)
        assert np.array_equal(a.frames, b.frames)
        assert np.array_equal(a.samples, b.samples)

    def test_tone_burst_is_trimmed(self) -> None:
        burst = tone_burst(3.0, voice_fraction=0.7)
        pp = preprocess(_canonical(burst))
        # Trimmed output must be shorter than the original 3-second waveform.
        assert pp.samples.size < 3.0 * 16000

    def test_voice_fraction_reported(self) -> None:
        burst = tone_burst(2.0, voice_fraction=0.6)
        pp = preprocess(_canonical(burst))
        assert 0.0 < pp.vad.voice_fraction <= 1.0

    def test_output_is_float32(self) -> None:
        pp = preprocess(_canonical(sine(220.0, 1.0)))
        assert pp.frames.dtype == np.float32
