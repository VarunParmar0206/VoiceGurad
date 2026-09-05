"""VoiceGuard V2 — Phase 5 tests: end-to-end pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from tests.unit.voice_helpers import noise, sine, tone_burst, wav_bytes
from voiceguard.voice import (
    AudioDecodeError,
    AudioDurationError,
    AudioNumericError,
    AudioValidationError,
    FeatureResult,
    UnsupportedAudioError,
    process,
)


class TestProcessPipeline:
    def test_valid_synthetic_audio_produces_feature_result(self) -> None:
        r = process(sine(220.0, 1.0))
        assert isinstance(r, FeatureResult)
        assert r.sample_rate == 16000
        assert r.num_samples > 0
        assert r.duration_seconds > 0.5
        assert r.mel.log_mel.shape[0] == 80
        assert r.statistical.dimension > 0
        assert np.isfinite(r.mel.log_mel).all()
        assert np.isfinite(r.statistical.values).all()

    def test_stereo_input_downmixed(self) -> None:
        mono = sine(220.0, 1.0)
        stereo = np.stack([mono, mono], axis=1)
        r = process(stereo)
        assert r.sample_rate == 16000
        assert r.mel.log_mel.shape[0] == 80

    def test_44k1_input_resampled(self) -> None:
        r = process(sine(220.0, 1.0, sample_rate=44100), sample_rate=44100)
        assert r.sample_rate == 16000
        assert abs(r.duration_seconds - 1.0) < 0.05

    def test_wav_bytes_input(self) -> None:
        raw = wav_bytes(sine(220.0, 1.0), 16000)
        r = process(raw)
        assert r.duration_seconds > 0.5

    def test_noise_input_is_finite(self) -> None:
        r = process(noise(1.0, amplitude=0.2, seed=0))
        assert np.isfinite(r.mel.log_mel).all()
        assert np.isfinite(r.statistical.values).all()

    def test_tone_burst_trimmed_quality_report(self) -> None:
        burst = tone_burst(3.0, voice_fraction=0.7)
        r = process(burst)
        assert isinstance(r.quality.trimmed, bool)
        assert r.quality.canonical_duration_seconds <= 3.0

    def test_repeated_identical_input_equivalent_output(self) -> None:
        a = process(sine(220.0, 1.5))
        b = process(sine(220.0, 1.5))
        assert np.array_equal(a.mel.log_mel, b.mel.log_mel)
        assert np.array_equal(a.statistical.values, b.statistical.values)

    def test_empty_array_fails_cleanly(self) -> None:
        with pytest.raises((AudioValidationError, UnsupportedAudioError)):
            process(np.array([], dtype=np.float32))

    def test_nan_fails_cleanly(self) -> None:
        x = sine(220.0, 1.0)
        x[5] = np.nan
        with pytest.raises((AudioDecodeError, AudioNumericError)):
            process(x)

    def test_oversized_duration_fails_cleanly(self) -> None:
        long = np.zeros(16000 * 60, dtype=np.float32)
        long[1000:4000] = 0.5
        with pytest.raises(AudioDurationError):
            process(long)

    def test_no_raw_waveform_in_result(self) -> None:
        r = process(sine(220.0, 1.0))
        fields = set(r.__dict__.keys())
        # FeatureResult must not carry a raw waveform array.
        assert "waveform" not in fields
        assert "samples" not in fields
        assert "raw" not in fields
