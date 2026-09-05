"""VoiceGuard V2 — Phase 5 tests: mel + statistical feature extraction."""

from __future__ import annotations

import numpy as np
import pytest

from tests.unit.voice_helpers import sine
from voiceguard.voice.canonical import canonicalize
from voiceguard.voice.errors import AudioSilenceError
from voiceguard.voice.features import extract_features, extract_mel, extract_statistical
from voiceguard.voice.io import decode_array
from voiceguard.voice.preprocessing import preprocess

EXPECTED_STATISTICAL_DIM = 259  # 240 MFCC + 5 spectral + 2 ZCR + 4 pitch + 2 RMS + 6 formants


class TestExtractMel:
    def _mel(self, x: np.ndarray):
        c = canonicalize(decode_array(x, sample_rate=16000))
        return extract_mel(preprocess(c))

    def test_shape_80_rows_variable_time(self) -> None:
        m1 = self._mel(sine(220.0, 1.0))
        assert m1.log_mel.shape[0] == 80
        assert m1.log_mel.shape[1] > 0

    def test_short_and_long_inputs_time_axis_scales(self) -> None:
        short = self._mel(sine(220.0, 1.0))
        long = self._mel(sine(220.0, 2.5))
        assert long.time_frames > short.time_frames

    def test_dtype_is_float32(self) -> None:
        m = self._mel(sine(220.0, 1.0))
        assert m.log_mel.dtype == np.float32

    def test_finite_output(self) -> None:
        m = self._mel(sine(220.0, 1.0))
        assert np.isfinite(m.log_mel).all()

    def test_deterministic_repeated(self) -> None:
        a = self._mel(sine(220.0, 1.5))
        b = self._mel(sine(220.0, 1.5))
        assert np.array_equal(a.log_mel, b.log_mel)

    def test_metadata(self) -> None:
        m = self._mel(sine(220.0, 1.0))
        assert m.n_mels == 80
        assert m.hop_length == 160
        assert m.n_fft == 512

    def test_very_short_signal_produces_finite_single_frame(self) -> None:
        # Upstream validation rejects < 0.5 s; at the feature layer a short
        # but non-empty signal still yields a finite, well-shaped spectrogram.
        x = sine(220.0, 0.03)
        m = self._mel(x)
        assert m.log_mel.shape[0] == 80
        assert m.log_mel.shape[1] >= 1
        assert np.isfinite(m.log_mel).all()


class TestExtractStatistical:
    def _stats(self, x: np.ndarray):
        c = canonicalize(decode_array(x, sample_rate=16000))
        return extract_statistical(preprocess(c))

    def test_expected_dimension(self) -> None:
        s = self._stats(sine(220.0, 1.0))
        assert s.dimension == EXPECTED_STATISTICAL_DIM

    def test_names_match_dimension(self) -> None:
        s = self._stats(sine(220.0, 1.0))
        assert len(s.names) == s.dimension

    def test_finite_values(self) -> None:
        s = self._stats(sine(220.0, 1.0))
        assert np.isfinite(s.values).all()

    def test_deterministic_repeated(self) -> None:
        a = self._stats(sine(220.0, 1.5))
        b = self._stats(sine(220.0, 1.5))
        assert np.array_equal(a.values, b.values)

    def test_expected_names_present(self) -> None:
        s = self._stats(sine(220.0, 1.0))
        joined = " ".join(s.names)
        for token in (
            "mfcc_mean_0",
            "delta_std_10",
            "spectral_centroid_mean",
            "zcr_mean",
            "pitch_mean",
            "rms_mean",
            "f1_mean",
            "f3_std",
        ):
            assert token in joined


class TestExtractFeaturesTogether:
    def test_returns_both_paths(self) -> None:
        c = canonicalize(decode_array(sine(220.0, 1.0), sample_rate=16000))
        mel, stats = extract_features(c)
        assert mel.log_mel.shape == (80, mel.time_frames)
        assert stats.dimension == EXPECTED_STATISTICAL_DIM


class TestPathological:
    def _stats(self, x: np.ndarray):
        c = canonicalize(decode_array(x, sample_rate=16000))
        return extract_statistical(preprocess(c))

    def test_zero_amplitude_rejected(self) -> None:
        x = sine(220.0, 1.0, amplitude=0.0)
        with pytest.raises(AudioSilenceError):
            canonicalize(decode_array(x, sample_rate=16000))

    def test_extremely_quiet_is_finite(self) -> None:
        x = sine(220.0, 2.0, amplitude=1e-5)
        assert np.isfinite(self._stats(x).values).all()

    def test_clipping_amplitude_is_finite(self) -> None:
        x = np.clip(sine(220.0, 1.0, amplitude=0.9), -1.0, 1.0)
        assert np.isfinite(self._stats(x).values).all()

    def test_repeated_identical_samples_handled(self) -> None:
        # A 200 Hz square wave (hard-clipped, extreme) must yield finite,
        # deterministic statistical features.
        t = np.arange(16000) / 16000
        x = (np.sin(2 * np.pi * 200 * t) > 0).astype(np.float32)
        a = self._stats(x)
        b = self._stats(x)
        assert np.isfinite(a.values).all()
        assert np.array_equal(a.values, b.values)
