"""VoiceGuard V2 — Phase 5 tests: canonicalization to 16 kHz mono float32."""

from __future__ import annotations

import numpy as np

from tests.unit.voice_helpers import sine
from voiceguard.voice.canonical import canonicalize
from voiceguard.voice.io import decode_array


class TestCanonicalize:
    def test_returns_float32_mono_16k(self) -> None:
        c = canonicalize(decode_array(sine(220.0, 1.0), sample_rate=16000))
        assert c.samples.dtype == np.float32
        assert c.samples.ndim == 1
        assert c.sample_rate == 16000

    def test_mono_input_stays_mono(self) -> None:
        c = canonicalize(decode_array(sine(220.0, 1.0), sample_rate=16000))
        assert c.samples.ndim == 1

    def test_stereo_downmix_to_mono(self) -> None:
        mono = sine(220.0, 1.0)
        stereo = np.stack([mono, mono], axis=1)
        c = canonicalize(decode_array(stereo, sample_rate=16000))
        assert c.samples.ndim == 1
        assert c.num_samples == 16000

    def test_resample_44100_to_16000(self) -> None:
        src = decode_array(sine(220.0, 1.0, sample_rate=44100), sample_rate=44100)
        c = canonicalize(src)
        assert c.sample_rate == 16000
        # ~1s at 16 kHz within tolerance.
        assert abs(c.duration_seconds - 1.0) < 0.05

    def test_already_canonical_untouched_shape(self) -> None:
        x = sine(440.0, 2.0)
        c = canonicalize(decode_array(x, sample_rate=16000))
        assert c.num_samples == 32000
        assert c.sample_rate == 16000

    def test_deterministic_repeated_conversion(self) -> None:
        src = decode_array(sine(330.0, 1.5, sample_rate=44100), sample_rate=44100)
        a = canonicalize(src)
        b = canonicalize(src)
        assert np.array_equal(a.samples, b.samples)

    def test_peak_normalized_to_dbfs_within_range(self) -> None:
        c = canonicalize(decode_array(sine(220.0, 1.0, amplitude=0.3), sample_rate=16000))
        peak = float(np.max(np.abs(c.samples)))
        assert peak <= 1.0
        assert peak > 0.7  # ≈ 0.891 (-1 dBFS)

    def test_near_full_scale_input_never_exceeds_unity(self) -> None:
        # Even a near-full-scale input must never exceed the [-1, 1] contract
        # after peak normalization (clipping guard).
        x = np.full(16000, 0.99, dtype=np.float32)
        c = canonicalize(decode_array(x, sample_rate=16000))
        assert float(np.max(np.abs(c.samples))) <= 1.0

    def test_default_dtype_is_float32(self) -> None:
        x = sine(220.0, 1.0)
        c = canonicalize(decode_array(x, sample_rate=16000))
        assert c.samples.dtype == np.float32
