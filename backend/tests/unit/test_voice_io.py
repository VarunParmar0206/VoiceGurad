"""VoiceGuard V2 — Phase 5 tests: input decoding + validation."""

from __future__ import annotations

import numpy as np
import pytest

from tests.unit.voice_helpers import silence, sine, wav_bytes
from voiceguard.voice import (
    AudioDecodeError,
    AudioDurationError,
    AudioNumericError,
    AudioSilenceError,
    AudioValidationError,
    UnsupportedAudioError,
    decode_array,
    decode_bytes,
    decode_source,
    validate,
)
from voiceguard.voice.result import DecodedAudio


class TestDecodeArray:
    def test_mono_float32_passthrough(self) -> None:
        x = sine(220.0, 1.0)
        d = decode_array(x, sample_rate=16000)
        assert d.channels == 1
        assert d.sample_rate == 16000
        assert d.num_samples == 16000

    def test_stereo_passthrough_keeps_channels(self) -> None:
        mono = sine(220.0, 1.0)
        stereo = np.stack([mono, mono], axis=1)
        d = decode_array(stereo, sample_rate=16000)
        assert d.channels == 2
        assert d.samples.shape == (16000, 2)

    def test_int16_array_decodes_to_float(self) -> None:
        x = (sine(220.0, 1.0) * 32767).astype(np.int16)
        d = decode_array(x, sample_rate=16000)
        assert d.samples.dtype == np.float32
        assert float(np.max(np.abs(d.samples))) <= 1.0

    def test_empty_array_rejected(self) -> None:
        with pytest.raises((AudioValidationError, UnsupportedAudioError)):
            decode_array(np.array([], dtype=np.float32), sample_rate=16000)

    def test_0d_array_rejected(self) -> None:
        with pytest.raises(UnsupportedAudioError):
            decode_array(np.float32(0.5), sample_rate=16000)

    def test_3d_array_rejected(self) -> None:
        with pytest.raises(UnsupportedAudioError):
            decode_array(np.zeros((4, 4, 4), dtype=np.float32), sample_rate=16000)

    def test_unsupported_dtype_rejected(self) -> None:
        with pytest.raises(UnsupportedAudioError):
            decode_array(np.zeros(100, dtype=np.complex64), sample_rate=16000)

    def test_invalid_sample_rate_rejected(self) -> None:
        with pytest.raises(UnsupportedAudioError):
            decode_array(sine(220.0, 1.0), sample_rate=0)

    def test_nan_rejected(self) -> None:
        x = sine(220.0, 1.0)
        x[10] = np.nan
        with pytest.raises((AudioDecodeError, AudioNumericError)):
            decode_array(x, sample_rate=16000)

    def test_inf_rejected(self) -> None:
        x = sine(220.0, 1.0)
        x[10] = np.inf
        with pytest.raises((AudioDecodeError, AudioNumericError)):
            decode_array(x, sample_rate=16000)

    def test_oversized_duration_rejected(self) -> None:
        long = np.zeros(16000 * 60, dtype=np.float32)
        with pytest.raises(AudioDurationError):
            decode_array(long, sample_rate=16000)


class TestDecodeBytes:
    def test_valid_wav_bytes(self) -> None:
        raw = wav_bytes(sine(220.0, 1.0), 16000)
        d = decode_bytes(raw)
        assert d.sample_rate == 16000
        assert d.channels == 1
        assert d.num_samples >= 15000

    def test_corrupted_bytes_rejected(self) -> None:
        with pytest.raises(AudioDecodeError):
            decode_bytes(b"\x00\x01not-a-wav\xff\xfe")

    def test_empty_bytes_rejected(self) -> None:
        with pytest.raises(AudioDecodeError):
            decode_bytes(b"")


class TestDecodeSource:
    def test_pathlib_path(self, tmp_path) -> None:
        p = tmp_path / "tone.wav"
        p.write_bytes(wav_bytes(sine(220.0, 0.8), 16000))
        d = decode_source(p)
        assert d.sample_rate == 16000

    def test_str_path(self, tmp_path) -> None:
        p = tmp_path / "tone.wav"
        p.write_bytes(wav_bytes(sine(220.0, 0.8), 16000))
        d = decode_source(str(p))
        assert d.channels == 1

    def test_missing_file(self) -> None:
        with pytest.raises(AudioDecodeError):
            decode_source("nonexistent.wav")

    def test_invalid_source_type(self) -> None:
        with pytest.raises(UnsupportedAudioError):
            decode_source(12345)

    def test_numpy_array_source(self) -> None:
        d = decode_source(sine(220.0, 1.0))
        assert isinstance(d, DecodedAudio)


class TestValidate:
    def test_valid_passes(self) -> None:
        d = decode_array(sine(220.0, 1.0), sample_rate=16000)
        assert validate(d) is d

    def test_out_of_range_amplitude_rejected(self) -> None:
        x = np.array([0.0, 1.2, -1.5, 0.3], dtype=np.float32)
        d = decode_array(x, sample_rate=16000)
        with pytest.raises(AudioNumericError):
            validate(d)

    def test_nan_rejected(self) -> None:
        x = sine(220.0, 1.0)
        x[5] = np.nan
        with pytest.raises((AudioDecodeError, AudioNumericError)):
            d = decode_array(x, sample_rate=16000)
            validate(d)

    def test_too_short_rejected(self) -> None:
        d = decode_array(sine(220.0, 0.3), sample_rate=16000)
        with pytest.raises(AudioValidationError):
            validate(d)

    def test_silence_rejected(self) -> None:
        d = decode_array(silence(1.0), sample_rate=16000)
        with pytest.raises(AudioSilenceError):
            validate(d)

    def test_too_long_rejected(self) -> None:
        with pytest.raises(AudioDurationError):
            d = decode_array(np.zeros(16000 * 60, dtype=np.float32), sample_rate=16000)
            validate(d)
