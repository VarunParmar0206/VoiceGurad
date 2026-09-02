"""Tests for shared.voiceguard.constants."""

from __future__ import annotations

import voiceguard_common.constants as c


def test_enrollment_phrases_count() -> None:
    assert len(c.ENROLLMENT_PHRASES) == 5


def test_enrollment_phrases_are_strings() -> None:
    for phrase in c.ENROLLMENT_PHRASES:
        assert isinstance(phrase, str)
        assert len(phrase) > 0


def test_challenge_phrases_non_empty() -> None:
    assert len(c.CHALLENGE_PHRASES) >= 10


def test_audio_defaults() -> None:
    assert c.DEFAULT_SAMPLE_RATE == 16000
    assert c.DEFAULT_CHANNELS == 1


def test_feature_defaults() -> None:
    assert c.DEFAULT_N_MFCC == 40
    assert c.DEFAULT_N_MELS == 80
    assert c.DEFAULT_HOP_LENGTH == 160
    assert c.DEFAULT_N_FFT == 512
    assert c.DEFAULT_F_MIN == 50
    assert c.DEFAULT_F_MAX == 8000


def test_currency() -> None:
    assert c.CURRENCY_SYMBOL == "₹"
    assert c.CURRENCY_CODE == "INR"
