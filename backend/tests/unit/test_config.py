"""Tests for voiceguard.config — backend Settings."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError


def _make_env() -> dict[str, str]:
    """Return a minimal valid environment for Settings."""
    return {
        "VG_DATABASE_URL": "postgresql+asyncpg://user:pass@localhost:5432/testdb",
        "VG_JWT_SECRET_KEY": "a-very-long-secret-key-for-jwt-signing-operations",
        "VG_ENCRYPTION_KEY": "gAAAAABh0Z0TestKeyThatIsLongEnoughForFernetValidation!",
    }


class TestSettingsDefaults:
    """Verify that all default values are sane."""

    @patch.dict(os.environ, _make_env(), clear=False)
    def test_defaults_load(self) -> None:
        from voiceguard.config import Settings

        s = Settings()
        assert s.APP_NAME == "VoiceGuard"
        assert s.APP_VERSION == "2.0.0"
        assert s.AUDIO_SAMPLE_RATE == 16000
        assert s.N_MFCC == 40
        assert s.N_MELS == 80
        assert s.HOP_LENGTH == 160
        assert s.N_FFT == 512
        assert s.EMBEDDING_DIM == 256
        assert s.VERIFICATION_THRESHOLD == 0.82
        assert s.GMM_N_COMPONENTS == 8
        assert s.DEFAULT_BALANCE == 10000.00
        assert s.SESSION_TIMEOUT_SECONDS == 900
        assert s.MAX_FAILED_ATTEMPTS == 5
        assert s.CHALLENGE_EXPIRY_SECONDS == 30
        assert s.ANTI_SPOOF_THRESHOLD == 0.70


class TestSettingsRequired:
    """Verify that required fields raise errors when missing."""

    def test_missing_database_url(self) -> None:
        from voiceguard.config import Settings

        env = _make_env()
        del env["VG_DATABASE_URL"]
        with patch.dict(os.environ, env, clear=False), pytest.raises(ValidationError):
            Settings()

    def test_missing_jwt_secret(self) -> None:
        from voiceguard.config import Settings

        env = _make_env()
        del env["VG_JWT_SECRET_KEY"]
        with patch.dict(os.environ, env, clear=False), pytest.raises(ValidationError):
            Settings()

    def test_missing_encryption_key(self) -> None:
        from voiceguard.config import Settings

        env = _make_env()
        del env["VG_ENCRYPTION_KEY"]
        with patch.dict(os.environ, env, clear=False), pytest.raises(ValidationError):
            Settings()


class TestSettingsValidation:
    """Verify field-level validators."""

    @patch.dict(os.environ, _make_env(), clear=False)
    def test_encryption_key_too_short(self) -> None:
        from voiceguard.config import Settings

        env = _make_env()
        env["VG_ENCRYPTION_KEY"] = "short"
        with patch.dict(os.environ, env, clear=False), pytest.raises(ValidationError):
            Settings()

    @patch.dict(os.environ, _make_env(), clear=False)
    def test_env_prefix_override(self) -> None:
        from voiceguard.config import Settings

        env = _make_env()
        env["VG_PORT"] = "9000"
        with patch.dict(os.environ, env, clear=False):
            s = Settings()
            assert s.PORT == 9000
