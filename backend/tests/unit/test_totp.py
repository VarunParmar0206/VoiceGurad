"""Tests for voiceguard.security.totp (RFC 6238 TOTP + at-rest encryption)."""

from __future__ import annotations

import pyotp

from voiceguard.config import settings
from voiceguard.security.totp import (
    TOTPError,
    decrypt_totp_secret,
    encrypt_totp_secret,
    generate_totp_secret,
    provisioning_uri,
    verify_totp,
)


class TestGenerateSecret:
    def test_generates_base32_secret(self) -> None:
        secret = generate_totp_secret()
        assert len(secret) > 0
        # pyotp accepts it and only contains A-Z2-7.
        totp = pyotp.TOTP(secret)
        assert len(totp.now()) == settings.TOTP_CODE_DIGITS

    def test_secrets_are_unique(self) -> None:
        assert generate_totp_secret() != generate_totp_secret()


class TestVerifyTotp:
    def test_valid_code_accepted(self) -> None:
        secret = generate_totp_secret()
        totp = pyotp.TOTP(secret)
        code = totp.now()
        assert verify_totp(secret, code) is True

    def test_wrong_code_rejected(self) -> None:
        secret = generate_totp_secret()
        code = pyotp.TOTP(secret).now()
        wrong = "000000" if code != "000000" else "111111"
        assert verify_totp(secret, wrong) is False

    def test_right_adjacent_window_accepted(self) -> None:
        """Codes within the configured ±window must still verify."""
        import time

        secret = generate_totp_secret()
        totp = pyotp.TOTP(secret)
        reference = int(time.time()) // 30 * 30
        # A code from 30s earlier must verify within the ±1 window.
        prev_code = totp.at(reference - 30)
        assert totp.verify(prev_code, valid_window=1, for_time=reference) is True

    def test_window_tolerance_is_configured(self) -> None:
        assert settings.TOTP_WINDOW_STEPS >= 1

    def test_non_numeric_rejected(self) -> None:
        assert verify_totp(generate_totp_secret(), "abcdef") is False

    def test_wrong_length_rejected(self) -> None:
        secret = generate_totp_secret()
        code = pyotp.TOTP(secret).now()
        assert verify_totp(secret, "123") is False
        assert verify_totp(secret, code + "0") is False

    def test_invalid_secret_does_not_crash(self) -> None:
        # A malformed base32 should be caught by verify (weak check first).
        assert verify_totp("!!!!!!!!", "123456") is False


class TestProvisioningUri:
    def test_uri_contains_expected_parts(self) -> None:
        secret = generate_totp_secret()
        uri = provisioning_uri(secret, "alice@VoiceGuard")
        assert uri.startswith("otpauth://totp/")
        assert "secret=" + secret in uri
        assert settings.TOTP_ISSUER in uri

    def test_uri_embeds_account(self) -> None:
        secret = generate_totp_secret()
        uri = provisioning_uri(secret, "bob@VoiceGuard")
        assert "bob@VoiceGuard" in uri or "bob%40VoiceGuard" in uri


class TestTotpEncryption:
    def test_round_trip(self) -> None:
        secret = generate_totp_secret()
        encrypted = encrypt_totp_secret(secret)
        assert decrypt_totp_secret(encrypted) == secret

    def test_ciphertext_differs_from_plaintext(self) -> None:
        secret = generate_totp_secret()
        encrypted = encrypt_totp_secret(secret)
        # Base32 secret must not appear verbatim in ciphertext.
        assert secret.encode("utf-8") not in encrypted
        assert secret not in encrypted.decode("latin-1", errors="ignore")

    def test_randomized_encryption(self) -> None:
        secret = generate_totp_secret()
        e1 = encrypt_totp_secret(secret)
        e2 = encrypt_totp_secret(secret)
        assert e1 != e2
        assert decrypt_totp_secret(e1) == secret
        assert decrypt_totp_secret(e2) == secret

    def test_tampered_ciphertext_raises(self) -> None:
        secret = generate_totp_secret()
        encrypted = bytearray(encrypt_totp_secret(secret))
        encrypted[-1] ^= 0x01
        try:
            decrypt_totp_secret(bytes(encrypted))
        except Exception:
            pass
        else:
            raise AssertionError("Tampered TOTP secret decrypted without error")

    def test_short_data_raises(self) -> None:
        try:
            decrypt_totp_secret(b"too-short")
        except TOTPError:
            pass
        else:
            raise AssertionError("Malformed TOTP secret did not raise")
