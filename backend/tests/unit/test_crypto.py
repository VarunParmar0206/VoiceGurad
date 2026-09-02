"""Tests for voiceguard.security.crypto.BiometricEncryptor."""

from __future__ import annotations

import base64
import os

import pytest
from cryptography.exceptions import InvalidTag

from voiceguard.security.crypto import BiometricEncryptor

# A valid base64 Fernet key (44 chars decode to 32 bytes signing key).
# This is a test secret only — never use in production.
FERNET_KEY = base64.urlsafe_b64encode(os.urandom(32)).decode()


def _make_encryptor(key: str = FERNET_KEY) -> BiometricEncryptor:
    return BiometricEncryptor(key)


class TestEncryption:
    def test_round_trip(self) -> None:
        enc = _make_encryptor()
        plaintext = b"sensitive-embedding-data"
        ciphertext = enc.encrypt(plaintext)
        assert enc.decrypt(ciphertext) == plaintext

    def test_ciphertext_differs_from_plaintext(self) -> None:
        enc = _make_encryptor()
        ciphertext = enc.encrypt(b"payload")
        assert ciphertext != b"payload"

    def test_nonce_is_random_per_call(self) -> None:
        """Two encryptions of the same plaintext must produce different
        ciphertext (random nonce), preventing replay of identical bytes."""
        enc = _make_encryptor()
        c1 = enc.encrypt(b"same")
        c2 = enc.encrypt(b"same")
        assert c1 != c2

    def test_ciphertext_contains_nonce_prefix(self) -> None:
        enc = _make_encryptor()
        c = enc.encrypt(b"x")
        # The first 12 bytes are the nonce.
        assert len(c) >= 12

    def test_empty_plaintext(self) -> None:
        enc = _make_encryptor()
        c = enc.encrypt(b"")
        assert enc.decrypt(c) == b""


class TestTamperDetection:
    def test_flipped_byte_raises_invalid_tag(self) -> None:
        """A single flipped byte in the ciphertext must cause decryption
        to fail — verifying GCM provides integrity, not just secrecy."""
        enc = _make_encryptor()
        c = bytearray(enc.encrypt(b"authenticated-data"))
        # Flip a bit in the body (past the nonce).
        c[-1] ^= 0x01
        with pytest.raises(InvalidTag):
            enc.decrypt(bytes(c))

    def test_truncated_data_raises(self) -> None:
        from cryptography.exceptions import InvalidTag

        enc = _make_encryptor()
        c = enc.encrypt(b"some-data")
        with pytest.raises((InvalidTag, ValueError)):
            enc.decrypt(c[:5])


class TestKeyHandling:
    def test_wrong_key_raises_invalid_tag(self) -> None:
        enc1 = _make_encryptor(FERNET_KEY)
        other_key = base64.urlsafe_b64encode(os.urandom(32)).decode()
        enc2 = _make_encryptor(other_key)
        c = enc1.encrypt(b"secret")
        with pytest.raises(InvalidTag):
            enc2.decrypt(c)

    def test_short_key_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            BiometricEncryptor("short-key")

    def test_key_material_too_short_after_decode(self) -> None:
        # base64 of a short raw string.
        with pytest.raises(ValueError):
            BiometricEncryptor(base64.urlsafe_b64encode(b"tiny").decode())
