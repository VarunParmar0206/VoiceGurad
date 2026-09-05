"""VoiceGuard V2 — TOTP backup / secondary authentication.

Design
******
- Uses ``pyotp`` (RFC 6238) for TOTP secret generation, provisioning-URI
  building, and code verification.
- The base32 TOTP secret is **sensitive** (it grants account access).  It is
  encrypted at rest with AES-256-GCM before being written to the database and
  decrypted only in memory during verification.
- The TOTP encryption key is **domain-separated** from the biometric key via
  HKDF-SHA256 over the configured ``VG_ENCRYPTION_KEY`` with an application-
  specific ``info`` label.  Per the security notes in ``security/crypto.py``,
  a single key must not be reused across purposes; this derivation gives the
  TOTP column its own key while reusing the same secret-management pipeline.

Clock skew
**********
- TOTP verification accepts the code from the current 30-second window plus
  a configurable number of neighbouring windows (``TOTP_WINDOW_STEPS``),
  defaulting to ±1 step to tolerate mild clock drift (Architecture §15 risk
  mitigation).

Security notes
**************
- TOTP secrets are never returned to the client after provisioning (the
  provisioning URI / QR secret is only issued once at setup time).
- Code strings are validated as 6-digit numeric before comparison.
- No custom cryptography is implemented — only ``pyotp`` and
  ``cryptography.hazmat`` (AES-GCM + HKDF) primitives.
"""

from __future__ import annotations

import base64
import binascii
import os

import pyotp
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from voiceguard.config import settings

# HKDF ``info`` label -- binds this key to the TOTP purpose.
_TOTP_KEY_INFO = b"voiceguard-totp-encryption-v1"

# AES-256-GCM parameters (mirrors the biometric encryptor).
_KEY_LENGTH = 32
_NONCE_LENGTH = 12


class TOTPError(Exception):
    """Base exception for TOTP operations."""


class _TOTPSecretEncryptor:
    """AES-256-GCM encryptor for TOTP secrets using a domain-separated key."""

    def __init__(self, base_key_material: str) -> None:
        raw = base64.urlsafe_b64decode(base_key_material)
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=_KEY_LENGTH,
            salt=None,
            info=_TOTP_KEY_INFO,
        )
        self._key = hkdf.derive(raw)
        self._aesgcm = AESGCM(self._key)

    def encrypt(self, secret: str) -> bytes:
        nonce = os.urandom(_NONCE_LENGTH)
        ct_with_tag = self._aesgcm.encrypt(nonce, secret.encode("utf-8"), None)
        return nonce + ct_with_tag

    def decrypt(self, data: bytes) -> str:
        if len(data) <= _NONCE_LENGTH:
            raise TOTPError("Encrypted TOTP secret is malformed.")
        nonce = data[: _NONCE_LENGTH]
        ct_with_tag = data[_NONCE_LENGTH:]
        plaintext = self._aesgcm.decrypt(nonce, ct_with_tag, None)
        return plaintext.decode("utf-8")


_encryptor: _TOTPSecretEncryptor | None = None


def _get_encryptor() -> _TOTPSecretEncryptor:
    global _encryptor
    if _encryptor is None:
        _encryptor = _TOTPSecretEncryptor(settings.ENCRYPTION_KEY)
    return _encryptor


def generate_totp_secret() -> str:
    """Generate a new base32 TOTP secret (RFC 6238).

    ``TOTP_SECRET_BYTES`` is interpreted as the desired key length in
    *bytes* (default 20 bytes == 160 bits of entropy, the RFC 4226/6238
    recommended strength).  The base32 character length is derived so the
    resulting secret carries that many bits (20 bytes -> 32 base32 chars).
    """
    bits = settings.TOTP_SECRET_BYTES * 8
    chars = (bits + 4) // 5  # 5 bits per base32 character
    return pyotp.random_base32(length=chars)


def encrypt_totp_secret(secret: str) -> bytes:
    """Encrypt a base32 TOTP secret for at-rest storage.

    Returns ``nonce ‖ tag ‖ ciphertext``.
    """
    return _get_encryptor().encrypt(secret)


def decrypt_totp_secret(encrypted: bytes) -> str:
    """Decrypt an at-rest TOTP secret back to its base32 form.

    Raises:
        TOTPError: If the ciphertext is malformed or has been tampered with.
    """
    return _get_encryptor().decrypt(encrypted)


def provisioning_uri(secret: str, account_name: str) -> str:
    """Build otpauth:// provisioning URI for authenticator app enrollment.

    ``account_name`` is typically ``username@issuer``.
    """
    return pyotp.totp.TOTP(secret).provisioning_uri(
        name=account_name, issuer_name=settings.TOTP_ISSUER
    )


def verify_totp(secret: str, code: str) -> bool:
    """Verify a TOTP code against *secret* within the allowed window.

    Args:
        secret: The base32 TOTP secret.
        code: The 6-digit code supplied by the user.

    Returns:
        ``True`` if the code is valid within the configured ±window.
    """
    if not _validate_code(code):
        return False
    try:
        totp = pyotp.TOTP(secret)
        return totp.verify(code, valid_window=settings.TOTP_WINDOW_STEPS)
    except (ValueError, binascii.Error, TypeError):
        # A malformed/illegal base32 secret can never yield a matching code.
        return False


def _validate_code(code: str) -> bool:
    """Return ``True`` if *code* looks like a valid numeric TOTP code."""
    if not isinstance(code, str):
        return False
    if not code.isdigit():
        return False
    return len(code) == settings.TOTP_CODE_DIGITS
