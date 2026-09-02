"""VoiceGuard V2 — AES-256-GCM authenticated encryption for biometric columns.

Design
******
- ``BiometricEncryptor`` wraps a single symmetric key.
- The key is derived from ``VG_ENCRYPTION_KEY`` (a base64 Fernet key) by
  taking the raw 32 bytes of the Fernet key material.
- Each ``encrypt()`` call generates a fresh 12-byte nonce.  The ciphertext
  is stored as ``nonce ‖ tag ‖ ciphertext`` (all raw bytes in a single
  ``BYTEA`` column).
- ``decrypt()`` verifies the 16-byte GCM tag before returning plaintext.
- The nonce is **not** secret — it is prepended to the ciphertext so that
  decryption is stateless.

What is encrypted
*****************
- ``voice_templates.template_data``  — cancelable biometric embedding
- ``voice_models.model_data``        — encrypted GMM / model parameters

What remains plaintext (non-sensitive metadata)
************************************************
- ``user_id``, ``model_version``, ``enrollment_samples``, ``quality_scores``
- ``salt`` (random per-user, used for key derivation of cancelable transform)
- ``created_at``, ``is_active``

Security notes
**************
- AES-256-GCM provides both confidentiality **and** integrity (authenticated
  encryption).  Tampering with ciphertext causes decryption to fail with
  ``InvalidTag``.
- The same key MUST NOT be used for different purposes (e.g., do not reuse
  it for JWT signing or session encryption).
- Key rotation is planned but not yet implemented — the current design
  stores a single key.  A future version should prepend a key-identifier
  byte to allow multiple keys to coexist.
"""

from __future__ import annotations

import base64
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class BiometricEncryptor:
    """AES-256-GCM encryptor/decryptor for biometric template columns.

    Parameters
    ----------
    key_material : str
        A base64-encoded Fernet key (44 characters, ``gAAAAA…``).
        The first 32 raw bytes after base64 decoding are used as the
        AES-256 key.  This reuses the same secret that the legacy
        ``SecureDatabase`` used for Fernet, ensuring backward-compatible
        key management.
    """

    _KEY_LENGTH = 32  # AES-256
    _NONCE_LENGTH = 12  # Recommended nonce size for GCM
    _TAG_LENGTH = 16  # GCM authentication tag

    def __init__(self, key_material: str) -> None:
        raw = base64.urlsafe_b64decode(key_material)
        if len(raw) < self._KEY_LENGTH:
            raise ValueError(
                f"Decoded key material is {len(raw)} bytes; "
                f"need at least {self._KEY_LENGTH} for AES-256."
            )
        # Take exactly 32 bytes — Fernet keys are 32 bytes of signing key
        # followed by a 1-byte version flag, but we only need the 32-byte
        # signing portion.
        self._key = raw[: self._KEY_LENGTH]
        self._aesgcm = AESGCM(self._key)

    def encrypt(self, plaintext: bytes) -> bytes:
        """Encrypt *plaintext* and return ``nonce ‖ tag ‖ ciphertext``.

        A fresh nonce is generated for every call.
        """
        nonce = os.urandom(self._NONCE_LENGTH)
        # AESGCM.encrypt returns ciphertext with the tag appended.
        ct_with_tag = self._aesgcm.encrypt(nonce, plaintext, None)
        return nonce + ct_with_tag

    def decrypt(self, data: bytes) -> bytes:
        """Decrypt ``nonce ‖ tag ‖ ciphertext`` and return plaintext.

        Raises ``cryptography.exceptions.InvalidTag`` if the data has been
        tampered with or the wrong key is used.
        """
        nonce = data[: self._NONCE_LENGTH]
        ct_with_tag = data[self._NONCE_LENGTH :]
        return self._aesgcm.decrypt(nonce, ct_with_tag, None)
