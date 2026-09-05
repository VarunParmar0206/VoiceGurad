"""VoiceGuard V2 — Security primitives.

Submodules
**********
- ``password``: Argon2id password hashing/verification.
- ``tokens``: JWT access tokens + opaque, hashed refresh tokens.
- ``totp``: TOTP backup/secondary authentication.
- ``crypto``: AES-256-GCM encryption for biometric columns.
- ``rate_limit``: Redis-backed rate limiting.
"""

from __future__ import annotations

from voiceguard.security.crypto import BiometricEncryptor
from voiceguard.security.password import (
    hash_password,
    password_needs_rehash,
    verify_password,
)
from voiceguard.security.tokens import (
    ACCESS_TOKEN_TYPE,
    REFRESH_TOKEN_TYPE,
    InvalidTokenError,
    TokenError,
    create_access_token,
    decode_access_token,
    generate_login_token,
    generate_refresh_token,
    hash_login_token,
    hash_refresh_token,
)
from voiceguard.security.totp import (
    decrypt_totp_secret,
    encrypt_totp_secret,
    generate_totp_secret,
    provisioning_uri,
    verify_totp,
)

__all__ = [
    "BiometricEncryptor",
    "hash_password",
    "password_needs_rehash",
    "verify_password",
    "decrypt_totp_secret",
    "encrypt_totp_secret",
    "generate_totp_secret",
    "provisioning_uri",
    "verify_totp",
    "ACCESS_TOKEN_TYPE",
    "REFRESH_TOKEN_TYPE",
    "InvalidTokenError",
    "TokenError",
    "create_access_token",
    "decode_access_token",
    "generate_refresh_token",
    "hash_refresh_token",
    "generate_login_token",
    "hash_login_token",
]
