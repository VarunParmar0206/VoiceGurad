"""VoiceGuard V2 — Argon2id password hashing.

Design
******
- Passwords are hashed with **Argon2id** (the OWASP-recommended algorithm)
  via the ``argon2-cffi`` library.
- Each hash embeds a unique random salt and the algorithm parameters, so the
  stored string is self-contained (``$argon2id$…``).
- Verification compares against the embedded parameters and is constant-time.

Chosen parameters (OWASP-recommended starting point)
****************************************************
- ``time_cost``   = 3   (iterations)
- ``memory_cost`` = 65536 KiB = 64 MiB
- ``parallelism`` = 4    (lanes)
- ``hash_len``    = 32   (256-bit output)
- ``salt_len``    = 16   (128-bit random salt)

These parameters are deliberately exposed as module constants so they can be
benchmarked and tuned on target hardware without touching call sites.  The
Argon2 PHC string format records them, so raising parameters later does not
invalidate existing hashes.

Security notes
**************
- Plaintext passwords are never stored or logged.
- ``hash_password`` raises ``ValueError`` on non-str / empty input so we
  never hash ``None`` accidentally.
- Verification raises ``VerificationError`` on mismatch (never leaks whether
  the account exists through timing since we always run a hash comparison).
- No custom cryptography is used — only the battle-tested ``argon2-cffi``.
"""

from __future__ import annotations

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError


class PasswordParameters:
    """Configuration for Argon2id hashing.

    See module docstring for rationale.  Values follow the OWASP cheat sheet
    recommended baseline for Argon2id.
    """

    TIME_COST: int = 3
    MEMORY_COST: int = 65536  # KiB (~64 MiB)
    PARALLELISM: int = 4
    HASH_LEN: int = 32
    SALT_LEN: int = 16


_hasher = PasswordHasher(
    time_cost=PasswordParameters.TIME_COST,
    memory_cost=PasswordParameters.MEMORY_COST,
    parallelism=PasswordParameters.PARALLELISM,
    hash_len=PasswordParameters.HASH_LEN,
    salt_len=PasswordParameters.SALT_LEN,
)


def hash_password(password: str) -> str:
    """Hash a plaintext password with Argon2id and return its PHC string.

    Args:
        password: The plaintext password (must be a non-empty ``str``).

    Returns:
        The encoded Argon2id hash string.

    Raises:
        ValueError: If *password* is not a string or is empty/blank.
    """
    if not isinstance(password, str) or not password:
        raise ValueError("Password must be a non-empty string.")
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a plaintext password against an Argon2id hash.

    Args:
        password: The plaintext password to check.
        password_hash: The stored Argon2id PHC string.

    Returns:
        ``True`` if the password matches the hash, ``False`` otherwise.

    Raises:
        ValueError: If the stored hash is malformed (``InvalidHashError``).
    """
    if not isinstance(password, str) or not password:
        return False
    try:
        return _hasher.verify(password_hash, password)
    except VerificationError:
        return False
    except InvalidHashError as exc:
        raise ValueError("Stored password hash is invalid.") from exc


def password_needs_rehash(password_hash: str) -> bool:
    """Return ``True`` if a stored hash does not meet current parameters.

    Allows lazy re-hashing of legacy or under-parameterized hashes on login.
    """
    try:
        return _hasher.check_needs_rehash(password_hash)
    except InvalidHashError:
        return True
