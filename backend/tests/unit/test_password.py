"""Tests for voiceguard.security.password (Argon2id hashing)."""

from __future__ import annotations

import pytest

from voiceguard.security.password import (
    PasswordParameters,
    hash_password,
    password_needs_rehash,
    verify_password,
)


class TestHashPassword:
    def test_round_trip(self) -> None:
        h = hash_password("S3cure!Password")
        assert verify_password("S3cure!Password", h) is True

    def test_wrong_password_fails(self) -> None:
        h = hash_password("correct-horse")
        assert verify_password("wrong-password", h) is False

    def test_hash_is_salted_and_unique(self) -> None:
        h1 = hash_password("same-password")
        h2 = hash_password("same-password")
        assert h1 != h2
        assert verify_password("same-password", h1) is True
        assert verify_password("same-password", h2) is True

    def test_hash_starts_with_argon2id_phc(self) -> None:
        h = hash_password("password")
        assert h.startswith("$argon2id$")

    def test_empty_password_rejected(self) -> None:
        with pytest.raises(ValueError):
            hash_password("")

    def test_non_string_password_rejected(self) -> None:
        with pytest.raises(ValueError):
            hash_password(None)  # type: ignore[arg-type]

    def test_hash_uses_configured_parameters(self) -> None:
        h = hash_password("password")
        # PHC format: $argon2id$v=19$m=65536,t=3,p=4$...
        parts = h.split("$")
        assert parts[1] == "argon2id"
        params = parts[3]
        assert "m=65536" in params
        assert "t=3" in params
        assert "p=4" in params


class TestVerifyPassword:
    def test_verify_garbage_hash_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            verify_password("pw", "not-a-valid-hash")

    def test_verify_empty_password_returns_false(self) -> None:
        h = hash_password("password")
        assert verify_password("", h) is False

    def test_parameters_defaults_sane(self) -> None:
        assert PasswordParameters.TIME_COST == 3
        assert PasswordParameters.MEMORY_COST == 65536
        assert PasswordParameters.PARALLELISM == 4
        assert PasswordParameters.HASH_LEN == 32
        assert PasswordParameters.SALT_LEN == 16


class TestNeedsRehash:
    def test_fresh_hash_does_not_need_rehash(self) -> None:
        h = hash_password("password")
        assert password_needs_rehash(h) is False

    def test_invalid_hash_needs_rehash(self) -> None:
        assert password_needs_rehash("garbage") is True
