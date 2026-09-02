"""Integration tests for auth registration with a real (SQLite) DB.

Uses the ``client`` fixture from ``conftest.py`` which overrides the
DB session dependency with an in-memory SQLite database.
"""

from __future__ import annotations

from voiceguard.repositories import UserRepository


class TestRegisterIntegration:
    async def test_register_creates_user(self, client, session_factory) -> None:
        """POST /api/v1/auth/register creates a user in the database."""
        resp = client.post(
            "/api/v1/auth/register",
            json={
                "username": "jane_doe",
                "email": "jane@example.com",
                "password": "securepass123",
                "display_name": "Jane",
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert "user_id" in body
        assert body["username"] == "jane_doe"

        # Verify the user exists in the DB
        async with session_factory() as s:
            repo = UserRepository(s)
            user = await repo.get_by_username("jane_doe")
            assert user is not None
            assert user.email == "jane@example.com"
            assert user.balance == __import__("decimal").Decimal("10000.00")
            assert user.is_active is True

    async def test_register_duplicate_username_conflict(
        self, client, session_factory
    ) -> None:
        """Duplicate username returns 409."""
        payload = {
            "username": "dup_user",
            "email": "dup@example.com",
            "password": "securepass123",
        }
        first = client.post("/api/v1/auth/register", json=payload)
        assert first.status_code == 201

        second = client.post("/api/v1/auth/register", json=payload)
        assert second.status_code == 409
        body = second.json()
        assert body["error"] in ("conflict", "request_error")
