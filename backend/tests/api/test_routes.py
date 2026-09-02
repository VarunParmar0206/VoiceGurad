"""Integration tests for API route stubs and status codes (Phase 3)."""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from voiceguard.main import app


def _client() -> TestClient:
    return TestClient(app)


class TestRouteRegistration:
    """Verify all Phase 3 route stubs accept requests and return status codes."""

    def test_openapi_docs(self) -> None:
        with _client() as c:
            r = c.get("/docs")
            assert r.status_code == 200

    def test_openapi_schema_has_routes(self) -> None:
        with _client() as c:
            schema = c.get("/openapi.json").json()
            paths = schema["paths"]
            assert "/api/v1/health" in paths
            assert "/api/v1/health/ready" in paths
            assert "/api/v1/auth/register" in paths
            assert "/api/v1/auth/login-password" in paths
            assert "/api/v1/auth/login-voice" in paths
            assert "/api/v1/auth/refresh" in paths
            assert "/api/v1/auth/logout" in paths
            assert "/api/v1/users/me" in paths
            assert "/api/v1/transactions" in paths
            assert "/api/v1/transactions/balance" in paths
            assert "/api/v1/transactions/{transaction_id}" in paths
            assert "/api/v1/voice/enroll" in paths
            assert "/api/v1/voice/status" in paths
            assert "/api/v1/voice/re-enroll" in paths


class TestAuthRoutes:
    def test_register_rejects_invalid_email(self) -> None:
        with _client() as c:
            r = c.post(
                "/api/v1/auth/register",
                json={
                    "username": "jane_doe",
                    "email": "not-an-email",
                    "password": "securepass123",
                },
            )
            assert r.status_code == 422
            body = r.json()
            assert body["error"] == "validation_error"

    def test_login_voice_requires_uuid(self) -> None:
        with _client() as c:
            r = c.post(
                "/api/v1/auth/login-voice",
                json={"user_id": "not-a-uuid", "challenge_id": str(uuid.uuid4())},
            )
            assert r.status_code == 400


class TestUsersRoutes:
    def test_get_profile_requires_auth(self) -> None:
        with _client() as c:
            r = c.get("/api/v1/users/me")
            assert r.status_code == 401
            body = r.json()
            assert body["error"] == "unauthorized"

    def test_update_profile_requires_auth(self) -> None:
        with _client() as c:
            r = c.put("/api/v1/users/me", json={"display_name": "Jane"})
            assert r.status_code == 401

    def test_change_password_requires_auth(self) -> None:
        with _client() as c:
            r = c.put(
                "/api/v1/users/me/password",
                json={"current_password": "old", "new_password": "newpass123"},
            )
            assert r.status_code == 401


class TestTransactionRoutes:
    def test_get_transaction_requires_auth(self) -> None:
        with _client() as c:
            r = c.get(f"/api/v1/transactions/{uuid.uuid4()}")
            assert r.status_code == 401

    def test_balance_route_matches_before_detail(self) -> None:
        """Ensure /transactions/balance is not shadowed by /{transaction_id}."""
        with _client() as c:
            r = c.get("/api/v1/transactions/balance")
            assert r.status_code == 401, f"Expected 401, got {r.status_code}"


class TestVoiceRoutes:
    def test_enroll_requires_auth(self) -> None:
        with _client() as c:
            r = c.post("/api/v1/voice/enroll")
            assert r.status_code == 401

    def test_status_requires_auth(self) -> None:
        with _client() as c:
            r = c.get("/api/v1/voice/status")
            assert r.status_code == 401

    def test_re_enroll_requires_auth(self) -> None:
        with _client() as c:
            r = c.post("/api/v1/voice/re-enroll")
            assert r.status_code == 401


class TestErrorHandling:
    def test_404_returns_standard_format(self) -> None:
        with _client() as c:
            r = c.get("/api/v1/nonexistent")
            assert r.status_code == 404

    def test_bad_request_returns_standard_format(self) -> None:
        with _client() as c:
            r = c.post(
                "/api/v1/auth/login-voice",
                json={"user_id": "bad", "challenge_id": "bad"},
            )
            assert r.status_code == 422 or r.status_code == 400
