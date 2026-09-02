"""Tests for the FastAPI application factory and middleware (Phase 3)."""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from voiceguard.config import settings
from voiceguard.main import app


def _client() -> TestClient:
    return TestClient(app)


class TestAppStartup:
    def test_openapi_docs_available(self) -> None:
        with _client() as c:
            resp = c.get("/docs")
            assert resp.status_code == 200

    def test_openapi_schema(self) -> None:
        with _client() as c:
            resp = c.get("/openapi.json")
            assert resp.status_code == 200
            assert resp.json()["info"]["title"] == settings.APP_NAME


class TestHealthEndpoint:
    def test_health_returns_200(self) -> None:
        with _client() as c:
            resp = c.get("/api/v1/health")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "ok"
            assert body["version"] == settings.APP_VERSION
            assert body["service"] == "voiceguard-api"

    def test_health_ready_endpoint(self) -> None:
        with _client() as c:
            resp = c.get("/api/v1/health/ready")
            assert resp.status_code == 200
            body = resp.json()
            assert "checks" in body
            assert isinstance(body["checks"], dict)


class TestRequestID:
    def test_request_id_generated(self) -> None:
        with _client() as c:
            resp = c.get("/api/v1/health")
            assert resp.headers.get("X-Request-ID") is not None

    def test_request_id_propagated(self) -> None:
        rid = str(uuid.uuid4())
        with _client() as c:
            resp = c.get("/api/v1/health", headers={"X-Request-ID": rid})
            assert resp.headers["X-Request-ID"] == rid

    def test_invalid_request_id_replaced(self) -> None:
        with _client() as c:
            resp = c.get("/api/v1/health", headers={"X-Request-ID": "not-a-uuid"})
            assert resp.headers["X-Request-ID"] is not None
            # Should be a valid UUID
            uuid.UUID(resp.headers["X-Request-ID"])


class TestValidationErrors:
    def test_invalid_email_returns_422(self) -> None:
        with _client() as c:
            resp = c.post(
                "/api/v1/auth/register",
                json={
                    "username": "jane_doe",
                    "email": "not-an-email",
                    "password": "securepass123",
                },
            )
            assert resp.status_code == 422
            body = resp.json()
            assert body["error"] == "validation_error"
            assert "field" in body

    def test_missing_field_returns_422(self) -> None:
        with _client() as c:
            resp = c.post(
                "/api/v1/auth/register",
                json={"username": "jane_doe"},
            )
            assert resp.status_code == 422
            body = resp.json()
            assert body["error"] == "validation_error"
            assert body["field"] is not None


class TestErrorFormat:
    def test_error_response_structure(self) -> None:
        """Ensure all error responses have {error, detail, field} structure."""
        with _client() as c:
            # 401 from a protected route
            resp = c.get("/api/v1/users/me")
            assert resp.status_code == 401
            body = resp.json()
            assert "error" in body
            assert "detail" in body
            assert "field" in body
