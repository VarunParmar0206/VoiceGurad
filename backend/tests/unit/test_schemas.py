"""Tests for Pydantic request/response schemas (Phase 3)."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from voiceguard.schemas.auth import (
    LoginPasswordRequest,
    LoginVoiceRequest,
    RegisterRequest,
)
from voiceguard.schemas.common import ErrorResponse
from voiceguard.schemas.health import HealthResponse, ReadyResponse
from voiceguard.schemas.transaction import (
    BalanceResponse,
    TransactionCreateRequest,
)
from voiceguard.schemas.user import (
    PasswordChangeRequest,
    UserUpdateRequest,
)
from voiceguard.schemas.voice import (
    VoiceEnrollResponse,
    VoiceStatusResponse,
    VoiceVerifyResponse,
)


class TestRegisterSchema:
    def test_valid_registration(self) -> None:
        req = RegisterRequest(
            username="jane_doe",
            email="jane@example.com",
            password="securepass123",
            display_name="Jane",
        )
        assert req.username == "jane_doe"
        assert req.email == "jane@example.com"

    def test_rejects_malformed_email(self) -> None:
        with pytest.raises(ValidationError):
            RegisterRequest(
                username="jane_doe",
                email="not-an-email",
                password="securepass123",
            )

    def test_rejects_short_username(self) -> None:
        with pytest.raises(ValidationError):
            RegisterRequest(
                username="ab",
                email="jane@example.com",
                password="securepass123",
            )

    def test_rejects_username_with_invalid_chars(self) -> None:
        with pytest.raises(ValidationError):
            RegisterRequest(
                username="bad-name!",
                email="jane@example.com",
                password="securepass123",
            )

    def test_rejects_short_password(self) -> None:
        with pytest.raises(ValidationError):
            RegisterRequest(
                username="jane_doe",
                email="jane@example.com",
                password="short",
            )

    def test_rejects_missing_fields(self) -> None:
        with pytest.raises(ValidationError):
            RegisterRequest(username="jane_doe")  # type: ignore[call-arg]


class TestLoginPasswordSchema:
    def test_valid(self) -> None:
        req = LoginPasswordRequest(username="jane", password="pass1234")
        assert req.username == "jane"

    def test_rejects_missing_password(self) -> None:
        with pytest.raises(ValidationError):
            LoginPasswordRequest(username="jane")  # type: ignore[call-arg]


class TestLoginVoiceSchema:
    def test_valid(self) -> None:
        import uuid

        req = LoginVoiceRequest(
            user_id="jane",
            challenge_id=uuid.uuid4(),
        )
        assert req.user_id == "jane"

    def test_rejects_invalid_challenge_id(self) -> None:
        with pytest.raises(ValidationError):
            LoginVoiceRequest(user_id="jane", challenge_id="not-a-uuid")


class TestTransactionSchema:
    def test_rejects_negative_amount(self) -> None:
        import uuid

        with pytest.raises(ValidationError):
            TransactionCreateRequest(
                recipient_name="Bob",
                amount=Decimal("-5.00"),
                request_id=uuid.uuid4(),
            )

    def test_rejects_zero_amount(self) -> None:
        import uuid

        with pytest.raises(ValidationError):
            TransactionCreateRequest(
                recipient_name="Bob",
                amount=Decimal("0.00"),
                request_id=uuid.uuid4(),
            )

    def test_valid_amount(self) -> None:
        import uuid

        req = TransactionCreateRequest(
            recipient_name="Bob",
            amount=Decimal("100.00"),
            request_id=uuid.uuid4(),
        )
        assert req.amount == Decimal("100.00")

    def test_rejects_empty_recipient(self) -> None:
        import uuid

        with pytest.raises(ValidationError):
            TransactionCreateRequest(
                recipient_name="",
                amount=Decimal("10.00"),
                request_id=uuid.uuid4(),
            )

    def test_rejects_too_many_decimal_places(self) -> None:
        import uuid

        with pytest.raises(ValidationError):
            TransactionCreateRequest(
                recipient_name="Bob",
                amount=Decimal("10.123"),
                request_id=uuid.uuid4(),
            )

    def test_request_id_defaults_to_uuid(self) -> None:
        import uuid

        req = TransactionCreateRequest(recipient_name="Bob", amount=Decimal("1.00"))
        assert isinstance(req.request_id, uuid.UUID)


class TestUserSchema:
    def test_valid_update(self) -> None:
        req = UserUpdateRequest(display_name="New Name")
        assert req.display_name == "New Name"

    def test_rejects_invalid_email(self) -> None:
        with pytest.raises(ValidationError):
            UserUpdateRequest(email="not-an-email")

    def test_password_change_rejects_short_new(self) -> None:
        with pytest.raises(ValidationError):
            PasswordChangeRequest(
                current_password="oldpass123",
                new_password="short",
            )


class TestBalanceSchema:
    def test_valid_balance(self) -> None:
        resp = BalanceResponse(
            balance=Decimal("500.00"),
            daily_limit=Decimal("50000.00"),
        )
        assert resp.balance == Decimal("500.00")


class TestHealthSchema:
    def test_health_response(self) -> None:
        resp = HealthResponse(version="2.0.0")
        assert resp.status == "ok"
        assert resp.service == "voiceguard-api"

    def test_ready_response(self) -> None:
        resp = ReadyResponse(
            status="ready",
            checks={"database": "ok", "redis": "ok"},
        )
        assert resp.checks["database"] == "ok"


class TestErrorSchema:
    def test_error_response(self) -> None:
        resp = ErrorResponse(
            error="validation_error",
            detail="Invalid input",
            field="email",
        )
        assert resp.error == "validation_error"


class TestVoiceSchema:
    def test_valid_enroll_response(self) -> None:
        import uuid

        resp = VoiceEnrollResponse(
            user_id=uuid.uuid4(),
            template_id=uuid.uuid4(),
            enrollment_samples=5,
        )
        assert resp.status == "enrolled"

    def test_valid_status_response(self) -> None:
        import uuid

        resp = VoiceStatusResponse(
            user_id=uuid.uuid4(),
            is_enrolled=True,
            enrollment_samples=5,
        )
        assert resp.is_enrolled is True

    def test_valid_verify_response(self) -> None:

        resp = VoiceVerifyResponse(
            status="verified",
            voice_score=0.95,
            threshold=0.82,
        )
        assert resp.voice_score == 0.95
