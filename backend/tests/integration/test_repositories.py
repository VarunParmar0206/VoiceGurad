"""Integration tests for the repository (data-access) layer.

Uses an in-memory SQLite database with foreign-key enforcement, which is
isolated per test and requires no external database service.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from voiceguard.models import (
    AuditLog,
    AuthAttempt,
    Challenge,
    Session,
    Transaction,
    User,
    VoiceModel,
    VoiceTemplate,
)
from voiceguard.repositories import (
    AuditLogRepository,
    AuthAttemptRepository,
    ChallengeRepository,
    SessionRepository,
    TransactionRepository,
    UserRepository,
    VoiceModelRepository,
    VoiceTemplateRepository,
)

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_user(username: str = "alice", email: str = "alice@example.com") -> User:
    return User(
        username=username,
        email=email,
        password_hash="hashed-password",
        display_name="Alice",
        is_active=True,
        is_locked=False,
    )


# ── User repository ──────────────────────────────────────────────────────


class TestUserRepository:
    async def test_create_and_get(self, session) -> None:
        repo = UserRepository(session)
        user = _make_user()
        await repo.add(user)
        fetched = await repo.get(user.id)
        assert fetched is not None
        assert fetched.username == "alice"
        assert fetched.email == "alice@example.com"

    async def test_get_missing_returns_none(self, session) -> None:
        repo = UserRepository(session)
        import uuid as uuid_mod

        assert await repo.get(uuid_mod.uuid4()) is None

    async def test_get_by_username_and_email(self, session) -> None:
        repo = UserRepository(session)
        user = _make_user()
        await repo.add(user)

        by_name = await repo.get_by_username("alice")
        assert by_name is not None and by_name.id == user.id

        by_email = await repo.get_by_email("alice@example.com")
        assert by_email is not None and by_email.id == user.id

    async def test_exists_checks(self, session) -> None:
        repo = UserRepository(session)
        await repo.add(_make_user())

        assert await repo.exists_username("alice") is True
        assert await repo.exists_username("bob") is False
        assert await repo.exists_email("alice@example.com") is True

    async def test_delete(self, session) -> None:
        repo = UserRepository(session)
        user = _make_user()
        await repo.add(user)
        assert await repo.delete_by_id(user.id) is True
        # The user removed from the store; expire cached state and re-query.
        session.expire_all()
        assert await repo.get(user.id) is None

    async def test_unique_username_conflict_raises(self, session) -> None:
        from sqlalchemy.exc import IntegrityError

        repo = UserRepository(session)
        await repo.add(_make_user("alice", "x@x.com"))
        await session.commit()

        repo2 = UserRepository(session)
        with pytest.raises(IntegrityError):
            await repo2.add(_make_user("alice", "y@y.com"))
            await session.commit()


# ── VoiceTemplate / VoiceModel repo ──────────────────────────────────────


class TestVoiceRepository:
    async def test_voice_template_crud(self, session) -> None:
        user_repo = UserRepository(session)
        user = _make_user()
        await user_repo.add(user)

        vt_repo = VoiceTemplateRepository(session)
        vt = VoiceTemplate(
            user_id=user.id,
            model_version="v1.0",
            template_data=b"\x01\x02\x03",
            enrollment_samples=5,
            quality_scores={"avg": 0.8},
            salt=b"salt",
            is_active=True,
        )
        await vt_repo.add(vt)

        found = await vt_repo.get_active_for_user(user.id)
        assert found is not None and found.id == vt.id

    async def test_only_active_template_returned(self, session) -> None:
        user_repo = UserRepository(session)
        user = _make_user()
        await user_repo.add(user)

        vt_repo = VoiceTemplateRepository(session)
        await vt_repo.add(
            VoiceTemplate(
                user_id=user.id,
                model_version="v1.0",
                template_data=b"inactive",
                enrollment_samples=5,
                salt=b"s1",
                is_active=False,
            )
        )
        active = await vt_repo.add(
            VoiceTemplate(
                user_id=user.id,
                model_version="v1.1",
                template_data=b"active",
                enrollment_samples=5,
                salt=b"s2",
                is_active=True,
            )
        )

        found = await vt_repo.get_active_for_user(user.id)
        assert found is not None and found.id == active.id

    async def test_deactivate_for_user(self, session) -> None:
        user_repo = UserRepository(session)
        user = _make_user()
        await user_repo.add(user)

        vt_repo = VoiceTemplateRepository(session)
        await vt_repo.add(
            VoiceTemplate(
                user_id=user.id,
                model_version="v1.0",
                template_data=b"a",
                enrollment_samples=5,
                salt=b"s1",
                is_active=True,
            )
        )
        await vt_repo.deactivate_for_user(user.id)

        found = await vt_repo.get_active_for_user(user.id)
        assert found is None

    async def test_voice_model_linked_to_template(self, session) -> None:
        user_repo = UserRepository(session)
        user = _make_user()
        await user_repo.add(user)

        vt_repo = VoiceTemplateRepository(session)
        vt = await vt_repo.add(
            VoiceTemplate(
                user_id=user.id,
                model_version="v1.0",
                template_data=b"x",
                enrollment_samples=5,
                salt=b"s",
            )
        )

        vm_repo = VoiceModelRepository(session)
        vm = await vm_repo.add(
            VoiceModel(
                user_id=user.id,
                template_id=vt.id,
                model_type="gmm",
                model_data=b"params",
                parameters={"n": 8},
            )
        )

        by_type = await vm_repo.get_for_user(user.id, "gmm")
        assert by_type is not None and by_type.id == vm.id


# ── Session repository ───────────────────────────────────────────────────


class TestSessionRepository:
    async def test_create_and_fetch_by_token(self, session) -> None:
        user = _make_user()
        await UserRepository(session).add(user)

        repo = SessionRepository(session)
        s = await repo.add(
            Session(
                user_id=user.id,
                refresh_token="abc-token",
                user_agent="test-agent",
                ip_address="127.0.0.1",
                expires_at=datetime.now(UTC) + timedelta(hours=1),
            )
        )
        found = await repo.get_by_refresh_token("abc-token")
        assert found is not None and found.id == s.id

    async def test_expired_session_not_active(self, session) -> None:
        user = _make_user()
        await UserRepository(session).add(user)

        repo = SessionRepository(session)
        s = await repo.add(
            Session(
                user_id=user.id,
                refresh_token="expired",
                expires_at=datetime.now(UTC) - timedelta(hours=1),
            )
        )
        assert s.is_active is False

    async def test_revoke_and_count(self, session) -> None:
        user = _make_user()
        await UserRepository(session).add(user)

        repo = SessionRepository(session)
        await repo.add(
            Session(
                user_id=user.id,
                refresh_token="t1",
                expires_at=datetime.now(UTC) + timedelta(hours=1),
            )
        )
        await repo.add(
            Session(
                user_id=user.id,
                refresh_token="t2",
                expires_at=datetime.now(UTC) + timedelta(hours=1),
            )
        )

        assert await repo.count_active_for_user(user.id) == 2
        n = await repo.revoke_all_for_user(user.id)
        assert n == 2
        assert await repo.count_active_for_user(user.id) == 0


# ── Transaction repository ───────────────────────────────────────────────


class TestTransactionRepository:
    async def test_create_and_fetch_by_request_id(self, session) -> None:
        sender = _make_user("sender", "sender@x.com")
        await UserRepository(session).add(sender)

        repo = TransactionRepository(session)
        import uuid as uuid_mod

        request_id = uuid_mod.uuid4()
        tx = await repo.add(
            Transaction(
                sender_id=sender.id,
                recipient_name="Charlie",
                amount=100.50,
                status="completed",
                request_id=request_id,
            )
        )
        by_req = await repo.get_by_request_id(request_id)
        assert by_req is not None and by_req.id == tx.id

    async def test_request_id_idempotency_duplicate_raises(self, session) -> None:
        sender = _make_user("sender", "sender@x.com")
        await UserRepository(session).add(sender)

        repo = TransactionRepository(session)
        import uuid as uuid_mod

        request_id = uuid_mod.uuid4()
        await repo.add(
            Transaction(
                sender_id=sender.id,
                recipient_name="C",
                amount=10,
                status="completed",
                request_id=request_id,
            )
        )
        await session.commit()

        from sqlalchemy.exc import IntegrityError

        repo2 = TransactionRepository(session)
        with pytest.raises(IntegrityError):
            await repo2.add(
                Transaction(
                    sender_id=sender.id,
                    recipient_name="C",
                    amount=10,
                    status="completed",
                    request_id=request_id,
                )
            )
            await session.commit()

    async def test_sum_sent_since(self, session) -> None:
        sender = _make_user("sender", "sender@x.com")
        await UserRepository(session).add(sender)

        repo = TransactionRepository(session)
        await repo.add(
            Transaction(sender_id=sender.id, recipient_name="A", amount=50, status="completed")
        )
        await repo.add(
            Transaction(sender_id=sender.id, recipient_name="B", amount=30, status="completed")
        )
        await repo.add(
            Transaction(sender_id=sender.id, recipient_name="C", amount=20, status="declined")
        )

        total = await repo.sum_sent_since(sender.id, datetime.now(UTC) - timedelta(days=1))
        # Only completed transactions sum (50 + 30); declined excluded.
        assert total == 80


# ── Audit log repository ─────────────────────────────────────────────────


class TestAuditRepository:
    async def test_add_and_list(self, session) -> None:
        user = _make_user()
        await UserRepository(session).add(user)

        repo = AuditLogRepository(session)
        await repo.add(
            AuditLog(user_id=user.id, event_type="auth.login.success", event_detail={"ip": "x"})
        )
        await repo.add(AuditLog(user_id=user.id, event_type="transaction.created"))

        rows = await repo.list_by_user(user.id)
        assert len(rows) == 2

        by_type = await repo.list_by_event_type("transaction.created")
        assert len(by_type) == 1


# ── Challenge repository ─────────────────────────────────────────────────


class TestChallengeRepository:
    async def test_valid_unexpired(self, session) -> None:
        user = _make_user()
        await UserRepository(session).add(user)

        repo = ChallengeRepository(session)
        c = await repo.add(
            Challenge(
                user_id=user.id,
                challenge_text="verify 1234",
                challenge_type="number",
                expires_at=datetime.now(UTC) + timedelta(minutes=1),
            )
        )
        found = await repo.get_valid(c.id)
        assert found is not None

    async def test_used_challenge_not_returned(self, session) -> None:
        user = _make_user()
        await UserRepository(session).add(user)

        repo = ChallengeRepository(session)
        c = await repo.add(
            Challenge(
                user_id=user.id,
                challenge_text="verify 9999",
                challenge_type="number",
                expires_at=datetime.now(UTC) + timedelta(minutes=1),
            )
        )
        await repo.mark_used(c)
        found = await repo.get_valid(c.id)
        assert found is None

    async def test_expired_challenge_not_returned(self, session) -> None:
        user = _make_user()
        await UserRepository(session).add(user)

        repo = ChallengeRepository(session)
        c = await repo.add(
            Challenge(
                user_id=user.id,
                challenge_text="old",
                challenge_type="phrase",
                expires_at=datetime.now(UTC) - timedelta(minutes=1),
            )
        )
        found = await repo.get_valid(c.id)
        assert found is None


# ── Auth attempt repository ──────────────────────────────────────────────


class TestAuthAttemptRepository:
    async def test_count_failures(self, session) -> None:
        user = _make_user()
        await UserRepository(session).add(user)

        repo = AuthAttemptRepository(session)
        await repo.add(
            AuthAttempt(
                user_id=user.id, attempt_type="voice", success=False, failure_reason="low-score"
            )
        )
        await repo.add(
            AuthAttempt(
                user_id=user.id, attempt_type="voice", success=False, failure_reason="replay"
            )
        )
        await repo.add(AuthAttempt(user_id=user.id, attempt_type="voice", success=True))

        n = await repo.count_failures_since(user.id, datetime.now(UTC) - timedelta(hours=1))
        assert n == 2

    async def test_count_failures_by_ip(self, session) -> None:
        user = _make_user()
        await UserRepository(session).add(user)

        repo = AuthAttemptRepository(session)
        await repo.add(
            AuthAttempt(
                user_id=user.id, attempt_type="password", success=False, ip_address="10.0.0.1"
            )
        )
        await repo.add(
            AuthAttempt(
                user_id=user.id, attempt_type="password", success=False, ip_address="10.0.0.1"
            )
        )

        n = await repo.count_failures_for_ip_since(
            "10.0.0.1", datetime.now(UTC) - timedelta(hours=1)
        )
        assert n == 2
