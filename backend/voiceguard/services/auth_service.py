"""VoiceGuard V2 — Auth service.

Full authentication and session business logic for Phase 4:

- Registration (Argon2id password hashing).
- Two-step login: password verification (step 1) followed by a secondary
  factor (step 2).  The secondary factor is either voice verification
  (Phase 6 ML; a stub here) or TOTP backup codes (Phase 4).
- JWT access-token issuance + opaque refresh-token session creation.
- Refresh-token rotation and session revocation.
- Account lockout with escalating cooldown after repeated failures.
- Per-attempt audit logging (without logging secrets).

Design note
***********
Identity/session management is kept **separate** from voice biometric
verification.  This service never interprets audio or embeddings; voice
verification (Phase 6) sits behind the ``login_voice`` route and, when it
succeeds, the service simply issues tokens for the authenticated user.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from voiceguard.config import settings
from voiceguard.models.audit_log import AuditLog
from voiceguard.models.pending_login import PendingLogin
from voiceguard.models.user import User
from voiceguard.repositories import (
    AuditLogRepository,
    AuthAttemptRepository,
    PendingLoginRepository,
    SessionRepository,
    UserRepository,
)
from voiceguard.security.password import verify_password
from voiceguard.security.tokens import (
    create_access_token,
    generate_login_token,
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
from voiceguard.services.session_service import (
    InvalidRefreshTokenError,
    SessionError,
    SessionService,
)
from voiceguard.services.user_service import UserService


class AuthError(Exception):
    """Base exception for authentication failures."""


class InvalidCredentialsError(AuthError):
    """Username/email + password combination did not match."""


class AccountLockedError(AuthError):
    """Account is temporarily locked due to repeated failures. Carries
    the remaining cooldown in ``cooldown_seconds``."""

    def __init__(self, message: str, cooldown_seconds: int) -> None:
        super().__init__(message)
        self.cooldown_seconds = cooldown_seconds


class AccountInactiveError(AuthError):
    """Account exists but is not active."""


class TOTPNotEnabledError(AuthError):
    """TOTP is not set up for the account."""


class TOTPVerificationError(AuthError):
    """The provided TOTP code was invalid."""


class PendingLoginError(AuthError):
    """No valid, unexpired server-side login state for the attempt."""


class TokenIssueError(AuthError):
    """Tokens could not be issued (e.g. concurrent session limit)."""


@dataclass
class TokenPair:
    """Result of a successful token issuance."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60

    @property
    def refresh_token_hash(self) -> str:
        return hash_refresh_token(self.refresh_token)


class AuthService:
    """Encapsulates authentication and session business logic."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._user_repo = UserRepository(session)
        self._session_repo = SessionRepository(session)
        self._attempt_repo = AuthAttemptRepository(session)
        self._audit_repo = AuditLogRepository(session)
        self._pending_login_repo = PendingLoginRepository(session)
        self._user_service = UserService(session)
        self._session_service = SessionService(session)

    # ── Registration ──────────────────────────────────────────────────────

    async def register_user(
        self,
        username: str,
        email: str,
        password: str,
        display_name: str | None = None,
    ) -> uuid.UUID:
        """Register a new user with an Argon2id-hashed password."""
        if await self._user_service.username_exists(username):
            raise ValueError("Username already taken")
        if await self._user_service.email_exists(email):
            raise ValueError("Email already registered")

        from voiceguard.security.password import hash_password

        user = await self._user_service.create_user(
            username=username,
            email=email,
            password_hash=hash_password(password),
            display_name=display_name,
        )
        return user.id

    # ── Step 1: Password authentication (with lockout) ────────────────────

    async def authenticate_password(
        self,
        username_or_email: str,
        password: str,
        *,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> tuple[uuid.UUID, str]:
        """Verify credentials (step 1) and return ``(user_id, login_token)``.

        On success a short-lived, one-time server-side login state is created
        and its opaque token returned.  The secondary-factor step consumes
        this token to derive the target account, so it is never taken from
        client input.

        Applies account-lockout / escalating-cooldown and records every
        attempt to ``auth_attempts`` and the audit log.

        Raises:
            AccountLockedError: If the account is locked; carries the
                remaining cooldown.
            AccountInactiveError: If the account is disabled.
            InvalidCredentialsError: If the password is wrong or the user
                does not exist.
        """
        user = await self._user_repo.get_by_username(username_or_email)
        if user is None:
            user = await self._user_repo.get_by_email(username_or_email)

        if user is not None:
            if not user.is_active:
                raise AccountInactiveError("Account is not active.")
            if await self._is_locked(user):
                cooldown = await self._remaining_cooldown(user)
                raise AccountLockedError(
                    "Too many failed attempts. Try again later.",
                    cooldown_seconds=cooldown,
                )

        if user is None or not verify_password(password, user.password_hash):
            if user is not None:
                await self._record_failure(
                    user,
                    "password",
                    failure_reason="invalid_password",
                    ip_address=ip_address,
                    user_agent=user_agent,
                )
                await self._maybe_lock_account(user, ip_address, user_agent)
            else:
                # Do not reveal whether the account exists.
                await self._record_failure(
                    None,
                    "password",
                    failure_reason="unknown_user",
                    ip_address=ip_address,
                    user_agent=user_agent,
                )
            # Commit immediately so audit records and lockout state persist
            # even though the caller will receive an error response whose
            # session rollback would otherwise undo the flush.
            await self._session.commit()
            raise InvalidCredentialsError("Invalid username or password.")

        # A successful password step also verifies/clears lockout state.
        await self._record_success(
            user,
            "password",
            ip_address=ip_address,
            user_agent=user_agent,
        )
        login_token = await self._create_pending_login(
            user.id, ip_address=ip_address, user_agent=user_agent
        )
        return user.id, login_token

    async def _create_pending_login(
        self,
        user_id: uuid.UUID,
        *,
        ip_address: str | None,
        user_agent: str | None,
    ) -> str:
        """Create a one-time server-side login state; return its opaque token."""
        plaintext = generate_login_token()
        pending = PendingLogin(
            id=uuid.uuid4(),
            user_id=user_id,
            token_hash=hash_login_token(plaintext),
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=datetime.now(UTC)
            + timedelta(seconds=settings.PENDING_LOGIN_EXPIRE_SECONDS),
        )
        await self._pending_login_repo.add(pending)
        return plaintext

    async def _consume_pending_login(self, login_token: str) -> PendingLogin | None:
        """Resolve and consume a one-time login state to its user.

        Returns the ``PendingLogin`` row (marked used) if it is valid and
        unexpired, otherwise ``None``.
        """
        pending = await self._pending_login_repo.get_by_token_hash(
            hash_login_token(login_token)
        )
        if pending is None or not pending.is_valid:
            return None
        await self._pending_login_repo.mark_used(pending)
        return pending

    async def issue_token_pair(
        self,
        user_id: uuid.UUID,
        *,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> TokenPair:
        """Issue a JWT access token + create a refresh session."""
        try:
            session = await self._session_service.create_session(
                user_id,
                user_agent=user_agent,
                ip_address=ip_address,
            )
        except SessionError as exc:
            raise TokenIssueError(str(exc)) from exc

        access_token = create_access_token(user_id)
        # Pull the one-time plaintext refresh token off the created session.
        refresh_token = session.new_refresh_token  # type: ignore[attr-defined]
        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )

    async def refresh_token_pair(
        self,
        refresh_token: str,
        *,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> TokenPair:
        """Rotate a refresh token and issue a fresh token pair."""
        try:
            session = await self._session_service.rotate(
                refresh_token,
                user_agent=user_agent,
                ip_address=ip_address,
            )
        except InvalidRefreshTokenError as exc:
            raise AuthError(str(exc)) from exc
        except SessionError as exc:
            raise TokenIssueError(str(exc)) from exc

        access_token = create_access_token(session.user_id)
        refresh_token = session.new_refresh_token  # type: ignore[attr-defined]
        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )

    async def logout(self, refresh_token: str | None, user_id: uuid.UUID | None) -> None:
        """Revoke one session (if a token is given) or all for the user.

        If ``refresh_token`` is provided, only that session is revoked;
        otherwise every active session for the user is revoked.
        """
        if refresh_token:
            await self._session_service.revoke(refresh_token)
        else:
            if user_id is None:
                raise AuthError(
                    "User identification is required to revoke all sessions."
                )
            await self._session_service.revoke_all_for_user(user_id)

    # ── TOTP backup / secondary auth ──────────────────────────────────────

    async def setup_totp(self, user_id: uuid.UUID) -> tuple[str, str]:
        """Generate a TOTP secret and provisioning URI for a user.

        Returns ``(base32_secret, otpauth_uri)``.  The secret is the only
        opportunity to be shown as a QR / manual entry.
        """
        user = await self._user_service.get_by_id(user_id)
        if user is None:
            raise AuthError("User not found.")

        secret = generate_totp_secret()
        await self._user_repo.set_totp_secret(user, encrypt_totp_secret(secret))
        uri = provisioning_uri(secret, f"{user.username}@{settings.TOTP_ISSUER}")
        return secret, uri

    async def confirm_totp(self, user_id: uuid.UUID, code: str) -> None:
        """Activate TOTP for a user after verifying their first code."""
        user = await self._user_service.get_by_id(user_id)
        if user is None:
            raise AuthError("User not found.")
        if user.totp_secret is None:
            raise TOTPNotEnabledError("TOTP setup has not been started.")
        secret = decrypt_totp_secret(user.totp_secret)
        if not verify_totp(secret, code):
            raise TOTPVerificationError("Invalid TOTP code.")
        await self._user_repo.set_totp_enabled(user, True)

    async def authenticate_totp(
        self,
        login_token: str,
        code: str,
        *,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> TokenPair:
        """Authenticate with TOTP as the secondary factor and issue tokens.

        The target user is derived **server-side** from the one-time
        ``login_token`` issued by the password step.  It is never taken from
        a client-supplied user UUID, so a TOTP code cannot be paired with an
        arbitrary account without first completing that account's password
        step.
        """
        pending = await self._consume_pending_login(login_token)
        if pending is None:
            await self._audit(
                None,
                "auth.login.failure",
                ip_address,
                user_agent,
                detail={"reason": "invalid_pending_login"},
            )
            raise PendingLoginError("Login state is invalid or expired.")
        user = await self._user_service.get_by_id(pending.user_id)
        if user is None:
            raise InvalidCredentialsError("Invalid user.")
        if not user.totp_enabled or user.totp_secret is None:
            raise TOTPNotEnabledError("TOTP is not enabled for this account.")
        secret = decrypt_totp_secret(user.totp_secret)
        if not verify_totp(secret, code):
            await self._record_failure(
                user, "mfa", failure_reason="invalid_totp",
                ip_address=ip_address, user_agent=user_agent,
            )
            raise TOTPVerificationError("Invalid TOTP code.")
        await self._record_success(
            user, "mfa", ip_address=ip_address, user_agent=user_agent
        )
        return await self.issue_token_pair(
            user.id, user_agent=user_agent, ip_address=ip_address
        )

    # ── Lockout / cooldown ────────────────────────────────────────────────

    async def _is_locked(self, user: User) -> bool:
        return bool(user.is_locked)

    async def _failed_attempts(self, user: User) -> int:
        since = datetime.now(UTC) - timedelta(
            seconds=settings.LOCKOUT_COOLDOWN_MAX_SECONDS
        )
        return await self._attempt_repo.count_failures_since(user.id, since)

    async def _remaining_cooldown(self, user: User) -> int:
        """Remaining cooldown seconds based on recent consecutive failures."""
        failures = await self._failed_attempts(user)
        if failures < settings.MAX_FAILED_ATTEMPTS:
            return 0
        # Escalation multipliers -> 30s, 60s, 300s.
        stages = [1, 2, 10]
        idx = min(failures - settings.MAX_FAILED_ATTEMPTS, len(stages) - 1)
        cooldown = settings.LOCKOUT_COOLDOWN_BASE_SECONDS * stages[idx]
        return min(cooldown, settings.LOCKOUT_COOLDOWN_MAX_SECONDS)

    async def _maybe_lock_account(
        self, user: User, ip_address: str | None, user_agent: str | None
    ) -> None:
        failures = await self._failed_attempts(user)
        if failures >= settings.MAX_FAILED_ATTEMPTS:
            await self._user_repo.set_locked(user, True)
            await self._audit(
                user.id, "auth.lockout.activated", ip_address, user_agent
            )

    # ── Attempt + audit recording ─────────────────────────────────────────

    async def _record_failure(
        self,
        user: User | None,
        attempt_type: str,
        *,
        failure_reason: str,
        ip_address: str | None,
        user_agent: str | None,
    ) -> None:
        from voiceguard.models.auth_attempt import AuthAttempt

        attempt = AuthAttempt(
            user_id=user.id if user is not None else None,
            attempt_type=attempt_type,
            success=False,
            failure_reason=failure_reason,
            ip_address=ip_address,
        )
        self._session.add(attempt)
        await self._session.flush()
        await self._audit(
            user.id if user is not None else None,
            "auth.login.failure",
            ip_address,
            user_agent,
            detail={"reason": failure_reason},
        )

    async def _record_success(
        self,
        user: User,
        attempt_type: str,
        *,
        ip_address: str | None,
        user_agent: str | None,
    ) -> None:
        from voiceguard.models.auth_attempt import AuthAttempt

        attempt = AuthAttempt(
            user_id=user.id,
            attempt_type=attempt_type,
            success=True,
            failure_reason=None,
            ip_address=ip_address,
        )
        self._session.add(attempt)
        await self._session.flush()
        # A successful password step clears the lockout flag.
        await self._user_repo.set_locked(user, False)
        await self._audit(
            user.id, "auth.login.success", ip_address, user_agent
        )

    async def _audit(
        self,
        user_id: uuid.UUID | None,
        event_type: str,
        ip_address: str | None,
        user_agent: str | None,
        *,
        detail: dict[str, object] | None = None,
    ) -> None:
        """Append an audit-log row.  Never logs secrets."""
        entry = AuditLog(
            user_id=user_id,
            event_type=event_type,
            event_detail=detail,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        self._session.add(entry)
        await self._session.flush()
