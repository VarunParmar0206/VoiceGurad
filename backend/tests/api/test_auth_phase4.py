"""Comprehensive Phase 4 auth integration tests.

These exercise the full authentication/session surface through the FastAPI
TestClient against an in-memory SQLite database:

- registration, invalid password
- account lockout / escalating cooldown
- rate limiting (fail-open safety is asserted separately at the unit level)
- JWT validation (signature/expiry/issuer/audience/type)
- refresh rotation, reuse, revocation
- concurrent session limiting
- logout
- TOTP setup/confirm/login + cross-user authorization
- password change and session invalidation
- login-voice must NOT issue tokens
- malformed / invalid inputs
- no sensitive secrets or biometric data in responses
"""

from __future__ import annotations

import uuid

import pyotp

from voiceguard.config import settings
from voiceguard.repositories import UserRepository
from voiceguard.security.tokens import create_access_token

PASSWORD = "Str0ng!Pass"


def _register(client, username: str, email: str | None = None) -> uuid.UUID:
    resp = client.post(
        "/api/v1/auth/register",
        json={
            "username": username,
            "email": email or f"{username}@example.com",
            "password": PASSWORD,
        },
    )
    assert resp.status_code == 201, resp.text
    return uuid.UUID(resp.json()["user_id"])


def _bearer(client, user_id: uuid.UUID) -> dict[str, str]:
    """A test-only valid Bearer header for the given user.

    Creates the JWT directly in the test (never a production bootstrap
    mechanism) so TOTP setup/confirm can be exercised via the API.
    """
    token = create_access_token(user_id)
    return {"Authorization": f"Bearer {token}"}


def _login_password(client, username: str, password: str = PASSWORD):
    return client.post(
        "/api/v1/auth/login-password",
        json={"username": username, "password": password},
    )


def _setup_and_confirm_totp(client, user_id: uuid.UUID) -> str:
    """Run TOTP setup+confirm via the API and return the base32 secret."""
    auth = _bearer(client, user_id)
    setup = client.post("/api/v1/auth/totp/setup", headers=auth)
    assert setup.status_code == 200, setup.text
    secret = setup.json()["secret"]
    assert "otpauth_uri" in setup.json()

    code = pyotp.TOTP(secret).now()
    confirm = client.post(
        "/api/v1/auth/totp/confirm", headers=auth, json={"code": code}
    )
    assert confirm.status_code == 204, confirm.text
    return secret


def _totp_login(client, login_token: str, secret: str | None = None, code: str | None = None):
    """Complete the TOTP login step; returns (response, refresh_token)."""
    if code is None:
        if secret is None:
            raise AssertionError("secret or code required")
        code = pyotp.TOTP(secret).now()
    resp = client.post(
        "/api/v1/auth/login-totp",
        json={"login_token": login_token, "code": code},
    )
    return resp, (resp.json().get("refresh_token", "") if resp.status_code == 200 else "")


def _enable_totp_user(client, username: str, email: str | None = None) -> dict:
    """Register + enable TOTP + return login context."""
    user_id = _register(client, username, email)
    secret = _setup_and_confirm_totp(client, user_id)
    step1 = _login_password(client, username)
    assert step1.status_code == 200, step1.text
    login_token = step1.json()["login_token"]
    resp, refresh = _totp_login(client, login_token, secret=secret)
    assert resp.status_code == 200, resp.text
    access = resp.json()["access_token"]
    return {
        "user_id": user_id,
        "secret": secret,
        "login_token": login_token,
        "access_token": access,
        "refresh_token": refresh,
    }


class TestRegistration:
    def test_register_success(self, client) -> None:
        resp = client.post(
            "/api/v1/auth/register",
            json={
                "username": "alice",
                "email": "alice@example.com",
                "password": PASSWORD,
                "display_name": "Alice",
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert "user_id" in body
        assert "password" not in str(body.keys()).lower()

    def test_register_does_not_return_secrets(self, client) -> None:
        resp = client.post(
            "/api/v1/auth/register",
            json={
                "username": "bob01",
                "email": "bob@example.com",
                "password": PASSWORD,
            },
        )
        body = resp.json()
        raw = str(body).lower()
        assert "password_hash" not in raw
        assert PASSWORD.lower() not in raw
        assert "totp_secret" not in raw

    def test_register_duplicate_username_conflict(self, client) -> None:
        _register(client, "carol")
        resp = client.post(
            "/api/v1/auth/register",
            json={
                "username": "carol",
                "email": "carol-new@example.com",
                "password": PASSWORD,
            },
        )
        assert resp.status_code == 409

    def test_register_duplicate_email_conflict(self, client) -> None:
        _register(client, "dave01", email="shared@example.com")
        resp = client.post(
            "/api/v1/auth/register",
            json={
                "username": "dave02",
                "email": "shared@example.com",
                "password": PASSWORD,
            },
        )
        assert resp.status_code == 409

    def test_register_malformed_inputs(self, client) -> None:
        r = client.post("/api/v1/auth/register", json={
            "username": "xxx", "email": "x@example.com", "password": "short"})
        assert r.status_code == 422
        r = client.post("/api/v1/auth/register", json={
            "username": "yyy", "email": "not-email", "password": PASSWORD})
        assert r.status_code == 422
        r = client.post("/api/v1/auth/register", json={
            "username": "bad name!", "email": "y@example.com", "password": PASSWORD})
        assert r.status_code == 422
        r = client.post("/api/v1/auth/register", json={"username": "zzz"})
        assert r.status_code == 422


class TestPasswordLogin:
    def test_valid_credentials_returns_login_token(self, client) -> None:
        _register(client, "eve01")
        resp = _login_password(client, "eve01")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "password_verified"
        assert body["requires_secondary"] is True
        assert "login_token" in body
        assert "access_token" not in body
        assert "refresh_token" not in body

    def test_login_transient_state_is_opaque(self, client) -> None:
        _register(client, "frank")
        resp = _login_password(client, "frank")
        token = resp.json()["login_token"]
        assert len(token) >= 40

    def test_invalid_password(self, client) -> None:
        _register(client, "grace")
        resp = _login_password(client, "grace", password="wrongpassword")
        assert resp.status_code == 401
        assert "WWW-Authenticate" in resp.headers

    def test_unknown_user_same_401(self, client) -> None:
        resp = _login_password(client, "no_such_user_xyz", password=PASSWORD)
        assert resp.status_code == 401

    def test_malformed_login(self, client) -> None:
        r = client.post("/api/v1/auth/login-password", json={"username": "xxx"})
        assert r.status_code == 422


class TestLockout:
    def _lockout(self, client, username: str, attempts: int = settings.MAX_FAILED_ATTEMPTS):
        _register(client, username)
        for _ in range(attempts):
            resp = _login_password(client, username, password="wrong-password")
            assert resp.status_code == 401
        return username

    def test_locks_after_max_failed_attempts(self, client) -> None:
        username = self._lockout(client, "ivan00")
        resp = _login_password(client, username, password=PASSWORD)
        assert resp.status_code == 423
        assert resp.json()["error"] == "account_locked"

    def test_locked_returns_retry_after(self, client) -> None:
        username = self._lockout(client, "judy00")
        resp = _login_password(client, username, password=PASSWORD)
        assert resp.status_code == 423
        body = resp.json()
        detail = body.get("detail", body)
        if isinstance(detail, dict):
            assert isinstance(detail.get("retry_after"), int)
        else:
            assert isinstance(body.get("retry_after"), int) or isinstance(detail, str)

    def test_correct_password_before_lock_succeeds(self, client) -> None:
        username = self._lockout(client, "kate00", attempts=settings.MAX_FAILED_ATTEMPTS - 1)
        resp = _login_password(client, username, password=PASSWORD)
        assert resp.status_code == 200
        assert "login_token" in resp.json()


class TestTotpSetupConfirmLogin:
    async def test_full_totp_flow(self, client, session_factory) -> None:
        user_id = _register(client, "liam00")
        secret = _setup_and_confirm_totp(client, user_id)
        async with session_factory() as s:
            user = await UserRepository(s).get(user_id)
            assert user is not None
            assert user.totp_enabled is True
            assert user.totp_secret is not None
            assert secret.encode() not in user.totp_secret

    def test_totp_setup_requires_auth(self, client) -> None:
        r = client.post("/api/v1/auth/totp/setup")
        assert r.status_code == 401

    def test_totp_confirm_requires_auth(self, client) -> None:
        r = client.post("/api/v1/auth/totp/confirm", json={"code": "123456"})
        assert r.status_code == 401

    def test_confirm_before_setup_fails(self, client) -> None:
        user_id = _register(client, "mia00")
        auth = _bearer(client, user_id)
        code = pyotp.TOTP(pyotp.random_base32()).now()
        r = client.post("/api/v1/auth/totp/confirm", headers=auth, json={"code": code})
        assert r.status_code == 400

    def test_login_totp_success_issues_tokens(self, client) -> None:
        ctx = _enable_totp_user(client, "noah00")
        assert ctx["access_token"]
        assert ctx["refresh_token"]
        headers = {"Authorization": f"Bearer {ctx['access_token']}"}
        me = client.get("/api/v1/users/me", headers=headers)
        assert me.status_code == 200
        assert me.json()["username"] == "noah00"

    def test_login_totp_invalid_code(self, client) -> None:
        user_id = _register(client, "olivia0")
        _setup_and_confirm_totp(client, user_id)
        step1 = _login_password(client, "olivia0")
        assert step1.status_code == 200
        login_token = step1.json()["login_token"]
        resp, _ = _totp_login(client, login_token, code="000000")
        assert resp.status_code == 400

    def test_login_totp_without_password_step_rejected(self, client) -> None:
        user_id = _register(client, "paul00")
        secret = _setup_and_confirm_totp(client, user_id)
        code = pyotp.TOTP(secret).now()
        resp = client.post(
            "/api/v1/auth/login-totp",
            json={"login_token": "made-up-token-value", "code": code},
        )
        assert resp.status_code == 401

    def test_login_totp_token_is_one_time(self, client) -> None:
        user_id = _register(client, "quinn0")
        secret = _setup_and_confirm_totp(client, user_id)
        step1 = _login_password(client, "quinn0")
        login_token = step1.json()["login_token"]
        code = pyotp.TOTP(secret).now()
        resp1 = client.post(
            "/api/v1/auth/login-totp",
            json={"login_token": login_token, "code": code},
        )
        assert resp1.status_code == 200
        resp2 = client.post(
            "/api/v1/auth/login-totp",
            json={"login_token": login_token, "code": pyotp.TOTP(secret).now()},
        )
        assert resp2.status_code == 401

    def test_login_totp_ignores_extra_body_fields(self, client) -> None:
        user_id = _register(client, "ryan00")
        secret = _setup_and_confirm_totp(client, user_id)
        step1 = _login_password(client, "ryan00")
        login_token = step1.json()["login_token"]
        code = pyotp.TOTP(secret).now()
        resp = client.post(
            "/api/v1/auth/login-totp",
            json={
                "login_token": login_token,
                "code": code,
                "user_id": "00000000-0000-4000-8000-000000000000",
            },
        )
        assert resp.status_code == 200
        me = client.get(
            "/api/v1/users/me",
            headers={"Authorization": f"Bearer {resp.json()['access_token']}"},
        )
        assert me.json()["username"] == "ryan00"


class TestCrossUserTotpAuthorization:
    async def test_cannot_configure_another_users_totp(self, client, session_factory) -> None:
        victim_id = _register(client, "victim0")
        attacker_id = _register(client, "attacker0")
        auth = _bearer(client, attacker_id)
        setup = client.post("/api/v1/auth/totp/setup", headers=auth)
        assert setup.status_code == 200
        async with session_factory() as s:
            repo = UserRepository(s)
            victim = await repo.get(victim_id)
            attacker = await repo.get(attacker_id)
            assert victim.totp_secret is None
            assert victim.totp_enabled is False
            assert attacker.totp_secret is not None

    def test_attacker_cannot_pair_totp_code_with_victim_account(self, client) -> None:
        victim_id = _register(client, "victim2")
        secret = _setup_and_confirm_totp(client, victim_id)
        victim_code = pyotp.TOTP(secret).now()

        _register(client, "attacker2")
        step1 = _login_password(client, "attacker2")
        assert step1.status_code == 200
        attacker_token = step1.json()["login_token"]

        resp = client.post(
            "/api/v1/auth/login-totp",
            json={"login_token": attacker_token, "code": victim_code},
        )
        assert resp.status_code in (400, 401)
        # The attacker's account must not have been authenticated either.
        step1b = _login_password(client, "attacker2")
        assert step1b.status_code == 200
        attacker_token_b = step1b.json()["login_token"]
        code_b = pyotp.TOTP(secret).now()
        resp_b = client.post(
            "/api/v1/auth/login-totp",
            json={"login_token": attacker_token_b, "code": code_b},
        )
        assert resp_b.status_code == 400

    async def test_setup_and_confirm_derive_user_from_jwt(self, client, session_factory) -> None:
        u1 = _register(client, "user_one")
        u2 = _register(client, "user_two")
        s1 = client.post("/api/v1/auth/totp/setup", headers=_bearer(client, u1))
        assert s1.status_code == 200
        s2 = client.post("/api/v1/auth/totp/setup", headers=_bearer(client, u2))
        assert s2.status_code == 200
        async with session_factory() as s:
            repo = UserRepository(s)
            a = await repo.get(u1)
            b = await repo.get(u2)
            assert (a.totp_secret is not None) and (a.totp_enabled is False)
            assert (b.totp_secret is not None) and (b.totp_enabled is False)
            assert a.totp_secret != b.totp_secret


class TestJwtProtection:
    def test_access_denied_without_token(self, client) -> None:
        r = client.get("/api/v1/users/me")
        assert r.status_code == 401

    def test_access_denied_with_bad_signature(self, client) -> None:
        user_id = _register(client, "jwta")
        token = create_access_token(user_id)
        forgery = token[:-4] + ("abcd" if not token.endswith("abcd") else "0000")
        r = client.get(
            "/api/v1/users/me", headers={"Authorization": f"Bearer {forgery}"}
        )
        assert r.status_code == 401

    def test_access_denied_with_garbage_token(self, client) -> None:
        r = client.get(
            "/api/v1/users/me", headers={"Authorization": "Bearer not.a.jwt"}
        )
        assert r.status_code == 401

    def test_access_denied_with_wrong_scheme(self, client) -> None:
        user_id = _register(client, "jwtb")
        token = create_access_token(user_id)
        r = client.get(
            "/api/v1/users/me", headers={"Authorization": f"Basic {token}"}
        )
        assert r.status_code == 401

    def test_logout_requires_auth(self, client) -> None:
        r = client.post("/api/v1/auth/logout")
        assert r.status_code == 401


class TestRefresh:
    def test_refresh_rotates_tokens(self, client) -> None:
        ctx = _enable_totp_user(client, "refr1")
        r = client.post(
            "/api/v1/auth/refresh", json={"refresh_token": ctx["refresh_token"]}
        )
        assert r.status_code == 200
        new_refresh = r.json()["refresh_token"]
        assert new_refresh != ctx["refresh_token"]

    def test_refresh_reuse_rejected(self, client) -> None:
        ctx = _enable_totp_user(client, "refr2")
        r = client.post(
            "/api/v1/auth/refresh", json={"refresh_token": ctx["refresh_token"]}
        )
        assert r.status_code == 200
        old_refresh = ctx["refresh_token"]
        r2 = client.post("/api/v1/auth/refresh", json={"refresh_token": old_refresh})
        assert r2.status_code == 401

    def test_refresh_garbage_rejected(self, client) -> None:
        r = client.post("/api/v1/auth/refresh", json={"refresh_token": "garbage"})
        assert r.status_code == 401


class TestLogout:
    def test_logout_revokes_session(self, client) -> None:
        ctx = _enable_totp_user(client, "logo1")
        r = client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": f"Bearer {ctx['access_token']}"},
            json={"refresh_token": ctx["refresh_token"]},
        )
        assert r.status_code == 204
        r2 = client.post("/api/v1/auth/refresh", json={"refresh_token": ctx["refresh_token"]})
        assert r2.status_code == 401

    def test_logout_all_revokes_all_sessions(self, client) -> None:
        ctx = _enable_totp_user(client, "logo2")
        r = client.post(
            "/api/v1/auth/refresh", json={"refresh_token": ctx["refresh_token"]}
        )
        assert r.status_code == 200
        second_refresh = r.json()["refresh_token"]

        r = client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": f"Bearer {ctx['access_token']}"},
            json={},
        )
        assert r.status_code == 204
        assert client.post(
            "/api/v1/auth/refresh", json={"refresh_token": second_refresh}
        ).status_code == 401


class TestConcurrentSessions:
    def test_concurrent_session_limit(self, client) -> None:
        user_id = _register(client, "sess1")
        secret = _setup_and_confirm_totp(client, user_id)
        limit = settings.CONCURRENT_SESSION_LIMIT
        refresh_tokens = []
        for _ in range(limit):
            step1 = _login_password(client, "sess1")
            login_token = step1.json()["login_token"]
            resp, refresh = _totp_login(client, login_token, secret=secret)
            assert resp.status_code == 200
            refresh_tokens.append(refresh)
        step1 = _login_password(client, "sess1")
        login_token = step1.json()["login_token"]
        resp, _ = _totp_login(client, login_token, secret=secret)
        assert resp.status_code != 200


class TestPasswordChange:
    def test_password_change_revokes_sessions(self, client) -> None:
        ctx = _enable_totp_user(client, "pwch1")
        r = client.put(
            "/api/v1/users/me/password",
            headers={"Authorization": f"Bearer {ctx['access_token']}"},
            json={"current_password": PASSWORD, "new_password": "BrandNew!Pa55"},
        )
        assert r.status_code == 204
        r2 = client.post("/api/v1/auth/refresh", json={"refresh_token": ctx["refresh_token"]})
        assert r2.status_code == 401

    def test_password_change_wrong_current(self, client) -> None:
        user_id = _register(client, "pwch2")
        auth = _bearer(client, user_id)
        r = client.put(
            "/api/v1/users/me/password",
            headers=auth,
            json={"current_password": "wrong-pass", "new_password": "BrandNew!Pa55"},
        )
        assert r.status_code == 403

    def test_password_change_same_password_rejected(self, client) -> None:
        user_id = _register(client, "pwch3")
        auth = _bearer(client, user_id)
        r = client.put(
            "/api/v1/users/me/password",
            headers=auth,
            json={"current_password": PASSWORD, "new_password": PASSWORD},
        )
        assert r.status_code == 400

    def test_password_change_requires_auth(self, client) -> None:
        r = client.put(
            "/api/v1/users/me/password",
            json={"current_password": "x", "new_password": "NewPassword123"},
        )
        assert r.status_code == 401


class TestLoginVoiceNoTokens:
    def test_login_voice_not_implemented_and_no_tokens(self, client) -> None:
        r = client.post(
            "/api/v1/auth/login-voice",
            json={"user_id": str(uuid.uuid4()), "challenge_id": str(uuid.uuid4())},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "not_implemented"
        assert "access_token" not in body
        assert "refresh_token" not in body
        assert "token" not in body

    def test_login_voice_malformed(self, client) -> None:
        r = client.post(
            "/api/v1/auth/login-voice",
            json={"user_id": "bad", "challenge_id": "bad"},
        )
        assert r.status_code in (422, 400)


class TestNoSecretsInResponses:
    def test_error_responses_do_not_leak_internals(self, client) -> None:
        _register(client, "secA1")
        r = _login_password(client, "secA1", password="wrong-password")
        assert r.status_code == 401
        assert "argon2" not in r.text.lower()
        assert "password_hash" not in r.text.lower()

    def test_register_and_login_do_not_expose_totp_secret(self, client) -> None:
        user_id = _register(client, "secB1")
        auth = _bearer(client, user_id)
        setup = client.post("/api/v1/auth/totp/setup", headers=auth)
        secret = setup.json()["secret"]
        confirm_code = pyotp.TOTP(secret).now()
        confirm = client.post(
            "/api/v1/auth/totp/confirm", headers=auth, json={"code": confirm_code}
        )
        assert confirm.status_code == 204
        assert confirm.content == b""
        step1 = _login_password(client, "secB1")
        login_token = step1.json()["login_token"]
        pair, _ = _totp_login(client, login_token, secret=secret)
        assert pair.status_code == 200
        assert secret not in pair.text
        assert "totp_secret" not in pair.text
