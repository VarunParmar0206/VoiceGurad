"""Tests for the rate limiter (Phase 3)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from voiceguard.security.rate_limit import (
    RateLimitResult,
    check_rate_limit,
    make_rate_limit_key,
)


class TestRateLimitKey:
    def test_key_format(self) -> None:
        assert make_rate_limit_key("login", "127.0.0.1") == "ratelimit:login:127.0.0.1"

    def test_key_with_uuid(self) -> None:
        import uuid

        uid = uuid.uuid4()
        assert make_rate_limit_key("voice", str(uid)) == f"ratelimit:voice:{uid}"


class TestRateLimitResult:
    def test_allowed_default(self) -> None:
        r = RateLimitResult(allowed=True, limit=5, remaining=4)
        assert r.allowed is True
        assert r.retry_after_seconds is None

    def test_blocked_with_retry(self) -> None:
        r = RateLimitResult(
            allowed=False, limit=5, remaining=0, retry_after_seconds=30
        )
        assert r.allowed is False
        assert r.retry_after_seconds == 30


class TestCheckRateLimit:
    """Test rate limiter counter and blocking behaviour with mocked Redis."""

    async def test_first_request_allowed(self) -> None:
        redis = AsyncMock()
        redis.incr = AsyncMock(return_value=1)
        redis.expire = AsyncMock(return_value=True)
        redis.ttl = AsyncMock(return_value=60)

        with patch("voiceguard.security.rate_limit.get_redis", return_value=redis):
            result = await check_rate_limit(
                key="ratelimit:test:user",
                limit=5,
                window_seconds=60,
            )
        assert result.allowed is True
        assert result.remaining == 4
        redis.incr.assert_awaited_once()
        redis.expire.assert_awaited_once_with("ratelimit:test:user", 60)

    async def test_blocks_after_threshold(self) -> None:
        redis = AsyncMock()
        redis.incr = AsyncMock(return_value=6)
        redis.expire = AsyncMock(return_value=True)
        redis.ttl = AsyncMock(return_value=30)

        with patch("voiceguard.security.rate_limit.get_redis", return_value=redis):
            result = await check_rate_limit(
                key="ratelimit:test:user",
                limit=5,
                window_seconds=60,
            )
        assert result.allowed is False
        assert result.remaining == 0
        assert result.retry_after_seconds == 30

    async def test_fail_open_when_redis_unavailable(self) -> None:
        """When Redis is unreachable, requests should pass through."""
        redis = AsyncMock()
        redis.incr = AsyncMock(side_effect=Exception("connection refused"))

        with patch("voiceguard.security.rate_limit.get_redis", return_value=redis):
            result = await check_rate_limit(
                key="ratelimit:test:unavailable",
                limit=1,
                window_seconds=60,
            )
        assert result.allowed is True
